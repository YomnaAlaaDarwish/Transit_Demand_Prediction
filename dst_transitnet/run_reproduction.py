"""End-to-end DST-TransitNet reproduction experiment runner.

Trains DST-TransitNet, DST-TransitNetV2, and four baselines (FFNN, LSTM,
DLinear, simplified iTransformer) on the Bogota TransMilenio BRT dataset,
evaluates on the Normal / Protest / COVID test periods (Table 1), runs the
long-term iterative-forecasting comparison (Table 2), and saves all
checkpoints/metrics/predictions/logs under dst_transitnet/outputs/.
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dst_transitnet.config import DataConfig, ModelConfig, TrainConfig
from dst_transitnet.data import (load_station_series, build_adjacency, gcn_normalize,
                                  make_splits, build_windows, MinMaxScaler, scale_dataframe)
from dst_transitnet.models import DSTTransitNet, DSTTransitNetV2
from dst_transitnet.baselines import FFNNBaseline, LSTMBaseline, DLinearBaseline, SimpleITransformerBaseline
from dst_transitnet.train import train_model, evaluate, long_term_forecast, predict
from dst_transitnet.metrics import per_station_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_reproduction")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def build_model(name, cfg: DataConfig, mcfg: ModelConfig):
    if name == "dst_transitnet":
        return DSTTransitNet(cfg.hist_len, mcfg)
    if name == "dst_transitnet_v2":
        return DSTTransitNetV2(cfg.hist_len, mcfg)
    if name == "ffnn":
        return FFNNBaseline(cfg.recent_len)
    if name == "lstm":
        return LSTMBaseline()
    if name == "dlinear":
        return DLinearBaseline(cfg.recent_len)
    if name == "itransformer":
        return SimpleITransformerBaseline(cfg.recent_len, cfg.hist_len)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "dst_transitnet", "dst_transitnet_v2", "ffnn", "lstm", "dlinear", "itransformer"])
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--train_stride", type=int, default=None)
    parser.add_argument("--skip_long_term", action="store_true")
    parser.add_argument("--test_stride", type=int, default=1,
                        help="Evaluate on every Nth timestamp of each test period (compute budget control).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    dcfg = DataConfig()
    mcfg = ModelConfig()
    tcfg = TrainConfig()
    if args.max_epochs is not None:
        tcfg.max_epochs = args.max_epochs
    if args.train_stride is not None:
        tcfg.train_stride = args.train_stride

    torch.set_num_threads(os.cpu_count() or 4)

    logger.info("Loading Bogota BRT station series ...")
    df = load_station_series(dcfg)
    A = build_adjacency(dcfg, df.columns)
    A_norm = gcn_normalize(A)

    splits = make_splits(df, dcfg, val_fraction=tcfg.val_fraction)
    scaler = MinMaxScaler().fit(splits.train)
    full_scaled = scale_dataframe(df, scaler)

    logger.info("Building windowed samples ...")
    w_train = build_windows(full_scaled, splits.train.index, dcfg, stride=tcfg.train_stride)
    w_val = build_windows(full_scaled, splits.val.index, dcfg)

    test_periods = {
        "normal": splits.normal,
        "protest": splits.protest,
        "covid": splits.covid,
    }
    w_test = {name: build_windows(full_scaled, frame.index, dcfg, stride=args.test_stride)
              for name, frame in test_periods.items()}
    w_train_eval = build_windows(full_scaled, splits.train.index, dcfg, stride=max(1, args.test_stride))

    for name, w in {"train": w_train_eval, **w_test}.items():
        logger.info("%s windows: Xo=%s y=%s", name, w.Xo.shape, w.y.shape)

    results_table = {}
    per_station_results = {}
    timing = {}

    for model_name in args.models:
        logger.info("=== Training %s ===", model_name)
        model = build_model(model_name, dcfg, mcfg)
        n_params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        model, history = train_model(
            model, w_train, w_val, A_norm, tcfg,
            log_path=os.path.join(LOG_DIR, f"{model_name}_train_log.json"))
        train_time_min = (time.time() - t0) / 60.0

        ckpt_path = os.path.join(CKPT_DIR, f"{model_name}.pt")
        torch.save(model.state_dict(), ckpt_path)
        model_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        timing[model_name] = {"train_time_min": train_time_min, "n_params": n_params, "size_mb": model_size_mb}
        logger.info("%s: %d params, %.1f MB, trained in %.2f min",
                    model_name, n_params, model_size_mb, train_time_min)

        model_results = {}
        model_station_results = {}
        for period_name, w in {"train": w_train_eval, **w_test}.items():
            metrics, pred = evaluate(model, w, A_norm)
            model_results[period_name] = metrics
            r2_s, maape_s = per_station_scores(w.y, pred)
            model_station_results[period_name] = {"r2": r2_s.tolist(), "maape": maape_s.tolist()}
            np.save(os.path.join(OUT_DIR, f"{model_name}_{period_name}_pred.npy"), pred)
            np.save(os.path.join(OUT_DIR, f"{model_name}_{period_name}_true.npy"), w.y)
            logger.info("%s / %s: R2=%.4f MAAPE=%.4f", model_name, period_name, metrics["r2"], metrics["maape"])

        results_table[model_name] = model_results
        per_station_results[model_name] = model_station_results

        if not args.skip_long_term:
            logger.info("Long-term iterative forecast for %s ...", model_name)
            lt_ratio = {}
            for period_name, frame in test_periods.items():
                w_lt = build_windows(full_scaled, frame.index, dcfg, long_term=True,
                                      stride=max(1, args.test_stride * 4))
                if w_lt.Xo.shape[0] == 0:
                    continue
                preds = long_term_forecast(model, w_lt, A_norm, dcfg.decomposition_kernel,
                                            dcfg.long_term_max_lag, dcfg.hist_len)
                from dst_transitnet.metrics import maape as maape_fn
                maape_per_lag = [maape_fn(w_lt.y[:, lag, :], preds[:, lag, :])
                                  for lag in range(dcfg.long_term_max_lag)]
                ratio = maape_per_lag[-1] / maape_per_lag[0] if maape_per_lag[0] > 0 else float("nan")
                lt_ratio[period_name] = {"maape_per_lag": maape_per_lag, "ratio_12_vs_1": ratio}
                logger.info("%s / %s long-term MAAPE ratio (lag12/lag1) = %.4f",
                            model_name, period_name, ratio)
            with open(os.path.join(OUT_DIR, f"{model_name}_long_term.json"), "w") as f:
                json.dump(lt_ratio, f, indent=2)

    with open(os.path.join(OUT_DIR, "results_table.json"), "w") as f:
        json.dump(results_table, f, indent=2)
    with open(os.path.join(OUT_DIR, "per_station_results.json"), "w") as f:
        json.dump(per_station_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    logger.info("Done. Results written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
