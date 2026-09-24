"""End-to-end reproduction runner: builds the pooled (station, 15-min)
tabular dataset, trains the paper's five algorithms, evaluates them plus
four ensemble strategies on the Normal/Protest/COVID periods (same as the
DST-TransitNet / TSB-Forecast reproductions), and saves all metrics.
"""
import argparse
import gc
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dst_transitnet.data import load_station_series, make_splits
from apc_tabular.config import DataConfig, FeatureConfig, ModelConfig, RunConfig
from apc_tabular.features import (load_station_metadata, build_calendar_table, melt_to_long,
                                    ALL_FEATURE_COLS)
from apc_tabular.models import build_model
from apc_tabular.metrics import all_metrics
from apc_tabular.ensemble import simple_average, best_pair_average, weighted_least_squares, ridge_stack

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("apc_run_reproduction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MODEL_NAMES = ["random_forest", "xgboost", "catboost", "lightgbm", "tabular_dnn"]
SHORT = {"random_forest": "RF", "xgboost": "XGB", "catboost": "CBM", "lightgbm": "LGB", "tabular_dnn": "TAB"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--train_stride", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    dcfg = DataConfig()
    fcfg = FeatureConfig()
    mcfg = ModelConfig()
    if args.train_stride is not None:
        fcfg.train_stride = args.train_stride

    logger.info("Loading Bogota BRT station series (raw counts, unscaled) ...")
    df = load_station_series(dcfg)  # raw boarding counts, NOT min-max scaled (paper predicts raw counts)
    splits = make_splits(df, dcfg, val_fraction=0.1)

    meta = load_station_metadata(dcfg, df.columns)
    calendar_df = build_calendar_table(df.index)

    logger.info("Building pooled tabular frames ...")
    t0 = time.time()
    train_long = melt_to_long(splits.train, meta, calendar_df, stride=fcfg.train_stride)
    val_long = melt_to_long(splits.val, meta, calendar_df, stride=1)
    test_periods = {
        "normal": melt_to_long(splits.normal, meta, calendar_df, stride=1),
        "protest": melt_to_long(splits.protest, meta, calendar_df, stride=1),
        "covid": melt_to_long(splits.covid, meta, calendar_df, stride=1),
    }
    logger.info("train=%s val=%s normal=%s protest=%s covid=%s (built in %.1fs)",
                train_long.shape, val_long.shape, test_periods["normal"].shape,
                test_periods["protest"].shape, test_periods["covid"].shape, time.time() - t0)

    y_train = train_long["y"].values.astype(np.float32)

    results = {}
    timing = {}
    test_preds = {name: {} for name in args.models}

    for name in args.models:
        logger.info("=== Training %s ===", name)
        t0 = time.time()
        model = build_model(name, mcfg)
        if name == "tabular_dnn":
            model.fit(train_long[ALL_FEATURE_COLS], y_train,
                      X_val=val_long[ALL_FEATURE_COLS], y_val=val_long["y"].values.astype(np.float32))
        else:
            model.fit(train_long[ALL_FEATURE_COLS], y_train)
        dt = time.time() - t0
        timing[name] = dt
        joblib.dump(model, os.path.join(CKPT_DIR, f"{name}.joblib"))
        logger.info("%s trained in %.1fs", name, dt)

        model_results = {}
        train_pred = model.predict(train_long[ALL_FEATURE_COLS])
        model_results["train"] = all_metrics(y_train, train_pred)
        for period_name, frame in test_periods.items():
            pred = model.predict(frame[ALL_FEATURE_COLS])
            test_preds[name][period_name] = pred
            model_results[period_name] = all_metrics(frame["y"].values.astype(np.float32), pred)
            logger.info("%s / %s: R2=%.4f RMSE=%.4f MAAPE=%.4f",
                        name, period_name, model_results[period_name]["r2"],
                        model_results[period_name]["rmse"], model_results[period_name]["maape"])
        results[name] = model_results
        del model, train_pred
        gc.collect()

    with open(os.path.join(OUT_DIR, "results_table.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT_DIR, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    # ---- Ensembles (per period, using all successfully trained models)
    if len(test_preds) >= 2:
        ensemble_results = {}
        for period_name, frame in test_periods.items():
            y_true = frame["y"].values.astype(np.float32)
            preds = {SHORT[n]: test_preds[n][period_name] for n in test_preds}

            avg = simple_average(preds)
            best_pair, best_pair_pred = best_pair_average(
                preds, y_true, lambda yt, yp: all_metrics(yt, yp)["r2"])
            wls_weights, wls_pred = weighted_least_squares(preds, y_true)
            ridge_weights, ridge_pred, _ = ridge_stack(preds, y_true)

            ensemble_results[period_name] = {
                "simple_average": all_metrics(y_true, avg),
                "best_pair": {"pair": best_pair, **all_metrics(y_true, best_pair_pred)},
                "weighted_least_squares": {"weights": wls_weights, **all_metrics(y_true, wls_pred)},
                "ridge_stack": {"weights": ridge_weights, **all_metrics(y_true, ridge_pred)},
            }
            logger.info("Ensemble / %s: simple_avg R2=%.4f, best_pair=%s R2=%.4f, ridge R2=%.4f",
                        period_name, ensemble_results[period_name]["simple_average"]["r2"], best_pair,
                        ensemble_results[period_name]["best_pair"]["r2"],
                        ensemble_results[period_name]["ridge_stack"]["r2"])

        with open(os.path.join(OUT_DIR, "ensemble_results.json"), "w") as f:
            json.dump(ensemble_results, f, indent=2)

    logger.info("Done. Results written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
