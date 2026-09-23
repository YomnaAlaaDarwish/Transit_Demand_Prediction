"""End-to-end TSB-Forecast-base reproduction runner.

Trains one independent stacked ensemble (ETR + XGBoost + LinearRegression
meta-learner) per BRT station, using the paper's Time2Vec-inspired learned
temporal embedding (shared/global, see time2vec.py) plus lag and calendar
features -- explicitly WITHOUT the paper's SBERT/news module or weather
features (no equivalent data source available; see
TSB_FORECAST_REPRODUCTION_NOTES.md). Same station set / periods / splits as
the DST-TransitNet reproduction for direct comparability.
"""
import argparse
import json
import logging
import os
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dst_transitnet.data import load_station_series, make_splits, MinMaxScaler, scale_dataframe
from tsb_forecast.config import DataConfig, FeatureConfig, ModelConfig, RunConfig
from tsb_forecast.features import build_calendar_features, build_time2vec_pointwise_inputs
from tsb_forecast.time2vec import train_time2vec, embed_features
from tsb_forecast.ensemble import tune_hyperparameters
from tsb_forecast.train import train_and_evaluate_station, long_term_forecast_station
from tsb_forecast.metrics import all_metrics, maape

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("tsb_run_reproduction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations_limit", type=int, default=None)
    parser.add_argument("--time2vec_sample_stations", type=int, default=20,
                         help="Number of stations pooled to train the shared Time2Vec encoder.")
    parser.add_argument("--long_term_stride", type=int, default=20,
                         help="Evaluate long-term rollout on every Nth test timestamp per station.")
    parser.add_argument("--skip_long_term", action="store_true")
    parser.add_argument("--save_station_models", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    dcfg = DataConfig()
    fcfg = FeatureConfig()
    mcfg = ModelConfig()
    rcfg = RunConfig(stations_limit=args.stations_limit)

    logger.info("Loading Bogota BRT station series ...")
    df = load_station_series(dcfg)
    splits = make_splits(df, dcfg, val_fraction=0.1)
    scaler = MinMaxScaler().fit(splits.train)
    full_scaled = scale_dataframe(df, scaler)

    calendar_df = build_calendar_features(full_scaled.index)

    test_periods = {"normal": splits.normal.index, "protest": splits.protest.index, "covid": splits.covid.index}

    # ---- Step 1: train the shared Time2Vec encoder (pooled across a sample of stations, train period only)
    logger.info("Training shared Time2Vec encoder ...")
    sample_cols = list(full_scaled.columns[:args.time2vec_sample_stations])
    xs, ys = [], []
    train_scaled = full_scaled.loc[splits.train.index]
    for col in sample_cols:
        x, x_next, valid = build_time2vec_pointwise_inputs(train_scaled[col], fcfg)
        xs.append(x[valid])
        ys.append(x_next[valid])
    X_t2v = np.concatenate(xs, axis=0)
    Y_t2v = np.concatenate(ys, axis=0)
    logger.info("Time2Vec training pairs: %s", X_t2v.shape)
    embedder = train_time2vec(X_t2v, Y_t2v, fcfg)
    joblib.dump(embedder, os.path.join(CKPT_DIR, "time2vec_encoder.joblib"))

    # ---- Step 2: tune ETR/XGBoost hyperparameters once, on the system-wide aggregate series
    logger.info("Tuning ETR/XGBoost hyperparameters on system-wide aggregate series ...")
    aggregate_raw = df.sum(axis=1)
    agg_scaler = MinMaxScaler().fit(splits.train.sum(axis=1).to_frame("agg"))
    agg_scaled = ((aggregate_raw - agg_scaler.min_[0]) / agg_scaler.range_[0]).rename("agg")

    from tsb_forecast.features import build_lag_table
    agg_lags = build_lag_table(agg_scaled, fcfg).join(calendar_df)
    agg_lags["y"] = agg_scaled.shift(-1)
    x_agg, _, _ = build_time2vec_pointwise_inputs(agg_scaled, fcfg)
    emb_agg = embed_features(embedder, x_agg)
    for i in range(emb_agg.shape[1]):
        agg_lags[f"embed_{i}"] = emb_agg[:, i]
    agg_train = agg_lags.loc[agg_lags.index.intersection(splits.train.index)].dropna()
    feat_cols_tune = [c for c in agg_train.columns if c != "y"]
    etr_params, xgb_params = tune_hyperparameters(agg_train[feat_cols_tune].values, agg_train["y"].values, mcfg)
    logger.info("Tuned params: ETR=%s XGB=%s", etr_params, xgb_params)
    with open(os.path.join(OUT_DIR, "tuned_hyperparameters.json"), "w") as f:
        json.dump({"etr": etr_params, "xgb": xgb_params}, f, indent=2)

    # ---- Step 3: per-station training/eval/long-term
    stations = list(full_scaled.columns)
    if rcfg.stations_limit:
        stations = stations[:rcfg.stations_limit]
    logger.info("Training %d per-station stacked ensembles ...", len(stations))

    per_station_results = {}
    per_station_long_term = {}
    timing = {}

    for i, station in enumerate(stations):
        t0 = time.time()
        series = full_scaled[station]
        model, results, preds, feature_cols = train_and_evaluate_station(
            series, calendar_df, splits.train.index, test_periods,
            embedder, etr_params, xgb_params, fcfg, mcfg)
        dt = time.time() - t0
        timing[station] = dt
        per_station_results[station] = results

        if args.save_station_models:
            joblib.dump(model, os.path.join(CKPT_DIR, f"station_{station.strip('()').split(')')[0]}.joblib"))

        if not args.skip_long_term:
            lt_ratios = {}
            for period_name, idx in test_periods.items():
                strided_idx = idx[::args.long_term_stride]
                if len(strided_idx) == 0:
                    continue
                lt_preds = long_term_forecast_station(
                    series, calendar_df, model, feature_cols, embedder, fcfg, rcfg, strided_idx)

                series_values = series.values
                pos_map = {ts: p for p, ts in enumerate(series.index)}
                true_vals = np.full((len(strided_idx), rcfg.long_term_max_lag), np.nan, dtype=np.float32)
                for row_i, t in enumerate(strided_idx):
                    p0 = pos_map.get(t)
                    if p0 is None:
                        continue
                    for lag in range(rcfg.long_term_max_lag):
                        p = p0 + lag + 1  # lag=0 predicts t0+1, matching the short-term y=shift(-1) convention
                        if p < len(series_values):
                            true_vals[row_i, lag] = series_values[p]
                mask = ~np.isnan(lt_preds).any(axis=1) & ~np.isnan(true_vals).any(axis=1)
                if mask.sum() == 0:
                    continue
                maape_per_lag = [maape(true_vals[mask, lag], lt_preds[mask, lag])
                                  for lag in range(rcfg.long_term_max_lag)]
                ratio = maape_per_lag[-1] / maape_per_lag[0] if maape_per_lag[0] > 0 else float("nan")
                lt_ratios[period_name] = {"maape_per_lag": maape_per_lag, "ratio_12_vs_1": ratio,
                                           "n_samples": int(mask.sum())}
            per_station_long_term[station] = lt_ratios

        if (i + 1) % 10 == 0 or i == len(stations) - 1:
            logger.info("[%d/%d] %s done in %.1fs (train R2=%.3f)",
                        i + 1, len(stations), station, dt, results["train"]["r2"])

    with open(os.path.join(OUT_DIR, "per_station_results.json"), "w") as f:
        json.dump(per_station_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "per_station_long_term.json"), "w") as f:
        json.dump(per_station_long_term, f, indent=2)
    with open(os.path.join(OUT_DIR, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    # ---- Aggregate across stations for a system-wide summary table
    summary = {}
    for period in ["train", "normal", "protest", "covid"]:
        vals = {m: [] for m in ["mae", "rmse", "smape", "r2", "maape"]}
        for station, res in per_station_results.items():
            if period in res:
                for m in vals:
                    vals[m].append(res[period][m])
        summary[period] = {m: {"mean": float(np.mean(v)), "median": float(np.median(v))}
                            for m, v in vals.items() if len(v) > 0}
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. Results written to %s", OUT_DIR)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
