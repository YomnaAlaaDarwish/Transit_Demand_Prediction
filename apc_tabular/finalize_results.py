"""Reloads all 5 already-trained model checkpoints (saved separately across
two runs -- tree models in one process, Tabular DNN in an isolated process
to work around the OOM documented in TRONDHEIM_APC_REPRODUCTION_NOTES.md --
so no single results_table.json held all 5 at once) and rebuilds the
unified results_table.json, timing.json, and ensemble_results.json.
"""
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dst_transitnet.data import load_station_series, make_splits
from apc_tabular.config import DataConfig, FeatureConfig
from apc_tabular.features import load_station_metadata, build_calendar_table, melt_to_long, ALL_FEATURE_COLS
from apc_tabular.metrics import all_metrics
from apc_tabular.ensemble import simple_average, best_pair_average, weighted_least_squares, ridge_stack

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

MODEL_NAMES = ["random_forest", "xgboost", "catboost", "lightgbm", "tabular_dnn"]
SHORT = {"random_forest": "RF", "xgboost": "XGB", "catboost": "CBM", "lightgbm": "LGB", "tabular_dnn": "TAB"}

# From the two runs' logs (full_run.log for the tree models, dnn_only.log for the DNN)
TIMING = {"random_forest": 141.3, "xgboost": 32.8, "catboost": 288.6, "lightgbm": 12.0, "tabular_dnn": 1498.7}


def main():
    dcfg = DataConfig()
    fcfg = FeatureConfig()

    df = load_station_series(dcfg)
    splits = make_splits(df, dcfg, val_fraction=0.1)
    meta = load_station_metadata(dcfg, df.columns)
    calendar_df = build_calendar_table(df.index)

    train_long = melt_to_long(splits.train, meta, calendar_df, stride=fcfg.train_stride)
    test_periods = {
        "normal": melt_to_long(splits.normal, meta, calendar_df, stride=1),
        "protest": melt_to_long(splits.protest, meta, calendar_df, stride=1),
        "covid": melt_to_long(splits.covid, meta, calendar_df, stride=1),
    }
    y_train = train_long["y"].values.astype(np.float32)

    results = {}
    test_preds = {}
    for name in MODEL_NAMES:
        ckpt_path = os.path.join(CKPT_DIR, f"{name}.joblib")
        print(f"Loading {ckpt_path} ...")
        model = joblib.load(ckpt_path)

        model_results = {}
        train_pred = model.predict(train_long[ALL_FEATURE_COLS])
        model_results["train"] = all_metrics(y_train, train_pred)

        test_preds[name] = {}
        for period_name, frame in test_periods.items():
            pred = model.predict(frame[ALL_FEATURE_COLS])
            test_preds[name][period_name] = pred
            model_results[period_name] = all_metrics(frame["y"].values.astype(np.float32), pred)
        results[name] = model_results
        print(f"  {name}: normal R2={model_results['normal']['r2']:.4f}")
        del model

    with open(os.path.join(OUT_DIR, "results_table.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT_DIR, "timing.json"), "w") as f:
        json.dump(TIMING, f, indent=2)

    ensemble_results = {}
    for period_name, frame in test_periods.items():
        y_true = frame["y"].values.astype(np.float32)
        preds = {SHORT[n]: test_preds[n][period_name] for n in MODEL_NAMES}

        avg = simple_average(preds)
        best_pair, best_pair_pred = best_pair_average(preds, y_true, lambda yt, yp: all_metrics(yt, yp)["r2"])
        wls_weights, wls_pred = weighted_least_squares(preds, y_true)
        ridge_weights, ridge_pred, _ = ridge_stack(preds, y_true)

        ensemble_results[period_name] = {
            "simple_average": all_metrics(y_true, avg),
            "best_pair": {"pair": best_pair, **all_metrics(y_true, best_pair_pred)},
            "weighted_least_squares": {"weights": wls_weights, **all_metrics(y_true, wls_pred)},
            "ridge_stack": {"weights": ridge_weights, **all_metrics(y_true, ridge_pred)},
        }
        print(f"Ensemble / {period_name}: simple_avg R2={ensemble_results[period_name]['simple_average']['r2']:.4f}, "
              f"ridge R2={ensemble_results[period_name]['ridge_stack']['r2']:.4f}")

    with open(os.path.join(OUT_DIR, "ensemble_results.json"), "w") as f:
        json.dump(ensemble_results, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
