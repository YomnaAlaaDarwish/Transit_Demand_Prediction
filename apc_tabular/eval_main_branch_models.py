"""Evaluates the pre-existing main-branch daily models (ARIMA, SARIMA, Dense,
CNN, LSTM -- committed under `output/day/static/{single,multioutput}/`, from
before this thesis's reproduction work) against the same Normal/Protest/COVID
periods used by the DST-TransitNet / TSB-Forecast / APC-tabular reproductions,
so they can appear as a "what already existed" baseline row in the final
cross-reproduction comparison.

IMPORTANT CAVEAT (kept in the output and in the final comparison doc): these
models operate on a fundamentally different task -- DAILY aggregation,
7-day-ahead multi-step forecasting (settings.yaml: aggregation=day,
steps_back=14, forecast_window=7) -- vs. the three reproductions' 15-minute,
1-step-ahead task. R2/MAAPE are still well-defined and computed the same way,
but are not a strictly fair apples-to-apples comparison; they answer "how did
the repository's original daily models do on the same real-world periods",
not "how does a 15-min model compare to a daily one at the same horizon."

The original pipeline's raw input (`data/clean_transactions.csv`) is not
present in this repo (gitignored, `*.csv`) -- only `data/transmilenio_
transactions.parquet` (used by dst_transitnet/tsb_forecast) survives. Both
ultimately derive from the same raw transaction CSVs
(`data/transactions/*.csv`, untouched), so we reconstruct the day-aggregated
true series from the parquet instead of re-running data.py's CSV-based
read_data(), matching stations by numeric code (robust to the
accent-stripped/lowercased station-name strings used in the saved JSON
filenames).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dst_transitnet.config import DataConfig
from dst_transitnet.data import load_station_series
from apc_tabular.metrics import all_metrics

STEPS_BACK = 14
FORECAST_WINDOW = 7
TRAIN_DATE = "2018-08-01"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _code(name: str):
    m = re.match(r"\((\d+)\)", name)
    return int(m.group(1)) if m else None


def build_daily_true_series() -> pd.DataFrame:
    dcfg = DataConfig()
    quarter_hourly = load_station_series(dcfg)  # hours 4-22 already filtered, raw counts
    daily = quarter_hourly.resample("D").sum()
    return daily


def true_label_windows(daily_series: pd.Series) -> tuple:
    """Replicates WindowGenerator's test-window construction (data.py) in
    plain numpy: test_df starts at train_idx - steps_back - forecast_window + 1;
    window i's label = test_df[i+steps_back : i+steps_back+forecast_window].
    Returns (labels: (n_windows, forecast_window), label_start_dates).
    """
    idx = daily_series.index.get_loc(pd.Timestamp(TRAIN_DATE))
    start = idx - STEPS_BACK - FORECAST_WINDOW + 1
    test_df = daily_series.iloc[start:]
    total_window = STEPS_BACK + FORECAST_WINDOW
    n_windows = len(test_df) - total_window + 1
    values = test_df.values
    labels = np.stack([values[i + STEPS_BACK: i + total_window] for i in range(n_windows)])
    label_start_dates = test_df.index[STEPS_BACK: STEPS_BACK + n_windows]
    return labels, label_start_dates


def load_predictions(model_dir: str) -> dict:
    """Returns {accent_stripped_lower_station_name: prediction array (n_windows, forecast_window)}."""
    preds = {}
    for path in glob.glob(os.path.join(model_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        name = os.path.splitext(os.path.basename(path))[0]
        preds[name] = np.array(d["prediction"])
    return preds


def match_station_names(daily_columns, json_names):
    """Map our (accented, mixed-case) parquet column names to the JSON
    files' (accent-stripped, lowercase) names, via numeric station code."""
    code_to_col = {_code(c): c for c in daily_columns}
    code_to_json = {}
    for jn in json_names:
        c = _code(jn)
        if c is not None:
            code_to_json[c] = jn
    matched = {code_to_col[c]: code_to_json[c] for c in code_to_col if c in code_to_json}
    return matched


def evaluate_variant(model_dir: str, daily: pd.DataFrame, label_start_dates: pd.DatetimeIndex,
                      period_ranges: dict) -> dict:
    preds = load_predictions(model_dir)
    if not preds:
        return {}
    name_map = match_station_names(daily.columns, preds.keys())

    per_period_true, per_period_pred = {p: [] for p in period_ranges}, {p: [] for p in period_ranges}
    per_period_true["train"], per_period_pred["train"] = [], []

    for col, jname in name_map.items():
        labels, _ = true_label_windows(daily[col])
        pred = preds[jname]
        n = min(len(labels), len(pred))
        labels, pred = labels[:n], pred[:n]
        dates = label_start_dates[:n]

        for period, (start, end) in period_ranges.items():
            mask = (dates >= start) & (dates < end)
            if mask.sum() == 0:
                continue
            per_period_true[period].append(labels[mask].ravel())
            per_period_pred[period].append(pred[mask].ravel())

    results = {}
    for period in per_period_true:
        if not per_period_true[period]:
            continue
        yt = np.concatenate(per_period_true[period])
        yp = np.concatenate(per_period_pred[period])
        results[period] = all_metrics(yt, yp)
        results[period]["n_points"] = int(len(yt))
    return results


def main():
    dcfg = DataConfig()
    daily = build_daily_true_series()
    _, label_start_dates = true_label_windows(daily.iloc[:, 0])

    period_ranges = {
        "normal": (pd.Timestamp(dcfg.normal_start), pd.Timestamp(dcfg.normal_end)),
        "protest": (pd.Timestamp(dcfg.protest_start), pd.Timestamp(dcfg.protest_end)),
        "covid": (pd.Timestamp(dcfg.covid_start), pd.Timestamp(dcfg.covid_end)),
    }

    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "day", "static")
    variants = {
        "arima_single": os.path.join(base, "single", "arima"),
        "sarima_single": os.path.join(base, "single", "sarima"),
        "dense_single": os.path.join(base, "single", "dense"),
        "cnn_single": os.path.join(base, "single", "cnn"),
        "lstm_single": os.path.join(base, "single", "lstm"),
        "dense_multioutput": os.path.join(base, "multioutput", "dense"),
        "cnn_multioutput": os.path.join(base, "multioutput", "cnn"),
        "lstm_multioutput": os.path.join(base, "multioutput", "lstm"),
    }

    all_results = {}
    for name, path in variants.items():
        if not os.path.isdir(path):
            print(f"skip {name}: {path} not found")
            continue
        print(f"Evaluating {name} ...")
        res = evaluate_variant(path, daily, label_start_dates, period_ranges)
        all_results[name] = res
        for period, m in res.items():
            print(f"  {period}: R2={m['r2']:.4f} RMSE={m['rmse']:.4f} MAAPE={m['maape']:.4f} n={m['n_points']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "main_branch_daily_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved to", os.path.join(OUT_DIR, "main_branch_daily_results.json"))


if __name__ == "__main__":
    main()
