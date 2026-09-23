"""Per-station training/evaluation pipeline, plus the iterative long-term
forecasting rollout (matching DST-TransitNet's lag-1..12 framework, Sec.
III.B.4 of the DST-TransitNet paper, applied here to TSB-Forecast-base's
per-station stacked ensemble for the same comparability table).
"""
import numpy as np
import pandas as pd

from .config import FeatureConfig, ModelConfig, RunConfig
from .features import build_lag_table, build_time2vec_pointwise_inputs
from .time2vec import embed_features
from .ensemble import fit_station_model
from .metrics import all_metrics


FEATURE_COLUMNS_BASE = None  # set dynamically per assembled frame


def build_feature_target_frame(series: pd.Series, calendar_df: pd.DataFrame, fcfg: FeatureConfig,
                                embedder, device: str = "cpu") -> pd.DataFrame:
    lags = build_lag_table(series, fcfg)
    df = lags.join(calendar_df.loc[series.index])
    df["y"] = series.shift(-1)

    x, _, _ = build_time2vec_pointwise_inputs(series, fcfg)
    emb = embed_features(embedder, x, device=device)
    for i in range(emb.shape[1]):
        df[f"embed_{i}"] = emb[:, i]

    return df


def split_rows(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = df.index.intersection(index)
    return df.loc[idx].dropna()


def train_and_evaluate_station(series: pd.Series, calendar_df: pd.DataFrame,
                                train_index, test_indices: dict,
                                embedder, etr_params, xgb_params,
                                fcfg: FeatureConfig, mcfg: ModelConfig, device: str = "cpu"):
    frame = build_feature_target_frame(series, calendar_df, fcfg, embedder, device=device)
    feature_cols = [c for c in frame.columns if c != "y"]

    train_df = split_rows(frame, train_index)
    X_train, y_train = train_df[feature_cols].values, train_df["y"].values

    model = fit_station_model(X_train, y_train, etr_params, xgb_params, mcfg)

    results = {}
    preds = {}
    for name, idx in test_indices.items():
        test_df = split_rows(frame, idx)
        if len(test_df) == 0:
            continue
        Xte, yte = test_df[feature_cols].values, test_df["y"].values
        pred = model.predict(Xte)
        results[name] = all_metrics(yte, pred)
        preds[name] = {"pred": pred, "true": yte, "index": test_df.index}

    train_pred = model.predict(X_train)
    results["train"] = all_metrics(y_train, train_pred)

    return model, results, preds, feature_cols


def long_term_forecast_station(series: pd.Series, calendar_df: pd.DataFrame, model,
                                feature_cols: list, embedder, fcfg: FeatureConfig,
                                run_cfg: RunConfig, target_index: pd.DatetimeIndex,
                                device: str = "cpu") -> np.ndarray:
    """Iterative rollout, matching DST-TransitNet's long-term framework:
    lag_1week (true previous-week data) is never touched; lag_0/lag_1/4/8/
    1day and the Time2Vec "current value" input are rolled forward using the
    model's own previous predictions once the lag distance is inside the
    already-predicted horizon.

    For efficiency (147 stations x thousands of target timestamps x 12
    iterative lags), each starting point only copies the small numpy window
    it actually needs (`lag_1week_steps` back to `max_lag` forward) instead
    of the full per-station series.
    """
    values = series.values.astype(np.float32)
    pos_map = pd.Series(np.arange(len(series)), index=series.index)
    max_lag = run_cfg.long_term_max_lag
    preds = np.full((len(target_index), max_lag), np.nan, dtype=np.float32)

    back = fcfg.lag_1week_steps + 1
    calendar_index = calendar_df.index

    for row_i, t0 in enumerate(target_index):
        if t0 not in pos_map.index:
            continue
        p0 = int(pos_map[t0])
        if p0 - back < 0:
            continue

        # window[k] holds the (real, then increasingly predicted) value at
        # absolute position (p0 - back + 1 + k), i.e. window[back-1] = the
        # true value AT p0 (matches the short-term convention: a feature row
        # timestamped t0 predicts the value at t0+1, so lag=0's target is
        # p0+1 and its "current value"/lag_0 feature is the true value at p0).
        window = np.concatenate([values[p0 - back + 1: p0 + 1], np.full(max_lag, np.nan, dtype=np.float32)])

        for lag in range(max_lag):
            target_pos_abs = p0 + lag + 1
            target_local = back + lag  # index into `window` for target_pos_abs

            def rel(offset_back):
                idx = target_local - offset_back
                return window[idx] if 0 <= idx < len(window) else np.nan

            row = {"lag_0": rel(1)}
            for L in fcfg.lag_steps:
                row[f"lag_{L}"] = rel(L)
            row["lag_1day"] = rel(fcfg.lag_1day_steps)
            row["lag_1week"] = rel(fcfg.lag_1week_steps)

            if target_pos_abs < len(series.index):
                target_ts = series.index[target_pos_abs]
            else:
                target_ts = series.index[-1] + pd.Timedelta(minutes=15) * (target_pos_abs - len(series.index) + 1)
            cal_row = calendar_df.loc[target_ts] if target_ts in calendar_index else calendar_df.iloc[-1]
            for c in cal_row.index:
                row[c] = cal_row[c]

            cur_val = row["lag_0"]
            week_val = row["lag_1week"]
            if np.isnan(cur_val) or np.isnan(week_val):
                break
            emb = embed_features(embedder, np.array([[cur_val, week_val]], dtype=np.float32), device=device)[0]
            for i, v in enumerate(emb):
                row[f"embed_{i}"] = v

            x_vec = np.array([[row[c] for c in feature_cols]], dtype=np.float32)
            if np.isnan(x_vec).any():
                break
            pred = model.predict(x_vec)[0]
            preds[row_i, lag] = pred
            window[target_local] = pred

    return preds
