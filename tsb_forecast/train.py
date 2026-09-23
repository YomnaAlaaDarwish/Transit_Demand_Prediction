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

    Vectorized across ALL starting points in `target_index` at once, one
    batched `model.predict()` call per lag step (12 calls total) rather than
    one call per (starting point, lag) pair -- with hundreds of starting
    points x 147 stations, per-row Python-level predict() calls made the
    naive version intractable (see TSB_FORECAST_REPRODUCTION_NOTES.md
    compute-budget section).
    """
    values = series.values.astype(np.float32)
    pos_map = pd.Series(np.arange(len(series)), index=series.index)
    max_lag = run_cfg.long_term_max_lag
    n = len(target_index)
    preds = np.full((n, max_lag), np.nan, dtype=np.float32)

    back = fcfg.lag_1week_steps + 1

    p0 = np.array([pos_map.get(t, -1) for t in target_index], dtype=np.int64)
    valid_start = (p0 - back >= 0) & (p0 + max_lag < len(values)) & (p0 >= 0)
    if not valid_start.any():
        return preds
    idx_valid = np.where(valid_start)[0]
    p0v = p0[idx_valid]
    n_valid = len(p0v)

    # window[i, k] = value at absolute position (p0v[i] - back + 1 + k);
    # window[:, back-1] = true value AT p0 (see short-term y=shift(-1) convention).
    window = np.full((n_valid, back + max_lag), np.nan, dtype=np.float32)
    for i in range(n_valid):
        window[i, :back] = values[p0v[i] - back + 1: p0v[i] + 1]

    alive = np.ones(n_valid, dtype=bool)  # rows that haven't hit a NaN/invalid feature yet

    for lag in range(max_lag):
        target_local = back + lag
        target_pos_abs = p0v + lag + 1
        target_ts = series.index[target_pos_abs]  # bounds already checked via valid_start

        row_feats = {"lag_0": window[:, target_local - 1]}
        for L in fcfg.lag_steps:
            row_feats[f"lag_{L}"] = window[:, target_local - L]
        row_feats["lag_1day"] = window[:, target_local - fcfg.lag_1day_steps]
        row_feats["lag_1week"] = window[:, target_local - fcfg.lag_1week_steps]

        cal_rows = calendar_df.reindex(target_ts)
        for c in cal_rows.columns:
            row_feats[c] = cal_rows[c].values

        cur_val = row_feats["lag_0"]
        week_val = row_feats["lag_1week"]
        row_valid = alive & ~np.isnan(cur_val) & ~np.isnan(week_val)
        if not row_valid.any():
            break

        emb = np.zeros((n_valid, embedder.encoder[-1].out_features), dtype=np.float32)
        safe_cur = np.nan_to_num(cur_val)
        safe_week = np.nan_to_num(week_val)
        emb[row_valid] = embed_features(embedder, np.stack([safe_cur, safe_week], axis=1)[row_valid], device=device)
        for i in range(emb.shape[1]):
            row_feats[f"embed_{i}"] = emb[:, i]

        X = np.stack([row_feats[c] for c in feature_cols], axis=1).astype(np.float32)
        row_valid &= ~np.isnan(X).any(axis=1)
        if not row_valid.any():
            alive &= row_valid
            break

        pred = model.predict(X[row_valid])
        preds[idx_valid[row_valid], lag] = pred
        window[np.where(row_valid)[0], target_local] = pred
        alive &= row_valid

    return preds
