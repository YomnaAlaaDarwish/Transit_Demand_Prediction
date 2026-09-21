"""Data loading, graph construction, and windowed-sample generation for the
DST-TransitNet reproduction on the Bogota TransMilenio BRT dataset.

Reuses the raw parquet file already produced by this repository's existing
preprocessing (data/transmilenio_transactions.parquet) -- raw per-year CSVs
under data/transactions/ are left untouched. Station dropping / hour
filtering mirrors the convention already used in run.py / data.py so this
reproduction stays consistent with the existing codebase.
"""
import re
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig

logger = logging.getLogger(__name__)


def _station_code(col: str):
    m = re.match(r"\((\d+)\)", col)
    return int(m.group(1)) if m else None


def load_station_series(cfg: DataConfig) -> pd.DataFrame:
    """Load the 15-min station x time matrix, dropping cable-car stations
    and night hours. Returns a DataFrame indexed by timestamp with one
    column per station (147 columns), sorted by station code for a
    deterministic, reproducible station ordering.
    """
    df = pd.read_parquet(cfg.parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    drop = [c for c in cfg.drop_stations if c in df.columns]
    df = df.drop(columns=drop)

    # A handful of 15-min slots have duplicate raw rows; sum them, matching
    # the existing repository's read_data() convention (data.py).
    station_cols = [c for c in df.columns if c != "timestamp"]
    df = df.groupby("timestamp")[station_cols].sum()
    df = df.sort_index()

    hour = df.index.hour
    df = df[~hour.isin(cfg.excluded_hours)]

    # Deterministic station ordering by numeric station code (ascending).
    codes = {c: _station_code(c) for c in df.columns}
    ordered_cols = sorted(df.columns, key=lambda c: codes[c])
    df = df[ordered_cols]

    df = df.fillna(0.0).astype("float32")
    return df


def station_order(df: pd.DataFrame):
    return list(df.columns), [_station_code(c) for c in df.columns]


def build_adjacency(cfg: DataConfig, columns) -> np.ndarray:
    """Builds a symmetric 0/1 adjacency matrix (with self-loops) aligned to
    `columns` (the station ordering used for the series matrix), from the
    BRT line-segment edge list.

    Edge endpoints not present among the 147 kept stations (e.g. an edge
    referencing a cable-car station, or a stray id typo in the source CSV)
    are dropped; this is a documented, minor data-cleaning decision -- see
    reproduction notes.
    """
    edges = pd.read_csv(cfg.edges_path)
    codes = [_station_code(c) for c in columns]
    code_to_idx = {c: i for i, c in enumerate(codes)}

    n = len(columns)
    A = np.eye(n, dtype=np.float32)  # self-loops (\tilde{A} = A + I)
    kept, dropped = 0, 0
    for _, row in edges.iterrows():
        a, b = int(row["node_1"]), int(row["node_2"])
        if a in code_to_idx and b in code_to_idx:
            i, j = code_to_idx[a], code_to_idx[b]
            A[i, j] = 1.0
            A[j, i] = 1.0
            kept += 1
        else:
            dropped += 1
    logger.info("Adjacency: kept %d edges, dropped %d (missing endpoints)", kept, dropped)
    return A


def gcn_normalize(A: np.ndarray) -> np.ndarray:
    """Symmetric normalization \tilde{D}^-1/2 \tilde{A} \tilde{D}^-1/2 used by
    the GCN aggregation formula in the paper (Sec. III.A.2). `A` is assumed
    to already include self-loops.
    """
    deg = A.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    d_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    normal: pd.DataFrame
    protest: pd.DataFrame
    covid: pd.DataFrame
    full: pd.DataFrame  # full series, used to source Xh (previous-week) context for any split


def make_splits(df: pd.DataFrame, cfg: DataConfig, val_fraction: float) -> SplitFrames:
    train_end = pd.Timestamp(cfg.train_end)
    train_full = df[df.index < train_end]

    n_val = int(len(train_full) * val_fraction)
    train = train_full.iloc[: len(train_full) - n_val]
    val = train_full.iloc[len(train_full) - n_val:]

    normal = df[(df.index >= cfg.normal_start) & (df.index < cfg.normal_end)]
    protest = df[(df.index >= cfg.protest_start) & (df.index < cfg.protest_end)]
    covid = df[(df.index >= cfg.covid_start) & (df.index < cfg.covid_end)]

    return SplitFrames(train=train, val=val, normal=normal, protest=protest, covid=covid, full=df)


class MinMaxScaler:
    """Per-station min-max scaler fit on the training split only (mirrors
    the existing repository's `min_max` convention in data.py, and matches
    the paper's own references to "scaled ridership" in Fig. 13/17/20/21).
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, df: pd.DataFrame):
        self.min_ = df.min(axis=0).values.astype(np.float32)
        self.max_ = df.max(axis=0).values.astype(np.float32)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_).astype(np.float32)
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.min_) / self.range_

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.range_ + self.min_


def scale_dataframe(df: pd.DataFrame, scaler: "MinMaxScaler") -> pd.DataFrame:
    """Applies a fitted per-station MinMaxScaler to a DataFrame, preserving
    index/columns. Both R2 and MAAPE are invariant to this (positive,
    per-station-consistent) affine rescaling, so metrics computed on scaled
    values equal those on raw values -- this mirrors the paper's own
    reporting on "scaled ridership" (Figs. 13/17/20/21).
    """
    scaled = scaler.transform(df.values.astype(np.float32))
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def moving_average_decompose(x: np.ndarray, kernel: int) -> tuple:
    """Xt = AvgPool1D(Xo), Xr = Xo - Xt, applied along the time axis.

    x: array of shape (..., T) -- decomposition is applied on the last axis
    (the lookback/time dimension), independently per station/sample, as in
    the paper's temporal decomposition layer (Sec III.B.1) and the
    DLinear/Autoformer moving-average decomposition it cites.
    """
    pad = kernel // 2
    x_padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, kernel - 1 - pad)], mode="edge")
    csum = np.cumsum(x_padded, axis=-1)
    csum = np.concatenate([np.zeros_like(csum[..., :1]), csum], axis=-1)
    trend = (csum[..., kernel:] - csum[..., :-kernel]) / kernel
    residual = x - trend
    return trend.astype(np.float32), residual.astype(np.float32)


@dataclass
class WindowedSamples:
    """All tensors are numpy arrays.

    Xo: (N, recent_len, S)   recent original ridership
    Xh: (N, hist_len, S)     historical (same weekday/time, previous week) ridership
    Xt: (N, recent_len, S)   trend component of Xo
    Xr: (N, recent_len, S)   residual component of Xo
    y:  (N, S)               target at t+1 (1-step) OR (N, max_lag, S) if long_term
    timestamps: (N,) target timestamps, for bookkeeping / peak-hour analysis
    """
    Xo: np.ndarray
    Xh: np.ndarray
    Xt: np.ndarray
    Xr: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray


def build_windows(full_df: pd.DataFrame, target_index: pd.DatetimeIndex, cfg: DataConfig,
                   long_term: bool = False, stride: int = 1) -> WindowedSamples:
    """Builds windowed samples for every timestamp in `target_index` that has
    sufficient history in `full_df` (recent_len steps back, and a full
    previous-week historical window). `full_df` must be the *entire*
    (train+test) series so that historical (previous week) context is
    always drawn from real observed data -- this matches the paper's
    long-term deployment design, where Xh never depends on the model's own
    predictions (Sec III.C).
    """
    values = full_df.values.astype(np.float32)  # (T, S)
    index = full_df.index
    pos_map = pd.Series(np.arange(len(index)), index=index)

    recent_len = cfg.recent_len
    hist_len = cfg.hist_len
    horizon = cfg.long_term_max_lag if long_term else cfg.pred_horizon

    target_index = target_index[::stride]
    target_pos = pos_map.reindex(target_index)

    hist_end_ts = target_index - pd.Timedelta(days=cfg.hist_week_offset_days)
    hist_end_pos = pos_map.reindex(hist_end_ts)

    Xo_list, Xh_list, y_list, ts_list = [], [], [], []
    S = values.shape[1]

    for t_ts, t_pos, h_pos in zip(target_index, target_pos.values, hist_end_pos.values):
        if np.isnan(t_pos) or np.isnan(h_pos):
            continue
        t_pos = int(t_pos)
        h_pos = int(h_pos)
        if t_pos - recent_len < 0:
            continue
        # For long_term, we need a *sliding* historical window: lag k's own
        # previous-week window ends `k` steps after lag 0's, so we fetch a
        # hist_len + horizon - 1 slab and let the caller slide over it.
        hist_extra = (horizon - 1) if long_term else 0
        if h_pos - hist_len + 1 < 0:
            continue
        if h_pos + hist_extra + 1 > len(values):
            continue
        if t_pos + horizon > len(values):
            continue

        xo = values[t_pos - recent_len: t_pos]  # (recent_len, S)
        xh = values[h_pos - hist_len + 1: h_pos + hist_extra + 1]  # (hist_len[+horizon-1], S)
        if long_term:
            y = values[t_pos: t_pos + horizon]  # (horizon, S)
        else:
            y = values[t_pos]  # (S,)

        Xo_list.append(xo)
        Xh_list.append(xh)
        y_list.append(y)
        ts_list.append(t_ts)

    Xo = np.stack(Xo_list).astype(np.float32) if Xo_list else np.zeros((0, recent_len, S), np.float32)
    Xh = np.stack(Xh_list).astype(np.float32) if Xh_list else np.zeros((0, hist_len, S), np.float32)
    y = np.stack(y_list).astype(np.float32) if y_list else (
        np.zeros((0, horizon, S), np.float32) if long_term else np.zeros((0, S), np.float32))
    ts = np.array(ts_list)

    # Xo is (N, recent_len, S); decomposition operates along time axis ->
    # transpose to (N, S, recent_len), decompose, transpose back.
    xo_t = np.transpose(Xo, (0, 2, 1))
    trend_t, resid_t = moving_average_decompose(xo_t, cfg.decomposition_kernel)
    Xt = np.transpose(trend_t, (0, 2, 1))
    Xr = np.transpose(resid_t, (0, 2, 1))

    return WindowedSamples(Xo=Xo, Xh=Xh, Xt=Xt, Xr=Xr, y=y, timestamps=ts)
