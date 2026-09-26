"""Dataset builder for Track A (daily) and Track B (15-min).

    build_dataset(track, fmt, feature_groups, splits=None, origin_stride=1)

Definitions (DATA_LOG.md section 7):
  * The series is a matrix Y[t, s]: t indexes the track's time axis, s follows
    data/interim/station_order.csv.
      Track A: daily sums over hours 04:00-22:45 (the benchmark's filter), 2015-08-01..2021-04-30.
      Track B: 15-min values with hours 23:00-03:45 removed (76 slots/day), as in the
               DST-TransitNet reproduction. Lookback steps are counted on this axis.
  * A forecast origin p is the index of the FIRST target step. Targets are
    Y[p .. p+H-1]; everything observed must come from steps < p (for 15-min data an
    interval starting at p-1 has ended by the start of p).
  * Leakage rule. Known in advance (calendar, station static) -> may use the target
    time. Everything else (ridership, weather, disruption flags) -> only steps < p.
  * Splits by target time: train = all targets before train_end; val = the last
    val_fraction of train origins (train origins whose targets reach into val are
    purged); test = first target at/after train_end, last target <= test_end.
    Test data is never used for tuning.

Feature groups:
  lags               Track A: lag_1..lag_L (lag_k = Y[p-k]).
                     Track B: lag_1..lag_20 plus wk_0..wk_19 (wk_j = Y[p - W - j],
                     W = 7 days of steps, i.e. the target's time one week earlier and
                     the 19 steps before it -- dst_transitnet's Xh window).
  calendar_benchmark data.py's features at the target time (calendar_features.py).
  calendar_rich      long weekends, day before/after holiday, Holy Week, year end.
  station_static     id_trazado, tipo_esta, num_vag, area_est, num_acc, acc_puent.
  weather            SLOT ONLY -- raises until data/interim/weather_hourly.parquet
                     exists and the feature is implemented (DATA_LOG 6).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .calendar_features import calendar_benchmark, calendar_rich
from .periods import dst_window_label, period_label

FEATURE_GROUPS = ("lags", "calendar_benchmark", "calendar_rich", "station_static", "weather")


# --------------------------------------------------------------------------- series
@dataclass
class Series:
    """Y[t, s] with its time index, station codes and per-step metadata."""
    track: str
    Y: np.ndarray                      # (T, S) float32
    times: pd.DatetimeIndex            # (T,)
    stations: list                     # (S,) 5-char codes, station_order.csv order
    meta: pd.DataFrame = field(default=None)  # (T, ...) e.g. is_service_interval


def station_order() -> list:
    return pd.read_csv(C.STATION_ORDER, dtype={"code": str}).code.tolist()


def load_series(track: str) -> Series:
    order = station_order()
    r = pd.read_parquet(C.RIDERSHIP_15MIN).set_index("timestamp")
    tf = pd.read_parquet(C.TIME_FEATURES_15MIN).set_index("timestamp")
    assert list(r.columns) == order, "ridership columns differ from station_order.csv"
    keep = tf.in_benchmark_hours.to_numpy()
    if track == "A":
        d = r[keep].resample("D").sum()
        meta = pd.DataFrame(index=d.index)
        return Series("A", d.to_numpy("float32"), d.index, order, meta)
    if track == "B":
        rb = r[keep]
        meta = tf.loc[keep, ["is_service_interval", "system_suspended", "filled_zero"]]
        return Series("B", rb.to_numpy("float32"), rb.index, order, meta)
    raise ValueError(track)


def week_steps(series: Series) -> int:
    if series.track == "A":
        return 7
    per_day = int(pd.Series(series.times.normalize()).value_counts().iloc[0])
    return 7 * per_day


# -------------------------------------------------------------------------- windows
def make_windows(series: Series, origin_stride: int = 1) -> pd.DataFrame:
    """One row per valid forecast origin, with its split and period labels."""
    cfg = C.TRACKS[series.track]
    L, H, K = cfg["lookback"], cfg["horizon"], cfg["weekly_hist"]
    T = len(series.times)
    W = week_steps(series)
    first = max(L, W + K - 1 if K else 0)
    p = np.arange(first, T - H + 1)
    t_first = series.times[p]
    t_last = series.times[p + H - 1]
    train_end, test_end = pd.Timestamp(cfg["train_end"]), pd.Timestamp(cfg["test_end"])

    split = np.full(len(p), "", dtype=object)
    is_train = np.asarray(t_last < train_end)
    split[is_train] = "train"
    test = np.asarray((t_first >= train_end) & (t_last <= test_end))
    split[test] = "test"
    tr = np.flatnonzero(is_train)
    n_val = int(len(tr) * cfg["val_fraction"])
    val = tr[len(tr) - n_val:]
    split[val] = "val"
    val_start = series.times[p[val[0]]]
    purge = is_train & np.asarray(t_last >= val_start) & (split == "train")
    split[purge] = ""  # train origins whose targets overlap validation

    w = pd.DataFrame({"origin_pos": p, "origin": t_first, "last_target": t_last, "split": split})
    w = w[w.split != ""].reset_index(drop=True)
    w["period"] = np.where(w.split == "test", period_label(w.origin), "")
    if series.track == "B":
        w["dst_window"] = np.where(w.split == "test", dst_window_label(w.origin), "")
    if origin_stride > 1:
        w = w[w.groupby("split").cumcount() % origin_stride == 0]  # every k-th origin per split
    return w.reset_index(drop=True)


def input_positions(series: Series, p: np.ndarray) -> dict:
    """Index arrays (N, n_steps) of every observed step used for origins p."""
    cfg = C.TRACKS[series.track]
    L, K = cfg["lookback"], cfg["weekly_hist"]
    out = {"lag": p[:, None] - np.arange(1, L + 1)[None, :]}          # lag_k = p - k
    if K:
        W = week_steps(series)
        out["wk"] = p[:, None] - W - np.arange(K)[None, :]             # wk_j = p - W - j
    for v in out.values():
        assert (v < p[:, None]).all() and (v >= 0).all(), "input step at/after origin"
    return out


# ------------------------------------------------------------------------- features
def _calendar(series: Series, groups) -> pd.DataFrame:
    parts = []
    if "calendar_benchmark" in groups:
        parts.append(calendar_benchmark(series.times, daily=series.track == "A"))
    if "calendar_rich" in groups:
        parts.append(calendar_rich(series.times))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=series.times)


def _static(stations) -> pd.DataFrame:
    st = pd.read_csv(C.STATIONS, dtype={"code": str}).set_index("code").loc[stations, C.STATIC_FEATURES]
    assert not st.isna().any().any(), "missing static attributes for a benchmark station"
    return st


def _check_groups(groups):
    bad = set(groups) - set(FEATURE_GROUPS)
    if bad:
        raise ValueError(f"unknown feature groups {bad}; choose from {FEATURE_GROUPS}")
    if "weather" in groups:
        if not C.WEATHER_HOURLY.exists():
            raise NotImplementedError(
                "weather slot: data/interim/weather_hourly.parquet does not exist (download blocked, "
                "DATA_LOG 6). When implemented, weather must use only hours with interval_end <= origin.")
        raise NotImplementedError("weather features not implemented yet (data present; define them first)")


# ------------------------------------------------------------------------- builders
def build_tabular(series: Series, groups, windows: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per station x origin x horizon step."""
    cfg = C.TRACKS[series.track]
    H = cfg["horizon"]
    S = len(series.stations)
    p = windows.origin_pos.to_numpy()
    N = len(p)
    rows_origin = np.repeat(np.arange(N), H * S)                       # origin-major
    rows_h = np.tile(np.repeat(np.arange(H), S), N)
    rows_s = np.tile(np.arange(S), N * H)
    tpos = p[rows_origin] + rows_h

    df = pd.DataFrame({
        "code": pd.Categorical(np.asarray(series.stations)[rows_s], categories=series.stations),
        "origin": series.times[p][rows_origin],
        "h": (rows_h + 1).astype("int8"),
        "target_time": series.times[tpos],
        "y": series.Y[tpos, rows_s],
        "split": windows.split.to_numpy()[rows_origin],
        "period": windows.period.to_numpy()[rows_origin],
    })
    if series.track == "B":
        df["dst_window"] = windows.dst_window.to_numpy()[rows_origin]
        df["is_service_interval"] = series.meta.is_service_interval.to_numpy()[tpos]
    if "lags" in groups:
        pos = input_positions(series, p)
        for name, P in pos.items():
            for j in range(P.shape[1]):
                col = f"lag_{j + 1}" if name == "lag" else f"wk_{j}"
                df[col] = series.Y[P[rows_origin, j], rows_s]
    cal = _calendar(series, groups)
    for c in cal.columns:
        df[c] = cal[c].to_numpy()[tpos]
    if "station_static" in groups:
        st = _static(series.stations)
        for c in C.STATIC_FEATURES:
            v = st[c].to_numpy()[rows_s]
            df[c] = pd.Categorical(v) if c == "id_trazado" else v.astype("float32")
    return df


def build_tensor(series: Series, groups, windows: pd.DataFrame) -> dict:
    """Arrays with the station axis in station_order.csv order.

    values    (T, S, 1)  ridership (the only dynamic station feature so far)
    calendar  (T, C)     calendar features; index them at TARGET steps
    static    (S, K)     numeric static features, id_trazado one-hot
    windows   DataFrame  origin_pos + split/period labels
    Use window_arrays() to cut X/Y for a set of windows.
    """
    out = {"track": series.track, "times": series.times, "stations": list(series.stations),
           "values": series.Y[:, :, None].copy(), "value_features": ["ridership"],
           "windows": windows, "lookback": C.TRACKS[series.track]["lookback"],
           "horizon": C.TRACKS[series.track]["horizon"],
           "weekly_hist": C.TRACKS[series.track]["weekly_hist"], "week_steps": week_steps(series),
           "groups": tuple(groups)}
    cal = _calendar(series, groups)
    out["calendar"] = cal.to_numpy("float32")
    out["calendar_features"] = list(cal.columns)
    if "station_static" in groups:
        st = _static(series.stations)
        num = st.drop(columns="id_trazado").astype("float32")
        oh = pd.get_dummies(st.id_trazado, prefix="tz").astype("float32")
        s = pd.concat([num, oh], axis=1)
        out["static"], out["static_features"] = s.to_numpy("float32"), list(s.columns)
    if series.track == "B":
        out["is_service_interval"] = series.meta.is_service_interval.to_numpy()
    return out


def window_arrays(ds: dict, windows: pd.DataFrame = None) -> dict:
    """X/Y arrays for tensor datasets: X_lag (N, L, S, F), [X_wk (N, K, S, F)],
    Y (N, H, S), cal_target (N, H, C), positions of every step used."""
    w = ds["windows"] if windows is None else windows
    p = w.origin_pos.to_numpy()
    series = Series(ds["track"], ds["values"][:, :, 0], ds["times"], ds["stations"])
    pos = input_positions(series, p)
    H = ds["horizon"]
    tpos = p[:, None] + np.arange(H)[None, :]
    lag_pos = pos["lag"][:, ::-1]                                      # chronological order
    out = {"X_lag": ds["values"][lag_pos], "Y": ds["values"][tpos, :, 0],
           "cal_target": ds["calendar"][tpos], "target_pos": tpos, "lag_pos": lag_pos}
    if "wk" in pos:
        wk_pos = pos["wk"][:, ::-1]
        out["X_wk"], out["wk_pos"] = ds["values"][wk_pos], wk_pos
    return out


def build_dataset(track: str, fmt: str, feature_groups=("lags", "calendar_benchmark"),
                  splits=None, origin_stride: int = 1, series: Series = None):
    """track 'A'|'B', fmt 'tabular'|'tensor'. `series` overrides the data (tests)."""
    _check_groups(feature_groups)
    series = load_series(track) if series is None else series
    windows = make_windows(series, origin_stride)
    if splits is not None:
        windows = windows[windows.split.isin(splits)].reset_index(drop=True)
    if fmt == "tabular":
        return build_tabular(series, feature_groups, windows)
    if fmt == "tensor":
        return build_tensor(series, feature_groups, windows)
    raise ValueError(fmt)
