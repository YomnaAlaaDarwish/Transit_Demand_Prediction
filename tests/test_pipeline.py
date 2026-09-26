"""Tests for the thesis dataset builder and metrics.  Run:  python -m pytest -q tests/

Synthetic tests encode the time index into the ridership values (Y[t, s] = t), so any
feature value can be traced back to the step it came from. Tests marked `real` need
the rebuilt data/interim files and are skipped if they are missing.
"""
import numpy as np
import pandas as pd
import pytest

from thesis_pipeline import config as C
from thesis_pipeline.dataset import (Series, build_dataset, load_series, make_windows,
                                     station_order, window_arrays)
from thesis_pipeline.metrics import arctan_ape, evaluate_long, maape

GROUPS = ("lags", "calendar_benchmark", "calendar_rich", "station_static")
real = pytest.mark.skipif(not C.RIDERSHIP_15MIN.exists(), reason="data/interim not built")


def synthetic(track, fill="time"):
    order = station_order()
    S = len(order)
    if track == "A":
        times = pd.date_range("2015-08-01", C.DATA_END[:10], freq="D")
    else:
        g = pd.date_range("2015-08-01", C.DATA_END, freq="15min")
        times = g[~g.hour.isin(C.BENCHMARK_EXCLUDED_HOURS)]
    T = len(times)
    t = np.arange(T, dtype="float32")[:, None]
    s = np.arange(S, dtype="float32")[None, :]
    Y = np.broadcast_to(t if fill == "time" else s, (T, S)).astype("float32").copy()
    meta = pd.DataFrame({"is_service_interval": True}, index=times)
    return Series(track, Y, times, order, meta)


# ---------------------------------------------------------------- 1. leakage
@pytest.mark.parametrize("track", ["A", "B"])
def test_tabular_uses_no_ridership_at_or_after_origin(track):
    s = synthetic(track)
    df = build_dataset(track, "tabular", GROUPS, origin_stride=(1 if track == "A" else 37), series=s)
    pos = pd.Series(np.arange(len(s.times)), index=s.times)
    o = pos[df.origin].to_numpy()
    tt = pos[df.target_time].to_numpy()
    assert np.array_equal(df.y.to_numpy(), tt), "target is not the value at target_time"
    assert (tt >= o).all()
    obs = [c for c in df.columns if c.startswith(("lag_", "wk_"))]
    assert obs, "no lag features built"
    assert (df[obs].to_numpy() < o[:, None]).all(), "a ridership feature comes from origin or later"
    assert (df["lag_1"].to_numpy() == o - 1).all(), "lag_1 must be the last step before the origin"


@pytest.mark.parametrize("track", ["A", "B"])
def test_tensor_windows_use_no_ridership_at_or_after_origin(track):
    s = synthetic(track)
    ds = build_dataset(track, "tensor", GROUPS, series=s)
    w = ds["windows"].iloc[:: (1 if track == "A" else 29)]
    a = window_arrays(ds, w)
    p = w.origin_pos.to_numpy()
    assert (a["X_lag"][..., 0] < p[:, None, None]).all()
    assert (a["X_lag"][:, -1, :, 0] == (p - 1)[:, None]).all(), "last input step must be origin-1"
    if "X_wk" in a:
        assert (a["X_wk"][..., 0] < p[:, None, None]).all()
    assert (a["Y"][:, 0, :] == p[:, None]).all(), "first target must be the origin step"


def test_weather_slot_is_a_stub_until_data_exists():
    if C.WEATHER_HOURLY.exists():
        pytest.skip("weather downloaded; replace this test with a real weather leakage test")
    with pytest.raises(NotImplementedError):
        build_dataset("A", "tabular", ("lags", "weather"), series=synthetic("A"))


@pytest.mark.parametrize("track", ["A", "B"])
def test_splits_are_chronological_and_disjoint(track):
    w = make_windows(synthetic(track))
    tr, va, te = (w[w.split == k] for k in ("train", "val", "test"))
    train_end = pd.Timestamp(C.TRACKS[track]["train_end"])
    assert tr.last_target.max() < va.origin.min(), "train targets overlap validation"
    assert va.last_target.max() < train_end, "validation reaches into the test period"
    assert te.origin.min() >= train_end
    assert te.last_target.max() <= pd.Timestamp(C.TRACKS[track]["test_end"])
    assert set(te.period) <= {"stable", "protest", "covid"}


# ---------------------------------------------------------------- 2. station axes
@pytest.mark.parametrize("track", ["A", "B"])
def test_station_axis_follows_station_order(track):
    s = synthetic(track, fill="station")
    order = station_order()
    ds = build_dataset(track, "tensor", GROUPS, series=s)
    assert ds["stations"] == order
    assert (ds["values"][0, :, 0] == np.arange(len(order))).all()
    df = build_dataset(track, "tabular", ("lags",), splits=("test",), origin_stride=500, series=s)
    idx = {c: i for i, c in enumerate(order)}
    assert (df.y.to_numpy() == df.code.astype(str).map(idx).to_numpy()).all()
    assert list(df.code.cat.categories) == order


@real
def test_real_files_follow_station_order():
    order = station_order()
    assert order == sorted(order) and len(order) == 147 and all(len(c) == 5 for c in order)
    r = pd.read_parquet(C.RIDERSHIP_15MIN)
    assert list(r.columns[1:]) == order
    for name in ("benchmark", "physical_clean"):
        f = C.INTERIM / "graphs" / f"adj_{name}.npy"
        if not f.exists():
            pytest.skip("graphs not built")
        A = np.load(f)
        e = pd.read_csv(C.INTERIM / "graphs" / f"edges_{name}.csv", dtype={"code_1": str, "code_2": str})
        i = {c: k for k, c in enumerate(order)}
        assert A.shape == (147, 147) and (A == A.T).all() and A.sum() == 2 * len(e)
        assert all(A[i[a], i[b]] == 1 for a, b in zip(e.code_1, e.code_2))


# ---------------------------------------------------------------- 3. daily sums
@real
def test_track_a_daily_equals_15min_and_raw_source():
    A = load_series("A")
    r = pd.read_parquet(C.RIDERSHIP_15MIN).set_index("timestamp")
    keep = ~r.index.hour.isin(C.BENCHMARK_EXCLUDED_HOURS)
    d15 = r[keep].groupby(r.index[keep].date).sum()
    assert np.array_equal(A.Y, d15.to_numpy("float32"))
    # independent path from the untouched raw parquet (benchmark read_data logic)
    raw = pd.read_parquet(C.ROOT / "data/transmilenio_transactions.parquet")
    raw["timestamp"] = pd.to_datetime(raw.timestamp)
    cols = {c[1:6]: c for c in raw.columns if c.startswith("(")}
    raw = raw.groupby("timestamp")[[cols[c] for c in A.stations]].sum()
    raw = raw[raw.index <= pd.Timestamp(C.DATA_END)]
    raw = raw[~raw.index.hour.isin(C.BENCHMARK_EXCLUDED_HOURS)]
    daily_raw = raw.resample("D").sum()
    assert np.array_equal(A.Y, daily_raw.to_numpy("float32"))
    assert A.Y.astype("float64").sum() == r[keep].to_numpy().sum()


# ---------------------------------------------------------------- 4. metrics
def test_maape_matches_definition_and_zero_rules():
    y = np.array([100.0, 100.0, 0.0, 0.0])
    p = np.array([100.0, 50.0, 10.0, 0.0])
    a = arctan_ape(y, p)
    assert np.allclose(a, [0.0, np.arctan(0.5), np.pi / 2, 0.0])
    assert np.isclose(maape(y, p), a.mean())


def test_system_maape_averages_per_origin_first():
    df = pd.DataFrame({"origin": [0, 0, 1], "period": "stable", "y": [10.0, 10.0, 10.0],
                       "yhat": [10.0, 0.0, 10.0], "is_service_interval": [True, False, True]})
    r = evaluate_long(df).iloc[0]
    assert np.isclose(r.maape, np.mean([np.mean([0, np.pi / 4]), 0.0]))
    rs = evaluate_long(df, service_only=True).iloc[0]
    assert np.isclose(rs.maape, 0.0) and rs.n_cells == 2
