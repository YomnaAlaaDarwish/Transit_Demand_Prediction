"""06_download_weather.py -- hourly historical weather for Bogota, per model grid cell.

Stores facts only (no model features). Source: Open-Meteo Historical Weather API
https://archive-api.open-meteo.com/v1/archive, models era5 and era5_land, each
downloaded separately.

Steps (run all by default; each step reads only files written by earlier steps):
  probe     for every station in data/interim/stations.csv, ask each model which
            grid cell serves its coordinates (1-day request, several locations per
            call); save raw responses; write data/interim/station_weather_cell.csv
  download  for each model, download the full period once per UNIQUE cell
            (requested at the cell's own coordinates); save raw responses
  process   raw responses -> data/interim/weather_hourly.parquet
  checks    gaps, seasonal/diurnal rain, era5 vs era5_land -> data/interim/weather_checks/

Request settings, identical for every call:
  timezone=America/Bogota   times are local (UTC-5, no DST); the response's
                            utc_offset_seconds is asserted to be -18000
  cell_selection=nearest    the nearest grid cell (the default "land" may pick a
                            farther cell with similar terrain height)
  elevation=nan             no statistical downscaling: values are the grid cell's
                            own, and the returned elevation is the cell's mean height,
                            so every station in one cell gets the same series
  timeformat=iso8601, format=json (raw responses stay human-readable)

HTTP: requests-cache (on-disk SQLite cache, never expires) + retry-requests
(5 retries, exponential backoff) -- the same caching/retry stack that the
openmeteo-requests client wraps. JSON is used instead of that client's
FlatBuffers so the raw responses are kept exactly as downloaded, in readable form.

Time convention (Open-Meteo): each hourly value carries a label T (local time).
  precipitation, rain  = sum over the PRECEDING hour  -> interval_start=T-1h, interval_end=T
  temperature_2m, relative_humidity_2m, cloud_cover, wind_speed_10m
                       = instantaneous at T          -> obs_time=T
weather_hourly.parquet keeps obs_time (=T), interval_start (=T-1h) and interval_end (=T)
on every row; use interval_* for the accumulated variables and obs_time for the others.

Outputs:
  data/raw/weather/probe_<model>_<k>.json, <model>_<cell_id>.json, http_cache.sqlite
  data/interim/station_weather_cell.csv  code, model, cell_id, cell_lat, cell_lon, cell_elevation, distance_km
  data/interim/weather_hourly.parquet    model, cell_id, obs_time, interval_start, interval_end, <variables>
  data/interim/weather_checks/*.png, *.csv

Run from the repo root:  python scripts/06_download_weather.py [--step probe|download|process|checks|all]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ROOT / "data/interim/stations.csv"
RAW = ROOT / "data/raw/weather"
OUT = ROOT / "data/interim"
CHECKS = OUT / "weather_checks"

URL = "https://archive-api.open-meteo.com/v1/archive"
MODELS = ["era5", "era5_land"]
START, END = "2015-07-01", "2021-06-30"
VARIABLES = ["precipitation", "rain", "temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m"]
ACCUMULATED = ["precipitation", "rain"]
COMMON = {"timezone": "America/Bogota", "cell_selection": "nearest", "elevation": "nan",
          "timeformat": "iso8601", "format": "json"}
PROBE_CHUNK = 50  # locations per probe call
UTC_OFFSET = -18000


def session():
    import requests_cache
    from retry_requests import retry
    s = requests_cache.CachedSession(str(RAW / "http_cache"), backend="sqlite", expire_after=-1)
    return retry(s, retries=5, backoff_factor=0.5)


def get(sess, params):
    r = sess.get(URL, params=params, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def cell_id(lat, lon):
    return f"{lat:.4f}_{lon:.4f}"


def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


# --------------------------------------------------------------------------- probe
def probe(sess):
    st = pd.read_csv(STATIONS, dtype={"code": str})
    st = st[st.lat.notna()].reset_index(drop=True)
    rows = []
    for model in MODELS:
        for k in range(0, len(st), PROBE_CHUNK):
            part = st.iloc[k:k + PROBE_CHUNK]
            params = {**COMMON, "models": model, "start_date": START, "end_date": START,
                      "hourly": "temperature_2m",
                      "latitude": ",".join(f"{x:.6f}" for x in part.lat),
                      "longitude": ",".join(f"{x:.6f}" for x in part.lon)}
            resp = get(sess, params)
            resp = resp if isinstance(resp, list) else [resp]
            assert len(resp) == len(part), (len(resp), len(part))
            (RAW / f"probe_{model}_{k // PROBE_CHUNK}.json").write_text(
                json.dumps({"request": params, "response": resp}, ensure_ascii=False))
            for i, ((_, s), r) in enumerate(zip(part.iterrows(), resp)):
                assert r.get("location_id", i) == i, "multi-location response out of order"
                assert r["utc_offset_seconds"] == UTC_OFFSET, r["utc_offset_seconds"]
                rows.append({"code": s.code, "model": model, "cell_id": cell_id(r["latitude"], r["longitude"]),
                             "cell_lat": r["latitude"], "cell_lon": r["longitude"],
                             "cell_elevation": r["elevation"],
                             "distance_km": round(haversine_km(s.lat, s.lon, r["latitude"], r["longitude"]), 3)})
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "station_weather_cell.csv", index=False)
    b147 = set(st.loc[st.in_benchmark_147, "code"])
    for model, g in m.groupby("model"):
        g147 = g[g.code.isin(b147)]
        print(f"[probe] {model}: {g.cell_id.nunique()} unique cells for {len(g)} stations; "
              f"{g147.cell_id.nunique()} for the 147; station->cell distance "
              f"median {g147.distance_km.median():.2f} km, max {g147.distance_km.max():.2f} km; "
              f"cell elevation {g.cell_elevation.min():.0f}-{g.cell_elevation.max():.0f} m")
    return m


# ------------------------------------------------------------------------ download
def download(sess):
    m = pd.read_csv(OUT / "station_weather_cell.csv", dtype={"code": str})
    cells = m.drop_duplicates(["model", "cell_id"])
    for _, c in cells.iterrows():
        f = RAW / f"{c.model}_{c.cell_id}.json"
        params = {**COMMON, "models": c.model, "start_date": START, "end_date": END,
                  "hourly": ",".join(VARIABLES), "latitude": f"{c.cell_lat:.6f}", "longitude": f"{c.cell_lon:.6f}"}
        r = get(sess, params)
        assert r["utc_offset_seconds"] == UTC_OFFSET
        back = cell_id(r["latitude"], r["longitude"])
        if back != c.cell_id:
            raise RuntimeError(f"{c.model}: requesting cell {c.cell_id} returned cell {back}")
        f.write_text(json.dumps({"request": params, "response": r}, ensure_ascii=False))
        print(f"[download] {c.model} {c.cell_id}: {len(r['hourly']['time'])} hours -> {f.name}")
    print(f"[download] {len(cells)} files ({cells.groupby('model').size().to_dict()})")


# ------------------------------------------------------------------------- process
def process():
    m = pd.read_csv(OUT / "station_weather_cell.csv", dtype={"code": str})
    frames = []
    for _, c in m.drop_duplicates(["model", "cell_id"]).iterrows():
        r = json.loads((RAW / f"{c.model}_{c.cell_id}.json").read_text())["response"]
        h = r["hourly"]
        df = pd.DataFrame({v: pd.to_numeric(pd.Series(h[v]), errors="coerce").astype("float32") for v in VARIABLES})
        units = r["hourly_units"]
        assert units["precipitation"] == "mm" and units["temperature_2m"] == "°C", units
        df.insert(0, "obs_time", pd.to_datetime(h["time"], format="%Y-%m-%dT%H:%M"))
        df.insert(0, "cell_id", c.cell_id)
        df.insert(0, "model", c.model)
        frames.append(df)
    w = pd.concat(frames, ignore_index=True)
    w.insert(3, "interval_start", w.obs_time - pd.Timedelta(hours=1))
    w.insert(4, "interval_end", w.obs_time)
    w = w.sort_values(["model", "cell_id", "obs_time"]).reset_index(drop=True)
    assert not w.duplicated(["model", "cell_id", "obs_time"]).any()
    w.to_parquet(OUT / "weather_hourly.parquet", index=False)
    print(f"[process] weather_hourly.parquet: {len(w)} rows, {w.groupby('model').cell_id.nunique().to_dict()} cells")
    return w


# -------------------------------------------------------------------------- checks
def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color="#1a1a19")
    ax.grid(axis="y", color="#e6e5df", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors="#6b6a63")


def checks(w=None):
    CHECKS.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(OUT / "weather_hourly.parquet") if w is None else w
    m = pd.read_csv(OUT / "station_weather_cell.csv", dtype={"code": str})
    st = pd.read_csv(STATIONS, dtype={"code": str})
    b147 = set(st.loc[st.in_benchmark_147, "code"])
    colors = {"era5": "#2a78d6", "era5_land": "#eb6834"}
    report = {}

    # 1. gaps: every cell must have every hour of the period, and no null values
    expected = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta(hours=23), freq="h")
    gap_rows = []
    for (model, cid), g in w.groupby(["model", "cell_id"]):
        missing = expected.difference(pd.DatetimeIndex(g.obs_time))
        extra = pd.DatetimeIndex(g.obs_time).difference(expected)
        nulls = g[VARIABLES].isna().sum().to_dict()
        gap_rows.append({"model": model, "cell_id": cid, "n_hours": len(g), "expected": len(expected),
                         "missing_hours": len(missing), "extra_hours": len(extra),
                         "first_missing": missing[:5].astype(str).tolist(), **{f"null_{k}": v for k, v in nulls.items()}})
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(CHECKS / "gaps_and_nulls.csv", index=False)
    print("[checks] hours per cell vs expected, and null counts per variable:")
    print(gaps.drop(columns=["first_missing"]).to_string(index=False))

    # station-weighted area mean over the 147 stations' cells (a cell counts once per station it serves)
    weights = m[m.code.isin(b147)].groupby(["model", "cell_id"]).size().rename("n_st").reset_index()
    ww = w.merge(weights, on=["model", "cell_id"])
    area = (ww.assign(**{v: ww[v] * ww.n_st for v in ACCUMULATED})
              .groupby(["model", "obs_time", "interval_start"])[ACCUMULATED + ["n_st"]].sum(min_count=1))
    area = area[ACCUMULATED].div(area.n_st, axis=0).reset_index()

    # 2. monthly rain climatology (by month of interval_start)
    area["month"] = area.interval_start.dt.month
    area["year"] = area.interval_start.dt.year
    mon = (area.groupby(["model", "year", "month"]).precipitation.sum().groupby(["model", "month"]).mean()
               .unstack(0))
    mon.to_csv(CHECKS / "monthly_precip_mean_mm.csv")
    report["wettest_months"] = {mdl: mon[mdl].nlargest(4).index.tolist() for mdl in mon}
    print(f"[checks] mean monthly precipitation (mm), station-weighted:\n{mon.round(1).to_string()}")
    print(f"[checks] 4 wettest months: {report['wettest_months']}")
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    x = np.arange(1, 13)
    for i, mdl in enumerate(mon):
        ax.bar(x + (i - 0.5) * 0.38, mon[mdl], width=0.36, color=colors[mdl], label=mdl)
    ax.set_xticks(x, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("mm / month", fontsize=8, color="#6b6a63")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, f"Mean monthly precipitation, {START[:4]}-{END[:4]} (station-weighted cells)")
    fig.tight_layout()
    fig.savefig(CHECKS / "monthly_precipitation.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # 3. diurnal rain cycle (by hour of interval_start, local time)
    area["hour"] = area.interval_start.dt.hour
    diu = area.groupby(["model", "hour"]).precipitation.mean().unstack(0)
    diu.to_csv(CHECKS / "diurnal_precip_mean_mm_per_h.csv")
    report["peak_hour_start"] = {mdl: int(diu[mdl].idxmax()) for mdl in diu}
    print(f"[checks] precipitation peak (interval start hour, local): {report['peak_hour_start']}")
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    for mdl in diu:
        ax.plot(diu.index, diu[mdl], color=colors[mdl], lw=2, marker="o", ms=4, label=mdl)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("hour of interval start (local)", fontsize=8, color="#6b6a63")
    ax.set_ylabel("mm / h", fontsize=8, color="#6b6a63")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Mean precipitation by hour of day")
    fig.tight_layout()
    fig.savefig(CHECKS / "diurnal_precipitation.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # 4. era5 vs era5_land daily precipitation (station-weighted area mean, local days by interval_start)
    daily = area.groupby(["model", area.interval_start.dt.date]).precipitation.sum().unstack(0).dropna()
    daily.to_csv(CHECKS / "daily_precip_by_model_mm.csv")
    if set(MODELS) <= set(daily.columns):
        a, b = daily["era5"], daily["era5_land"]
        report["daily_pearson"] = round(float(a.corr(b)), 3)
        report["daily_spearman"] = round(float(a.rank().corr(b.rank())), 3)  # rank Pearson = Spearman
        report["mean_daily_mm"] = {"era5": round(float(a.mean()), 2), "era5_land": round(float(b.mean()), 2)}
        # per station: correlation of its era5 cell vs its era5_land cell
        piv = w.pivot_table(index=w.interval_start.dt.date, columns=["model", "cell_id"],
                            values="precipitation", aggfunc="sum")
        cm = m[m.code.isin(b147)].pivot(index="code", columns="model", values="cell_id")
        r = [piv[("era5", e)].corr(piv[("era5_land", l)]) for e, l in zip(cm.era5, cm.era5_land)]
        report["per_station_daily_pearson"] = {"min": round(float(np.min(r)), 3),
                                               "median": round(float(np.median(r)), 3),
                                               "max": round(float(np.max(r)), 3)}
        print(f"[checks] era5 vs era5_land daily precipitation: {report}")
        fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=150)
        ax.scatter(a, b, s=8, color="#2a78d6", alpha=0.35, edgecolors="none")
        lim = float(max(a.max(), b.max())) * 1.03
        ax.plot([0, lim], [0, lim], color="#9a9990", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("era5 daily precipitation (mm)", fontsize=8, color="#6b6a63")
        ax.set_ylabel("era5_land daily precipitation (mm)", fontsize=8, color="#6b6a63")
        _style(ax, f"Daily precipitation, r = {report['daily_pearson']}")
        fig.tight_layout()
        fig.savefig(CHECKS / "era5_vs_era5_land_daily.png", facecolor="white", bbox_inches="tight")
        plt.close(fig)
    (CHECKS / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["probe", "download", "process", "checks", "all"])
    step = ap.parse_args().step
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"[run] {date.today()} step={step}")
    if step in ("probe", "download", "all"):
        sess = session()
    if step in ("probe", "all"):
        probe(sess)
    if step in ("download", "all"):
        download(sess)
    w = process() if step in ("process", "all") else None
    if step in ("checks", "all"):
        checks(w)


if __name__ == "__main__":
    main()
