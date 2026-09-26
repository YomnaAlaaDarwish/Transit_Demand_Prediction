"""07_build_dataset.py -- write model-ready datasets for Track A (daily) and Track B (15-min).

The logic lives in thesis_pipeline/dataset.py (build_dataset); this script only saves
its output. No model is trained here.

Outputs (rebuildable, not committed):
  data/processed/track_a_daily/tabular_<groups>.parquet
  data/processed/track_a_daily/tensor_<groups>.npz  (+ windows_<groups>.parquet)
  data/processed/track_b_15min/tensor_<groups>.npz  (+ windows_<groups>.parquet)
  data/processed/track_b_15min/tabular_<groups>.parquet   only with --b-tabular
       (about 11 M rows per full pass; written in chunks; use --b-stride to thin origins)
  data/processed/<track>/summary_<groups>.json  split sizes and date ranges

Run from the repo root:
  python scripts/07_build_dataset.py [--groups lags calendar_benchmark ...] [--b-tabular] [--b-stride N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thesis_pipeline import config as C  # noqa: E402
from thesis_pipeline.dataset import (build_tabular, build_tensor, load_series,  # noqa: E402
                                     make_windows, _check_groups)

OUT = {"A": C.PROCESSED / "track_a_daily", "B": C.PROCESSED / "track_b_15min"}


def summary(windows):
    g = windows.groupby("split")
    s = {k: {"n_origins": int(len(v)), "first_origin": str(v.origin.min()),
             "last_target": str(v.last_target.max())} for k, v in g}
    t = windows[windows.split == "test"]
    s["test_periods"] = t.period.value_counts().to_dict()
    if "dst_window" in t:
        s["test_dst_windows"] = t[t.dst_window != ""].dst_window.value_counts().to_dict()
    return s


def save_tensor(ds, path: Path):
    np.savez_compressed(path.with_suffix(".npz"), values=ds["values"], calendar=ds["calendar"],
                        static=ds.get("static", np.zeros((len(ds["stations"]), 0), "float32")),
                        times=ds["times"].astype("datetime64[ns]").to_numpy(),
                        stations=np.array(ds["stations"]),
                        calendar_features=np.array(ds["calendar_features"]),
                        static_features=np.array(ds.get("static_features", [])),
                        lookback=ds["lookback"], horizon=ds["horizon"],
                        weekly_hist=ds["weekly_hist"], week_steps=ds["week_steps"],
                        **({"is_service_interval": ds["is_service_interval"]} if "is_service_interval" in ds else {}))
    ds["windows"].to_parquet(path.parent / f"windows_{path.stem.split('_', 1)[1]}.parquet", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["lags", "calendar_benchmark", "calendar_rich", "station_static"])
    ap.add_argument("--b-tabular", action="store_true")
    ap.add_argument("--b-stride", type=int, default=1)
    a = ap.parse_args()
    _check_groups(a.groups)
    tag = "-".join(g.replace("calendar_", "cal") .replace("station_", "") for g in a.groups)

    for track in ("A", "B"):
        OUT[track].mkdir(parents=True, exist_ok=True)
        series = load_series(track)
        windows = make_windows(series)
        s = summary(windows)
        (OUT[track] / f"summary_{tag}.json").write_text(json.dumps(s, indent=2))
        print(f"[{track}] {json.dumps(s)}")

        ds = build_tensor(series, a.groups, windows)
        save_tensor(ds, OUT[track] / f"tensor_{tag}")
        print(f"[{track}] tensor values {ds['values'].shape}, calendar {ds['calendar'].shape}")

        if track == "A":
            df = build_tabular(series, a.groups, windows)
            df.to_parquet(OUT[track] / f"tabular_{tag}.parquet", index=False)
            print(f"[A] tabular {df.shape}")
        elif a.b_tabular:
            w = windows.iloc[::a.b_stride].reset_index(drop=True)
            path = OUT[track] / f"tabular_{tag}_stride{a.b_stride}.parquet"
            writer, n = None, 0
            for k in range(0, len(w), 4000):
                df = build_tabular(series, a.groups, w.iloc[k:k + 4000])
                t = pa.Table.from_pandas(df, preserve_index=False)
                writer = writer or pq.ParquetWriter(path, t.schema)
                writer.write_table(t)
                n += len(df)
            writer.close()
            print(f"[B] tabular {n} rows -> {path.name}")


if __name__ == "__main__":
    main()
