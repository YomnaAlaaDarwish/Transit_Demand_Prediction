"""04_build_ridership_15min.py -- clean 15-min station ridership + per-timestamp flags.

Decisions applied (approved after the 02 audit, see DATA_LOG.md section 4):
  * duplicate timestamps: drop the all-zero overflow row (identical to summing);
  * end at 2021-04-30 23:45 -- 2021-05-01 is an empty overflow column of the
    April-2021 raw file;
  * missing slots (all overnight, 00:00-03:45) are added and filled with 0;
  * whole-system zeros in core service hours are kept as real (flagged);
  * timestamps are local Bogota time (UTC-5, no DST), naive, marking the START
    of the 15-min interval (inferred from data, see DATA_LOG.md section 1b(c)).

Inputs (read-only): data/transmilenio_transactions.parquet, data/interim/station_order.csv
Outputs:
  data/interim/ridership_15min.parquet      timestamp + 147 station columns (5-char codes,
                                            station_order.csv order), int32 validations
  data/interim/time_features_15min.parquet  one row per timestamp: calendar facts + flags

Run from the repo root:  python scripts/04_build_ridership_15min.py
"""
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RIDERSHIP = ROOT / "data/transmilenio_transactions.parquet"
ORDER = ROOT / "data/interim/station_order.csv"
OUT = ROOT / "data/interim"

START = pd.Timestamp("2015-08-01 00:00")
END = pd.Timestamp("2021-04-30 23:45")
OVERNIGHT_HOURS = [23, 0, 1, 2, 3]  # = the benchmark's dropped hours [0,1,2,3,23]
EDGE_HOURS = [4, 22]                # opening / closing hour: open or closed depending on timetable


def main():
    order = pd.read_csv(ORDER, dtype={"code": str}).code.tolist()
    raw = pd.read_parquet(RIDERSHIP)
    raw["timestamp"] = pd.to_datetime(raw.timestamp, format="%Y-%m-%d %H:%M:%S")
    col_of = {c[1:6]: c for c in raw.columns if c.startswith("(")}
    cols = [col_of[c] for c in order]
    raw_total = raw[cols].to_numpy().sum()

    # --- duplicates: every pair is (data row, all-zero row) -> keep the data row
    d = raw[raw.timestamp.duplicated(keep=False)]
    rowsum = d[cols].sum(axis=1)
    per_ts = rowsum.groupby(d.timestamp).agg(["size", "min"])
    assert (per_ts["size"] == 2).all() and (per_ts["min"] == 0).all(), "a duplicate pair is not (x, 0)"
    df = (raw.assign(_s=raw[cols].sum(axis=1))
             .sort_values(["timestamp", "_s"], ascending=[True, False])
             .drop_duplicates("timestamp", keep="first"))
    print(f"[dup] dropped {len(raw) - len(df)} all-zero duplicate rows")

    # --- period ------------------------------------------------------------
    tail = df[df.timestamp > END]
    assert tail[cols].to_numpy().sum() == 0, "data after END is not all zero"
    df = df[(df.timestamp >= START) & (df.timestamp <= END)]
    print(f"[period] dropped {len(tail)} rows after {END} (all zero)")

    # --- full grid, fill missing with 0 --------------------------------------
    grid = pd.date_range(START, END, freq="15min", name="timestamp")
    wide = df.set_index("timestamp")[cols].reindex(grid)
    filled = wide.isna().all(axis=1)
    assert not wide[~filled].isna().any().any(), "partially missing rows"
    assert filled[filled].index.hour.isin([0, 1, 2, 3]).all(), "a filled slot is in service hours"
    wide = wide.fillna(0)
    assert (wide.to_numpy() % 1 == 0).all() and wide.to_numpy().min() >= 0
    wide = wide.astype("int32")
    wide.columns = order
    assert wide.to_numpy().sum() == raw_total, "total validations changed"
    print(f"[grid] {len(grid)} slots ({len(grid) // 96} days x 96); filled {int(filled.sum())} missing "
          f"overnight slots with 0; total validations preserved ({raw_total:,.0f})")

    # --- per-timestamp facts and flags ---------------------------------------
    co = holidays.Colombia(years=range(2015, 2022))
    t = pd.DataFrame(index=grid)
    t["date"] = grid.date
    t["slot_of_day"] = (grid.hour * 4 + grid.minute // 15).astype("int16")
    t["hour"] = grid.hour.astype("int8")
    t["minute"] = grid.minute.astype("int8")
    t["dayofweek"] = grid.dayofweek.astype("int8")  # Monday=0
    t["holiday_name"] = [co.get(x) for x in grid.date]
    t["is_holiday"] = t.holiday_name.notna()
    t["day_type"] = np.where(t.is_holiday | (t.dayofweek == 6), "sunday_holiday",
                             np.where(t.dayofweek == 5, "saturday", "weekday"))
    t["system_total"] = wide.sum(axis=1).astype("int64")
    t["service_band"] = np.where(t.hour.isin(OVERNIGHT_HOURS), "overnight",
                                 np.where(t.hour.isin(EDGE_HOURS), "edge", "core"))
    t["in_benchmark_hours"] = ~t.hour.isin(OVERNIGHT_HOURS)
    t["is_service_interval"] = (t.service_band == "core") | ((t.service_band == "edge") & (t.system_total > 0))
    t["system_suspended"] = (t.service_band == "core") & (t.system_total == 0)
    t["filled_zero"] = filled.to_numpy()
    t = t.reset_index()

    n = t.service_band.value_counts().to_dict()
    closed_edge = int(((t.service_band == "edge") & (t.system_total == 0)).sum())
    print(f"[flags] bands {n}; edge slots closed (system zero) {closed_edge}; "
          f"is_service_interval {int(t.is_service_interval.sum())}; system_suspended "
          f"{int(t.system_suspended.sum())} on {t[t.system_suspended].date.nunique()} dates; "
          f"filled_zero {int(t.filled_zero.sum())}")
    print(t[t.system_suspended].groupby("date").timestamp.agg(["size", "min", "max"]).to_string())

    wide.reset_index().to_parquet(OUT / "ridership_15min.parquet", index=False)
    t.to_parquet(OUT / "time_features_15min.parquet", index=False)
    print(f"[write] ridership_15min.parquet {wide.shape}, time_features_15min.parquet {t.shape}")


if __name__ == "__main__":
    main()
