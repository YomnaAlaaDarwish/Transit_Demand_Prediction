"""Builds the pooled (station, 15-min interval) tabular dataset -- the
adaptation of the paper's per-(trip, stop-visit) tabular row to our
station-level-only Bogota data (see config.py for the full rationale).

Unlike DST-TransitNet (one many-to-many graph model) and TSB-Forecast (147
independent per-station models), this reproduction trains a SINGLE model
per algorithm on ALL stations pooled together, with station identity as
just another categorical input -- exactly matching the paper's own
non-spatial, non-recurrent design (their Sec. 4.4 explicitly notes "stop-
level modelling lacks inductive biases on the spatial structure of transit
networks", which this reproduction faithfully inherits).
"""
import re

import holidays_co
import numpy as np
import pandas as pd

from .config import DataConfig, FeatureConfig


def _station_code(col):
    if not isinstance(col, str):
        return None
    m = re.match(r"\((\d+)\)", col)
    return int(m.group(1)) if m else None


def load_station_metadata(dcfg: DataConfig, columns) -> pd.DataFrame:
    """Per-station categorical/continuous metadata: coordinates, primary
    "zone" (paper's Line analogue), and a transfer-stop count computed from
    the number of zones/corridors serving each station (paper's
    TransferStop feature, Sec. 3.4.1) -- the one external-factor-style
    feature we CAN faithfully reproduce with data already in this repo.
    """
    raw = pd.read_csv(dcfg.stations_db_path)
    raw["code"] = raw["station_name"].apply(_station_code)

    zones = raw.groupby("code")["nombrelinea"].apply(lambda s: sorted(set(s)))
    coords = raw.groupby("code")[["latitude", "longitude"]].first()

    codes = [_station_code(c) for c in columns]
    meta = pd.DataFrame(index=columns)
    meta["station_code"] = codes
    meta["primary_zone"] = [zones.get(c, ["Unknown"])[0] if c in zones.index else "Unknown" for c in codes]
    meta["transfer_stop_count"] = [max(len(zones.get(c, [])) - 1, 0) for c in codes]
    meta["stop_type"] = np.where(meta["transfer_stop_count"] > 0, "Transfer", "Ordinary")
    meta["latitude"] = [coords["latitude"].get(c, np.nan) for c in codes]
    meta["longitude"] = [coords["longitude"].get(c, np.nan) for c in codes]
    return meta


def build_calendar_table(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Temporal/calendar input features (paper Table 10: Date-derived
    categoricals + boolean flags): hour, day-of-week, month, year, holiday,
    weekend. No school-vacation equivalent is available for Bogota.
    """
    years = range(index.year.min(), index.year.max() + 1)
    holidays = set()
    for y in years:
        for h in holidays_co.get_colombia_holidays_by_year(y):
            holidays.add(h.date)
    is_holiday = pd.Series(index.normalize().date, index=index).isin(holidays).astype(np.int8)

    return pd.DataFrame({
        "hour": index.hour.astype(np.int16),
        "day_of_week": index.dayofweek.astype(np.int8),
        "month": index.month.astype(np.int8),
        "year": index.year.astype(np.int16),
        "is_holiday": is_holiday.values,
        "is_weekend": (index.dayofweek >= 5).astype(np.int8),
    }, index=index)


CATEGORICAL_COLS = ["station_code", "primary_zone", "stop_type", "hour", "day_of_week", "month", "year"]
CONTINUOUS_COLS = ["latitude", "longitude", "transfer_stop_count"]
BOOLEAN_COLS = ["is_holiday", "is_weekend"]
ALL_FEATURE_COLS = CATEGORICAL_COLS + CONTINUOUS_COLS + BOOLEAN_COLS


def melt_to_long(df: pd.DataFrame, meta: pd.DataFrame, calendar_df: pd.DataFrame,
                  stride: int = 1) -> pd.DataFrame:
    """df: wide (time x station) raw boarding counts (NOT scaled -- the
    paper's targets/metrics are in raw passenger-count units).
    Returns a long dataframe: one row per (timestamp, station), with all
    input features + target `y` = boarding count.
    """
    sub = df.iloc[::stride] if stride > 1 else df
    long_df = sub.reset_index().melt(id_vars="timestamp", var_name="station", value_name="y")
    # IMPORTANT: after melt(), the "timestamp" column repeats (once per
    # station), so it can no longer be used as a join index -- pd.concat(...,
    # axis=1) between frames that both carry that duplicated index would
    # align per duplicate GROUP and silently explode into a cartesian
    # product (375 unique timestamps x 147 rows each side -> 147x147 per
    # group instead of 147). Every lookup below is immediately
    # reset_index(drop=True)'d so all concats are purely positional.
    long_df = long_df.reset_index(drop=True)

    cal = calendar_df.loc[long_df["timestamp"]].reset_index(drop=True)
    long_df = pd.concat([long_df.drop(columns=["timestamp"]), cal], axis=1)

    meta_aligned = meta.loc[long_df["station"]].reset_index(drop=True)
    long_df = pd.concat([long_df, meta_aligned], axis=1)

    long_df["station_code"] = long_df["station_code"].astype("category")
    long_df["primary_zone"] = long_df["primary_zone"].astype("category")
    long_df["stop_type"] = long_df["stop_type"].astype("category")

    return long_df
