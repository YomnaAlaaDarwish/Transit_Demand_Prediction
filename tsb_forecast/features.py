"""Feature engineering for the TSB-Forecast-base reproduction: lag features,
calendar features, and (via time2vec.py) the learned temporal embedding --
assembled into the per-station tabular dataset the stacked ensemble trains
on (Sec. III.B.2 "Unified Feature Set" of the paper, minus the SBERT/news
block).
"""
import numpy as np
import pandas as pd
import holidays_co

from .config import FeatureConfig
from .time2vec import build_pointwise_features


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour-of-day / day-of-week cyclical encodings, weekend flag, and
    Colombian holiday flag -- native calendar features already used
    elsewhere in this repo (data.py:add_cycles/add_holidays), NOT part of
    the excluded "external factors" (weather) or "events module" (SBERT).
    """
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek

    years = range(index.year.min(), index.year.max() + 1)
    holidays = set()
    for y in years:
        for h in holidays_co.get_colombia_holidays_by_year(y):
            holidays.add(h.date)
    is_holiday = pd.Series(index.normalize().date, index=index).isin(holidays).astype(np.float32)

    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "is_weekend": (dow >= 5).astype(np.float32),
        "is_holiday": is_holiday.values,
    }, index=index)


def build_lag_table(series: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    """Explicit scalar lag features for one station's (scaled) series,
    analogous to the paper's ['actual_load_lag8', 'actual_load_lag_1day',
    'actual_load_lag_1week', ...] selection (Sec. III.B.1 step 1), adapted
    to our 15-min resolution and lookback (see config.py FeatureConfig).
    """
    out = {"lag_0": series.shift(0)}  # most recent known value at prediction time (paper's 'actual_load_MW')
    for lag in cfg.lag_steps:
        out[f"lag_{lag}"] = series.shift(lag)
    out["lag_1day"] = series.shift(cfg.lag_1day_steps)
    out["lag_1week"] = series.shift(cfg.lag_1week_steps)
    return pd.DataFrame(out, index=series.index)


def build_time2vec_pointwise_inputs(series: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    """The 2-D raw feature vector [current value, same-time-last-week value]
    fed to the Time2Vec encoder at every timestep (time2vec.py), plus its
    own 1-step-ahead target (for training the encoder self-supervised).
    """
    value = series.values
    week_ago = series.shift(cfg.lag_1week_steps).values
    x = build_pointwise_features(value, week_ago)  # (T, 2)
    x_next = np.roll(x, -1, axis=0)
    valid = np.ones(len(series), dtype=bool)
    valid[-1] = False
    valid &= ~np.isnan(week_ago)
    return x, x_next, valid


# NOTE: the actual per-station feature+target frame assembly used by the
# runner lives in train.py:build_feature_target_frame (it needs the trained
# Time2Vec embedder in scope); this module only provides the building
# blocks (lag table, calendar features, Time2Vec pointwise inputs).
