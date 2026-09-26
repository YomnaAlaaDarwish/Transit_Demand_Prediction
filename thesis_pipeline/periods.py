"""Test-period labels."""
import numpy as np
import pandas as pd

from .config import DST_WINDOWS, PERIODS


def period_label(times) -> np.ndarray:
    """'stable' / 'protest' / 'covid' for each timestamp (config.PERIODS)."""
    t = pd.DatetimeIndex(times)
    lab = np.full(len(t), "stable", dtype=object)
    for name, (start, end) in PERIODS.items():
        m = t >= pd.Timestamp(start)
        if end is not None:
            m &= t < pd.Timestamp(end)
        lab[m] = name
    return lab


def dst_window_label(times) -> np.ndarray:
    """DST-TransitNet reproduction window ('normal'/'protest'/'covid') or '' outside them."""
    t = pd.DatetimeIndex(times)
    lab = np.full(len(t), "", dtype=object)
    for name, (start, end) in DST_WINDOWS.items():
        lab[(t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))] = name
    return lab
