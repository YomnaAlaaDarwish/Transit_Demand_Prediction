"""Paper's primary metrics (R2, RMSE, Sec. 2.2) plus MAAPE, imported from
the DST-TransitNet reproduction for cross-reproduction comparability.
"""
import numpy as np

from dst_transitnet.metrics import r2_score, maape  # re-exported


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"r2": r2_score(y_true, y_pred), "rmse": rmse(y_true, y_pred), "maape": maape(y_true, y_pred)}
