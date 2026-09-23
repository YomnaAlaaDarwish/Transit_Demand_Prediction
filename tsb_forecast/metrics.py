"""Evaluation metrics: the paper's own trio (MAE, RMSE, SMAPE, Sec. IV.D)
plus R2/MAAPE (imported from the DST-TransitNet reproduction) so results are
directly comparable across the two reproductions on the same stations/periods.
"""
import numpy as np

from dst_transitnet.metrics import r2_score, maape  # re-exported for comparability


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE (%), Eq. 4 of the paper."""
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "maape": maape(y_true, y_pred),
    }
