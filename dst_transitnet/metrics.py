"""Evaluation metrics used by the paper (Sec. IV.B): R^2 and MAAPE."""
import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. (1). Computed over the flattened (time, station) array, matching
    the paper's dataset-level R2 reporting (Table 1)."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def maape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Eq. (2): Mean Arctangent Absolute Percentage Error."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    ratio = (y_true - y_pred) / denom
    return float(np.mean(np.arctan(np.abs(ratio))))


def per_station_scores(y_true: np.ndarray, y_pred: np.ndarray):
    """y_true, y_pred: (N, S). Returns per-station R2 and MAAPE arrays,
    matching Fig. 14's network-wide station-level distributions."""
    S = y_true.shape[1]
    r2 = np.array([r2_score(y_true[:, s], y_pred[:, s]) for s in range(S)])
    ma = np.array([maape(y_true[:, s], y_pred[:, s]) for s in range(S)])
    return r2, ma
