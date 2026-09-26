"""Evaluation metrics (DATA_LOG.md section 7.4).

MAAPE (Kim & Kim 2016), as coded in the benchmark (evaluation_metrics.ipynb,
experiments/result_analysis.ipynb):  mean( arctan( |(y - yhat) / y| ) ).
Zero targets: y = 0, yhat != 0 gives arctan(inf) = pi/2, the same as the benchmark.
y = 0 and yhat = 0 is 0/0: the benchmark's np.mean returns NaN for the whole set. Here
it counts as a perfect forecast (0), and the count of such cells is reported.

"System-wide MAAPE" (benchmark definition): for each forecast origin, average over
all stations and all horizon steps; then average over the origins of a period. With
equal numbers of stations x steps per origin this is the plain mean over all cells.

maape_system_total is a separate diagnostic: MAAPE of the SUM over stations (one
series for the whole system). It is not the benchmark's definition.
"""
import numpy as np
import pandas as pd


def arctan_ape(y, yhat) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    err = np.abs(y - yhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.arctan(err / np.abs(y))
    a[(y == 0) & (err == 0)] = 0.0
    return a


def maape(y, yhat) -> float:
    return float(np.mean(arctan_ape(y, yhat)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, "float64") - np.asarray(yhat, "float64"))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, "float64") - np.asarray(yhat, "float64")) ** 2)))


def evaluate_long(df: pd.DataFrame, pred_col: str = "yhat", by=("period",), service_only: bool = False) -> pd.DataFrame:
    """Metrics from a long table (one row per station x origin x horizon step).

    Needs columns: origin, y, <pred_col>, the `by` columns, and is_service_interval when
    service_only=True (Track B). MAAPE is averaged per origin first, then over origins.
    """
    d = df
    if service_only:
        d = d[d.is_service_interval.astype(bool)]
    d = d.assign(_a=arctan_ape(d.y, d[pred_col]), _e=d.y.astype("float64") - d[pred_col].astype("float64"))
    rows = []
    for key, g in d.groupby(list(by), observed=True):
        per_origin = g.groupby("origin", observed=True)._a.mean()
        tot = g.groupby("origin", observed=True)[["y", pred_col]].sum()
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                     "n_origins": int(len(per_origin)), "n_cells": int(len(g)),
                     "maape": float(per_origin.mean()),
                     "mae": float(g._e.abs().mean()),
                     "rmse": float(np.sqrt((g._e ** 2).mean())),
                     "maape_system_total": maape(tot.y, tot[pred_col]),
                     "zero_zero_cells": int(((g.y == 0) & (g[pred_col] == 0)).sum())})
    return pd.DataFrame(rows)


def evaluate_arrays(Y, P, origins, period, service_mask=None) -> pd.DataFrame:
    """Metrics from arrays Y, P of shape (N_origins, H, S); period: (N,) labels.
    service_mask (N, H) bool restricts to service intervals (Track B)."""
    Y, P = np.asarray(Y, "float64"), np.asarray(P, "float64")
    a = arctan_ape(Y, P)
    e = Y - P
    rows = []
    for p in pd.unique(np.asarray(period)):
        m = np.asarray(period) == p
        if service_mask is not None:
            keep = np.broadcast_to(service_mask[m][:, :, None], a[m].shape)
            per_origin = np.array([a[m][i][keep[i]].mean() for i in range(m.sum()) if keep[i].any()])
            ee = e[m][keep]
            ys, ps = (Y[m] * keep).sum(2).ravel(), (P[m] * keep).sum(2).ravel()
        else:
            per_origin = a[m].mean(axis=(1, 2))
            ee = e[m].ravel()
            ys, ps = Y[m].sum(2).ravel(), P[m].sum(2).ravel()
        rows.append({"period": p, "n_origins": int(m.sum()), "maape": float(per_origin.mean()),
                     "mae": float(np.abs(ee).mean()), "rmse": float(np.sqrt((ee ** 2).mean())),
                     "maape_system_total": maape(ys, ps)})
    return pd.DataFrame(rows)
