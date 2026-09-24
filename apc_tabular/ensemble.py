"""Ensemble strategies from Sec. 2.2/4.1.5: simple averaging, best-pair
averaging, constrained least-squares weighting, and ridge-regression
stacking, applied to the out-of-sample predictions of the five base models.
"""
import itertools

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


def simple_average(preds: dict) -> np.ndarray:
    return np.mean(list(preds.values()), axis=0)


def best_pair_average(preds: dict, y_true: np.ndarray, metric_fn) -> tuple:
    """Exhaustive search over all pairs; returns (names, averaged_preds)."""
    best_score, best_pair, best_avg = -np.inf, None, None
    for a, b in itertools.combinations(preds.keys(), 2):
        avg = (preds[a] + preds[b]) / 2.0
        score = metric_fn(y_true, avg)
        if score > best_score:
            best_score, best_pair, best_avg = score, (a, b), avg
    return best_pair, best_avg


def weighted_least_squares(preds: dict, y_true: np.ndarray) -> tuple:
    """Constrained convex combination (weights >= 0, sum to 1), Sec. 2.2."""
    names = list(preds.keys())
    P = np.stack([preds[n] for n in names], axis=1)  # (N, k)

    def objective(w):
        return np.mean((P @ w - y_true) ** 2)

    k = len(names)
    w0 = np.ones(k) / k
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * k
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = dict(zip(names, res.x))
    combined = P @ res.x
    return weights, combined


def ridge_stack(preds: dict, y_true: np.ndarray, alpha: float = 1.0) -> tuple:
    """Ridge regression meta-learner over base predictions (Sec. 2.2/4.1.5):
    unlike weighted_least_squares, weights can be negative and need not sum
    to 1 -- the paper finds this the best-performing ensemble strategy.
    """
    names = list(preds.keys())
    P = np.stack([preds[n] for n in names], axis=1)
    model = Ridge(alpha=alpha)
    model.fit(P, y_true)
    weights = dict(zip(names, model.coef_))
    weights["intercept"] = float(model.intercept_)
    combined = model.predict(P)
    return weights, combined, model
