"""Two-layer stacked ensemble: ExtraTreesRegressor + XGBoost base learners,
LinearRegression meta-learner, 5-fold TimeSeriesSplit CV (Sec. III.C /
"Phase 3: Model Building" of the paper).

Compute-budget note (documented, analogous to the DST-TransitNet
reproduction's baseline adaptation): the paper tunes ETR/XGBoost via grid
search "within the TSV framework" for its single system-wide series. Running
a full grid search independently for each of our 147 per-station models
would multiply the paper's own tuning cost by 147x, which is not tractable
here. Instead we run the grid search ONCE on a representative pooled sample
(the system-wide aggregate series) to pick one fixed (etr_params, xgb_params)
configuration, then fit a fully independent stacked ensemble PER STATION
using those fixed hyperparameters -- each station's trees, and the
meta-learner's weights, are still trained from scratch on that station's own
data; only the hyperparameter *search* is shared.

Implementation note: sklearn's StackingRegressor generates out-of-fold base
predictions via `cross_val_predict`, which requires the CV splitter to
produce a full partition of the data -- a `TimeSeriesSplit` does NOT (its
first fold's training rows are never held out), so `StackingRegressor(cv=
TimeSeriesSplit(...))` raises `ValueError: cross_val_predict only works for
partitions`. We therefore implement the paper's "5-fold time series
cross-validation" stacking manually: TimeSeriesSplit still drives the OOF
meta-feature generation (only over the rows time-series CV can actually
hold out), and both base learners are refit on the full training set for
final inference -- the same net effect the paper describes, without
depending on an sklearn internal that assumes non-time-aware CV.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from .config import ModelConfig


def tune_hyperparameters(X: np.ndarray, y: np.ndarray, mcfg: ModelConfig):
    tscv = TimeSeriesSplit(n_splits=mcfg.n_splits)

    etr_grid = {"n_estimators": mcfg.etr_n_estimators, "max_depth": mcfg.etr_max_depth}
    etr_search = GridSearchCV(
        ExtraTreesRegressor(random_state=mcfg.random_state, n_jobs=-1),
        etr_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
    etr_search.fit(X, y)

    xgb_grid = {"n_estimators": mcfg.xgb_n_estimators, "max_depth": mcfg.xgb_max_depth,
                "learning_rate": mcfg.xgb_learning_rate}
    xgb_search = GridSearchCV(
        XGBRegressor(random_state=mcfg.random_state, n_jobs=-1, verbosity=0),
        xgb_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
    xgb_search.fit(X, y)

    return etr_search.best_params_, xgb_search.best_params_


class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """Manual ETR+XGBoost -> LinearRegression stack with TimeSeriesSplit OOF
    meta-features (see module docstring for why sklearn's own
    StackingRegressor can't take a TimeSeriesSplit directly)."""

    def __init__(self, etr, xgb, n_splits: int = 5):
        self.etr = etr
        self.xgb = xgb
        self.n_splits = n_splits

    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof_etr = np.full(len(y), np.nan)
        oof_xgb = np.full(len(y), np.nan)

        for train_idx, test_idx in tscv.split(X):
            etr_fold = clone(self.etr).fit(X[train_idx], y[train_idx])
            xgb_fold = clone(self.xgb).fit(X[train_idx], y[train_idx])
            oof_etr[test_idx] = etr_fold.predict(X[test_idx])
            oof_xgb[test_idx] = xgb_fold.predict(X[test_idx])

        covered = ~np.isnan(oof_etr)
        meta_X = np.stack([oof_etr[covered], oof_xgb[covered]], axis=1)
        self.meta_learner_ = LinearRegression().fit(meta_X, y[covered])

        # Refit base learners on the FULL training set for inference.
        self.etr_ = clone(self.etr).fit(X, y)
        self.xgb_ = clone(self.xgb).fit(X, y)
        return self

    def predict(self, X):
        base = np.stack([self.etr_.predict(X), self.xgb_.predict(X)], axis=1)
        return self.meta_learner_.predict(base)


def build_stacked_ensemble(etr_params: dict, xgb_params: dict, mcfg: ModelConfig) -> TimeSeriesStackingRegressor:
    etr = ExtraTreesRegressor(random_state=mcfg.random_state, n_jobs=-1, **etr_params)
    xgb = XGBRegressor(random_state=mcfg.random_state, n_jobs=-1, verbosity=0, **xgb_params)
    return TimeSeriesStackingRegressor(etr=etr, xgb=xgb, n_splits=mcfg.n_splits)


def fit_station_model(X: np.ndarray, y: np.ndarray, etr_params: dict, xgb_params: dict,
                       mcfg: ModelConfig) -> TimeSeriesStackingRegressor:
    model = build_stacked_ensemble(etr_params, xgb_params, mcfg)
    model.fit(X, y)
    return model
