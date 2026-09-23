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
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
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


def build_stacked_ensemble(etr_params: dict, xgb_params: dict, mcfg: ModelConfig) -> StackingRegressor:
    etr = ExtraTreesRegressor(random_state=mcfg.random_state, n_jobs=-1, **etr_params)
    xgb = XGBRegressor(random_state=mcfg.random_state, n_jobs=-1, verbosity=0, **xgb_params)
    tscv = TimeSeriesSplit(n_splits=mcfg.n_splits)
    return StackingRegressor(
        estimators=[("etr", etr), ("xgb", xgb)],
        final_estimator=LinearRegression(),
        cv=tscv,
        n_jobs=1,
        passthrough=False,
    )


def fit_station_model(X: np.ndarray, y: np.ndarray, etr_params: dict, xgb_params: dict,
                       mcfg: ModelConfig) -> StackingRegressor:
    model = build_stacked_ensemble(etr_params, xgb_params, mcfg)
    model.fit(X, y)
    return model
