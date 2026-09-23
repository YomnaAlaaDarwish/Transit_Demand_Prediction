"""Configuration for the TSB-Forecast-base reproduction (Hasan et al., 2025,
"TSB-Forecast: A Short-Term Load Forecasting Model in Smart Cities for
Integrating Time Series Embeddings and Large Language Models").

Scope of this reproduction ("the base"): the paper's Time2Vec-inspired
temporal-embedding feature extractor plus its two-layer stacked ensemble
(ExtraTreesRegressor + XGBoost base learners, LinearRegression meta-learner).
The SBERT/news semantic-embedding module and external weather features are
explicitly OUT of scope for this reproduction (no equivalent data source is
available for Bogota's BRT system) -- see TSB_FORECAST_REPRODUCTION_NOTES.md.
A follow-up pass can add an equivalent "context" module (e.g. local news /
event embeddings, weather) the same way the paper fuses SBERT, once that
data is available.

Dataset/period/station setup is intentionally identical to the
DST-TransitNet reproduction (dst_transitnet/config.py) so results are
directly comparable at the station level, as requested.
"""
from dataclasses import dataclass, field
from typing import List

# Reuse the exact same station/period/graph setup as the DST-TransitNet
# reproduction for direct comparability.
from dst_transitnet.config import DataConfig as _DSTDataConfig


DataConfig = _DSTDataConfig  # identical station list, splits, hour filtering


@dataclass
class FeatureConfig:
    # [PAPER-SPECIFIED, adapted to our 15-min resolution] Explicit scalar lag
    # features. The paper used ['actual_load_lag8', 'actual_load_lag_1day',
    # 'actual_load_lag_1week', 'actual_load_lag_1month', 'actual_load_lag_1year']
    # (at their 30-min resolution) alongside the raw current/day-ahead load.
    # At our 15-min resolution we keep the same *semantic* lags (2h, 1 day,
    # 1 week) and drop 1-month/1-year (many training windows, esp. early in
    # the Aug2015-Jul2018 training period, would have no valid 1-year lag,
    # and the paper's own 5-year dataset made those lags viable in a way ours
    # is not) -- documented assumption.
    lag_steps: List[int] = field(default_factory=lambda: [1, 4, 8])  # 15/60/120 min
    lag_1day_steps: int = 76   # one operating day (hours 4-22 kept, see DataConfig)
    lag_1week_steps: int = 76 * 7

    # [PAPER-SPECIFIED] Time2Vec-style sliding window length. Paper: 48 steps
    # at 30-min resolution (24h). We keep the DST-TransitNet reproduction's
    # 20-step recent window (5h) for the embedding input so the "same
    # environment by station" (identical lookback/target construction) holds
    # across both reproductions -- documented deviation from the paper's own
    # 48-step/24h window.
    time2vec_window: int = 20
    time2vec_hidden: int = 128
    time2vec_embed_dim: int = 64  # [PAPER-SPECIFIED]: "final 64-dimensional output"

    # [ASSUMPTION] Time2Vec encoder training (not fully specified by the
    # paper beyond "two linear layers with ReLU" and "MSE loss between
    # sequential embeddings"): trained once, globally (pooled across all
    # 147 stations) as a per-timestep next-step predictor -- see
    # tsb_forecast/time2vec.py docstring for the exact reconstruction.
    time2vec_epochs: int = 5
    time2vec_batch_size: int = 512
    time2vec_lr: float = 1e-3

    # Calendar features (native to this repo's existing pipeline via
    # holidays_co, NOT part of the excluded "external factors"): hour-of-day
    # and day-of-week cyclical encodings, weekend flag, Colombian holiday flag.
    use_calendar: bool = True


@dataclass
class ModelConfig:
    # [PAPER-SPECIFIED intent, ASSUMPTION on exact values -- the paper says
    # "grid search hyperparameter tuning" without listing the grid] Small,
    # CPU-tractable grids since this reproduction trains 147 independent
    # per-station stacks: unbounded-depth ExtraTrees on 60-70k rows proved
    # far too slow (see TSB_FORECAST_REPRODUCTION_NOTES.md compute-budget
    # section), so depth is capped and estimator counts kept modest.
    etr_n_estimators: List[int] = field(default_factory=lambda: [50, 100])
    etr_max_depth: List[int] = field(default_factory=lambda: [8, 15])

    xgb_n_estimators: List[int] = field(default_factory=lambda: [50, 100])
    xgb_max_depth: List[int] = field(default_factory=lambda: [3, 6])
    xgb_learning_rate: List[float] = field(default_factory=lambda: [0.1])

    # [PAPER-SPECIFIED intent: "5-fold time series split cross-validation";
    # reduced to 3 here -- a compute-budget adaptation, since each split adds
    # two more full tree-ensemble fits per station across 147 stations.]
    n_splits: int = 3
    random_state: int = 42


@dataclass
class RunConfig:
    pred_horizon: int = 1          # short-term target: t+1 (15 min), matches DST-TransitNet Table 1
    long_term_max_lag: int = 12    # matches DST-TransitNet Table 2 (up to 3h iterative)
    stations_limit: int = None     # None = all 147; set for smoke tests
