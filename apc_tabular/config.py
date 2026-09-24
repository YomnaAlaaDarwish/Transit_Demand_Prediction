"""Configuration for the "horizon-agnostic stop-level" reproduction of
Yusuf, Rasheed & Lindseth (2025), "Data-driven predictive modelling of
stop-level public transit patterns" (Transportation, DOI:
10.1007/s11116-025-10689-4), adapted to the Bogota TransMilenio dataset.

The paper has no proper model name -- it proposes a "horizon-agnostic ML
framework" for stop-level passenger-count/operational prediction. This
reproduction is named after the dataset/branch (`trondheim-apc-tabular-
reproduction`) rather than inventing a paper acronym that doesn't exist.

=== Scope of this "base" reproduction (documented dataset-driven limits) ===

The paper's setup and this reproduction's necessary departures from it:

1. **Row granularity**: the paper's unit of analysis is a (trip, stop-visit)
   -- an individual bus trip's arrival at one stop along one Line+Route.
   Our TransMilenio dataset has NO trip/route-level records at all -- only
   station-level totals aggregated over a 15-minute window (see
   `dst_transitnet/data.py`, reused directly here). The natural adaptation is
   a (station, 15-min interval) row -- the finest granularity our data
   actually supports -- with station identity, calendar/time-of-day, and
   coordinates as inputs. This is NOT a compute-budget shortcut; it's an
   unavoidable dataset constraint, and is the single biggest structural
   difference from the paper.
2. **Single target: Boarding only.** TransMilenio's fare-validation system
   only records boardings (tap-in / proof-of-payment); there is no
   alighting/tap-out data, unlike Trondheim's onboard APC sensors which
   count both. The paper's `Alighting` target and its boarding-alighting
   correlation analysis (Sec. 4.1) cannot be reproduced.
3. **No operational targets** (`StopActualArrival`, `StopTime`): our dataset
   has no per-trip GPS/schedule-adherence records, only aggregate ridership.
   Table 4 of the paper (XGBoost on operational targets) is out of scope.
4. **No GT/video validation model** (paper Sec. 3.2 / Table 2): Bogota has no
   equivalent manually/video-verified passenger count dataset. This
   correction-quality-assessment side task is out of scope, same pattern as
   excluding SBERT/weather in the TSB-Forecast reproduction.
5. **No weather, terrain, or demographics/land-use features** (paper Sec.
   3.4.2-3.4.4): weather requires a paid/region-specific API (Visual
   Crossing, Trondheim-only free-tier query used in the paper), terrain
   requires per-stop elevation data (not available for Bogota in this repo),
   and demographics/land-use requires Norwegian statistical grids with no
   Bogota equivalent available offline. All three are explicitly flagged by
   the user as "extra factors to add later" -- excluded from this base pass.
6. **Transfers/StopType ARE reproduced**: the paper's `TransferStop`/
   `StopType` features (count of other lines serving a stop, Sec. 3.4.1) map
   directly onto `clean_stations_database_v2.csv`'s `nombrelinea` (zone/
   corridor) column -- most Bogota stations belong to exactly one zone, a
   few to two, giving a natural, faithfully-computed transfer-stop signal.
7. **No lag/autoregressive features** -- matches the paper's own design
   exactly: Table 10's input features are purely categorical/temporal/
   spatial context, with NO historical passenger-count lags. This
   reproduction is deliberately "horizon-agnostic" in the same sense: it
   predicts from context alone, not from recent history.

Same station set (147), train cutoff, and Normal/Protest/COVID test periods
as the DST-TransitNet and TSB-Forecast-base reproductions (via
`dst_transitnet.config.DataConfig`), reused directly for comparability.
"""
from dataclasses import dataclass, field
from typing import List

from dst_transitnet.config import DataConfig as _DSTDataConfig

DataConfig = _DSTDataConfig


@dataclass
class FeatureConfig:
    # [compute budget] every Nth 15-min timestamp used for training (same
    # pattern/justification as TSB-Forecast's train_stride) -- pooling all
    # 147 stations x every training timestamp gives ~10.9M rows, too slow
    # for the paper's own heavier tree hyperparameters (Table 12) on this
    # session's 4-core CPU sandbox.
    train_stride: int = 4


@dataclass
class ModelConfig:
    """Hyperparameters. Where the paper reports an optimal value (Table 12),
    we use it directly rather than re-running the paper's own W&B/Optuna
    search (a full HPO sweep repeated here would cost far more than this
    session's compute budget allows, and the paper already reports the
    result of that search). Where compute forced a reduction (estimator
    counts / depth), this is flagged [REDUCED] with the paper's own value
    alongside.
    """
    random_state: int = 42

    # CatBoost: paper depth=10, l2_leaf_reg=9.0, learning_rate=0.06, max_ctr_complexity=8
    catboost_iterations: int = 200  # [REDUCED from unspecified paper iteration count]
    catboost_depth: int = 8  # [REDUCED from paper's 10]
    catboost_l2_leaf_reg: float = 9.0  # [PAPER-SPECIFIED]
    catboost_learning_rate: float = 0.06  # [PAPER-SPECIFIED]
    catboost_bagging_temperature: float = 0.0  # [PAPER-SPECIFIED]

    # Random Forest: paper n_estimators=320, max_depth=32, max_features=0.75,
    # max_samples=0.7, min_samples_leaf=48, min_samples_split=288
    rf_n_estimators: int = 100  # [REDUCED from paper's 320]
    rf_max_depth: int = 16  # [REDUCED from paper's 32]
    rf_max_features: float = 0.75  # [PAPER-SPECIFIED]
    rf_max_samples: float = 0.5  # [REDUCED from paper's 0.7, smaller bootstrap sample per tree]
    rf_min_samples_leaf: int = 48  # [PAPER-SPECIFIED]
    rf_min_samples_split: int = 288  # [PAPER-SPECIFIED]

    # XGBoost: paper colsample_bytree=0.5, eta=0.04, gamma=0.12, max_depth=16,
    # min_child_weight=8.5, reg_alpha=0.001, reg_lambda=0.0, subsample=0.9
    xgb_n_estimators: int = 200  # [REDUCED, paper doesn't report a fixed round count separately from eta]
    xgb_max_depth: int = 8  # [REDUCED from paper's 16]
    xgb_colsample_bytree: float = 0.5  # [PAPER-SPECIFIED]
    xgb_eta: float = 0.04  # [PAPER-SPECIFIED]
    xgb_gamma: float = 0.12  # [PAPER-SPECIFIED]
    xgb_min_child_weight: float = 8.5  # [PAPER-SPECIFIED]
    xgb_reg_alpha: float = 0.001  # [PAPER-SPECIFIED]
    xgb_reg_lambda: float = 0.0  # [PAPER-SPECIFIED]
    xgb_subsample: float = 0.9  # [PAPER-SPECIFIED]

    # LightGBM: paper max_depth=16, num_leaves=63, learning_rate=0.09,
    # min_child_samples=7, min_child_weight=9, reg_alpha=0.212, reg_lambda=3.2,
    # subsample=0.9, colsample_bytree=1.0
    lgbm_n_estimators: int = 200  # [REDUCED]
    lgbm_max_depth: int = 8  # [REDUCED from paper's 16]
    lgbm_num_leaves: int = 63  # [PAPER-SPECIFIED]
    lgbm_learning_rate: float = 0.09  # [PAPER-SPECIFIED]
    lgbm_min_child_samples: int = 7  # [PAPER-SPECIFIED]
    lgbm_reg_alpha: float = 0.212  # [PAPER-SPECIFIED]
    lgbm_reg_lambda: float = 3.2  # [PAPER-SPECIFIED]
    lgbm_subsample: float = 0.9  # [PAPER-SPECIFIED]
    lgbm_colsample_bytree: float = 1.0  # [PAPER-SPECIFIED]

    # Tabular DNN: paper batch_size=1024, lr=0.01, layer_dropout=0.1,
    # n_blocks=5, hidden_sizes=[128,2048,256,512,256]
    dnn_batch_size: int = 8192  # [REDUCED training-throughput adaptation: paper's optimal was 1024
    # (from their own search space [1024,2048,4096,8192]), but at ~2.7M pooled training rows on this
    # session's 4-core CPU, 1024 measured ~26.5s/epoch per 55k rows (>20 min/epoch at full scale);
    # 8192 measured ~51s/epoch per 551k rows (~4-5 min/epoch at full scale) -- picked for tractability
    # from within the paper's OWN searched batch-size options, not an arbitrary new value.]
    dnn_lr: float = 0.01  # [PAPER-SPECIFIED]
    dnn_dropout: float = 0.1  # [PAPER-SPECIFIED]
    dnn_hidden_sizes: List[int] = field(default_factory=lambda: [128, 2048, 256, 512, 256])  # [PAPER-SPECIFIED]
    dnn_embedding_dim: int = 16  # [ASSUMPTION -- not given per-categorical in the paper]
    dnn_max_epochs: int = 8  # [ASSUMPTION -- not specified by the paper; compute-budget capped]
    dnn_patience: int = 3  # [ASSUMPTION]


@dataclass
class RunConfig:
    pred_horizon: int = 1  # unused (no lag structure) -- kept for interface symmetry with other reproductions
