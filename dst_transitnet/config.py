"""Hyperparameters and experiment configuration for the DST-TransitNet reproduction.

Values are taken directly from the paper (Wang & Shalaby, 2024, arXiv:2410.15013)
wherever the paper specifies them explicitly. Values the paper leaves
unspecified are chosen as reasonable defaults and are flagged as
ASSUMPTION below; all assumptions are also documented in
DST_TRANSITNET_REPRODUCTION_NOTES.md.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    parquet_path: str = "data/transmilenio_transactions.parquet"
    edges_path: str = "preprocessing/Edges.csv"
    stations_db_path: str = "data/clean_stations_database_v2.csv"

    # Stations excluded because they belong to the TransMiCable cable-car
    # extension, not the original BRT trunk network. The base repository
    # (run.py) already drops these 4 stations, which brings the station
    # count from 151 to 147 -- matching the paper's "147 target stops"
    # exactly. [PAPER-SPECIFIED station count, station identity is our
    # mapping to the existing repo convention.]
    drop_stations: List[str] = field(default_factory=lambda: [
        "(40000) Cable Portal Tunal",
        "(40001) Juan Pablo II",
        "(40002) Manitas",
        "(40003) Mirador del Paraiso",
    ])

    # Hours excluded from the 15-minute series (matches existing repo
    # aggreagtion_func for '15-mins' aggregation, and approximates the
    # paper's statement that ridership records from ~12AM-6AM are removed).
    excluded_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 23])

    # [PAPER-SPECIFIED] Train / test split date -- "before July 2018" for
    # training (Fig. 10 caption states Aug 2015 - Jul 2018 training window).
    train_end: str = "2018-08-01 00:00:00"  # exclusive

    # [ASSUMPTION] Exact test-period boundaries are not given in the paper
    # beyond calendar month/year descriptions. We searched for boundaries
    # (starting immediately after the training cutoff / at well-documented
    # real-world event dates) that reproduce the paper's reported test-set
    # sizes as closely as possible (Table, Sec. IV.A):
    #   Normal:  (11750, 147)  -> found 2018-08-01..2019-01-01  => (11780,147)
    #   Protest: (2820, 147)   -> found 2019-11-21..2019-12-27  => (~2888,147)
    #   Covid:   (19317, 147)  -> found 2020-03-01..2020-11-05  => (19380,147)
    # These are within 1-3% of the published sizes; see reproduction notes.
    normal_start: str = "2018-08-01 00:00:00"
    normal_end: str = "2019-01-01 00:00:00"
    protest_start: str = "2019-11-21 00:00:00"
    protest_end: str = "2019-12-27 00:00:00"
    covid_start: str = "2020-03-01 00:00:00"
    covid_end: str = "2020-11-05 00:00:00"

    # [PAPER-SPECIFIED]
    recent_len: int = 20   # Xo / Xt / Xr lookback length (15-min steps)
    hist_len: int = 20     # Xh lookback length (15-min steps)
    hist_week_offset_days: int = 7
    pred_horizon: int = 1  # short-term: next 15-min step
    long_term_max_lag: int = 12  # up to 3 hours ahead, iterative

    # AvgPool1D kernel size for temporal decomposition (Xt = AvgPool1D(Xo)).
    # [ASSUMPTION] not specified in the paper; DLinear/Autoformer-style
    # decomposition commonly use an odd kernel with stride 1 and 'same'
    # padding. We use kernel=5 (~1.25 hours) as a reasonable moving-average
    # window for 15-minute transit data.
    decomposition_kernel: int = 5


@dataclass
class ModelConfig:
    # [ASSUMPTION] Hidden sizes / layer counts are not specified in the
    # paper. Chosen to keep the model "compact" (paper reports DST-TransitNet
    # as far smaller than iTransformer, Table 3) while giving the GRU/GNN/GAT
    # stack enough capacity to fit 147 stations jointly.
    gru_hidden: int = 32
    gat_hidden: int = 16
    gat_heads: int = 4
    gnn_hidden: int = 32
    ffnn_hidden: int = 64
    ffnn_layers: int = 2
    dropout: float = 0.1
    leaky_relu_slope: float = 0.01


@dataclass
class TrainConfig:
    # [ASSUMPTION] Loss/optimizer/schedule are not specified in the paper.
    # MSE is the standard default regression loss (the paper's R2/MAAPE are
    # evaluation metrics, not necessarily the training loss); Adam is the
    # de-facto default optimizer for this model family (GRU/GCN/GAT papers
    # referenced in Sec III.A all use Adam).
    loss: str = "mse"
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 25
    patience: int = 5
    grad_clip: float = 5.0
    seed: int = 42
    # Fraction of the training period reserved for validation / early
    # stopping (chronological, last slice of the training period).
    val_fraction: float = 0.1
    # [ASSUMPTION - compute budget] Training-set subsampling stride applied
    # only to keep this CPU-only reproduction tractable within the session's
    # compute budget. stride=1 uses every sample (full paper fidelity);
    # documented explicitly if changed from 1.
    train_stride: int = 1
