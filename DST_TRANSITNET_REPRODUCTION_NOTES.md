# DST-TransitNet Reproduction Notes

Reproduction of **"DST-TransitNet: A Dynamic Spatio-Temporal Deep Learning
Model for Scalable and Efficient Network-Wide Prediction of Station-Level
Transit Ridership"** (Wang & Shalaby, 2024, arXiv:2410.15013), implemented
on this repository's existing Bogotá TransMilenio BRT dataset.

Branch: `claude/sharp-mayer-izj5xv`. All code lives under `dst_transitnet/`;
the existing ARIMA/SARIMA/Dense/CNN/LSTM pipeline (`run.py`, `data.py`,
`models/`) and the raw data under `data/transactions/` are untouched.

This is a **clean baseline reproduction only** — no CAIL-RP thesis
contribution or new modeling ideas are included here.

## 1. Paper → implementation mapping

| Paper element | Section | Implementation |
|---|---|---|
| GRU | III.A.1 | `torch.nn.GRU` inside `layers.TemporalGRUEncoder` (equations match exactly) |
| GCN / k-GNN | III.A.2 | `layers.KGNNLayer`: `h_i' = σ(x_i W1 + Σ_j WE[j,i] x_j W2)` |
| GAT (dynamic spatial weights) | III.A.3 / III.B.2 | `layers.DynamicGATWeights`: multi-head attention over `X_h`, masked to the BRT graph, softmax over neighbors → `W_E` |
| Temporal decomposition (`X_t = AvgPool1D(X_o)`, `X_r = X_o - X_t`) | III.B.1 | `data.moving_average_decompose` (numpy, offline) and `layers.moving_average_decompose_torch` (torch, used inside the iterative long-term loop) |
| Spatio-temporal aggregation (4 GRU branches → 4 k-GNN branches → concat → FFNN) | III.B.3, Fig. 5 | `models.DSTTransitNet` |
| DST-TransitNetV2 (k-GNN per timestep, GRU moved to prediction layer) | III.B.3, Fig. 6 | `models.DSTTransitNetV2` |
| Long-term iterative forecasting | III.B.4, Fig. 7 | `train.long_term_forecast` |
| R², MAAPE | IV.B, Eq. 1–2 | `metrics.py` |
| FFNN / LSTM / DLinear / iTransformer baselines | IV.A, IV.C.1 | `baselines.py` |

## 2. Dataset and preprocessing

- **Source**: `data/transmilenio_transactions.parquet` (already produced by
  this repo's existing pipeline from `data/transactions/*.csv`; raw yearly
  CSVs were **not** modified or re-parsed).
- **Stations**: 151 raw station columns → drop 4 TransMiCable (cable-car)
  stations (`(40000)-(40003)`, same convention as `run.py`) → **147
  stations**, exactly matching the paper's reported station count. Stations
  are ordered by ascending numeric station code for a deterministic,
  reproducible ordering used consistently for the series matrix, the
  adjacency matrix, and all model tensors.
- **Duplicate timestamps**: ~1,900 fifteen-minute slots had duplicate raw
  rows; summed via `groupby(timestamp).sum()`, matching the existing
  `data.py:read_data` convention.
- **Hour filtering**: hours {0,1,2,3,23} dropped (matches
  `data.py:aggreagtion_func('15-mins')`), approximating the paper's "records
  from ~12AM-6AM are removed."
- **Graph**: built from `preprocessing/Edges.csv` (159 BRT line-segment
  edges between station codes), producing a symmetric adjacency matrix with
  self-loops. Two edges were dropped because they reference stations outside
  the kept 147 (the 4 dropped cable-car stations, and one edge referencing
  node id `7111`, which does not exist among the station codes and appears
  to be a data-entry typo in the source CSV — a documented, minor
  data-cleaning decision). Result: 157 kept undirected edges → 314 directed
  edges (+147 self-loops), vs. the paper's reported "320 edges" — within 2%.
- **Normalization**: per-station min-max scaling fit on the training split
  only (mirrors the existing repo's `data.py:min_max`, and the paper's own
  references to "scaled ridership" in Figs. 13/17/20/21). Both R² and MAAPE
  are invariant to this per-station positive affine rescaling, so metrics
  computed in scaled space equal those in raw units.

### Train / test split

[PAPER-SPECIFIED] Training: **August 2015 – July 2018** (Fig. 10 caption).
Test periods are only described qualitatively by the paper (calendar
months/years) and their exact boundaries are not given; Table in Sec. IV.A
reports test-set sizes `(rows, 147)`. We searched for period boundaries
(anchored either at the training cutoff or at well-documented real-world
event dates) that reproduce those sizes as closely as possible:

| Period | Paper size | Boundaries used here | Reconstructed size | Diff |
|---|---:|---|---:|---:|
| Normal | (11750, 147) | 2018-08-01 – 2019-01-01 | (11628, 147) | ~1.0% |
| Protest | (2820, 147) | 2019-11-21 – 2019-12-27 | (2736, 147)* | ~3.0% |
| COVID | (19317, 147) | 2020-03-01 – 2020-11-05 | (18924, 147)* | ~2.0% |

*(sizes after dedup of duplicate timestamps; the raw-row search that
motivated these boundaries is documented in `dst_transitnet/config.py`.)*

**[ASSUMPTION]** These boundaries are a best-effort reconstruction, not the
paper's actual (unpublished) split code. The Protest window matches the
well-documented Colombian national strike period (Nov 21 – Dec 2019); the
Normal/COVID windows were chosen purely by matching row counts, so their
exact start/end dates could differ from the paper's true split while still
representing genuine "normal" and "COVID-disrupted" ridership regimes.

## 3. Input construction

- **Recent input `X_o`**: last 20 steps (5 hours) immediately before the
  target time. [PAPER-SPECIFIED]
- **Historical input `X_h`**: 20 steps ending at (target time − 7 days),
  i.e. the same time-of-day and day-of-week one week prior.
  [PAPER-SPECIFIED, Sec. IV.A settings list]. Note: Sec. III.C's deployment
  narrative separately says "historical ridership input from the *previous
  day*", which is inconsistent with the quantitative settings list; we
  followed the quantitative, more specific "previous week, same weekday"
  definition since it is stated twice (Sec. III.B.2 and the settings list)
  and is the only version consistent with capturing weekday/weekend
  seasonality.
- **Trend/residual** (`X_t`, `X_r`): computed from `X_o` via a moving
  average with **kernel size 5** (~1.25h). **[ASSUMPTION]**: the paper does
  not specify the AvgPool1D kernel size; 5 was chosen as a reasonable
  short-window smoother for 15-minute data (consistent with the scale used
  by DLinear/Autoformer-style decomposition on comparable granularities).
- **Target**: value at the next 15-minute step (1-step-ahead), for all 147
  stations simultaneously (many-to-many prediction), matching the paper's
  "prediction time interval: 15 minutes."
- **Long-term (iterative) forecasting**: `X_o` is rolled forward using the
  model's own previous predictions; `X_h` is always the true, previously
  observed value from a week ago for the exact future target step (never a
  model prediction) — this matches the paper's Sec. III.C design, under
  which the previous-week historical input is "already in the dataset" for
  any future target time.

## 4. Model architecture (implemented exactly as specified where the paper
gives equations)

- **DST-TransitNet**: four independent branches for `X_o`, `X_h`, `X_t`,
  `X_r`. Each branch: a shared-weight GRU processes every station's scalar
  time series independently (temporal aggregation), then a k-GNN layer
  spatially aggregates each branch's hidden state using the *same* dynamic
  edge weights `W_E` — computed once per sample from `X_h` via the
  `DynamicGATWeights` GAT layer (Sec. III.B.2/III.B.3). The four spatially-
  aggregated branch outputs are concatenated and passed through a
  multi-layer FFNN to a scalar per station.
- **DST-TransitNetV2**: the k-GNN spatial aggregation is applied
  per-timestep (weight-shared across time) directly on a linear projection
  of each raw scalar input, for each of the four series; the four spatially
  fused sequences are concatenated at every timestep and fed to a single
  shared GRU — now placed in the prediction layer — whose final hidden
  state feeds the same FFNN head. This matches the paper's description
  ("the GRU layer ... is moved to the prediction layer, where it processes
  the spatially-aggregated features") and reproduces the paper's finding
  that V2 is more parameter-efficient than V1 (confirmed here: V2 has ~11%
  fewer parameters than V1 at matched hidden sizes).
- **[ASSUMPTION]** Hidden-layer sizes (`gru_hidden=32`, `gat_hidden=16`,
  `gat_heads=4`, `gnn_hidden=32`, FFNN 2×64) are not given in the paper and
  were chosen to keep the model "compact" (consistent with the paper's own
  Table 3 claim that DST-TransitNet is far smaller than iTransformer) while
  giving the model enough capacity to fit 147 stations jointly.

## 5. Baselines — compute-budget adaptation (important discrepancy)

The paper trains FFNN, LSTM, and iTransformer as **147 fully independent
one-to-one (single-station) models** and reports training time *summed*
across all 147 models (e.g. 723.8 minutes total for LSTM, Table 3). Training
147 independent instances per baseline is not tractable on this session's
CPU-only, 4-core sandbox. Instead, each baseline here uses the same
one-to-one architecture and inductive bias (no cross-station spatial term,
unlike DST-TransitNet/DLinear) but **shares parameters across stations**
(the station axis is folded into the batch axis, i.e. one shared-weight
model applied identically to each station's own history). This is the
single largest compute-driven simplification in this reproduction. It
preserves the "no spatial information" property that the paper's comparison
hinges on, but is not a strict re-implementation of "147 independent
models," and will tend to *understate* the true per-station specialization
FFNN/LSTM/iTransformer could achieve in the original paper (independent
models can overfit/specialize per station; shared-weight models cannot).
**DST-TransitNet, DST-TransitNetV2, and DLinear are trained exactly as the
paper specifies** — many-to-many, jointly over all 147 stations in a single
model — since that is the actual object of this reproduction.

The iTransformer baseline is further simplified: rather than the full
iTransformer (dataset-wide multivariate attention across many covariates),
we treat a single station's four available series (`X_o, X_h, X_t, X_r`) as
four "variate" tokens and run a small Transformer encoder attending across
these four tokens — preserving the paper-cited "inverted" attention idea
(attend across variates, not across time) at a scale this reproduction can
train.

## 6. Training configuration

| Setting | Value | Source |
|---|---|---|
| Loss | MSE | **[ASSUMPTION]** — not specified in the paper; standard default for this regression task |
| Optimizer | Adam, lr=1e-3 | **[ASSUMPTION]** — not specified; de-facto standard for GRU/GCN/GAT models |
| Batch size | 64 | **[ASSUMPTION]** |
| Max epochs / patience | 10 / 3 (this run) | **[ASSUMPTION + compute budget]** — config default is 25/5; reduced for this CPU-only session (see §7) |
| Train-set stride | 4 (this run) | **[compute budget]** — config default is 1 (full fidelity); every 4th training sample used to fit within session time constraints |
| Gradient clipping | 5.0 | **[ASSUMPTION]** |
| Validation | last 10% of the training period (chronological) | **[ASSUMPTION]** |

## 7. Compute-budget deviations (explicit)

This reproduction runs single-threaded/4-core CPU only, in a time-bounded
interactive session. Two deviations from "full-fidelity, unconstrained"
training were made and are the most likely source of any residual gap vs.
the paper's numbers:

1. **`train_stride=4`**: only every 4th training window is used (~18.6k of
   ~74.4k available windows), rather than the full training set.
2. **`max_epochs=10, patience=3`**: fewer epochs than the config's
   documented full-fidelity default (25/5).

Both are training-recipe/compute choices, not architecture changes — the
model definitions in `models.py` are the full, faithful DST-TransitNet /
DST-TransitNetV2 architecture described in the paper.

## 8. Results: paper vs. reproduction

*(filled in after the training run in `dst_transitnet/logs/full_run.log` and
`dst_transitnet/outputs/results_table.json` completes)*

### Table 1 — MAAPE / R² by period

| Model | Normal MAAPE | Normal R² | Protest MAAPE | Protest R² | COVID MAAPE | COVID R² |
|---|---:|---:|---:|---:|---:|---:|
| _to fill_ | | | | | | |

### Table 2 — Long-term MAAPE ratio (lag 12 vs. lag 1)

| Model | Normal | Protest | COVID |
|---|---:|---:|---:|
| _to fill_ | | | |

### Table 3 — Training time / model size

| Model | Training time (min) | Params | Size (MB) |
|---|---:|---:|---:|
| _to fill_ | | | |

## 9. Artifacts

- Code: `dst_transitnet/{config,data,layers,models,baselines,metrics,train,run_reproduction}.py`
- Checkpoints: `dst_transitnet/checkpoints/<model>.pt`
- Per-epoch training logs: `dst_transitnet/logs/<model>_train_log.json`
- Full run stdout/stderr: `dst_transitnet/logs/full_run.log`
- Predictions (scaled units): `dst_transitnet/outputs/<model>_<period>_{pred,true}.npy`
- Aggregate metrics: `dst_transitnet/outputs/results_table.json`,
  `per_station_results.json`, `timing.json`, `<model>_long_term.json`
