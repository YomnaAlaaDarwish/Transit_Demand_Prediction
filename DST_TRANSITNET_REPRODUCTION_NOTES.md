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

Full run: `dst_transitnet/logs/full_run.log`; raw numbers:
`dst_transitnet/outputs/{results_table,timing}.json`,
`dst_transitnet/outputs/<model>_long_term.json`. Training config actually
used for this run: `train_stride=4`, `max_epochs=10`, `patience=3` (see §7).

### Table 1 — MAAPE / R² by period

**Paper (Table 1):**

| Model | Normal MAAPE | Normal R² | Protest MAAPE | Protest R² | COVID MAAPE | COVID R² |
|---|---:|---:|---:|---:|---:|---:|
| FFNN | 0.1213 | 0.8825 | 0.1732 | 0.6591 | 0.1787 | 0.6230 |
| LSTM | 0.1124 | 0.9059 | 0.1693 | 0.7457 | 0.1729 | 0.7497 |
| iTransformer | 0.1177 | 0.8782 | 0.1699 | 0.8662 | 0.1842 | 0.7935 |
| DLinear | 0.1406 | 0.8579 | 0.1846 | 0.8594 | 0.1945 | 0.8162 |
| **DST-TransitNet** | **0.0937** | **0.9444** | **0.1519** | **0.9130** | **0.1515** | **0.8759** |
| **DST-TransitNetV2** | 0.0955 | 0.9362 | 0.1486 | 0.9037 | 0.1486 | 0.8777 |

**This reproduction:**

| Model | Normal MAAPE | Normal R² | Protest MAAPE | Protest R² | COVID MAAPE | COVID R² |
|---|---:|---:|---:|---:|---:|---:|
| FFNN | 0.2959 | 0.9187 | 0.4248 | 0.9255 | 0.4458 | 0.8974 |
| LSTM | 0.3168 | 0.9075 | 0.4422 | 0.9207 | 0.4670 | 0.8834 |
| iTransformer (simplified) | 0.2690 | 0.9421 | 0.4117 | 0.9326 | 0.4249 | 0.9247 |
| DLinear | 0.3208 | 0.9056 | 0.4471 | 0.9203 | 0.5458 | 0.8826 |
| **DST-TransitNet** | **0.2795** | **0.9470** | **0.4320** | **0.9294** | **0.4930** | **0.9248** |
| **DST-TransitNetV2** | 0.2845 | 0.9465 | 0.4341 | 0.9279 | 0.5119 | 0.9214 |

**Diff (reproduction − paper):**

| Model | ΔNormal MAAPE | ΔNormal R² | ΔProtest MAAPE | ΔProtest R² | ΔCOVID MAAPE | ΔCOVID R² |
|---|---:|---:|---:|---:|---:|---:|
| FFNN | +0.1746 | +0.0362 | +0.2516 | +0.2664 | +0.2671 | +0.2744 |
| LSTM | +0.2044 | +0.0016 | +0.2729 | +0.1750 | +0.2941 | +0.1337 |
| iTransformer | +0.1513 | +0.0639 | +0.2418 | +0.0664 | +0.2407 | +0.1312 |
| DLinear | +0.1802 | +0.0477 | +0.2625 | +0.0609 | +0.3513 | +0.0664 |
| DST-TransitNet | +0.1858 | +0.0026 | +0.2801 | +0.0164 | +0.3415 | +0.0489 |
| DST-TransitNetV2 | +0.1890 | +0.0103 | +0.2855 | +0.0242 | +0.3633 | +0.0437 |

**Assessment:**

- **R² is directionally close to the paper and, for most models, actually
  *higher*** — the shapes/rankings largely agree (DST-TransitNet is at or
  near the top on every period; all models keep the paper's ordering "Normal
  best, COVID/Protest harder"). DST-TransitNet's R² gap vs. the paper is
  small (+0.003 to +0.05).
- **MAAPE is 2–3.5× higher than the paper across every single model**,
  including DST-TransitNet. Since MAAPE is invariant to our min-max scaling
  choice, this is not a scaling artifact. The most likely cause is MAAPE's
  known sensitivity to samples where the true value `y` is small (the
  `arctan((y-ŷ)/y)` term blows up in relative terms even for small absolute
  errors); our reconstructed station/hour filtering may retain more
  low-ridership station-timesteps (e.g. very early/late operating hours,
  low-demand residential stations flagged as "challenge stations" in the
  paper's own analysis, Sec. IV.C.2) than the paper's version, and the paper
  does not state an epsilon/clipping convention for `y≈0`, which we had to
  choose ourselves (`metrics.maape`, `eps=1e-8`). A secondary contributor is
  the reduced training budget (§7): `train_stride=4` and only 10 epochs are
  well short of the (unspecified) full training paper authors likely used.
- **The relative ranking among baselines shifts**: our simplified
  iTransformer is competitive with (and by COVID R², marginally *ahead of*)
  DST-TransitNet, whereas the paper shows a clear gap with DST-TransitNet
  variants on top. This is consistent with the documented baseline
  adaptation (§5): sharing weights across all 147 stations turns FFNN/LSTM/
  iTransformer into effectively multi-task learners with far more effective
  training signal than the paper's 147-independent-model design, which
  narrows the gap to DST-TransitNet's spatial-aggregation advantage.

### Table 2 — Long-term MAAPE ratio (lag 12 vs. lag 1)

**Paper (Table 2):**

| Model | Normal | Protest | COVID |
|---|---:|---:|---:|
| **DST-TransitNet** | **1.265** | **1.423** | **1.365** |
| DST-TransitNetV2 | 1.424 | 1.473 | 1.294 |
| LSTM | 1.847 | 1.730 | 1.631 |
| FFNN | 1.906 | 1.783 | 1.624 |
| DLinear | 1.940 | 1.718 | 1.675 |
| iTransformer | 2.214 | 1.858 | 1.918 |

**This reproduction:**

| Model | Normal | Protest | COVID |
|---|---:|---:|---:|
| DST-TransitNet | 1.462 | 1.414 | 1.690 |
| DST-TransitNetV2 | 1.382 | 1.373 | 1.610 |
| FFNN | 2.186 | 1.750 | 1.720 |
| LSTM | 2.877 | 2.219 | 2.466 |
| DLinear | 2.549 | 1.998 | 2.366 |
| iTransformer (simplified) | **1.316** | **1.269** | **1.477** |

**Assessment:** partial qualitative match. DST-TransitNet/V2 are clearly and
consistently more stable over long-term iterative rollout than FFNN/LSTM/
DLinear (ratios ~1.4–1.7 vs ~1.7–2.9), reproducing the paper's central
long-term-stability claim relative to *those* baselines. However, our
simplified iTransformer is *more* stable than DST-TransitNet/V2 here, the
opposite of the paper's finding that DST-TransitNet variants are the most
stable of all six models. We attribute this to the iTransformer baseline's
much smaller, heavily-regularized 4-token structure (§5) being incidentally
robust to iterative error accumulation, rather than to any property the
paper's own (fully independent, per-station) iTransformer would have shared.

### Table 3 — Training time / model size

**Paper (Table 3):** absolute training time is the *sum across 147
independent per-station models* for FFNN/LSTM/iTransformer, vs. a *single*
many-to-many model for DST-TransitNet(V2)/DLinear — not directly comparable
to our shared-weight baselines (§5); shown for reference.

| Model | Training time (min) | Size (MB) |
|---|---:|---:|
| LSTM | 723.8 | 5.75 |
| FFNN | 355.6 | 1.83 |
| iTransformer | 292.5 | 3805 |
| DST-TransitNet | 44.7 | 0.8 |
| DLinear | 36.0 | 0.3 |
| DST-TransitNetV2 | 33.8 | 0.5 |

**This reproduction:**

| Model | Training time (min) | Params | Size (MB) |
|---|---:|---:|---:|
| DST-TransitNet | 26.61 | 35,649 | 0.148 |
| DST-TransitNetV2 | 34.09 | 31,873 | 0.132 |
| iTransformer (simplified) | 4.12 | 11,265 | 0.051 |
| LSTM | 7.13 | 4,513 | 0.020 |
| FFNN | 0.60 | 5,569 | 0.024 |
| DLinear | 0.08 | 42 | 0.003 |

**Assessment:** FFNN/LSTM/iTransformer are 60–1000× faster here than in the
paper purely because of the shared-weight adaptation (one model instead of
147) — expected and not a meaningful comparison point. More interesting:
**DST-TransitNet and DST-TransitNetV2's absolute training times land in the
same order of magnitude as the paper's** (26.6/34.1 min here vs. 44.7/33.8
min in the paper) despite `train_stride=4` and different hardware, which is
a reasonable consistency check that our many-to-many training loop's
per-epoch cost is architecturally comparable to the paper's. **One genuine
architectural discrepancy**: the paper reports V2 as *faster* to train than
V1 (33.8 < 44.7 min, i.e. moving the GRU to the prediction layer is a net
speed-up); our reproduction shows the **opposite** (34.09 > 26.61 min). This
traces to our specific interpretation of Fig. 6 (not available to us as an
image — only the caption/prose describing it): our V2 applies k-GNN spatial
aggregation at *every one of the 20 input timesteps, for all four series*
(80 k-GNN calls/sample) before the single prediction-layer GRU, whereas V1
applies k-GNN only *once per series* after each series' own GRU has already
temporally compressed the sequence (4 k-GNN calls/sample). Our V2 is more
parameter-efficient (31,873 vs 35,649 params, matching the paper's claim
that V2 is more compact) but more compute-intensive per sample under this
interpretation — a plausible, but not the only possible, reading of the
paper's V2 description, and the most likely source of this particular
mismatch.

## 9. Artifacts

Committed to git (branch `claude/sharp-mayer-izj5xv`):

- Code: `dst_transitnet/{config,data,layers,models,baselines,metrics,train,run_reproduction}.py`
- Checkpoints (all 6 models): `dst_transitnet/checkpoints/<model>.pt`
- Per-epoch training logs (all 6 models): `dst_transitnet/logs/<model>_train_log.json`
- Full run stdout/stderr: `dst_transitnet/logs/full_run.log`
- Aggregate metrics: `dst_transitnet/outputs/results_table.json` (Table 1
  source), `per_station_results.json` (per-station R²/MAAPE, cf. paper Fig.
  14), `timing.json` (Table 3 source), `<model>_long_term.json` (Table 2
  source)
- This document.

Kept locally but **not** committed (see `.gitignore`; large and/or
reproducible from the checkpoints):

- Raw per-timestep predictions/ground truth (scaled units):
  `dst_transitnet/outputs/<model>_<period>_{pred,true}.npy` — tens of MB per
  model/period, largely duplicated across models for the same period.
