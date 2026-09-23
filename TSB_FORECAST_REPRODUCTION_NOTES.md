# TSB-Forecast-base Reproduction Notes

Reproduction of the **base** model from **"TSB-Forecast: A Short-Term Load
Forecasting Model in Smart Cities for Integrating Time Series Embeddings and
Large Language Models"** (Hasan, El-Tazi, Moawad & Eissa, 2025,
IEEE Access, DOI: 10.1109/ACCESS.2025.3597421), adapted to this repository's
Bogotá TransMilenio BRT ridership dataset.

Branch: `tsb-forecast-base-reproduction` (created off
`claude/sharp-mayer-izj5xv`, which holds the DST-TransitNet reproduction —
not merged into that branch or into `main`). Code: `tsb_forecast/`.

## Scope of this pass ("the base")

Per explicit direction: implement the paper's core pipeline —
Time2Vec-inspired temporal embedding + two-layer stacked ensemble
(ExtraTreesRegressor + XGBoost, LinearRegression meta-learner) — **without**
the paper's SBERT/news semantic-embedding module and **without** its
external weather features, since Bogotá has no equivalent news corpus or
weather dataset available in this repo. Calendar features (hour-of-day,
day-of-week, Colombian holidays via the already-installed `holidays_co`
package) are kept, since those are native/structured, not "external
context," and the paper itself treats calendar variables as part of the
structured feature block, not the SBERT block.

**A follow-up pass, once suitable local-context data is available (e.g.
Bogotá news/event text, weather), will add an equivalent context-fusion
module the same way the paper fuses SBERT** — this reproduction is
deliberately the paper's "base," not a claim that context fusion doesn't
matter.

Same station set, splits, and Normal/Protest/COVID test periods as the
DST-TransitNet reproduction (`dst_transitnet/config.py`, reused directly via
`tsb_forecast/config.py`), per explicit direction, so the two reproductions'
results are directly comparable station-for-station.

## 1. Paper → implementation mapping

| Paper element | Section | Implementation |
|---|---|---|
| Time2Vec-inspired learned temporal embedding | III.B.1 | `tsb_forecast/time2vec.py`: pointwise 2-layer FFNN encoder (see §3 below for the exact reconstruction) |
| Lag features (`actual_load_lag8`, `_1day`, `_1week`, ...) | III.B.1 step 1 | `tsb_forecast/features.py:build_lag_table` |
| Calendar features | III.B.2 | `tsb_forecast/features.py:build_calendar_features` |
| SBERT / news embeddings | III.B.1 | **excluded** (no equivalent data source) |
| Weather (`tempC`) | III.B.1 | **excluded** ("external factor", no Bogotá weather data in this repo) |
| Two-layer stacked ensemble (ETR + XGBoost, LinearRegression meta) | III.C | `tsb_forecast/ensemble.py` |
| 5-fold TimeSeriesSplit CV | III.C.2 | `tsb_forecast/ensemble.py:TimeSeriesStackingRegressor` (see §5 — reduced to 3 folds, custom implementation) |
| MAE / RMSE / SMAPE | IV.D | `tsb_forecast/metrics.py` |
| R² / MAAPE (added for DST-TransitNet comparability) | — | `tsb_forecast/metrics.py` (imports from `dst_transitnet.metrics`) |

## 2. Dataset / splits / stations

Identical to the DST-TransitNet reproduction — reused directly, not
re-derived: 147 BRT stations, train Aug 2015–Jul 2018, test periods Normal
(2018-08-01–2019-01-01), Protest (2019-11-21–2019-12-27), COVID
(2020-03-01–2020-11-05); per-station min-max scaling fit on train. See
`DST_TRANSITNET_REPRODUCTION_NOTES.md` §2 for the full derivation and the
documented boundary-matching methodology.

## 3. Time2Vec-inspired embedding — reconstruction of an ambiguous description

The paper's own description (quoted in `time2vec.py`'s docstring) is a
**pointwise** feedforward encoder — "two linear layers with ReLU
activation" — not a recurrent or attention model. Read literally, it has no
mechanism to mix information *across* window positions; the "48-step sliding
window" serves as the *training* signal (many (t, t+1) pairs), and only the
*current* timestep's raw feature vector is actually encoded at inference
("the final 64-dimensional output corresponding to the latest time step").

Our reconstruction, made concrete and testable:

```
encoder(x_t) = Linear(hidden, embed_dim)(ReLU(Linear(in_dim, hidden)(x_t)))
decoder(encoder(x_t)) ≈ x_{t+1}          (MSE loss)
```

trained once, **globally** (pooled across a sample of 20 stations' training
data — not all 147, for compute-tractability, since the encoder is a shared
feature extractor, not a per-station model — analogous to the compute-budget
weight-sharing already documented for DST-TransitNet's baselines).

`x_t` = `[current scaled ridership, same-time-last-week scaled ridership]`
(2-D) — the paper used 8 selected numeric features (`actual_load_MW`,
`tempC`, several lags); we use the two temporally-relevant signals available
without weather/SBERT. `embed_dim = 64` **[PAPER-SPECIFIED]**.

## 4. Feature set actually used per station

- `lag_0` (current value, i.e. the paper's `actual_load_MW`), `lag_1`
  (15 min), `lag_4` (1h), `lag_8` (2h) **[PAPER-SPECIFIED lag_8, adapted
  resolution]**
- `lag_1day`, `lag_1week` **[PAPER-SPECIFIED, adapted]** — `lag_1month`/
  `lag_1year` dropped: many training windows (especially early in the
  Aug2015–Jul2018 training period) have no valid 1-year lag, unlike the
  paper's 5-year UK dataset where this was viable throughout.
- Calendar: `hour_sin/cos`, `dow_sin/cos`, `is_weekend`, `is_holiday`
- `embed_0`..`embed_63`: the shared Time2Vec embedding (§3)
- Target: `y` = value at `t+1` (15 min ahead) — matches DST-TransitNet's
  short-term target exactly, for direct comparability.

## 5. Stacked ensemble — implementation notes and compute-budget adaptations

**Bug found and fixed during implementation**: sklearn's `StackingRegressor`
generates out-of-fold base-learner predictions via `cross_val_predict`,
which requires the CV splitter to yield a full partition of the data.
`TimeSeriesSplit` does not (its first fold's training rows are never held
out), so `StackingRegressor(cv=TimeSeriesSplit(...))` raises `ValueError:
cross_val_predict only works for partitions`. Replaced with a manual
`TimeSeriesStackingRegressor` (`ensemble.py`) that drives OOF meta-feature
generation with `TimeSeriesSplit` directly (covering only the rows a
time-respecting CV can actually hold out) and refits both base learners on
the full training set for inference — the same net effect the paper
describes, without depending on an sklearn internal built for
non-time-aware CV.

**Compute-budget adaptations** (documented, same spirit as the
DST-TransitNet reproduction's baseline adaptations):

1. **Hyperparameter tuning done once, not per-station.** The paper tunes
   ETR/XGBoost via grid search "within the TSV framework" for its single
   system-wide series. Repeating a full grid search independently for 147
   stations would multiply the paper's own tuning cost 147×. We grid-search
   once on the **system-wide aggregate** series (sum of all 147 stations,
   its own min-max scaling) and reuse the resulting fixed hyperparameters
   for every station's *independently trained* stack (only the search is
   shared — every station's trees and meta-learner weights are fit from
   scratch on that station's own data). Tuned result this run: `ETR
   {max_depth: 15, n_estimators: 100}`, `XGB {max_depth: 6, n_estimators:
   100, learning_rate: 0.1}` (`tsb_forecast/outputs/tuned_hyperparameters.json`).
2. **Grid and CV-fold count reduced.** An initial run with unbounded-depth
   ETR, `n_estimators=200`, and 5-fold stacking CV measured ~12 min/station
   (~29 hours for 147 stations) — intractable. Capping ETR/XGB depth,
   reducing `n_estimators` to the grid `{50,100}`, and reducing the stacking
   CV to **3 folds** (down from the paper's 5) brought this to ~39s/station
   (~95 min total), with no material change in accuracy on the 3-station
   pilot used to diagnose the slowdown.
3. **Long-term forecast rollout vectorized.** The naive per-(starting-point,
   lag) prediction loop (one `model.predict()` call per row) was the other
   major bottleneck at 147-station scale; `train.py:long_term_forecast_station`
   batches all starting points for a given lag into a single `predict()`
   call (12 calls total per station instead of thousands).

## 6. Results

Full run: `tsb_forecast/logs/full_run.log` (147/147 stations, ~95 min
total). Raw numbers: `tsb_forecast/outputs/{summary,per_station_results,
per_station_long_term,timing,tuned_hyperparameters}.json`.

### Table 1 — Aggregate (mean across 147 stations) MAE / RMSE / SMAPE / R² / MAAPE

| Period | MAE | RMSE | SMAPE (%) | R² | MAAPE |
|---|---:|---:|---:|---:|---:|
| Train | 0.0184 | 0.0270 | 24.78 | 0.9689 | 0.2261 |
| Normal | 0.0189 | 0.0286 | 25.51 | 0.9463 | 0.2353 |
| Protest | 0.0227 | 0.0350 | 45.76 | 0.8742 | 0.3942 |
| COVID | 0.0144 | 0.0209 | 50.61 | 0.8754 | 0.4905 |

*(MAE/RMSE are in scaled [0,1] ridership units, since raw per-station min-max
scaling is used, matching the DST-TransitNet reproduction's convention —
R²/MAAPE are invariant to this scaling and thus directly comparable to
Table 1 in `DST_TRANSITNET_REPRODUCTION_NOTES.md`.)*

### Table 2 — Long-term MAAPE ratio (lag 12 vs. lag 1), mean across stations

| Period | Ratio |
|---|---:|
| Normal | 1.255 |
| Protest | 1.302 |
| COVID | 1.687 |

### TSB-Forecast-base vs. DST-TransitNet (same stations/periods/short-term target)

| Model | Normal R² | Normal MAAPE | Protest R² | Protest MAAPE | COVID R² | COVID MAAPE | Long-term ratio (Normal/Protest/COVID) |
|---|---:|---:|---:|---:|---:|---:|---|
| DST-TransitNet | 0.9470 | 0.2795 | 0.9294 | 0.4320 | 0.9248 | 0.4930 | 1.46 / 1.41 / 1.69 |
| **TSB-Forecast-base** | 0.9463 | 0.2353 | 0.8742 | 0.3942 | 0.8754 | 0.4905 | **1.26 / 1.30 / 1.69** |

**Assessment:**

- On the **Normal** period, TSB-Forecast-base is essentially tied with
  DST-TransitNet on R² (0.946 vs 0.947) and *better* on MAAPE (0.235 vs
  0.280) — a purely feature-engineered ensemble, with no spatial/graph
  component at all, matches a full GAT+GRU+k-GNN architecture on the easiest
  test period. This is a genuinely interesting finding for the thesis: most
  of the short-term, single-station predictability in this dataset comes
  from strong autoregressive/calendar structure that tree ensembles can
  already exploit well.
- On **Protest** and **COVID** — the two disruption periods the
  DST-TransitNet paper specifically credits its dynamic spatial-attention
  mechanism for handling better — TSB-Forecast-base's R² drops noticeably
  more than DST-TransitNet's (0.874 vs 0.929 protest; 0.875 vs 0.925 covid),
  while MAAPE stays roughly comparable or even slightly better. This is
  consistent with the expected story: **without any cross-station spatial
  signal**, a per-station model has less information to fall back on when a
  station's own recent history becomes atypical (system-wide disruptions),
  whereas DST-TransitNet's graph attention can borrow signal from
  structurally-related stations. This is the kind of gap a context-fusion
  module (weather/events, added later per the plan in this document's
  header) or a spatial extension could plausibly help close.
- The **long-term stability** result is a genuine surprise: TSB-Forecast-base
  is *more* stable (lower ratio = smaller relative growth in error from lag 1
  to lag 12) than DST-TransitNet on Normal (1.26 vs 1.46) and Protest (1.30
  vs 1.41), and tied on COVID (1.69 vs 1.69). This should be read with two
  caveats: (a) the long-term rollout here re-derives `lag_1week` fresh at
  every lag from true historical data (never predicted) exactly like
  DST-TransitNet, but the *other* lag features (`lag_0/1/4/8/1day`) are a
  small, explicit set rather than a full 20-step GRU input, so error
  accumulation has fewer channels to compound through; (b) the long-term
  evaluation subsamples target timestamps (`--long_term_stride 20`, i.e.
  every 20th timestep) for compute reasons on both reproductions, but with
  different absolute sample counts, so the two ratios are not computed on
  identically-sized samples. We report this finding as-is rather than
  discount it, but flag the caveat for anyone using it as a headline
  comparison.

## 7. Assumptions and discrepancies (summary)

- **[ASSUMPTION]** Time2Vec-inspired encoder architecture/training
  objective reconstructed from an underspecified description (§3).
- **[ASSUMPTION]** Trained once globally (20 pooled stations), not
  per-station or on all 147 — compute-budget choice, documented.
- **[ASSUMPTION]** Lag set adapted to 15-min resolution and our dataset's
  shorter usable history (dropped 1-month/1-year lags).
- **[ASSUMPTION]** ETR/XGBoost hyperparameter grids and their reduced
  values (§5) — the paper doesn't publish its grid.
- **[COMPUTE BUDGET]** Hyperparameters tuned once (system-wide aggregate),
  reused across all 147 independently-trained per-station stacks.
- **[COMPUTE BUDGET]** Stacking CV reduced from 5 to 3 folds.
- **[IMPLEMENTATION FIX]** Manual `TimeSeriesStackingRegressor` in place of
  `sklearn.ensemble.StackingRegressor` (incompatible with `TimeSeriesSplit`).
- **[SCOPE, per explicit user direction]** SBERT/news module and weather
  features excluded entirely for this pass; calendar features (holiday/
  day-of-week) kept as native structured features. A follow-up pass will add
  an equivalent context-fusion module once suitable data exists.

## 8. Artifacts

Committed to git (branch `tsb-forecast-base-reproduction`):

- Code: `tsb_forecast/{config,features,time2vec,ensemble,train,metrics,run_reproduction}.py`
- Shared Time2Vec encoder checkpoint: `tsb_forecast/checkpoints/time2vec_encoder.joblib`
- Full run log: `tsb_forecast/logs/full_run.log`
- Aggregate metrics: `tsb_forecast/outputs/{summary,per_station_results,per_station_long_term,timing,tuned_hyperparameters}.json`
- This document.

Not committed (large / not practical, and reproducible from the checkpoints
+ code): the 147 per-station fitted stacking models themselves (not saved by
default; pass `--save_station_models` to `run_reproduction.py` to persist
them if needed for a specific station's follow-up analysis).
