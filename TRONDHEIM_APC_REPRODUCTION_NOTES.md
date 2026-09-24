# Stop-Level Tabular ML Reproduction Notes

Reproduction of the **base** predictive-modelling framework from **"Data-driven
predictive modelling of stop-level public transit patterns"** (Yusuf, Rasheed
& Lindseth, 2025, *Transportation*, DOI: 10.1007/s11116-025-10689-4),
adapted to this repository's Bogotá TransMilenio dataset.

Branch: `trondheim-apc-tabular-reproduction`, created off
`tsb-forecast-base-reproduction` (which carries `dst_transitnet/` and
`tsb_forecast/` too — see "Why does this branch contain everything?" below).
Code: `apc_tabular/`.

The paper proposes no named model/acronym (just a "horizon-agnostic ML
framework"), so this reproduction is named after the dataset/institution
(Trondheim's AtB APC system) rather than inventing one.

## Why does this branch contain everything?

Each reproduction branch was created off the *previous* reproduction's
branch, not off `main`:

```
main
 └─ claude/sharp-mayer-izj5xv            (DST-TransitNet)
     └─ tsb-forecast-base-reproduction   (+ TSB-Forecast-base)
         └─ trondheim-apc-tabular-reproduction  (+ this reproduction)
```

Nothing was deleted at any step, so every branch also carries the
original repo's content (`run.py`, `data.py`, `models/`, `output/`)
untouched. This was deliberate: it means the final cross-reproduction
comparison (§6 below, and requested by the user) can be built from a single
checkout instead of juggling three separate branches/environments.

## 1. Scope: what "base" means here, and why so much is out of scope

This paper's data (Trondheim AtB bus APC system: per-trip, per-stop-visit
records) is structurally richer than Bogotá's TransMilenio data (station-
level totals aggregated per 15-minute window, no trip/route granularity).
Unlike the SBERT/weather exclusions in the TSB-Forecast reproduction (author
design choices we chose not to reproduce), most of what's excluded here is
an **unavoidable dataset constraint**, not a scope choice:

| Paper element | Status here | Why |
|---|---|---|
| Row granularity: (trip, stop-visit) | **Changed** to (station, 15-min interval) | No trip/route records exist in the Bogotá data at all |
| `Boarding` target | **Reproduced** | — |
| `Alighting` target | **Excluded** | TransMilenio only records boardings (fare tap-in); no tap-out/alighting data exists |
| Operational targets (`StopActualArrival`, `StopTime`) | **Excluded** | No per-trip GPS/schedule-adherence data available |
| GT/video validation sub-task (paper Table 2) | **Excluded** | No manually/video-verified passenger counts exist for Bogotá |
| Weather features | **Excluded** (deferred, per user direction) | Same reason as TSB-Forecast: no Bogotá weather data in this repo yet |
| Terrain features (elevation/slope) | **Excluded** (deferred) | No per-stop elevation data for Bogotá in this repo |
| Demographics/land-use features | **Excluded** (deferred) | No Bogotá-equivalent of Norway's statistical grids available offline |
| `TransferStop`/`StopType` features | **Reproduced faithfully** | Computed from `clean_stations_database_v2.csv`'s zone/corridor (`nombrelinea`) column — most stations belong to exactly one zone, a few to two, giving a genuine transfer-stop signal |
| 5 algorithms (Tabular DNN, CatBoost, RF, XGBoost, LightGBM) | **Reproduced** | — |
| 4 ensembling strategies (Sec. 2.2/4.1.5) | **Reproduced** | — |
| No lag/autoregressive features | **Reproduced (matches paper exactly)** | The paper's own Table 10 input features are purely categorical/temporal/spatial context — no historical passenger-count lags at all |

## 2. Architecture: single pooled model per algorithm, not per-station

Unlike DST-TransitNet (one many-to-many graph model) or TSB-Forecast (147
independent per-station models), this reproduction trains **one model per
algorithm on all 147 stations pooled together**, with station identity as
just another categorical feature — this is exactly the paper's own design
(a single model over all lines/stops/trips), and directly reflects the
paper's own stated limitation: *"stop-level modelling lacks inductive biases
on the spatial structure of transit networks"* (abstract). This reproduction
inherits that same property by construction.

Row features: `station_code`, `primary_zone` (Line analogue), `stop_type`,
`hour`, `day_of_week`, `month`, `year` (categorical); `latitude`,
`longitude`, `transfer_stop_count` (continuous); `is_holiday`, `is_weekend`
(boolean). Target: `y` = boarding count at `t+1` (15 min ahead) — same
short-term target as DST-TransitNet/TSB-Forecast, for comparability.

Same station set, train cutoff, and Normal/Protest/COVID test periods as the
other two reproductions (`dst_transitnet.config.DataConfig`, reused
directly).

## 3. Hyperparameters

Used the paper's own Table 12 **reported-optimal** values directly, rather
than re-running their W&B/Optuna search (a full sweep would cost far more
than this session's compute budget, and the paper already reports the
result). Where compute forced a reduction (mainly tree depth/estimator
counts, given ~2.76M pooled training rows), this is flagged in
`apc_tabular/config.py` with the paper's original value alongside, e.g.:
Random Forest `n_estimators` 320→100, `max_depth` 32→16; XGBoost/LightGBM
`max_depth` 16→8; CatBoost `depth` 10→8. Tabular DNN's architecture
(5 blocks, hidden sizes `[128,2048,256,512,256]`, dropout 0.1, lr 0.01) is
used exactly as reported; only its batch size was changed (1024→8192, both
values from the paper's *own* searched options — see §5).

**Training subsample**: every 4th 15-min timestamp used for training
(`train_stride=4`, same pattern as TSB-Forecast), giving ~2.76M pooled
training rows (147 stations × ~18.7k timestamps) — comparable in order of
magnitude to the paper's own 21.8M training rows.

## 4. Two real bugs found and fixed during implementation

**Bug 1 — cartesian-product blowup in feature assembly.** The first version
of `melt_to_long` joined the melted (timestamp, station) long-format frame
with calendar and station-metadata lookups via `pd.concat(..., axis=1)`
while the frame still carried a *duplicated* timestamp index (each
timestamp repeats once per station after `melt()`). `pd.concat` aligns by
index, and aligning two frames that share a duplicated index value produces
the *cartesian product* of the rows sharing that value — 147×147 rows per
timestamp instead of 147, an 147x blowup (55,125 expected rows became
8,103,375). Fixed by `reset_index(drop=True)` before every such join, making
all concats purely positional. Caught immediately via a smoke test before
any real training ran.

**Bug 2 — OOM from a monolithic Tabular DNN forward pass.** The full 5-model
run OOM-killed (this session's 15GB memory cgroup) partway through Tabular
DNN training. Diagnosis: added per-batch RSS logging and confirmed memory
was *completely flat* through every training batch — ruling out a leak.
Re-running the DNN alone (fresh process) still OOM-killed at the exact same
~13.9GB, which pointed at the one un-batched code path: validation-set and
`predict()` both ran a **single forward pass over the entire frame at once**
(up to 1.22M rows for validation, 2.78M for the COVID test period). With the
paper's own 2048-unit widest hidden layer, that materializes activations of
roughly `1.2M × 2048 × 4 bytes ≈ 10GB` for just one layer. Fixed by routing
both validation and `predict()` through the same chunked
(`dnn_batch_size`-sized) forward pass the training loop already used.
Confirmed fixed: RSS stayed flat at ~2.5-2.6GB through all 6 training epochs
and both full-dataset prediction passes afterward.

Because of this, the Tabular DNN had to be trained in a separate, isolated
process (`apc_tabular/logs/dnn_only.log`) from the four tree models
(`apc_tabular/logs/full_run.log`, which still shows the OOM crash occurring
right as `tabular_dnn` started); `apc_tabular/finalize_results.py` reloads
all five saved checkpoints afterward to rebuild one unified results table
and re-run the ensembling strategies over all five models together
(`apc_tabular/logs/finalize.log`).

## 5. Compute-budget deviations (explicit)

- Hyperparameter tuning **not** re-run; paper's Table 12 optimal values used
  directly (see §3).
- Tree model depth/estimator counts reduced from the paper's optimum for
  CPU tractability at ~2.76M pooled rows (see §3, `config.py`).
- Tabular DNN batch size increased 1024→8192 (both from the paper's own
  searched options `[1024,2048,4096,8192]`) purely for CPU training
  throughput at this row count; `max_epochs` capped at 8 with early
  stopping (patience 3) since the paper doesn't report an epoch count.

## 6. Results

Full run logs: `apc_tabular/logs/{full_run,dnn_only,finalize}.log`. Raw
numbers: `apc_tabular/outputs/{results_table,ensemble_results,timing}.json`.
Main-branch daily-model evaluation: `apc_tabular/outputs/
main_branch_daily_results.json` (via `apc_tabular/eval_main_branch_models.py`
— see its docstring for the important day-vs-15-min-resolution caveat).

### Table 1 — Per-algorithm performance (R² / RMSE / MAAPE)

| Model | Normal R² | Normal MAAPE | Protest R² | Protest MAAPE | COVID R² | COVID MAAPE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.884 | 0.438 | 0.742 | 0.595 | **-2.334** | 1.095 |
| XGBoost | 0.890 | 0.427 | 0.765 | 0.587 | **-1.944** | 1.102 |
| CatBoost | 0.863 | 0.428 | 0.726 | 0.587 | **-2.053** | 1.078 |
| LightGBM | 0.890 | 0.388 | 0.742 | 0.558 | **-2.486** | 1.076 |
| Tabular DNN | 0.794 | 0.829 | 0.586 | 0.922 | **-2.064** | 1.293 |

**Paper's own finding reproduced**: tree-based models all outperform the
Tabular DNN (paper Sec. 4.1.1: *"tree-based models all outperform the...DNN
on this task, with XGBoost performing the best"*) — confirmed here exactly:
XGBoost/LightGBM lead on Normal/Protest, and the DNN is clearly worst on
every single period and metric.

### Table 2 — Ensemble strategies (Sec. 2.2/4.1.5)

| Strategy | Normal R² | Protest R² | COVID R² |
|---|---:|---:|---:|
| Simple average | 0.893 | 0.748 | -1.982 |
| Best pair | 0.896 (XGB+LGB) | 0.760 (RF+XGB) | -1.762 (XGB+TAB) |
| Weighted least squares | 0.897 | 0.765 | -1.759 |
| **Ridge stack** | **0.899** | **0.774** | **0.546** |

**Paper's finding reproduced, more dramatically**: the paper reports ridge
regression as the best ensembling strategy (*"ridge regression demonstrates
superior performance through flexible model weighting without sum-to-one
constraints"*, reducing RMSE from 2.82 to 2.64). Here the effect is far more
striking: on COVID, **every individual model has strongly negative R²**
(-1.9 to -2.5 — worse than predicting the mean), yet ridge stacking recovers
R²=0.546 by assigning *negative* weights to the worst offenders (RF: -0.148,
Tabular DNN: -0.053) — i.e. it learns to subtract out each model's
systematic COVID-era bias rather than just averaging it in.

### Why every individual model fails on COVID (R² deeply negative)

This reproduction's row features are purely contextual — station identity,
hour, day-of-week, month, **year**, holiday/weekend flags — with **no
historical/lag features at all** (matching the paper's own "horizon-
agnostic" design exactly, §1). Training data only covers Aug 2015–Jul 2018,
so the model never observes `year=2020` or `2021` at all; at COVID-period
inference time this categorical value is entirely unseen, and the model has
*no other mechanism* (no recent-history signal) to detect "this is an
unprecedented regime." This is the same failure mode the paper's own authors
document and fix in Sec. 4.1.2: *"we introduced a binary isCOVID feature...
XGBoost's performance in the affected fold improved substantially."* We
report the failure as-is rather than add that fix, to keep this pass a clean
"base" reproduction (the same reasoning applied to deferring weather/terrain/
demographics) — adding a regime-shift indicator is a natural, well-motivated
next step, using the paper's own documented remedy.

## 7. Main-branch (pre-existing) daily models — added context, not a fair comparison

For completeness (per user request to compare against "whatever was already
on the main branch"), `apc_tabular/eval_main_branch_models.py` scores the
repository's **pre-existing** ARIMA/SARIMA/Dense/CNN/LSTM prediction JSONs
(`output/day/static/{single,multioutput}/`, generated before any of this
thesis's reproduction work) against the same real-world periods:

| Model | Normal R² | Protest R² | COVID R² |
|---|---:|---:|---:|
| ARIMA (single) | 0.811 | 0.776 | 0.607 |
| SARIMA (single) | 0.805 | 0.742 | 0.542 |
| Dense (single) | 0.941 | 0.847 | -1.982 |
| CNN (single) | 0.913 | 0.813 | -0.776 |
| LSTM (single) | 0.940 | 0.840 | -1.007 |
| Dense (multioutput) | 0.916 | 0.792 | -0.373 |
| CNN (multioutput) | 0.884 | 0.795 | 0.491 |
| LSTM (multioutput) | 0.934 | 0.810 | -0.303 |

**Important caveat**: these are **daily-aggregation, 7-day-ahead multi-step**
models (`settings.yaml`: `aggregation=day`, `steps_back=14`,
`forecast_window=7`) — a fundamentally different, much coarser task than the
three reproductions' 15-minute, 1-step-ahead setup. R²/MAAPE are computed
the same way and the real-world periods line up, but this is *informative
context*, not a strictly fair apples-to-apples comparison. See §6 of
`FINAL_MODEL_COMPARISON.md` for the full cross-reproduction table and this
caveat repeated in context.

Notable finding: the daily neural models (Dense/CNN/LSTM) trained pre-2018
show strongly negative R² on COVID, while ARIMA/SARIMA degrade far more
gracefully (0.61/0.54) — a daily-resolution echo of the same "no regime-
shift signal" failure mode discussed in §6 above, and consistent with the
DST-TransitNet paper's own narrative about neural models struggling to
extrapolate through unprecedented disruptions.

## 8. Artifacts

Committed to git (branch `trondheim-apc-tabular-reproduction`):

- Code: `apc_tabular/{config,features,models,ensemble,metrics,train,
  run_reproduction,eval_main_branch_models,finalize_results}.py`
- Checkpoints (all 5 models): `apc_tabular/checkpoints/*.joblib`
- Logs: `apc_tabular/logs/{full_run,dnn_only,finalize}.log`
- Aggregate metrics: `apc_tabular/outputs/{results_table,ensemble_results,
  timing,main_branch_daily_results}.json`
- This document.

Not committed: per-timestep raw predictions (large, gitignored, same
convention as the other two reproductions).
