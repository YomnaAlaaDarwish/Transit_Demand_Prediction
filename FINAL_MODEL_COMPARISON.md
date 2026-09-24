# Final Cross-Model Comparison — Bogotá TransMilenio Benchmark

Consolidates every model built or evaluated across this thesis's three
paper reproductions, plus the repository's pre-existing models, on the same
real-world dataset (Bogotá TransMilenio BRT) and, where the task allows,
the same station set, train cutoff, and Normal/Protest/COVID test periods.

Branch: `trondheim-apc-tabular-reproduction` (carries all three
reproductions — see `TRONDHEIM_APC_REPRODUCTION_NOTES.md` §"Why does this
branch contain everything?"). Source notes for each row's numbers:
`DST_TRANSITNET_REPRODUCTION_NOTES.md`, `TSB_FORECAST_REPRODUCTION_NOTES.md`,
`TRONDHEIM_APC_REPRODUCTION_NOTES.md`.

## Two comparability tiers — read this before the tables

**Tier A — directly comparable** (same station set, same train cutoff
Aug2015–Jul2018, same Normal/Protest/COVID test windows, same short-term
target: next 15-minute boarding count, same R²/MAAPE metrics which are
scale-invariant to each reproduction's own per-station min-max/raw-count
choice): DST-TransitNet + baselines, TSB-Forecast-base, and the new APC
tabular reproduction. These can be read against each other directly.

**Tier B — same real-world periods, different task** (repository's
pre-existing main-branch models): **daily** aggregation, **7-day-ahead**
multi-step forecasting (`settings.yaml`: `steps_back=14, forecast_window=7`)
— a much coarser, different-horizon task. R²/MAAPE are computed the same
way and line up on the same calendar periods, but this is *context*, not a
strictly fair comparison to Tier A. Kept separate below for that reason.

## Tier A — 15-minute, next-step boarding prediction (147 stations)

Sorted by COVID R² within each family, since COVID is where this dataset's
models are most differentiated (a regime shift none of the training data
covers).

| Model | Family | Normal R² | Normal MAAPE | Protest R² | Protest MAAPE | COVID R² | COVID MAAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| **DST-TransitNet** | DST-TransitNet | 0.947 | 0.280 | 0.929 | 0.432 | **0.925** | 0.493 |
| iTransformer (simplified baseline) | DST-TransitNet | 0.942 | 0.269 | 0.933 | 0.412 | 0.925 | 0.425 |
| DST-TransitNetV2 | DST-TransitNet | 0.947 | 0.285 | 0.928 | 0.434 | 0.921 | 0.512 |
| **TSB-Forecast-base** (mean/station) | TSB-Forecast | 0.946 | 0.235 | 0.874 | 0.394 | **0.875** | 0.491 |
| FFNN (shared-weight baseline) | DST-TransitNet | 0.919 | 0.296 | 0.926 | 0.425 | 0.897 | 0.446 |
| LSTM (shared-weight baseline) | DST-TransitNet | 0.908 | 0.317 | 0.921 | 0.442 | 0.883 | 0.467 |
| DLinear | DST-TransitNet | 0.906 | 0.321 | 0.920 | 0.447 | 0.883 | 0.546 |
| APC-tabular: **ridge ensemble** | APC-tabular | 0.899 | 0.439\* | 0.774 | 0.598\* | **0.546** | 0.705\* |
| APC-tabular: XGBoost | APC-tabular | 0.890 | 0.427 | 0.765 | 0.587 | -1.944 | 1.102 |
| APC-tabular: LightGBM | APC-tabular | 0.890 | 0.388 | 0.742 | 0.558 | -2.486 | 1.076 |
| APC-tabular: Random Forest | APC-tabular | 0.884 | 0.438 | 0.742 | 0.595 | -2.334 | 1.095 |
| APC-tabular: CatBoost | APC-tabular | 0.863 | 0.428 | 0.726 | 0.587 | -2.053 | 1.078 |
| APC-tabular: Tabular DNN | APC-tabular | 0.794 | 0.829 | 0.586 | 0.922 | -2.064 | 1.293 |

\* Ensemble MAAPE is computed directly from the ridge-combined predictions
(see `apc_tabular/outputs/ensemble_results.json`), not re-derived from R².

### What this table says

1. **DST-TransitNet and its simplified iTransformer baseline are the most
   robust models overall**, and the *only* family that stays strong through
   COVID without any special handling — direct evidence for the paper's
   central claim that dynamic cross-station spatial attention helps most
   exactly when a station's own recent history turns atypical.
2. **TSB-Forecast-base (Time2Vec + stacked ensemble, no spatial term) is
   competitive with DST-TransitNet on Normal** (0.946 vs 0.947) and clearly
   ahead of the plain APC-tabular models everywhere — its lag/history
   features give it something the fully history-free APC-tabular model
   lacks.
3. **APC-tabular's individual models collapse on COVID** (R² as low as
   -2.49) because that reproduction's design has **zero historical/lag
   features by construction** (faithfully matching the source paper's own
   "horizon-agnostic," purely contextual design) and never saw the `year`
   category 2020/2021 in training. This is the same failure the *paper's
   own authors* documented and fixed with an `isCOVID` flag — see
   `TRONDHEIM_APC_REPRODUCTION_NOTES.md` §6.
4. **Ridge-regression ensembling is the single most dramatic result in
   this whole comparison**: it takes APC-tabular from R²≈-2 (worse than
   predicting the mean) to R²=0.546 on COVID, by learning *negative*
   weights on the worst individual models to cancel out their systematic
   bias — a much larger effect than the same technique produces in the
   source paper, precisely because the individual-model failure it's
   correcting for is so much larger here.

## Tier B — repository's pre-existing daily models (different task, same real-world periods)

| Model | Normal R² | Protest R² | COVID R² |
|---|---:|---:|---:|
| Dense (single) | 0.941 | 0.847 | -1.982 |
| LSTM (single) | 0.940 | 0.840 | -1.007 |
| LSTM (multioutput) | 0.934 | 0.810 | -0.303 |
| CNN (single) | 0.913 | 0.813 | -0.776 |
| Dense (multioutput) | 0.916 | 0.792 | -0.373 |
| CNN (multioutput) | 0.884 | 0.795 | 0.491 |
| ARIMA (single) | 0.811 | 0.776 | **0.607** |
| SARIMA (single) | 0.805 | 0.742 | 0.542 |

Read alongside Tier A only as context, per the resolution/horizon caveat
above. One cross-cutting pattern does hold across *both* tiers, though: **on
COVID specifically, the simpler/more constrained models (ARIMA/SARIMA here;
DST-TransitNet's graph-attention design, TSB-Forecast's lag features, and
ridge stacking in Tier A) all degrade far more gracefully than models with
no mechanism for detecting a regime shift** (the plain neural daily models
here; the individual APC-tabular models in Tier A). The common thread is
informative regardless of resolution: a model needs *either* explicit
history/lag signal *or* explicit regime-shift features *or* an ensembling
step that can down-weight failing components, to survive a shift its
training data never covered.

## Practical takeaway for building on this as a thesis baseline

- If the next step is a **new spatial/graph-aware model**: DST-TransitNet
  is the strongest baseline to beat, especially on Protest/COVID.
- If the next step is a **lightweight, interpretable baseline**:
  TSB-Forecast-base is nearly as strong as DST-TransitNet on Normal and
  much cheaper to train/reason about.
- If the next step is **adding external context (weather/events/holidays
  beyond what's used today)**: the APC-tabular framework is the natural
  host for that experiment (it's explicitly designed around exactly this
  kind of contextual feature fusion), but needs at minimum the isCOVID-style
  regime-shift feature the source paper itself recommends before it's a
  fair comparison point on COVID.
- Ridge-style ensembling is cheap and should be considered as a standard
  final step regardless of which base model(s) are used — the size of its
  effect here suggests it's not just a marginal tweak.

## Sources

- `DST_TRANSITNET_REPRODUCTION_NOTES.md` — DST-TransitNet, DST-TransitNetV2,
  FFNN, LSTM, DLinear, iTransformer (`dst_transitnet/outputs/results_table.json`)
- `TSB_FORECAST_REPRODUCTION_NOTES.md` — TSB-Forecast-base
  (`tsb_forecast/outputs/summary.json`)
- `TRONDHEIM_APC_REPRODUCTION_NOTES.md` — the 5 APC-tabular models + 4
  ensembling strategies (`apc_tabular/outputs/{results_table,
  ensemble_results}.json`), and the Tier B main-branch daily models
  (`apc_tabular/outputs/main_branch_daily_results.json`)
