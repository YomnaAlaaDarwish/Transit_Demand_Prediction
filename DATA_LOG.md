# DATA_LOG

Thesis data pipeline for station-level short-term ridership prediction on
TransMilenio (Bogotá) trunk BRT. Track A is daily, like the jdcaicedo251 benchmark.
Track B is 15-minute, like DST-TransitNet. Branch: `thesis-data-pipeline`, created
from `trondheim-apc-tabular-reproduction`.

## Project rules

- Existing files are never modified, moved, renamed or deleted. Only new files are created.
- Raw inputs stay where they are and are recorded in `data/raw/README.md` (path, size, hash).
- Every derived file can be rebuilt by a numbered script in `scripts/`.
- A station code is always a 5-character string, e.g. `"02101"`. Never a number.
- There is one station order, `data/interim/station_order.csv`, with 147 stations. It is
  used for ridership columns, graph matrix rows and columns, and array station axes.
- Time is local Bogotá time (UTC−5, no DST), and a timestamp marks the **start** of its
  interval. See §1b(c) for the evidence behind this.
- Tables are Parquet, matrices are `.npy`, and small human-readable tables are CSV.
- Only small files are committed: `stations.csv`, `station_order.csv`, figures, scripts
  and docs. Everything else under `data/interim/` can be rebuilt and is not committed.
  The repo `.gitignore` ignores `*.csv`, so the two station CSVs are force-added.

Environment used: Python 3.11.15, pandas 3.0.6, pyarrow 25.0.1, matplotlib 3.11.2,
openpyxl 3.1.5, holidays 0.105.

## Folder layout

```
data/raw/README.md      manifest of raw inputs (referenced in place, nothing moved)
data/interim/           clean base tables
  stations.csv, station_order.csv
  ridership_15min.parquet            (pending approval of §1b)
  time_features_15min.parquet        (later)
  station_time_features_15min.parquet(later)
  graphs/*.npy                       (later)
  figures/                           diagnostic figures
  reports/                           audit CSVs written by the scripts (not committed)
data/processed/track_a_daily/, track_b_15min/   (later, by a builder script)
scripts/NN_*.py
```

## Script index

| # | Script | Writes | Status |
|---|---|---|---|
| 01 | `scripts/01_build_station_table.py` | `data/interim/stations.csv`, `station_order.csv`, `figures/stations_by_trazado.png`, `reports/01_*.csv` | done 2026-09-25 |
| 02 | `scripts/02_audit_time_grid.py` | `reports/02_*.csv`, `figures/02_profile_15min.png` (audit only) | done 2026-09-25 |
| 03 | `scripts/03_audit_edges.py` | `reports/03_edges_with_distance.csv`, `figures/03_edges.png` (audit only) | done 2026-09-25 |

Run each one from the repo root: `python scripts/NN_*.py`.

---

## 1. Station table (`01_build_station_table.py`), 2026-09-25

**Sources:** the ridership parquet columns; the GeoJSON; `preprocessing/Estaciones_Troncales_de_TRANSMILENIO.csv`
as the fallback; `data/clean_stations_database_v2.csv` for coordinates only; and the benchmark
result files `output/day/static/multioutput/dense/*.json` as the reference list of 147.

**Steps**
1. Parse the 151 ridership columns `(NNNNN) Name` into `code` and `bench_name`. The encoding
   repair (`ToberÃ­n` → `Toberín`, latin-1→utf-8 round trip applied only when `Ã`/`Â` is present)
   is in place but changed **0** names, because the parquet's names are already clean UTF-8.
2. Load the GeoJSON: 153 features. `num_est` is stored as a string, but 4 values have
   only 4 characters (`9005`, `5009`, `5010`, `5011`). They are zero-padded to 5.
   There are **0 duplicate codes**.
3. Match on `code == num_est`.
   - All 147 benchmark stations match the GeoJSON directly, so the fallback CSV was not needed.
   - The 4 cable stations (`40000` Cable Portal Tunal, `40001` Juan Pablo II, `40002`
     Manitas, `40003` Mirador del Paraiso) are in neither the GeoJSON nor the fallback CSV.
     Their lat/lon come from `clean_stations_database_v2.csv` by exact code, and each
     code has one consistent coordinate. They have no GeoJSON attributes.
     `match_status = coords_only_access_db`.
   - There are 6 GeoJSON stations with no ridership column: `07111` Ricaurte - NQS, `14003`
     Temporal AV. Jiménez - Inter Eléctricas, `09005` Danubio, `05009` Islandia,
     `05010` Los Laureles, `05011` Tibanica - Primavera. The last four are likely
     post-2021 openings or temporary stations. They are **not** in `stations.csv`.
4. Benchmark check. The original `data.py`/`settings.yaml` contain **no explicit cable
   filter**. `read_data` keeps every column containing `(`. Its input
   (`data/clean_transactions.csv`) is not in the repo. Still, all 14 benchmark result
   folders under `output/day/` contain exactly 147 station files with no `(400xx)`, and
   their codes **match our 147 exactly**. All 147 names also match after the benchmark's
   own `strip_accents`. So the cable stations were already missing from the CSV the
   benchmark used. `in_benchmark_147` is False only for 40000–40003.
5. Corridor check: the first 2 digits of the code crossed with `id_trazado` (`reports/01_prefix_vs_trazado.csv`):

   | prefix | id_trazado (n) | one-to-one? |
   |---|---|---|
   | 02 | TZ002 (17) | yes |
   | 03 | TZ003 (14) | yes |
   | 04 | TZ005 (13) | yes |
   | 05 | TZ009 (10) | yes |
   | 06 | TZ016 (14) | yes |
   | 07 | TZ010 (13), TZ008 (10), TZ011 (4), TZ007 (2) | **no**, split by 3rd digit: 070→TZ010, 071→TZ008 (+07112/07113→TZ010), 072→TZ007, 075→TZ011 |
   | 08 | TZ014 (3) | yes |
   | 09 | TZ001 (14), TZ012 (12), TZ013 (1) | **no**: 090→TZ012 (+09000 Portal Usme→TZ013), 091→TZ001/TZ012 |
   | 10 | TZ018 (10), TZ019 (1) | **no**: 10009 Museo Nacional→TZ019 |
   | 12 | TZ009 (6) | yes |
   | 14 | TZ015 (2), TZ009 (1) | **no**: 14001 La Sabana→TZ009 |

   TZ009 spans three prefixes (05, 12, 14). **The code prefix is not a corridor ID.** Use `id_trazado`.
6. Outputs
   - `data/interim/stations.csv` has 151 rows and the columns `code, bench_name, current_name, lat, lon,
     id_trazado, tipo_esta, num_vag, area_est, long_est, ancho_est, num_acc, acc_puent,
     esta_oper, match_status, in_benchmark_147`. `bench_name` is the ridership name;
     `current_name` is the GeoJSON `nom_est`, e.g. 02000 "Cabecera Autopista Norte" →
     "Portal Norte – Unicervantes".
   - `data/interim/station_order.csv` has 147 rows (`idx, code, bench_name`) in **ridership
     column order**. That order is by code except for `07010` Bosa, which sits at idx 146,
     after `14005`. DST-TransitNet sorts by code instead.
   - `data/interim/figures/stations_by_trazado.png`: 17 `id_trazado` values drawn with
     8 hues × 3 marker shapes, plus the cable stations as open diamonds.

**Checks passed:** codes unique; 147 = benchmark set; 147/147 GeoJSON matches; no duplicate `num_est`.

---

## 1b. Ridership time-grid audit (`02_audit_time_grid.py`), 2026-09-25, awaiting approval

Input: `data/transmilenio_transactions.parquet`, 186,378 rows, from **2015-08-01 00:00 to
2021-05-01 23:45** (the file's last timestamp). It has no NaN and no negative values.

### (a) Duplicate timestamps
- 2,188 timestamps appear exactly twice (4,376 rows).
- **In every pair, one row is entirely zero.** 2,076 pairs are nonzero + all-zero, and
  112 pairs are both all-zero (overnight). No pair has two different nonzero rows.
  Summing the pair and dropping the zero row therefore give **identical values in 2,188/2,188 cases**.
- The duplicates fall on 25 dates: the 1st of the month after every 30-day month
  (Oct, Dec, May, Jul 1st), and 1–3 March after February (only 1–2 March in the 2020 leap year),
  from 2017-10-01 to 2021-03-03.
- Cause, confirmed in the raw files: all 45 raw `.xlsx` reports with an Excel-date header (Aug 2017 – Apr 2021) have a
  fixed 31-day date header (`=+G7+1` formulas). A 30-day month's file therefore carries an
  extra column for the next month's 1st, and February's carries 1–3 March. Those columns
  are empty and become all-zero rows (`reports/02_raw_header_overflow.csv` lists all 19
  such files). The earlier, CSV-era reports end on the true last day and add nothing.
- **Proposal: drop the all-zero duplicate row** (keep one row per timestamp). This gives
  the same result as the benchmark's `groupby('timestamp').sum()`, and it records the real
  cause instead of adding a phantom row.

### The last day is an overflow artefact
- **2021-05-01 is all zeros** (its system-wide daily total is 0). It is the overflow column of
  the April-2021 file (30 days), and there is no May file to pair it with. The last real
  interval is **2021-04-30 23:45**, which is also the benchmark's hard cutoff in `data.read_data`
  (`df.index <= '2021-04-30 23:45:00'`).
- The file's end date stays recorded as 2021-05-01 23:45, as instructed. **Decision needed:**
  should `ridership_15min.parquet` end at 2021-04-30 23:45?

### (b) Gaps
- A full 15-min grid over the file's span has 201,696 slots; 184,190 unique timestamps are
  present and 17,506 are missing.
- **All 17,506 missing slots are overnight** (hour 0: 332, hour 1: 8,374, hour 2: 8,404,
  hour 3: 396). **0 are in service hours.** 01:15–02:45 never appears in the raw reports
  at all (no validations), and 01:00 appears on only 30 days. No calendar day is missing.
- There are also present-but-zero intervals inside 04:00–22:45 (the preprocessing did
  `fillna(0)`, so these could be closures or gaps). There are 509 of them, excluding 2021-05-01:
  - 480 at the service edges (04:xx/22:xx). Most are Sundays (347) and Mondays (73,
    Colombian holidays fall on Mondays), which fits a later opening on Sundays and holidays.
  - 29 in mid-service, on 7 dates: 2019-11-21 20:45–21:45 and 2019-11-22 19:15–21:45
    (national-strike unrest), 2019-11-23 05:00, 2019-12-31 20:30–21:45 (New Year's Eve),
    2020-12-27 20:00–21:45, 2020-09-21 21:45 and 2021-04-28 21:45 (2021 strike). These
    look like real service suspensions, not missing data. They are listed in
    `reports/02_zero_intervals_in_service.csv`.
- **Proposal:** keep the benchmark's overnight exclusion `[0,1,2,3,23]` for modelling.
  Reindex to a full grid inside the kept hours, which adds 0 rows, so there's nothing to impute.
  Keep the in-service zeros as true zeros, and flag the 7 event dates for later analysis.

### (c) Interval-start convention
- Raw labels are single clock times (`04:00`, `04:15`, …), not ranges. Each day's column
  runs from `00:00` to `23:45` and never includes `24:00`.
- Tue–Thu non-holiday days before Mar 2020 (701 days): the first interval with system total
  ≥100 is labelled **03:45** on 96.7 % of days. The mean profile rises 03:45 = 492 →
  04:00 = 1,760 → 04:15 = 4,858 → 05:00 = 25,748. The AM peak is the **06:30** label
  (06:15–07:00 are all ≈70 k), and the PM peak is the 17:15 label. Evening: 22:00 = 12,673 →
  22:45 = 2,623 → 23:00 = 974 → 23:45 = 86.
- Saturdays: first ≥100 at 04:15 (70 %). Sundays/holidays: first ≥100 at 04:45 (64 %).
- Reading: if the weekday opening is 04:00, a start-of-interval label puts the pre-opening
  trickle in 03:45 and the opening ramp in 04:00–04:15, which is what we see. The day's
  labels run 00:00–23:45 inside one date column, which is also the usual start-of-interval
  layout. **This is consistent with start-of-interval, but counts alone cannot prove it.**
  An end-of-interval reading would shift everything by 15 min and would not contradict any
  single number. The raw reports carry no timezone; local Bogotá time (UTC−5, no DST) is assumed.
- **Proposal:** adopt start-of-interval, as the project rules say, and record the caveat.
- Figure: `figures/02_profile_15min.png`.

**Nothing has been written to `ridership_15min.parquet` yet. This waits for approval of §1b.**

---

## 2. Benchmark graph audit (`03_audit_edges.py`), 2026-09-25

Input: `preprocessing/Edges.csv`, 159 rows. `preprocessing/Adjacency_Matrices.ipynb` reads
it into networkx, with nodes from `numero_estacion` in the old station CSV. The nodes are
integers, sorted by code.

- **Node IDs are station codes without the leading zero.** 274 IDs have 4 digits (e.g.
  `3000` → `"03000"`) and 44 have 5 digits (codes ≥ 10000). After zero-padding, all 148
  distinct nodes are valid codes in the old station CSV.
- Hygiene: 0 self-loops. 2 undirected edges appear twice (`03014–04107`, `09110–14001`,
  the second as `14001,09110`), which leaves 157 unique undirected edges.
- Coverage: **all 147 benchmark stations appear**. The one extra node is `07111`
  Ricaurte - NQS, which is in the GeoJSON but has no ridership column. Its 2 edges
  (`07110–07111`, `07111–07112`) are dropped when the graph is restricted to the 147,
  leaving 155 edges. **0 of the 147 have no edges.** Degrees: 1 → 14 stations
  (terminals, plus 07110 Paloquemao and 07112 Comuneros because of the 07111 cut),
  2 → 109, 3 → 20, 4 → 3 (03014, 09110, 09111), 6 → 1 (04108 El Polo).
- **Connectivity:** with 07111 dropped, the 147-node graph has **2 components (130 + 17)**.
  The 17 are the NQS-Sur/Soacha/Bosa group (07000–07010, 07112, 07113, 07503–07506).
  With 07111 kept as a pass-through, the graph is connected (148 nodes). Contracting it
  into a single edge `07110–07112` would give an edge of 1,814 m.
  DST-TransitNet (`dst_transitnet/data.py`) dropped the 07111 edges, so it trained on the
  disconnected graph.
- **Geometry** (haversine on GeoJSON coordinates, `reports/03_edges_with_distance.csv`):
  kept edges are 257 m min, 645 m median, 980 m p90 and 2,832 m max
  (`09000` Portal Usme – `09001` Molinos, a real long gap). **No edge exceeds 3 km.**
  23 kept edges join different `id_trazado` values; these are transfers and interlines. Worth a manual
  look: 04108 El Polo is linked to 02302 Virrey (877 m), 02303 Calle 85 (573 m) and
  02304 Héroes (500 m), plus 03014, 04107 and 07101. That's degree 6, which looks like
  service links rather than physical track adjacency.
- Figure: `figures/03_edges.png`.

---

## GTFS feed (`data/GTFS/`): recorded limitation, not used yet

Inspected during Phase 0, 2026-09-25:
- It has 9 files: agency 7 rows, calendar 7, calendar_dates 140, fare_attributes 7,
  fare_rules 678, frequencies 2, routes 678, stops 6,556 and trips 101,854 (data rows,
  header excluded). **There is no `stop_times.txt`, `shapes.txt` or `feed_info.txt`.**
- `calendar.txt` runs from 20000101 to 20990101, so it gives **no real validity period**.
- `routes.txt` only has agencies 3–7: URBANO 643, ALIMENTADOR 14, COMPLEMENTARIO 14,
  ESPECIAL 6, LINEACABLE 1. **Agencies 1 (Transmilenio-Troncal) and 2 (Dual) have no routes.**
- `stops.txt` has 6,552 zonal `Z_…` stops, 4 `cable…` stops and 1 `stop…` stop. All have
  `location_type = 0` and none have a `parent_station`. Trunk stations appear only
  by name inside zonal stop names, e.g. `CL 167 - KR 45A Estación Toberin`.
- **Consequence:** this feed can't produce a trunk route, sequence or frequency graph.
  A graph built from GTFS would need the full feed with the trunk routes and `stop_times.txt`.
