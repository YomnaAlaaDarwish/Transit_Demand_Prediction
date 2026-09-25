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
  interval. For the ridership data this is **inferred from data** (§1b(c)), not documented
  by the operator. Weather has its own explicit interval columns (§6).
- Tables are Parquet, matrices are `.npy`, and small human-readable tables are CSV.
- Only small files are committed: `stations.csv`, `station_order.csv`, figures, scripts
  and docs. Everything else under `data/interim/` (Parquet, `.npy`, edge-list and report
  CSVs) and `data/raw/weather/` can be rebuilt and is not committed. The repo `.gitignore`
  ignores `*.csv`, so the two station CSVs are force-added.

Environment used: Python 3.11.15, pandas 3.0.6, pyarrow 25.0.1, matplotlib 3.11.2,
openpyxl 3.1.5, holidays 0.105, requests-cache + retry-requests (for 06).

## Folder layout

```
data/raw/README.md      manifest of raw inputs (referenced in place, nothing moved)
data/raw/weather/       Open-Meteo responses as downloaded (§6)
data/interim/           clean base tables
  stations.csv, station_order.csv
  ridership_15min.parquet            §4
  time_features_15min.parquet        §4 (calendar facts + service flags per timestamp)
  station_time_features_15min.parquet(later)
  graphs/adj_*.npy, edges_*.csv      §5
  station_weather_cell.csv           §6 (pending download)
  weather_hourly.parquet             §6 (pending download)
  weather_checks/                    §6 (pending download)
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
| 04 | `scripts/04_build_ridership_15min.py` | `ridership_15min.parquet`, `time_features_15min.parquet` | done 2026-09-25 |
| 05 | `scripts/05_build_graphs.py` | `graphs/adj_benchmark.npy`, `graphs/adj_physical_clean.npy`, `graphs/edges_*.csv`, `graphs/removed_edges_physical_clean.csv` | done 2026-09-25 |
| 06 | `scripts/06_download_weather.py` | `data/raw/weather/*`, `station_weather_cell.csv`, `weather_hourly.parquet`, `weather_checks/*` | **written, not run: host blocked** (§6) |

The weather script was requested as `03_download_weather.py`. The number 03 was already
taken by the edge audit, and existing files are not renamed, so it is 06.

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
   - `data/interim/station_order.csv` has 147 rows (`idx, code, bench_name`) **sorted by code**
     (changed 2026-09-25, as decided). The first version used the parquet's column order,
     which is by code except that `07010` Bosa is appended after `14005`.
     **DST-TransitNet also uses sorted code order:** `dst_transitnet/data.py`
     `load_station_series` sorts columns by integer code, and `build_adjacency` follows
     that order. The two orders are identical because every code is 5 digits.
   - `data/interim/figures/stations_by_trazado.png`: 17 `id_trazado` values drawn with
     8 hues × 3 marker shapes, plus the cable stations as open diamonds.

**Checks passed:** codes unique; 147 = benchmark set; 147/147 GeoJSON matches; no duplicate `num_est`.

---

## 1b. Ridership time-grid audit (`02_audit_time_grid.py`), 2026-09-25 (decisions in §4)

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
- **Decision (2026-09-25):** `ridership_15min.parquet` ends at **2021-04-30 23:45**. The
  source file ends at 2021-05-01 23:45, but **2021-05-01 is an empty overflow column from the
  April-2021 raw file** (`…_2021_04 … al 30 Abr 2021 …xlsx`), not a real day of data.

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
- **Decision:** start-of-interval, recorded as **"inferred from data"**. Evidence: (1) the
  labels run 00:00–23:45 within each date, with no 24:00; (2) the weekday pre-opening trickle
  falls in the 03:45 label, and the opening ramp starts at 04:00; (3) the AM peak is at the
  06:30 label. It is not provable from the counts alone.
- Figure: `figures/02_profile_15min.png`.

All proposals in (a)–(c) were approved on 2026-09-25 and applied in §4.

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

## 4. Clean 15-min ridership (`04_build_ridership_15min.py`), 2026-09-25

Decisions applied (approved 2026-09-25):
1. **Duplicates:** keep the data row and drop the all-zero overflow row. 2,188 rows
   dropped. The script asserts that every pair is (data, all zero).
2. **Period:** 2015-08-01 00:00 to **2021-04-30 23:45**. The 86 rows of 2021-05-01 were
   dropped; they are all zero, which is asserted.
3. **Grid:** a full 15-min grid of 201,600 slots (2,100 days × 96). **17,496 missing
   overnight slots were filled with 0.** The script asserts that each one falls in
   00:00–03:45 and that no row is partially missing. (The audit counted 17,506 over the
   file's full span; the 10 slots of 2021-05-01 are no longer in range.)
4. **Total validations preserved:** 3,886,724,821 over the 147 stations, asserted equal
   before and after. Values are whole numbers, stored as `int32`.

`data/interim/ridership_15min.parquet` has 201,600 rows × (`timestamp` + 147 columns
named by 5-char code, in `station_order.csv` order). `timestamp` is naive local time and
marks the start of the interval.

`data/interim/time_features_15min.parquet` has one row per timestamp (201,600 × 15). It
stores facts and flags only:

| column | meaning |
|---|---|
| `timestamp`, `date`, `slot_of_day` (0–95), `hour`, `minute`, `dayofweek` (Mon=0) | calendar |
| `holiday_name`, `is_holiday` | Colombian public holidays (`holidays` 0.105) |
| `day_type` | `weekday` / `saturday` / `sunday_holiday` |
| `system_total` | sum of the 147 stations |
| `service_band` | `overnight` (23:00–03:45), `edge` (04:xx, 22:xx), `core` (05:00–21:45) |
| `in_benchmark_hours` | hour ∉ {0,1,2,3,23}, the benchmark's own filter |
| `is_service_interval` | normal operating interval: always `core`; `edge` only if `system_total > 0` (a zero opening or closing hour means closed by timetable: Sundays, holidays and the 2020 COVID timetable); never `overnight` |
| `system_suspended` | `core` interval with `system_total == 0` |
| `filled_zero` | slot absent from the raw data and filled with 0 by this script |

Counts: `core` 142,800, `edge` 16,800, `overnight` 42,000. There are 480 closed edge
slots. `is_service_interval` = 159,120 and `filled_zero` = 17,496.

**`system_suspended` = 30 slots on 7 dates.** The audit reported 29 because it summed all
151 stations. On 2020-12-27 at 20:15, the only validation in the whole system was 1 on
the cable car, which is not among the 147:

| date | slots | from–to |
|---|---:|---|
| 2019-11-21 | 3 | 20:45–21:45 |
| 2019-11-22 | 11 | 19:15–21:45 |
| 2019-11-23 | 1 | 05:00 |
| 2019-12-31 | 6 | 20:30–21:45 |
| 2020-09-21 | 1 | 21:45 |
| 2020-12-27 | 7 | 20:00–21:45 |
| 2021-04-28 | 1 | 21:45 |

These are kept as real zeros.

Note: a station-level zero that was missing in the raw reports cannot be told apart from a
real zero. The benchmark's preprocessing (`transactions_preprocess.ipynb`) applied
`fillna(0)` before the parquet was written. `filled_zero` therefore marks only whole
timestamps we added.

---

## 5. Station graphs (`05_build_graphs.py`), 2026-09-25

**Ricaurte check.** The only other Ricaurte code in the ridership data is **`12003` Ricaurte**
(GeoJSON "Ricaurte - CL 13", TZ009, 404 m from `07111`). The benchmark's station lookup
`data/clean_stations_database_v2.csv` books **every** access of `(07111) NQS - RICAURTE`
(9 rows) under `station_name = "(12003) Ricaurte"`. So since the benchmark's preprocessing,
**12003's series already contains 07111's validations**, and 07111 has no column of its own.
(One oddity: the Paloquemao row `(01) BATERIA UNO VAGON ORIENTE RICAURTE` is booked under
07110 Paloquemao.) The better bridge is therefore through 12003, not a direct 07110–07112 edge.

All matrices are 147 × 147 in `station_order.csv` order, symmetric, binary `float32`, with a
zero diagonal. Add self-loops in the model code if needed.

| version | edges | components | notes |
|---|---:|---|---|
| `adj_benchmark.npy` | 155 | **2** (130 + 17) | `Edges.csv` with 2 duplicates removed and 07111's 2 edges dropped. **Equal to DST-TransitNet's `build_adjacency` minus its self-loops (verified).** |
| `adj_physical_clean.npy` | 155 | **1** (147) | Ricaurte re-attached: `07110–12003` (0.446 km) and `12003–07112` (1.426 km), replacing `07110–07111` and `07111–07112`. El Polo filtered: 2 edges removed. |

El Polo rule. For `04108` (TZ005, at the junction of Calle 80, Autonorte, Suba and NQS),
keep neighbours on its own `id_trazado`, and for each other `id_trazado` keep only the
nearest station. The farther ones are reached along their own corridor from the nearest one.
- Kept: `04107` Escuela Militar (TZ005, 0.851 km), `02304` Héroes (TZ002, 0.500 km),
  `03014` San Martín (TZ003, 0.761 km), `07101` Castellana (TZ008, 0.606 km).
- **Removed** `04108–02303` Calle 85 (0.573 km). Reason: TZ002, but Héroes is the nearest
  TZ002 station, and Héroes–Calle 85 is an existing Autonorte edge.
- **Removed** `04108–02302` Virrey (0.877 km). Reason: TZ002, and it skips both Héroes and Calle 85.

Files: `graphs/edges_benchmark.csv` and `graphs/edges_physical_clean.csv` (`code_1, code_2,
distance_km, source`; `source` records re-attached edges), and
`graphs/removed_edges_physical_clean.csv`. Max edge length is 2.83 km in both
(Portal Usme–Molinos).

---

## 6. Hourly weather (`06_download_weather.py`): written, NOT yet downloaded

**Status 2026-09-25:** this session's network policy **blocks
`archive-api.open-meteo.com`** (the proxy answers 403 to CONNECT). No weather data has been
downloaded, and none of the outputs below exist yet. The script was tested end to end
(probe → download → process → checks) against a **synthetic** stand-in for the API, which
returns responses in the same JSON shape. It has not been tested against the real API.

Design:
- **Source:** Open-Meteo Historical Weather API, `https://archive-api.open-meteo.com/v1/archive`,
  free, no key. Two models are downloaded separately: `era5` (0.25°) and `era5_land` (0.1°).
- **Period:** 2015-07-01 to 2021-06-30 local time, with a buffer around the ridership
  period (2015-08-01 to 2021-04-30).
- **Hourly variables:** `precipitation, rain, temperature_2m, relative_humidity_2m,
  cloud_cover, wind_speed_10m`. Units are mm, °C, %, % and km/h. The response's units are
  asserted for precipitation and temperature.
- **Request parameters** (all calls): `timezone=America/Bogota` (`utc_offset_seconds = −18000`
  asserted), `cell_selection=nearest`, `elevation=nan` and `format=json`.
  With `elevation=nan` there is no lapse-rate downscaling to the station's own height, so
  the returned elevation is the grid cell's mean height and one cell gives one series.
- **Grid cells:**
  1. A 1-day "probe" request per station (50 locations per call) records the cell's
     lat/lon/elevation for each model.
  2. The cells are de-duplicated.
  3. Each unique cell is downloaded **once per model** at its own coordinates. The script
     asserts that the API returns the same cell.
  4. `station_weather_cell.csv` covers all 151 stations with coordinates; the counts
     reported are for the 147.
- **HTTP:** `requests-cache` (SQLite, no expiry, in `data/raw/weather/http_cache.sqlite`)
  and `retry-requests` (5 retries, backoff). This is the stack the `openmeteo-requests`
  client wraps. JSON is used instead of that client's FlatBuffers so the raw responses in
  `data/raw/weather/` stay exactly as downloaded and readable.
- **Time convention:** Open-Meteo labels each hour with a time T in local time.
  - `precipitation` and `rain` are the **sum over the preceding hour**. The value labelled
    07:00 covers 06:00–07:00, so it gets `interval_start = 06:00` and `interval_end = 07:00`.
  - Temperature, humidity, cloud cover and wind are **instantaneous at T** (`obs_time = T`).
  - `weather_hourly.parquet` stores `obs_time`, `interval_start = T − 1h` and
    `interval_end = T` on every row. Use `interval_*` for precipitation and rain, and
    `obs_time` for the other variables.
  - The first row, labelled 2015-07-01 00:00, therefore covers 2015-06-30 23:00–24:00.
- **Checks** (written to `weather_checks/`):
  - every expected hour is present for each cell, with null counts per variable;
  - mean monthly precipitation (the rainy seasons should peak in Apr–May and Oct–Nov);
  - mean precipitation by hour of `interval_start` (an afternoon peak is expected);
  - era5 vs era5_land daily precipitation: Pearson and Spearman correlation for the
    station-weighted area mean, plus the range of per-station correlations.
- **Expected caveat, to be confirmed on real data:** ERA5-Land has no cloud-cover field, so
  `cloud_cover` may be null for `era5_land`. The null counts in the gap check will show this.

**To finish:** allow `archive-api.open-meteo.com` in the environment's network settings,
then run `python scripts/06_download_weather.py`. The cell counts, download date and check
results will be filled in here.

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
