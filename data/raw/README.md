# data/raw — manifest of raw inputs (referenced in place)

No raw file has been moved or copied here. The existing pipeline (`settings.yaml`,
`dst_transitnet/config.py`, `tsb_forecast/`, `apc_tabular/`) reads these paths directly,
so they stay where they are. This file records what each raw input is. See `DATA_LOG.md`
for how the inputs are used.

Hashes are the first 16 hex characters of SHA-256, computed on 2026-09-25 on branch
`thesis-data-pipeline`. "Added in" is the last commit that touched the file.

| Path | Bytes | sha256[:16] | Added in | What it is |
|---|---:|---|---|---|
| `data/transmilenio_transactions.parquet` | 32,536,117 | `fa403139e358b0e4` | a04b620 (2025-07-01) | 15-min validations, wide: `timestamp` (text, `%Y-%m-%d %H:%M:%S`) + 151 `(NNNNN) Name` columns, 2015-08-01 00:00 → 2021-05-01 23:45 |
| `data/transactions/2015…2021/` | 1,479,202,471 (89 files, excluding 2 `~$` Excel lock files) | — | various | Raw monthly TransMilenio "Resumen de Validaciones … 15 min" reports (.csv to mid-2017, .xlsx after); source of the parquet |
| `data/Estaciones_Troncales_de_TRANSMILENIO.geojson` | 105,656 | `9321574633b6c13b` | 432b0cc (2026-09-25) | Current trunk-station layer, 153 point features (num_est, nom_est, latitud, longitud, id_trazado, …) |
| `preprocessing/Estaciones_Troncales_de_TRANSMILENIO.csv` | 35,437 | `b37d1ac86d18bf13` | 2d55ad9 (2022-11-18) | Older station export used by the benchmark's adjacency notebook (numero_estacion as integer) |
| `preprocessing/Edges.csv` | 1,808 | `deff031d1f437772` | e93000d (2022-11-18) | Benchmark's hand-made station graph, 159 rows `node_1,node_2` (integer codes) |
| `data/clean_stations_database_v2.csv` | 135,388 | `14ad8696559adc79` | 3962952 (2022-08-17) | Benchmark's station × access lookup (with lat/lon per access) |
| `data/GTFS/agency.txt` | 718 | `f1c93681affd402f` | 9cdeff7 (2026-09-25) | GTFS feed (zonal/SITP only; see DATA_LOG §GTFS) |
| `data/GTFS/calendar.txt` | 334 | `150331f8adce5633` | 9cdeff7 | 〃 |
| `data/GTFS/calendar_dates.txt` | 1,996 | `df9c31459f764dfc` | 9cdeff7 | 〃 |
| `data/GTFS/fare_attributes.txt` | 231 | `edd649eb5165ce01` | 9cdeff7 | 〃 |
| `data/GTFS/fare_rules.txt` | 9,347 | `7606520b7106bd22` | 9cdeff7 | 〃 |
| `data/GTFS/frequencies.txt` | 100 | `8963134b7be00e67` | 9cdeff7 | 〃 |
| `data/GTFS/routes.txt` | 36,646 | `c0213fed242ce91c` | 9cdeff7 | 〃 |
| `data/GTFS/stops.txt` | 536,862 | `97f5a651539b4c27` | 9cdeff7 | 〃 |
| `data/GTFS/trips.txt` | 4,076,909 | `970030e7c84e6676` | 9cdeff7 | 〃 |

Benchmark result files under `output/day/**` (147 JSON files per model) are used only
to confirm which stations the benchmark modelled. They are not a data source.
