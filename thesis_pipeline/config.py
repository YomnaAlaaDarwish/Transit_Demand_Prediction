"""Track definitions, splits and period labels (see DATA_LOG.md section 7).

Every date below is local Bogota time. A 15-min timestamp marks the start of its
interval; a daily timestamp is the calendar day.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
PROCESSED = ROOT / "data/processed"

RIDERSHIP_15MIN = INTERIM / "ridership_15min.parquet"
TIME_FEATURES_15MIN = INTERIM / "time_features_15min.parquet"
STATIONS = INTERIM / "stations.csv"
STATION_ORDER = INTERIM / "station_order.csv"
WEATHER_HOURLY = INTERIM / "weather_hourly.parquet"

# Hours the benchmark drops before any aggregation (data.aggreagtion_func), and
# that the DST-TransitNet reproduction drops from the 15-min series.
BENCHMARK_EXCLUDED_HOURS = (0, 1, 2, 3, 23)

DATA_END = "2021-04-30 23:45"

TRACKS = {
    # Track A -- the jdcaicedo251 benchmark: daily sums over hours 04:00-22:45,
    # 7-day-ahead multi-output forecast. train_date from settings.yaml.
    # lookback: settings.yaml has steps_back=14; 21 is the value specified for
    # this thesis (DATA_LOG 7.1 -- unresolved discrepancy, kept configurable).
    "A": {
        "freq": "D",
        "lookback": 21,
        "horizon": 7,
        "train_end": "2018-08-01",   # settings.yaml train_date; first test day (exclusive train end)
        "test_end": "2021-04-30",    # benchmark read_data cutoff (last target day)
        "val_fraction": 0.1,         # last 10% of training origins, purged
        "weekly_hist": 0,
    },
    # Track B -- the DST-TransitNet reproduction (dst_transitnet/config.py):
    # 15-min series without hours 23:00-03:45, next-step (15-min) forecast,
    # 20 recent steps + 20 steps ending at the same time one week earlier.
    "B": {
        "freq": "15min",
        "lookback": 20,
        "horizon": 1,
        "train_end": "2018-08-01 00:00",
        "test_end": "2021-04-30 23:45",
        "val_fraction": 0.1,         # dst_transitnet TrainConfig.val_fraction
        "weekly_hist": 20,           # dst_transitnet DataConfig.hist_len, offset 7 days
    },
}

# Period labels for the test set. The benchmark code contains no stable/protest/covid
# labelling (checked: this repo and every upstream branch). The definition below is the
# one specified for this thesis; see DATA_LOG 7.1.
PERIODS = {
    "protest": ("2019-11-01", "2020-01-01"),   # Nov-Dec 2019, [start, end)
    "covid": ("2020-03-01", None),             # from March 2020
}

# The DST-TransitNet reproduction's reconstructed test windows (dst_transitnet/config.py),
# kept so Track B results stay comparable with FINAL_MODEL_COMPARISON.md. [start, end).
DST_WINDOWS = {
    "normal": ("2018-08-01", "2019-01-01"),
    "protest": ("2019-11-21", "2019-12-27"),
    "covid": ("2020-03-01", "2020-11-05"),
}

STATIC_FEATURES = ["id_trazado", "tipo_esta", "num_vag", "area_est", "num_acc", "acc_puent"]
