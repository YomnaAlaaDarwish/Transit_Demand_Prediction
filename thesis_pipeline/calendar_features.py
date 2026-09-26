"""Calendar features. Known in advance, so they may be computed at the target time.

calendar_benchmark reproduces data.py (add_cycles, add_holidays, temporal_variables):
  * cycles from pd.Timestamp.timestamp of the (naive, local) index, i.e. seconds since
    1970-01-01 read as UTC -- exactly what the benchmark does;
  * year = 365.2524 days (sic, benchmark constant), week = 7 days, day = 1 day;
  * day_sin/day_cos only when the aggregation is not daily;
  * holiday = 1 on Sundays OR Colombian public holidays; saturday = 1 on Saturdays.
  Holidays: the benchmark calls holidays_co (environment.yml pins holidays-co==1.0.0).
  holidays-co 1.0.0 and the `holidays` package give the identical 125 dates for
  2015-2021 (verified 2026-09-26); `holidays` is used here. Note: holidays-co 1.1.3
  adds 7 non-official July dates ("Senora del Rosario de Chiquinquira").

calendar_rich (thesis additions, all per calendar day):
  * is_public_holiday   -- public holiday only (not Sundays)
  * long_weekend        -- day is in a run of >= 3 consecutive non-working days
                           (Saturday, Sunday or public holiday)
  * day_before_holiday / day_after_holiday -- working day immediately before / after a
                           public holiday
  * holy_week           -- Palm Sunday .. Easter Sunday (Semana Santa)
  * year_end            -- 20 Dec .. 6 Jan inclusive
"""
import numpy as np
import pandas as pd
from dateutil.easter import easter
import holidays as _holidays

BENCHMARK_COLUMNS_DAY = ["year_sin", "year_cos", "week_sin", "week_cos", "holiday", "saturday"]
BENCHMARK_COLUMNS_SUBDAY = ["year_sin", "year_cos", "week_sin", "week_cos", "day_sin", "day_cos",
                            "holiday", "saturday"]
RICH_COLUMNS = ["is_public_holiday", "long_weekend", "day_before_holiday", "day_after_holiday",
                "holy_week", "year_end"]


def public_holidays(years=range(2015, 2023)) -> set:
    return set(_holidays.Colombia(years=years))


def calendar_benchmark(index: pd.DatetimeIndex, daily: bool) -> pd.DataFrame:
    index = pd.DatetimeIndex(index)
    ts = np.asarray(index.map(pd.Timestamp.timestamp), dtype="float64")
    day = 24 * 60 * 60
    week = day * 7
    year = day * 365.2524
    out = pd.DataFrame(index=index)
    out["year_sin"] = np.sin(ts * (2 * np.pi / year))
    out["year_cos"] = np.cos(ts * (2 * np.pi / year))
    out["week_sin"] = np.sin(ts * (2 * np.pi / week))
    out["week_cos"] = np.cos(ts * (2 * np.pi / week))
    if not daily:
        out["day_sin"] = np.sin(ts * (2 * np.pi / day))
        out["day_cos"] = np.cos(ts * (2 * np.pi / day))
    hol = public_holidays()
    dates = index.normalize()
    out["holiday"] = ((index.weekday == 6) | np.isin(dates.date, list(hol))).astype("int8")
    out["saturday"] = (index.weekday == 5).astype("int8")
    return out.astype({c: "float32" for c in out.columns if c not in ("holiday", "saturday")})


def _daily_rich(days: pd.DatetimeIndex) -> pd.DataFrame:
    hol = public_holidays(range(days.min().year - 1, days.max().year + 2))
    pad = pd.date_range(days.min() - pd.Timedelta(days=10), days.max() + pd.Timedelta(days=10), freq="D")
    is_hol = pd.Series([d.date() in hol for d in pad], index=pad)
    off = is_hol | (pad.weekday >= 5)
    # length of the run of consecutive non-working days each day belongs to
    run_id = (off != off.shift()).cumsum()
    run_len = off.groupby(run_id).transform("size").where(off, 0)
    work = ~off
    r = pd.DataFrame(index=pad)
    r["is_public_holiday"] = is_hol
    r["long_weekend"] = run_len >= 3
    r["day_before_holiday"] = work & is_hol.shift(-1, fill_value=False)
    r["day_after_holiday"] = work & is_hol.shift(1, fill_value=False)
    hw = pd.Series(False, index=pad)
    for y in range(pad.min().year, pad.max().year + 1):
        e = pd.Timestamp(easter(y))
        hw |= (pad >= e - pd.Timedelta(days=7)) & (pad <= e)
    r["holy_week"] = hw
    md = pad.month * 100 + pad.day
    r["year_end"] = (md >= 1220) | (md <= 106)
    return r.loc[days].astype("int8")


def calendar_rich(index: pd.DatetimeIndex) -> pd.DataFrame:
    index = pd.DatetimeIndex(index)
    days = index.normalize()
    d = _daily_rich(pd.DatetimeIndex(days.unique()))
    out = d.reindex(days)
    out.index = index
    return out
