"""02_audit_time_grid.py -- audit the 15-min ridership time grid (read-only; writes reports only).

Does NOT write ridership_15min.parquet. It produces the evidence needed to decide
how to clean the grid:
  a) duplicate timestamps: identical vs different rows, where they occur, and the
     raw-file header evidence for where they come from;
  b) missing intervals and all-zero intervals: overnight/non-service vs in service;
  c) interval-start convention: first/last nonzero interval per day type and the
     hourly profile.

Inputs (read-only): data/transmilenio_transactions.parquet, data/interim/station_order.csv,
                    raw monthly .xlsx headers under data/transactions/ (header row only)
Outputs:            data/interim/reports/02_*.csv, data/interim/figures/02_profile_15min.png

Run from the repo root:  python scripts/02_audit_time_grid.py
"""
import glob
import re
from pathlib import Path

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RIDERSHIP = ROOT / "data/transmilenio_transactions.parquet"
RAW = ROOT / "data/transactions"
REP = ROOT / "data/interim/reports"
FIG = ROOT / "data/interim/figures"
NIGHT_HOURS = [0, 1, 2, 3, 23]  # the benchmark's own drop list (data.aggreagtion_func)


def load():
    df = pd.read_parquet(RIDERSHIP)
    df["ts"] = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
    stations = [c for c in df.columns if c.startswith("(")]
    df["total"] = df[stations].sum(axis=1)
    return df, stations


def audit_duplicates(df, stations):
    d = df[df.ts.duplicated(keep=False)]
    rows = []
    for ts, g in d.groupby("ts"):
        tot = g.total.tolist()
        if (g[stations].nunique() == 1).all():
            kind = "identical_all_zero" if tot[0] == 0 else "identical_nonzero"
        elif min(tot) == 0:
            kind = "one_row_all_zero"
        else:
            kind = "both_nonzero_different"
        rows.append({"ts": ts, "n_rows": len(g), "kind": kind, "row_totals": tot,
                     "sum_equals_max": sum(tot) == max(tot)})
    r = pd.DataFrame(rows)
    r["date"] = r.ts.dt.date
    by_day = r.groupby("date").agg(n_ts=("ts", "size"),
                                   kinds=("kind", lambda s: s.value_counts().to_dict()))
    by_day["day_of_month"] = [x.day for x in by_day.index]
    r.to_csv(REP / "02_duplicate_timestamps.csv", index=False)
    by_day.to_csv(REP / "02_duplicate_days.csv")
    print(f"\n[a] duplicate timestamps: {len(r)} (rows involved: {len(d)}); group sizes "
          f"{r.n_rows.value_counts().to_dict()}")
    print(f"[a] kinds: {r.kind.value_counts().to_dict()}; sum==drop-the-zero-row in "
          f"{r.sum_equals_max.sum()}/{len(r)} groups")
    print(f"[a] {len(by_day)} affected dates, day-of-month {sorted(by_day.day_of_month.unique())}:")
    print("    " + ", ".join(str(x) for x in by_day.index))
    return r


def raw_header_overflow():
    """For each raw .xlsx, list date columns in the header that fall outside the file's month."""
    rows = []
    for f in sorted(glob.glob(str(RAW / "*/*.xlsx"))):
        if Path(f).name.startswith("~$"):  # Excel lock files, not data
            continue
        rel = str(Path(f).relative_to(ROOT))
        try:
            wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        except Exception as e:  # e.g. not a real xlsx container
            rows.append({"file": rel, "month": None, "n_date_cols": 0, "outside_month": [],
                         "error": type(e).__name__})
            continue
        sheet = "Validaciones Tullave" if "Validaciones Tullave" in wb.sheetnames else wb.sheetnames[0]
        dates = []
        for header in wb[sheet].iter_rows(min_row=1, max_row=12, values_only=True):
            dates = [pd.Timestamp(x) for x in header if hasattr(x, "year")]
            if len(dates) >= 20:  # the date header row
                break
        wb.close()
        if len(dates) < 20:
            rows.append({"file": rel, "month": None, "n_date_cols": 0, "outside_month": [],
                         "error": "no Excel-date header (text-dated 2016 to Jul-2017 layout)"})
            continue
        month = pd.Series([(x.year, x.month) for x in dates]).mode()[0]
        extra = [x.date() for x in dates if (x.year, x.month) != month]
        rows.append({"file": rel, "month": f"{month[0]}-{month[1]:02d}",
                     "n_date_cols": len(dates), "outside_month": extra, "error": None})
    r = pd.DataFrame(rows)
    r.to_csv(REP / "02_raw_header_overflow.csv", index=False)
    ov = r[r.outside_month.str.len() > 0]
    bad = r[r.error.notna()]
    print(f"[a] raw .xlsx files: {len(r)}; without an Excel-date header: {len(bad)} "
          f"{bad.error.value_counts().to_dict()}; with date columns outside their month: {len(ov)}")
    for _, x in bad.iterrows():
        print(f"    no date header: {x.file}")
    for _, x in ov.iterrows():
        print(f"    {x.month}: {[str(d) for d in x.outside_month]}")
    return r


def audit_gaps(df):
    w = df.groupby("ts").total.sum()
    full = pd.date_range(w.index.min(), w.index.max(), freq="15min")
    miss = pd.Series(full.difference(w.index))
    miss_night = miss.dt.hour.isin(NIGHT_HOURS)
    print(f"\n[b] full 15-min grid {full[0]} .. {full[-1]}: {len(full)} slots; present {len(w)}; "
          f"missing {len(miss)}")
    print(f"[b] missing by hour: {miss.dt.hour.value_counts().sort_index().to_dict()}")
    print(f"[b] missing overnight (hours {NIGHT_HOURS}): {miss_night.sum()}; during service hours: "
          f"{(~miss_night).sum()}")
    pd.DataFrame({"ts": miss, "overnight": miss_night}).to_csv(REP / "02_missing_intervals.csv", index=False)

    # Present-but-zero intervals in service hours: the raw->wide step used fillna(0),
    # so a system-wide zero may hide a non-service slot or an outage.
    day = w.groupby(w.index.date).sum()
    zero_days = day[day == 0]
    print(f"[b] days whose system total is 0: {[str(x) for x in zero_days.index]}")
    ws = w[~w.index.hour.isin(NIGHT_HOURS) & ~pd.Index(w.index.date).isin(zero_days.index)]
    z = ws[ws == 0]
    zz = pd.DataFrame({"ts": z.index})
    zz["date"] = zz.ts.dt.date
    zz["hhmm"] = zz.ts.dt.strftime("%H:%M")
    zz["dow"] = zz.ts.dt.day_name()
    edge = zz.hhmm.isin(["04:00", "04:15", "04:30", "04:45", "22:00", "22:15", "22:30", "22:45"])
    zz["class"] = edge.map({True: "service_edge (04:xx / 22:xx)", False: "mid_service"})
    zz.to_csv(REP / "02_zero_intervals_in_service.csv", index=False)
    print(f"[b] all-zero intervals inside 04:00-22:45 (excluding zero days): {len(zz)}; "
          f"{zz['class'].value_counts().to_dict()}")
    print(f"[b]   edge zeros by weekday: {zz[edge].dow.value_counts().to_dict()}")
    mid = zz[~edge].groupby("date").hhmm.agg(["size", "min", "max"])
    print("[b]   mid-service zero intervals by date:\n" + mid.to_string())
    return w


def audit_convention(w):
    co = holidays.Colombia(years=range(2015, 2022))
    s = w[(w.index < "2020-03-01")]  # pre-COVID, stable timetable
    d = pd.DataFrame({"total": s})
    d["date"] = d.index.date
    d["dow"] = d.index.dayofweek
    d["hol"] = [x in co for x in d.date]
    d["type"] = "other"
    d.loc[d.dow.isin([1, 2, 3]) & ~d.hol, "type"] = "Tue-Thu"
    d.loc[(d.dow == 5) & ~d.hol, "type"] = "Saturday"
    d.loc[(d.dow == 6) | d.hol, "type"] = "Sunday/holiday"

    out = []
    for thr in (1, 100):
        nz = d[d.total >= thr]
        fl = nz.groupby(["type", "date"]).apply(lambda g: pd.Series({
            "first": g.index.min().strftime("%H:%M"), "last": g.index.max().strftime("%H:%M")}))
        for t in ["Tue-Thu", "Saturday", "Sunday/holiday"]:
            x = fl.loc[t]
            out.append({"threshold": thr, "day_type": t, "n_days": len(x),
                        "first_nonzero_mode": x["first"].mode()[0],
                        "first_share_at_mode": round((x["first"] == x["first"].mode()[0]).mean(), 3),
                        "last_nonzero_mode": x["last"].mode()[0],
                        "last_share_at_mode": round((x["last"] == x["last"].mode()[0]).mean(), 3)})
    fl = pd.DataFrame(out)
    fl.to_csv(REP / "02_first_last_nonzero.csv", index=False)
    print("\n[c] first/last interval with system total >= threshold (pre-2020-03, by day type):")
    print(fl.to_string(index=False))

    prof = d[d.type == "Tue-Thu"].groupby(d[d.type == "Tue-Thu"].index.strftime("%H:%M")).total.mean()
    prof.to_csv(REP / "02_profile_tue_thu.csv", header=["mean_validations"])
    am = prof[prof.index < "12:00"]
    pm = prof[prof.index >= "12:00"]
    print(f"[c] Tue-Thu mean profile: AM peak {am.idxmax()} ({am.max():.0f}), PM peak {pm.idxmax()} "
          f"({pm.max():.0f})")
    print("[c] early-morning / late-evening means:\n    " + ", ".join(
        f"{k}={v:.0f}" for k, v in prof.items() if k <= "05:00" or k >= "22:00"))

    grid = pd.date_range("2000-01-01", periods=96, freq="15min").strftime("%H:%M")
    prof = prof.reindex(grid)  # labels never reported (01:15-02:45) plot as gaps
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
    ax.bar(range(len(prof)), prof.fillna(0).values, width=0.8, color="#2a78d6")
    ticks = [i for i, k in enumerate(prof.index) if k.endswith(":00")]
    ax.set_xticks(ticks, [prof.index[i][:2] for i in ticks], fontsize=8, color="#6b6a63")
    ax.tick_params(axis="y", labelsize=8, colors="#6b6a63")
    ax.grid(axis="y", color="#e6e5df", linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("interval label (hour)", fontsize=9, color="#6b6a63")
    ax.set_title("Mean system validations per 15-min label, Tue-Thu non-holiday, Aug 2015 - Feb 2020",
                 fontsize=10, loc="left", color="#1a1a19")
    fig.tight_layout()
    fig.savefig(FIG / "02_profile_15min.png", facecolor="white", bbox_inches="tight")
    return fl, prof


def main():
    REP.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df, stations = load()
    print(f"[load] {len(df)} rows x {len(stations)} stations; {df.ts.min()} .. {df.ts.max()}")
    audit_duplicates(df, stations)
    raw_header_overflow()
    w = audit_gaps(df)
    audit_convention(w)


if __name__ == "__main__":
    main()
