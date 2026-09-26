"""08_validate_benchmark_metrics.py -- recompute MAAPE from the benchmark's saved predictions.

Reads every output/day/<training>/<output>/<model>/*.json (147 stations each), aligns the
predictions with the Track A daily series (data/interim/ridership_15min.parquet summed
over 04:00-22:45) and reports metrics per test period with thesis_pipeline.metrics.

Alignment (derived from run.py + data.WindowGenerator, then checked empirically):
  static: prediction k (k = 0..1002) targets days D0+k .. D0+k+6
  online: prediction k (k = 0..995)  targets days D0+k .. D0+k+6
  with D0 = train_date - 6 days = 2018-07-26. WindowGenerator's test_df starts
  input_width + shift - 1 rows before train_date, so the first label window ends ON
  train_date. The script scores offsets -8..+1 days around 2018-08-01 and reports the
  best one, to confirm this.
The benchmark's own analysis notebook (experiments/result_analysis.ipynb read_target)
aligns prediction k with days 2018-08-01+k.. -- that is also reported ("nb_align").

Periods are labelled by the first target day (thesis_pipeline.periods).

Outputs: data/interim/reports/08_benchmark_metrics.csv, 08_alignment_scan.csv
Run from the repo root:  python scripts/08_validate_benchmark_metrics.py
"""
import glob
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thesis_pipeline import config as C  # noqa: E402
from thesis_pipeline.dataset import load_series  # noqa: E402
from thesis_pipeline.metrics import arctan_ape, evaluate_arrays  # noqa: E402
from thesis_pipeline.periods import period_label  # noqa: E402

REP = C.INTERIM / "reports"
D0 = pd.Timestamp("2018-07-26")
NB = pd.Timestamp("2018-08-01")


def strip_accents(text):  # data.strip_accents
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8").lower()


def main():
    REP.mkdir(parents=True, exist_ok=True)
    s = load_series("A")
    st = pd.read_csv(C.STATIONS, dtype={"code": str}).set_index("code")
    by_name = {strip_accents(f"({c}) {st.at[c, 'bench_name']}"): i for i, c in enumerate(s.stations)}

    def targets(start, K):
        i0 = s.times.get_loc(start)
        return np.stack([s.Y[i0 + k:i0 + k + 7] for k in range(K)])       # (K, 7, S)

    rows, scan = [], []
    for d in sorted(glob.glob(str(ROOT / "output/day/*/*/*"))):
        training, output, model = Path(d).parts[-3:]
        files = glob.glob(d + "/*.json")
        P = [None] * len(s.stations)
        for f in files:
            j = json.load(open(f))
            P[by_name[strip_accents(j["name"])]] = np.asarray(j["prediction"], "float64")
        assert all(p is not None for p in P), f"{d}: station missing"
        P = np.stack(P, axis=-1)                                              # (K, 7, S)
        K = P.shape[0]
        # arima/sarima (static single) have 1004 rows; the extra row sits at the end
        for off in range(-8, 2):
            start = NB + pd.Timedelta(days=off)
            kk = min(K, len(s.times) - s.times.get_loc(start) - 6)
            scan.append({"training": training, "output": output, "model": model, "offset_days": off,
                         "first_target": str(start.date()), "maape_all": float(arctan_ape(targets(start, kk), P[:kk]).mean())})
        for label, start in (("derived", D0), ("nb_align", NB)):
            kk = min(K, len(s.times) - s.times.get_loc(start) - 6)
            origins = pd.date_range(start, periods=kk, freq="D")
            res = evaluate_arrays(targets(start, kk), P[:kk], origins, period_label(origins))
            for _, r in res.iterrows():
                rows.append({"training": training, "output": output, "model": model, "alignment": label,
                             "n_pred": K, **r.to_dict()})
    sc = pd.DataFrame(scan)
    sc.to_csv(REP / "08_alignment_scan.csv", index=False)
    best = sc.loc[sc.groupby(["training", "output", "model"]).maape_all.idxmin()]
    print("[alignment] best first-target day per model (lowest overall MAAPE):")
    print(best[["training", "output", "model", "first_target", "maape_all"]].to_string(index=False))
    out = pd.DataFrame(rows)
    out.to_csv(REP / "08_benchmark_metrics.csv", index=False)
    piv = out[out.alignment == "derived"].pivot_table(index=["training", "output", "model"], columns="period",
                                                      values=["maape", "maape_system_total"])
    print("\n[metrics] derived alignment, MAAPE (benchmark definition) and system-total MAAPE by period:")
    print(piv.round(3).to_string())
    print("\n[metrics] nb_align (result_analysis.ipynb alignment), MAAPE by period:")
    print(out[out.alignment == "nb_align"].pivot_table(index=["training", "output", "model"], columns="period",
                                                       values="maape").round(3).to_string())


if __name__ == "__main__":
    main()
