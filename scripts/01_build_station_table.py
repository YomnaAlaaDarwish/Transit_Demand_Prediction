"""01_build_station_table.py -- build the station master table and station order.

Inputs (read-only, never modified):
  data/transmilenio_transactions.parquet            15-min ridership, wide (timestamp + 151 station columns)
  data/Estaciones_Troncales_de_TRANSMILENIO.geojson current trunk-station layer (153 features)
  preprocessing/Estaciones_Troncales_de_TRANSMILENIO.csv  older station export (fallback match on numero_estacion)
  data/clean_stations_database_v2.csv               benchmark's station x access table (coords only, exact code)
  output/day/static/multioutput/dense/*.json        benchmark result files (one per station it modelled)

Outputs:
  data/interim/stations.csv          one row per ridership station (151), code as 5-char string
  data/interim/station_order.csv     the 147 benchmark stations in ridership column order
  data/interim/figures/stations_by_trazado.png
  data/interim/reports/01_*.csv      crosstab + match reports

Run from the repo root:  python scripts/01_build_station_table.py
"""
import json
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RIDERSHIP = ROOT / "data/transmilenio_transactions.parquet"
GEOJSON = ROOT / "data/Estaciones_Troncales_de_TRANSMILENIO.geojson"
OLD_CSV = ROOT / "preprocessing/Estaciones_Troncales_de_TRANSMILENIO.csv"
ACCESS_DB = ROOT / "data/clean_stations_database_v2.csv"
BENCH_OUT = ROOT / "output/day/static/multioutput/dense"

OUT = ROOT / "data/interim"
FIG = OUT / "figures"
REP = OUT / "reports"

CABLE_CODES = {"40000", "40001", "40002", "40003"}
GEO_FIELDS = ["id_trazado", "tipo_esta", "num_vag", "area_est", "long_est",
              "ancho_est", "num_acc", "acc_puent", "esta_oper"]
COL_RE = re.compile(r"^\((\d{5})\)\s*(.+)$")


def repair_mojibake(s: str) -> str:
    """Undo UTF-8-read-as-Latin-1 damage ("ToberÃ\xadn" -> "Toberín"); leave clean text alone."""
    if not any(m in s for m in ("Ã", "Â")):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def strip_accents(text: str) -> str:
    """Identical to the benchmark's data.strip_accents (used to name its output files)."""
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return text.lower()


def code5(x) -> str:
    """Station code as a 5-character zero-padded string."""
    return str(int(float(x))).zfill(5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)

    # --- 1. ridership columns -> code, bench_name ---------------------------
    cols = list(pd.read_parquet(RIDERSHIP).columns)
    assert cols[0] == "timestamp", cols[0]
    rows, repaired = [], 0
    for c in cols[1:]:
        fixed = repair_mojibake(c)
        repaired += fixed != c
        m = COL_RE.match(fixed)
        assert m, f"unparseable station column: {c!r}"
        rows.append({"code": m.group(1), "bench_name": m.group(2).strip(), "ridership_column": c})
    st = pd.DataFrame(rows)
    assert st.code.is_unique
    print(f"[ridership] {len(st)} station columns, {repaired} names repaired for encoding")

    # --- benchmark's own 147 (from its result files) ------------------------
    bench_files = {p.stem for p in BENCH_OUT.glob("*.json")}
    bench_codes = {n[1:6] for n in bench_files}
    st["in_benchmark_147"] = ~st.code.isin(CABLE_CODES)
    ours = set(st.loc[st.in_benchmark_147, "code"])
    name_hits = sum(strip_accents(f"({c}) {n}") in bench_files
                    for c, n in zip(st.code, st.bench_name) if c in ours)
    print(f"[benchmark] result files: {len(bench_files)}; codes identical to ours: {bench_codes == ours}; "
          f"names identical after strip_accents: {name_hits}/{len(ours)}")
    assert bench_codes == ours and len(ours) == 147

    # --- 2. GeoJSON ----------------------------------------------------------
    feats = json.load(open(GEOJSON, encoding="utf-8"))["features"]
    geo = pd.DataFrame([f["properties"] for f in feats])
    raw_len = geo.num_est.astype(str).str.len().value_counts().to_dict()
    geo["num_est"] = geo.num_est.map(code5)
    dups = geo[geo.num_est.duplicated(keep=False)]
    print(f"[geojson] {len(geo)} features; raw num_est lengths {raw_len} (zero-padded to 5); "
          f"duplicate codes: {len(dups)}")
    if len(dups):
        dups.to_csv(REP / "01_geojson_duplicate_codes.csv", index=False)
    geo = geo.drop_duplicates("num_est")

    # --- 3. match on code == num_est, fallback to the older station CSV -----
    g = geo.rename(columns={"num_est": "code", "nom_est": "current_name",
                            "latitud": "lat", "longitud": "lon"})
    st = st.merge(g[["code", "current_name", "lat", "lon"] + GEO_FIELDS], on="code", how="left")
    st["match_status"] = st.lat.notna().map({True: "geojson", False: "unmatched"})

    old = pd.read_csv(OLD_CSV, encoding="utf-8-sig")
    old["code"] = old.numero_estacion.map(code5)
    old = old.drop_duplicates("code").set_index("code")
    need = st.match_status == "unmatched"
    fb = st.loc[need, "code"].isin(old.index)
    for i in st.index[need][fb.values]:
        r = old.loc[st.at[i, "code"]]
        st.loc[i, ["current_name", "lat", "lon"]] = [r.nombre_estacion, r.latitud_estacion, r.longitud_estacion]
        st.at[i, "match_status"] = "preprocessing_csv"

    # Coordinates only (no GeoJSON attributes) from the benchmark's own access
    # table, exact code match on its "(NNNNN) name" field -- not a name guess.
    acc = pd.read_csv(ACCESS_DB)
    acc["code"] = acc.station_name.str.slice(1, 6)
    acc = acc.groupby("code")[["latitude", "longitude"]].agg(["min", "max"])
    need = st.match_status == "unmatched"
    for i in st.index[need]:
        c = st.at[i, "code"]
        if c in acc.index:
            la, lo = acc.loc[c, "latitude"], acc.loc[c, "longitude"]
            assert la["min"] == la["max"] and lo["min"] == lo["max"], f"inconsistent coords for {c}"
            st.loc[i, ["lat", "lon"]] = [la["min"], lo["min"]]
            st.at[i, "match_status"] = "coords_only_access_db"

    counts = st.match_status.value_counts().to_dict()
    print(f"[match] all 151: {counts}")
    print(f"[match] benchmark 147: {st[st.in_benchmark_147].match_status.value_counts().to_dict()}")
    unmatched_geo = st[st.match_status != "geojson"][["code", "bench_name", "match_status"]]
    print("[match] not in GeoJSON:\n" + unmatched_geo.to_string(index=False))
    extra = geo[~geo.num_est.isin(st.code)][["num_est", "nom_est", "id_trazado"]]
    extra.to_csv(REP / "01_geojson_not_in_ridership.csv", index=False)
    print(f"[match] GeoJSON features with no ridership column: {len(extra)}\n" + extra.to_string(index=False))

    # --- 4. corridor check: code prefix vs id_trazado ------------------------
    m = st[st.match_status == "geojson"].copy()
    m["prefix"] = m.code.str[:2]
    ct = pd.crosstab(m.prefix, m.id_trazado)
    ct.to_csv(REP / "01_prefix_vs_trazado.csv")
    per = (ct > 0).sum(axis=1)
    print("[corridor] prefix -> id_trazado:")
    for p, n in per.items():
        tz = {k: int(v) for k, v in ct.loc[p][ct.loc[p] > 0].items()}
        print(f"   {p}: {'ONE' if n == 1 else 'MULTIPLE'} {tz}")
    rev = (ct > 0).sum(axis=0)
    print(f"[corridor] id_trazado values spanning >1 prefix: {rev[rev > 1].to_dict()}")

    # --- 5. write tables -----------------------------------------------------
    out_cols = ["code", "bench_name", "current_name", "lat", "lon"] + GEO_FIELDS + \
               ["match_status", "in_benchmark_147"]
    for c in ["tipo_esta", "num_vag", "num_acc", "acc_puent", "esta_oper"]:
        st[c] = st[c].astype("Int64")  # integers, nullable for the 4 cable rows
    st[out_cols].to_csv(OUT / "stations.csv", index=False)
    order = st.loc[st.in_benchmark_147, ["code", "bench_name"]].reset_index(drop=True)
    order.insert(0, "idx", range(len(order)))
    order.to_csv(OUT / "station_order.csv", index=False)
    print(f"[write] stations.csv ({len(st)} rows), station_order.csv ({len(order)} rows)")

    # --- 6. map --------------------------------------------------------------
    plot_map(st)


def plot_map(st):
    # 17 corridors > 8 categorical hues: composite encoding (8 hues x 3 marker
    # shapes), fixed order by id_trazado, never cycled per-hue alone.
    hues = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    shapes = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(9, 11), dpi=150)
    ax.set_facecolor("white")
    tz = sorted(st.id_trazado.dropna().unique())
    for k, t in enumerate(tz):
        d = st[st.id_trazado == t]
        ax.scatter(d.lon, d.lat, s=34, c=hues[k % 8], marker=shapes[k // 8],
                   edgecolors="white", linewidths=0.8, label=f"{t} ({len(d)})", zorder=3)
        ax.annotate(t, (d.lon.median(), d.lat.median()), fontsize=7, color="#3d3d3a",
                    xytext=(6, 6), textcoords="offset points", zorder=4)
    d = st[st.id_trazado.isna() & st.lat.notna()]
    if len(d):
        ax.scatter(d.lon, d.lat, s=40, facecolors="none", edgecolors="#3d3d3a", marker="D",
                   linewidths=1.2, label=f"cable, no GeoJSON ({len(d)})", zorder=3)
    ax.set_aspect(1 / 0.9973)  # ~cos(4.6 deg): keep metric proportions at Bogota's latitude
    ax.grid(color="#e6e5df", linewidth=0.5, zorder=0)
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    ax.tick_params(colors="#6b6a63", labelsize=8)
    ax.set_xlabel("longitude", color="#6b6a63", fontsize=9)
    ax.set_ylabel("latitude", color="#6b6a63", fontsize=9)
    ax.set_title("TransMilenio stations in the ridership data, coloured by GeoJSON id_trazado",
                 fontsize=11, color="#1a1a19", loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), title="id_trazado (n)",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stations_by_trazado.png", facecolor="white", bbox_inches="tight")
    print(f"[write] {FIG / 'stations_by_trazado.png'}")


if __name__ == "__main__":
    main()
