"""03_audit_edges.py -- audit the benchmark's hand-made station graph (read-only; reports only).

preprocessing/Edges.csv (node_1,node_2) is what preprocessing/Adjacency_Matrices.ipynb
feeds to networkx, with node IDs taken from numero_estacion in
preprocessing/Estaciones_Troncales_de_TRANSMILENIO.csv (an integer, so the leading
zero is lost: 3000 == station "03000").

Checks: ID format, coverage of the 147 benchmark stations, isolated stations,
connectivity, duplicates/self-loops, and each edge's straight-line length from the
GeoJSON coordinates in data/interim/stations.csv (flag > 3 km).

Inputs (read-only): preprocessing/Edges.csv, preprocessing/Estaciones_Troncales_de_TRANSMILENIO.csv,
                    data/interim/stations.csv, data/interim/station_order.csv
Outputs:            data/interim/reports/03_*.csv, data/interim/figures/03_edges.png

Run from the repo root:  python scripts/03_audit_edges.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGES = ROOT / "preprocessing/Edges.csv"
GEOJSON = ROOT / "data/Estaciones_Troncales_de_TRANSMILENIO.geojson"
OLD_CSV = ROOT / "preprocessing/Estaciones_Troncales_de_TRANSMILENIO.csv"
STATIONS = ROOT / "data/interim/stations.csv"
ORDER = ROOT / "data/interim/station_order.csv"
REP = ROOT / "data/interim/reports"
FIG = ROOT / "data/interim/figures"
LONG_M = 3000


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def components(nodes, edges):
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v in nodes and v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return sorted(comps, key=len, reverse=True)


def main():
    REP.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(EDGES)
    st = pd.read_csv(STATIONS, dtype={"code": str}).set_index("code")
    # Graph nodes without a ridership column still have GeoJSON coordinates;
    # add them (distance checks only) so their edges can be measured.
    geo = pd.DataFrame([f["properties"] for f in json.load(open(GEOJSON, encoding="utf-8"))["features"]])
    geo["code"] = geo.num_est.astype(str).str.zfill(5)
    geo = geo.set_index("code")
    order = pd.read_csv(ORDER, dtype={"code": str}).code.tolist()
    b147 = set(order)

    # --- ID format -----------------------------------------------------------
    print(f"[ids] {len(raw)} edge rows; dtypes {raw.dtypes.astype(str).to_dict()}; "
          f"node range {raw.min().min()}..{raw.max().max()}")
    e = raw.astype(int).astype(str).apply(lambda s: s.str.zfill(5))
    e.columns = ["a", "b"]
    nodes = set(e.a) | set(e.b)
    old = pd.read_csv(OLD_CSV, encoding="utf-8-sig")
    old_codes = set(old.numero_estacion.astype(int).astype(str).str.zfill(5))
    print(f"[ids] distinct nodes {len(nodes)}; zero-padded to 5 chars: in 147 -> {len(nodes & b147)}, "
          f"in all 151 ridership -> {len(nodes & set(st.index))}, in old station CSV -> {len(nodes & old_codes)}")
    lens = raw.stack().astype(str).str.len().value_counts().to_dict()
    print(f"[ids] raw ID string lengths {lens} (4 = leading zero dropped, 5 = codes >= 10000)")

    # --- hygiene -------------------------------------------------------------
    self_loops = e[e.a == e.b]
    key = e.apply(lambda r: tuple(sorted((r.a, r.b))), axis=1)
    dup = e[key.duplicated(keep=False)]
    print(f"[hygiene] self-loops {len(self_loops)}; duplicate undirected edges {key.duplicated().sum()}"
          + (f": {dup.values.tolist()}" if len(dup) else ""))
    und = sorted(set(key))

    # --- coverage ------------------------------------------------------------
    outside = sorted(nodes - b147)
    for c in outside:
        if c not in st.index and c in geo.index:
            st.loc[c, ["bench_name", "lat", "lon", "id_trazado", "in_benchmark_147"]] = [
                None, geo.at[c, "latitud"], geo.at[c, "longitud"], geo.at[c, "id_trazado"], False]
    print(f"[coverage] nodes not among the 147: {outside}")
    for c in outside:
        nm = st.bench_name.get(c, None)
        oldn = old.loc[old.numero_estacion.astype(int).astype(str).str.zfill(5) == c, "nombre_estacion"]
        print(f"    {c}: ridership name={nm!r}; old-CSV name={oldn.iloc[0] if len(oldn) else None!r}; "
              f"edges={[x for x in und if c in x]}")
    kept = [x for x in und if x[0] in b147 and x[1] in b147]
    dropped = [x for x in und if x not in kept]
    print(f"[coverage] undirected edges {len(und)}: both ends in 147 -> {len(kept)}; dropped -> {len(dropped)}")
    deg = defaultdict(int)
    for a, b in kept:
        deg[a] += 1
        deg[b] += 1
    iso = [c for c in order if deg[c] == 0]
    print(f"[coverage] of the 147, stations with no edges: {len(iso)}")
    for c in iso:
        print(f"    {c} {st.at[c, 'bench_name']} ({st.at[c, 'id_trazado']})")
    dd = pd.Series([deg[c] for c in order]).value_counts().sort_index().to_dict()
    print(f"[coverage] degree distribution over the 147: {dd}")
    for k in sorted(set(deg[c] for c in order)):
        if k != 2:
            print(f"    degree {k}: " + ", ".join(f"{c} {st.at[c, 'bench_name']}" for c in order if deg[c] == k))

    # --- connectivity --------------------------------------------------------
    comps = components(order, kept)
    print(f"[connectivity] components over the 147 (edges within 147 only): {len(comps)}; "
          f"sizes {[len(c) for c in comps]}")
    for c in comps[1:]:
        names = [x + " " + st.at[x, "bench_name"] for x in c]
        print(f"    {names}")
    for c in outside:
        nb = sorted({y for x in und if c in x for y in x} - {c})
        if len(nb) == 2 and all(n in b147 for n in nb):
            d = haversine_m(st.at[nb[0], "lat"], st.at[nb[0], "lon"], st.at[nb[1], "lat"], st.at[nb[1], "lon"])
            print(f"[connectivity] {c} is a pass-through between {nb[0]} and {nb[1]}; a contracted edge "
                  f"{nb[0]}-{nb[1]} would be {d:.0f} m")
    comps_all = components(sorted(b147 | nodes), und)
    print(f"[connectivity] if the non-147 nodes are kept as pass-through: {len(comps_all)} components, "
          f"sizes {[len(c) for c in comps_all]}")

    # --- geometry ------------------------------------------------------------
    rows = []
    for a, b in und:
        la, lb = a in st.index, b in st.index
        d = haversine_m(st.at[a, "lat"], st.at[a, "lon"], st.at[b, "lat"], st.at[b, "lon"]) if la and lb else np.nan
        rows.append({"a": a, "a_name": st.bench_name.get(a), "a_trazado": st.id_trazado.get(a),
                     "b": b, "b_name": st.bench_name.get(b), "b_trazado": st.id_trazado.get(b),
                     "dist_m": round(d, 1) if d == d else np.nan,
                     "both_in_147": (a, b) in kept})
    g = pd.DataFrame(rows)
    g["same_trazado"] = g.a_trazado == g.b_trazado
    g.to_csv(REP / "03_edges_with_distance.csv", index=False)
    x = g[g.both_in_147].dist_m
    print(f"[geometry] edge length over kept edges (m): min {x.min():.0f}, median {x.median():.0f}, "
          f"p90 {x.quantile(.9):.0f}, max {x.max():.0f}")
    lng = g[g.dist_m > LONG_M].sort_values("dist_m", ascending=False)
    print(f"[geometry] edges longer than {LONG_M} m: {len(lng)}")
    if len(lng):
        print(lng[["a", "a_name", "b", "b_name", "a_trazado", "b_trazado", "dist_m"]].to_string(index=False))
    print(f"[geometry] kept edges joining different id_trazado: {(~g[g.both_in_147].same_trazado).sum()}")
    short = g[g.dist_m < 150]
    if len(short):
        print(f"[geometry] edges shorter than 150 m (possible same-site pairs):\n"
              + short[["a", "a_name", "b", "b_name", "dist_m"]].to_string(index=False))

    plot(st, g, iso)


def plot(st, g, iso):
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    for _, r in g.iterrows():
        if r.dist_m != r.dist_m:
            continue
        a, b = st.loc[r.a], st.loc[r.b]
        long_ = r.dist_m > LONG_M
        ax.plot([a.lon, b.lon], [a.lat, b.lat], lw=2 if long_ else 1,
                color="#e34948" if long_ else "#9a9990", zorder=2)
    s = st[st.in_benchmark_147]
    ax.scatter(s.lon, s.lat, s=14, color="#2a78d6", zorder=3, label="benchmark station (147)")
    if iso:
        i = st.loc[iso]
        ax.scatter(i.lon, i.lat, s=40, facecolors="none", edgecolors="#1a1a19", lw=1.2, zorder=4,
                   label=f"no edges ({len(iso)})")
    ax.plot([], [], color="#e34948", lw=2, label=f"edge > {LONG_M / 1000:.0f} km")
    ax.plot([], [], color="#9a9990", lw=1, label="edge")
    ax.set_aspect(1 / 0.9968)
    ax.grid(color="#e6e5df", lw=0.5, zorder=0)
    ax.tick_params(labelsize=8, colors="#6b6a63")
    ax.set_title("preprocessing/Edges.csv drawn on GeoJSON coordinates", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "03_edges.png", facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    main()
