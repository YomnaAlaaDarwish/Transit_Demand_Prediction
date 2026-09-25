"""05_build_graphs.py -- station adjacency matrices in station_order.csv order.

Two versions (see DATA_LOG.md section 5):
  adj_benchmark        preprocessing/Edges.csv, duplicate edges removed, node 07111
                       (Ricaurte - NQS, no ridership column) dropped with its edges.
                       Same edge set as dst_transitnet/data.py.
  adj_physical_clean   adj_benchmark, plus
                       * Ricaurte bridged through 12003: the benchmark's station
                         lookup (data/clean_stations_database_v2.csv) books every
                         (07111) NQS - RICAURTE access under station_name
                         "(12003) Ricaurte", so 12003 carries 07111's
                         validations. 07111's edges are re-attached to 12003.
                       * El Polo (04108) junction edges filtered to physical
                         corridor neighbours: keep same-id_trazado neighbours and,
                         for each other id_trazado, only the nearest station.

Matrices are symmetric, binary (float32 0/1), zero diagonal, shape 147 x 147.

Inputs (read-only): preprocessing/Edges.csv, data/interim/stations.csv,
                    data/interim/station_order.csv, data/clean_stations_database_v2.csv
Outputs: data/interim/graphs/adj_benchmark.npy, adj_physical_clean.npy,
         edges_benchmark.csv, edges_physical_clean.csv, removed_edges_physical_clean.csv

Run from the repo root:  python scripts/05_build_graphs.py
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGES = ROOT / "preprocessing/Edges.csv"
STATIONS = ROOT / "data/interim/stations.csv"
ORDER = ROOT / "data/interim/station_order.csv"
ACCESS_DB = ROOT / "data/clean_stations_database_v2.csv"
OUT = ROOT / "data/interim/graphs"

RICAURTE_NQS, RICAURTE = "07111", "12003"
EL_POLO = "04108"


def haversine_km(a, b, st):
    la1, lo1, la2, lo2 = map(np.radians, [st.at[a, "lat"], st.at[a, "lon"], st.at[b, "lat"], st.at[b, "lon"]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


def n_components(order, edges):
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, sizes = set(), []
    for n in order:
        if n in seen:
            continue
        stack, k = [n], 0
        seen.add(n)
        while stack:
            u = stack.pop()
            k += 1
            for v in adj[u] - seen:
                seen.add(v)
                stack.append(v)
        sizes.append(k)
    return sorted(sizes, reverse=True)


def to_matrix(order, edges):
    idx = {c: i for i, c in enumerate(order)}
    m = np.zeros((len(order), len(order)), dtype="float32")
    for a, b in edges:
        m[idx[a], idx[b]] = m[idx[b], idx[a]] = 1.0
    assert (m == m.T).all() and np.trace(m) == 0
    return m


def write(name, order, edges, sources, st):
    rows = [{"code_1": a, "code_2": b, "distance_km": round(haversine_km(a, b, st), 3),
             "source": sources[(a, b)]} for a, b in sorted(edges)]
    pd.DataFrame(rows).to_csv(OUT / f"edges_{name}.csv", index=False)
    np.save(OUT / f"adj_{name}.npy", to_matrix(order, edges))
    comps = n_components(order, edges)
    print(f"[{name}] {len(edges)} edges; {len(comps)} connected component(s), sizes {comps}; "
          f"max edge {max(r['distance_km'] for r in rows):.2f} km")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(STATIONS, dtype={"code": str}).set_index("code")
    order = pd.read_csv(ORDER, dtype={"code": str}).code.tolist()
    b147 = set(order)

    raw = pd.read_csv(EDGES).astype(int).astype(str).apply(lambda s: s.str.zfill(5))
    und = sorted({tuple(sorted(e)) for e in raw.itertuples(index=False)})
    print(f"[input] {len(raw)} rows -> {len(und)} unique undirected edges")

    # --- version 1: benchmark ------------------------------------------------
    bench = [e for e in und if e[0] in b147 and e[1] in b147]
    dropped = [e for e in und if e not in bench]
    print(f"[benchmark] dropped (endpoint not among the 147): {dropped}")
    write("benchmark", order, bench, {e: "Edges.csv" for e in bench}, st)

    # --- version 2: physical_clean -------------------------------------------
    acc = pd.read_csv(ACCESS_DB)
    booked = acc.loc[acc.nombreestacion.str.startswith(f"({RICAURTE_NQS})"), "station_name"].unique().tolist()
    assert booked == [f"({RICAURTE}) Ricaurte"], booked
    src = {e: "Edges.csv" for e in bench}
    for a, b in dropped:
        other = b if a == RICAURTE_NQS else a
        assert RICAURTE_NQS in (a, b) and other in b147
        e = tuple(sorted((other, RICAURTE)))
        src[e] = f"Edges.csv {a}-{b} re-attached: {RICAURTE_NQS} Ricaurte-NQS booked as {RICAURTE}"
    print(f"[physical_clean] Ricaurte: accesses of {RICAURTE_NQS} are booked as {booked[0]} -> "
          f"re-attached {len(dropped)} edges to {RICAURTE}")

    tz = st.id_trazado
    polo = [e for e in src if EL_POLO in e]
    nb = {(b if a == EL_POLO else a): e for e, (a, b) in zip(polo, polo)}
    removed = []
    for t in sorted({tz[x] for x in nb}):
        if t == tz[EL_POLO]:
            continue
        group = sorted((x for x in nb if tz[x] == t), key=lambda x: haversine_km(EL_POLO, x, st))
        for x in group[1:]:
            near = group[0]
            removed.append({"code_1": nb[x][0], "code_2": nb[x][1],
                            "distance_km": round(haversine_km(EL_POLO, x, st), 3),
                            "reason": f"El Polo junction: {x} {st.at[x, 'bench_name']} ({t}) is not the nearest "
                                      f"{t} station; nearest is {near} {st.at[near, 'bench_name']} "
                                      f"({haversine_km(EL_POLO, near, st):.3f} km), and {x} is reached from it "
                                      f"along {t}"})
            del src[nb[x]]
    pd.DataFrame(removed).to_csv(OUT / "removed_edges_physical_clean.csv", index=False)
    for r in removed:
        print(f"[physical_clean] removed {r['code_1']}-{r['code_2']} ({r['distance_km']} km): {r['reason']}")
    kept = sorted(x for x in nb if nb[x] in src)
    print(f"[physical_clean] El Polo keeps: " + ", ".join(f"{x} {st.at[x, 'bench_name']} ({tz[x]})" for x in kept))
    write("physical_clean", order, sorted(src), src, st)


if __name__ == "__main__":
    main()
