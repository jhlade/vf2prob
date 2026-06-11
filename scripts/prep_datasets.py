# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
"""
prep_datasets.py
----------------
Download a few SNAP graphs, turn them into property graphs with labels + edge probabilities,
and emit GraphML + S/M/L query lists (JSON) reproducibly.

Datasets:
- facebook_combined (undirected)
- email-Enron (directed; we undirect + take largest connected component)
- ca-HepTh (collaboration; undirected)

Edge probabilities:
- p = pmin + (1 - pmin) * clip(sigmoid(alpha * (Jaccard - 0.5))), alpha=6
Node labels:
- degree-quantile bins (5 bins) as categorical strings: "0".."4"
Edge labels:
- simple categorical: ((deg(u)+deg(v)) % 3) as "0","1","2"

Queries:
- S: |V_Q| ~ U[5,8],  n=16
- M: |V_Q| ~ U[9,12], n=16
- L: |V_Q| ~ U[13,20], n=12
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

SNAP = {
    "facebook": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "directed": False,
        "outfile": "facebook_combined.graphml",
    },
    "enron": {
        "url": "https://snap.stanford.edu/data/email-Enron.txt.gz",
        "directed": True,
        "outfile": "email_enron.graphml",
    },
    "hep-th": {
        "url": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
        "directed": False,
        "outfile": "ca_hepth.graphml",
    },
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path) -> None:
    import urllib.request
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def _read_snap_edgelist_gz(path: Path, directed: bool) -> nx.Graph:
    openf = gzip.open if path.suffix == ".gz" else open
    with openf(path, "rt", encoding="utf-8") as f:
        edges = []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            a, b = s.split()
            edges.append((int(a), int(b)))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edges)
    if directed:
        # undirect and keep largest weakly connected component
        UG = G.to_undirected(as_view=False)
    else:
        UG = G
    if not nx.is_connected(UG):
        # take LCC
        comp = max(nx.connected_components(UG), key=len)
        UG = UG.subgraph(comp).copy()
    # relabel nodes to consecutive ints for stability
    mapping = {n: i for i, n in enumerate(sorted(UG.nodes()))}
    UG = nx.relabel_nodes(UG, mapping, copy=True)
    return UG


def _assign_node_labels(G: nx.Graph, k: int = 5) -> None:
    degs = [G.degree[n] for n in G.nodes()]
    if not degs:
        return
    qs = [0, 20, 40, 60, 80, 100]
    pivots = []
    vals = sorted(degs)

    def pct(vs, p):
        if not vs: return 0
        idx = (len(vs) - 1) * (p / 100.0)
        import math
        f, c = math.floor(idx), math.ceil(idx)
        if f == c: return vs[int(idx)]
        return vs[f] * (c - idx) + vs[c] * (idx - f)

    for q in qs:
        pivots.append(pct(vals, q))
    # bins by thresholds pivots[1..-2]
    thr = [pivots[1], pivots[2], pivots[3], pivots[4]]
    for n in G.nodes():
        d = G.degree[n]
        b = 0
        while b < len(thr) and d > thr[b]:
            b += 1
        G.nodes[n]["label"] = str(b)


def _assign_edge_labels_and_probs(G: nx.Graph, *, pmin: float = 0.7, alpha: float = 6.0) -> None:
    # Precompute neighbor sets
    N = {u: set(G.neighbors(u)) for u in G.nodes()}

    def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

    for u, v in G.edges():
        # edge label
        elab = (G.degree[u] + G.degree[v]) % 3
        G.edges[u, v]["label"] = str(elab)
        # Jaccard over neighborhoods (excluding each other)
        Nu = N[u] - {v}
        Nv = N[v] - {u}
        inter = len(Nu & Nv)
        union = len(Nu | Nv) if Nu or Nv else 1
        j = inter / union
        base = sigmoid(alpha * (j - 0.5))  # ~0..1 centered at 0.5
        p = pmin + (1.0 - pmin) * base
        G.edges[u, v]["prob"] = float(max(0.0, min(1.0, p)))


def _sample_connected_queries(G: nx.Graph, size_min: int, size_max: int, nqueries: int, *, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    out = []
    for qi in range(nqueries):
        k = rng.randint(size_min, size_max)
        # BFS growth
        for _ in range(64):
            start = rng.choice(nodes)
            seen = {start}
            frontier = [start]
            while len(seen) < k and frontier:
                u = frontier.pop(0)
                for w in G.neighbors(u):
                    if w not in seen:
                        seen.add(w);
                        frontier.append(w)
                    if len(seen) >= k: break
            if len(seen) < k:
                continue
            H = G.subgraph(seen).copy()
            if nx.is_connected(H):
                out.append({"id": qi, "nodes": sorted(seen)})
                break
        else:
            # fallback: random sample (may be disconnected on rare cases)
            out.append({"id": qi, "nodes": sorted(rng.sample(nodes, min(k, len(nodes))))})
    return out


def _write_queries_json(path: Path, *, regime: str, size_range: Tuple[int, int], nqueries: int, seed: int,
                        queries: List[Dict]) -> None:
    payload = {
        "meta": {"regime": regime, "size_range": list(size_range), "nqueries": nqueries, "seed": seed},
        "queries": queries,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-data-dir", type=str, default="data/snap")
    ap.add_argument("--out-queries-dir", type=str, default="queries/snap")
    ap.add_argument("--datasets", type=str, default="facebook,enron,hep-th",
                    help="comma separated among: facebook,enron,hep-th")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pmin", type=float, default=0.7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.out_data_dir);
    _ensure_dir(data_dir)
    q_dir = Path(args.out_queries_dir);
    _ensure_dir(q_dir)
    dl_dir = Path(".tmp/snap");
    _ensure_dir(dl_dir)

    which = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for key in which:
        spec = SNAP.get(key)
        if not spec:
            print(f"[warn] unknown dataset key: {key} (skip)")
            continue
        url = spec["url"];
        directed = spec["directed"];
        ofile = spec["outfile"]
        gz_path = dl_dir / Path(url).name
        if not gz_path.exists():
            print(f"[dl] {key} ← {url}")
            _download(url, gz_path)
        print(f"[parse] {key}")
        G = _read_snap_edgelist_gz(gz_path, directed=directed)
        print(f"  nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
        _assign_node_labels(G, k=5)
        _assign_edge_labels_and_probs(G, pmin=args.pmin, alpha=6.0)
        out_graph = data_dir / ofile
        print(f"[save] {out_graph}")
        nx.write_graphml(G, out_graph)

        # queries: S/M/L
        reg = {
            "S": ((5, 8), 16),
            "M": ((9, 12), 16),
            "L": ((13, 20), 12),
        }
        for regime, (rng_sizes, nq) in reg.items():
            queries = _sample_connected_queries(G, rng_sizes[0], rng_sizes[1], nq,
                                                seed=args.seed + hash((key, regime)) % 1_000_000)
            out_q = q_dir / f"{key}_{regime}.json"
            _write_queries_json(out_q, regime=regime, size_range=rng_sizes, nqueries=nq, seed=args.seed,
                                queries=queries)
            print(f"[save] {out_q} (n={len(queries)})")


if __name__ == "__main__":
    main()
