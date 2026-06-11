# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
"""
prep_gmark.py
--------------
Schema-driven "gMark-like" property graph generator (no external deps).

Node types: Person (P), Post (Po), Comment (C), Forum (F), Tag (T)
Edge labels: knows, hasMember, hasPost, replyOf, hasTag
Undirected graph for compatibility with VF2 baselines.

Probabilities:
- knows: p = pmin + (1-pmin)*sigmoid(α*(Jaccard(P.neigh)-0.5)), α=6
- hasMember: base ~ 0.9 with small jitter
- hasPost:  base ~ 0.95
- replyOf:  base ~ 0.85, slightly higher when linking comment↔comment
- hasTag:   base ~ 0.80, more for popular tags

Outputs:
  data/gmark/gmark.graphml
  queries/gmark/gmark_S.json (…_M.json, …_L.json)
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import networkx as nx


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _percentile(vs, p):
    if not vs: return 0
    vs = sorted(vs)
    if p <= 0: return vs[0]
    if p >= 100: return vs[-1]
    k = (len(vs) - 1) * (p / 100.0)
    import math
    f, c = math.floor(k), math.ceil(k)
    if f == c: return vs[int(k)]
    return vs[f] * (c - k) + vs[c] * (k - f)


def _sample_connected_queries(G: nx.Graph, size_min: int, size_max: int, nqueries: int, seed: int):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    out = []
    for qi in range(nqueries):
        k = rng.randint(size_min, size_max)
        for _ in range(128):
            s = rng.choice(nodes)
            seen = {s};
            frontier = [s]
            while len(seen) < k and frontier:
                u = frontier.pop(0)
                for w in G.neighbors(u):
                    if w not in seen:
                        seen.add(w);
                        frontier.append(w)
                    if len(seen) >= k: break
            if len(seen) < k: continue
            H = G.subgraph(seen).copy()
            if nx.is_connected(H):
                out.append({"id": qi, "nodes": sorted(seen)})
                break
        else:
            out.append({"id": qi, "nodes": sorted(seen)})
    return out


def _write_queries_json(path: Path, *, regime, size_range, nqueries, seed, queries):
    # queries je list slovníků {"id": ..., "nodes": [...]}, kde nodes jsou aktuálně int
    norm = []
    for q in queries:
        norm.append({"id": q["id"], "nodes": [str(x) for x in q["nodes"]]})
    payload = {
        "meta": {"regime": regime, "size_range": list(size_range), "nqueries": nqueries, "seed": seed},
        "queries": norm,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_gmark_like(scale: float, seed: int, pmin: float) -> nx.Graph:
    rng = random.Random(seed)
    # node counts (mírné škálování)
    nP = int(2000 * scale)
    nPo = int(4000 * scale)
    nC = int(4000 * scale)
    nF = int(200 * scale)
    nT = int(300 * scale)

    G = nx.Graph()

    # add nodes with type labels
    # Person
    for i in range(nP):
        G.add_node(f"P{i}", label="Person")
    for i in range(nPo):
        G.add_node(f"Po{i}", label="Post")
    for i in range(nC):
        G.add_node(f"C{i}", label="Comment")
    for i in range(nF):
        G.add_node(f"F{i}", label="Forum")
    for i in range(nT):
        G.add_node(f"T{i}", label="Tag")

    # --- knows: small-worldish using random edges + triadic closure ---
    persons = [f"P{i}" for i in range(nP)]
    # initial random friendships
    target_deg = 8
    for u in persons:
        while G.degree(u) < target_deg:
            v = rng.choice(persons)
            if v == u: continue
            if G.has_edge(u, v): continue
            G.add_edge(u, v, label="knows", prob=0.5)  # temp; finalize later

    # triadic closure
    for u in persons:
        neigh = list(G.neighbors(u))
        if len(neigh) < 2: continue
        v = rng.choice(neigh);
        w = rng.choice(neigh)
        if v == w or G.has_edge(v, w): continue
        G.add_edge(v, w, label="knows", prob=0.5)

    # finalize knows probabilities via Jaccard-based logistic
    N = {u: set(G.neighbors(u)) for u in G.nodes()}
    for u, v, d in list(G.edges(data=True)):
        if d.get("label") == "knows":
            Nu = (N[u] - {v});
            Nv = (N[v] - {u})
            j = (len(Nu & Nv) / max(1, len(Nu | Nv))) if (Nu or Nv) else 0.0
            base = _sigmoid(6.0 * (j - 0.5))
            p = pmin + (1.0 - pmin) * base
            d["prob"] = float(max(pmin, min(1.0, p)))

    # --- forums & membership ---
    forums = [f"F{i}" for i in range(nF)]
    for u in persons:
        fcount = rng.choices([0, 1, 2, 3], weights=[0.1, 0.6, 0.25, 0.05])[0]
        for _ in range(fcount):
            f = rng.choice(forums)
            if not G.has_edge(u, f):
                p = max(pmin, 0.90 - rng.uniform(0.0, 0.05))
                G.add_edge(u, f, label="hasMember", prob=float(p))

    # --- posts (authorship) ---
    posts = [f"Po{i}" for i in range(nPo)]
    for po in posts:
        u = rng.choice(persons)
        p = max(pmin, 0.95 - rng.uniform(0.0, 0.05))
        G.add_edge(u, po, label="hasPost", prob=float(p))

    # --- comments & replyOf ---
    comments = [f"C{i}" for i in range(nC)]
    for c in comments:
        # link comment to either a post or a previous comment
        if rng.random() < 0.7:
            tgt = rng.choice(posts)
        else:
            tgt = rng.choice(comments[:max(1, comments.index(c))]) if comments.index(c) > 0 else rng.choice(posts)
        base = 0.88 if tgt.startswith("C") else 0.85
        p = max(pmin, base - rng.uniform(0.0, 0.05))
        G.add_edge(c, tgt, label="replyOf", prob=float(p))

    # --- tags ---
    tags = [f"T{i}" for i in range(nT)]
    # tag popularity (Zipf-ish)
    pop = [1.0 / (i + 1) for i in range(nT)]
    s = sum(pop);
    pop = [x / s for x in pop]
    for po in posts:
        for _ in range(rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]):
            t = rng.choices(tags, weights=pop, k=1)[0]
            p = max(pmin, 0.80 + 0.15 * pop[tags.index(t)] - rng.uniform(0.0, 0.05))
            if not G.has_edge(po, t):
                G.add_edge(po, t, label="hasTag", prob=float(min(1.0, p)))

    # ensure weak connectivity (bridge between communities)
    comps = list(nx.connected_components(G))
    if len(comps) > 1:
        reps = [next(iter(c)) for c in comps]
        for i in range(len(reps) - 1):
            u, v = reps[i], reps[i + 1]
            G.add_edge(u, v, label="bridge", prob=0.30)

    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-data-dir", type=str, default="data/gmark")
    ap.add_argument("--out-queries-dir", type=str, default="queries/gmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="Multiplies node counts; 1.0 ~ ~10k-12k nodes.")
    ap.add_argument("--pmin", type=float, default=0.7)
    args = ap.parse_args()

    _ensure_dir(Path(args.out_data_dir))
    _ensure_dir(Path(args.out_queries_dir))

    G = build_gmark_like(scale=args.scale, seed=args.seed, pmin=args.pmin)
    print(f"[prep_gmark] graph: |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
    out_graph = Path(args.out_data_dir) / "gmark.graphml"
    nx.write_graphml(G, out_graph)
    print(f"[prep_gmark] saved {out_graph}")

    regimes = {"S": ((5, 8), 16), "M": ((9, 12), 16), "L": ((13, 20), 12)}
    for reg, (rng_sz, nq) in regimes.items():
        qs = _sample_connected_queries(G, rng_sz[0], rng_sz[1], nq, seed=args.seed + hash(("gmark", reg)) % 1_000_000)
        out_q = Path(args.out_queries_dir) / f"gmark_{reg}.json"
        _write_queries_json(out_q, regime=reg, size_range=rng_sz, nqueries=nq, seed=args.seed, queries=qs)
        print(f"[prep_gmark] saved {out_q} (n={len(qs)})")


if __name__ == "__main__":
    main()
