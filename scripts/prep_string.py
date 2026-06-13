#!/usr/bin/env python3
"""Build a natively-uncertain GraphML from the STRING PPI network.

STRING edges carry a combined_score in 0..1000; we map p(e)=score/1000 onto the
edge-existence probability the C++ harness consumes (runs via --dataset graphml).
Node labels are degree-quantile classes (as in the SNAP prep). Queries are induced
connected subgraphs of the high-confidence backbone (queries/snap/*.json schema).

Input: a STRING 'protein.links' file (.txt or .txt.gz) with columns
   protein1 protein2 combined_score
e.g. the S. cerevisiae (taxid 4932) network from string-db.org; edges with score
below --min-score are dropped.

Usage:
  prep_string.py --links 4932.protein.links.v12.0.txt.gz \
     --out-graphml data/string/scerevisiae.graphml \
     --out-queries-dir queries/string --seed 324
"""
import argparse
import gzip
import json
import math
import random
from pathlib import Path


def _open(p):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)


def load_links(path, min_score):
    adj = {}
    prob = {}
    with _open(path) as f:
        header = f.readline()  # protein1 protein2 combined_score
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            a, b, sc = parts[0], parts[1], int(parts[2])
            if sc < min_score:
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
            prob[(min(a, b), max(a, b))] = sc / 1000.0
    return adj, prob


def largest_cc(adj):
    seen, best = set(), set()
    for s in adj:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u); seen.add(u)
            stack.extend(adj[u] - comp)
        if len(comp) > len(best):
            best = comp
    return best


def degree_quantile_labels(adj, nodes, k=5):
    deg = sorted((len(adj[n] & nodes), n) for n in nodes)
    lab = {}
    for rank, (_, n) in enumerate(deg):
        lab[n] = min(k - 1, rank * k // len(deg))
    return lab


def write_graphml(path, nodes, adj, prob, labels):
    idx = {n: i for i, n in enumerate(sorted(nodes))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        f.write("<graphml xmlns='http://graphml.graphdrawing.org/xmlns'>\n")
        f.write("<key id='d0' for='node' attr.name='label' attr.type='string'/>\n")
        f.write("<key id='d1' for='edge' attr.name='label' attr.type='string'/>\n")
        f.write("<key id='d2' for='edge' attr.name='prob' attr.type='double'/>\n")
        f.write("<graph edgedefault='undirected'>\n")
        for n in sorted(nodes):
            f.write(f"<node id='{idx[n]}'><data key='d0'>{labels[n]}</data></node>\n")
        done = set()
        for n in sorted(nodes):
            for m in adj[n]:
                if m not in idx:
                    continue
                e = (min(idx[n], idx[m]), max(idx[n], idx[m]))
                if e in done:
                    continue
                done.add(e)
                p = prob.get((min(n, m), max(n, m)), 1.0)
                f.write(f"<edge source='{e[0]}' target='{e[1]}'>"
                        f"<data key='d1'>0</data><data key='d2'>{p:.3f}</data></edge>\n")
        f.write("</graph>\n</graphml>\n")
    return idx


def sample_query(adj, nodes, idx, size, rng):
    start = rng.choice(list(nodes))
    comp = {start}
    frontier = list(adj[start] & nodes)
    while len(comp) < size and frontier:
        u = rng.choice(frontier)
        comp.add(u)
        frontier = [v for v in (frontier + list(adj[u] & nodes)) if v not in comp]
    if len(comp) < size:
        return None
    return [idx[n] for n in comp]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", required=True)
    ap.add_argument("--out-graphml", default="data/string/scerevisiae.graphml")
    ap.add_argument("--out-queries-dir", default="queries/string")
    ap.add_argument("--min-score", type=int, default=700)
    ap.add_argument("--seed", type=int, default=324)
    a = ap.parse_args()

    adj, prob = load_links(a.links, a.min_score)
    nodes = largest_cc(adj)
    print(f"high-confidence backbone: {len(nodes)} nodes "
          f"(score>={a.min_score}), {len(prob)} edges total")
    labels = degree_quantile_labels(adj, nodes, 5)
    idx = write_graphml(Path(a.out_graphml), nodes, adj, prob, labels)
    print(f"wrote {a.out_graphml}")

    qdir = Path(a.out_queries_dir); qdir.mkdir(parents=True, exist_ok=True)
    for regime, (lo, hi) in {"S": (5, 8), "M": (9, 12), "L": (13, 20)}.items():
        rng = random.Random(a.seed + hash(regime) % 1000)
        queries, qid = [], 0
        for _ in range(200):
            if len(queries) >= 32:
                break
            size = rng.randint(lo, hi)
            q = sample_query(adj, nodes, idx, size, rng)
            if q:
                queries.append({"id": qid, "nodes": q}); qid += 1
        out = qdir / f"string_{regime}.json"
        out.write_text(json.dumps(
            {"meta": {"regime": regime, "size_range": [lo, hi],
                      "nqueries": len(queries), "seed": a.seed}, "queries": queries},
            indent=2))
        print(f"wrote {out} ({len(queries)} queries)")


if __name__ == "__main__":
    main()
