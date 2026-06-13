#!/usr/bin/env python3
"""Substitution-tolerant subgraph isomorphism (Le Bodic et al. 2012 / GEM++) as a
Binary Linear Program, solved with HiGHS (scipy.optimize.milp).

Exact pattern topology + injective node assignment, minimising the substitution cost
sum of -log(s) over node and edge assignments (the MAP/log-likelihood objective under
cost = -log s). Node variables x[i,k] and edge variables y[(i,j),(k,l)]; the
O(|VQ||VG| + |EQ||EG|) binaries mean this is for small instances. Reads the GraphML
target + JSON queries of the C++ harness; writes results/blp_baseline.csv
(dataset,query,blp_ms,recovered,obj).
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
except Exception as e:  # pragma: no cover
    raise SystemExit(f"need scipy>=1.9 with HiGHS milp: {e}")

EPS = 1e-3
# defaults mirror compat.hpp (node_w0,node_w_label / edge_w0,edge_w_label,edge_w_prob)
NW0, NWL = -3.0, 6.0
EW0, EWL, EWP = -1.0, 3.0, 2.0


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def node_cost(ql, ul):
    s = max(EPS, sigmoid(NW0 + NWL * (1.0 if ql == ul else 0.0)))
    return -math.log(s)


def edge_cost(qlab, glab, p):
    p = min(max(p, EPS), 1.0 - 1e-9)
    s = max(EPS, sigmoid(EW0 + EWL * (1.0 if qlab == glab else 0.0) + EWP * math.log(p / (1.0 - p))))
    return -math.log(s)


def solve_blp(Qadj, Qelab, Gadj, qlab, glab, gprob, gelab, time_limit=30.0):
    """Exact-topology, substitution-tolerant min-cost matching via HiGHS.
    Qadj: query adjacency (idx->set); Qelab: {(i,j):label}; Gadj: target adjacency;
    qlab/glab: node labels; gprob/gelab: {(k,l):prob/label} on target edges (k<l).
    Returns (obj, None, assign|None)."""
    nq, ng = len(qlab), len(glab)
    col = {}
    cost = []

    def newvar(c):
        cost.append(c)
        return len(cost) - 1

    x = {(i, k): newvar(node_cost(qlab[i], glab[k])) for i in range(nq) for k in range(ng)}

    # edge variables: pattern edge (i,j),i<j -> oriented target edge (k,l)
    qedges = [(i, j) for i in range(nq) for j in Qadj[i] if i < j]
    gedir = []  # oriented target edges (k,l) for both directions of each undirected edge
    for (k, l) in gprob:
        gedir.append((k, l))
        gedir.append((l, k))
    y = {}
    for (i, j) in qedges:
        for (k, l) in gedir:
            key = (i, j, k, l)
            und = (min(k, l), max(k, l))
            y[key] = newvar(edge_cost(Qelab[(i, j)], gelab[und], gprob[und]))

    nx_ = len(cost)
    rows, lb, ub = [], [], []

    def addrow(coeffs, lo, hi):
        r = lil_matrix((1, nx_))
        for c, v in coeffs:
            r[0, c] = v
        rows.append(r); lb.append(lo); ub.append(hi)

    for i in range(nq):                                   # each pattern node assigned once
        addrow([(x[(i, k)], 1) for k in range(ng)], 1, 1)
    for k in range(ng):                                   # injective
        addrow([(x[(i, k)], 1) for i in range(nq)], None, 1)
    for (i, j) in qedges:                                 # each pattern edge hosted exactly once
        addrow([(y[(i, j, k, l)], 1) for (k, l) in gedir], 1, 1)
    for (i, j, k, l) in y:                                # link y -> x (both endpoints)
        addrow([(y[(i, j, k, l)], 1), (x[(i, k)], -1)], None, 0)
        addrow([(y[(i, j, k, l)], 1), (x[(j, l)], -1)], None, 0)

    A = lil_matrix((len(rows), nx_))
    for ri, r in enumerate(rows):
        A.rows[ri] = r.rows[0]; A.data[ri] = r.data[0]
    cons = LinearConstraint(A.tocsr(),
                            [(-np.inf if v is None else v) for v in lb],
                            [(np.inf if v is None else v) for v in ub])
    res = milp(c=np.array(cost), constraints=[cons], integrality=np.ones(nx_),
               bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if not res.success or res.x is None:
        return (None, None, None)
    assign = {i: k for (i, k), v in x.items() if res.x[v] > 0.5}
    return (res.fun, None, assign)


def load_graphml(path):
    import xml.etree.ElementTree as ET
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ET.parse(path).getroot()
    key = {k.get("id"): k.get("attr.name", "") for k in root.findall("g:key", ns)}
    node_label_keys = {kid for kid, nm in key.items() if nm == "label"}
    edge_label_keys = {kid for kid, nm in key.items() if nm == "label"}
    prob_keys = {kid for kid, nm in key.items() if nm == "prob"}
    g = root.find("g:graph", ns)
    idx, labels = {}, []
    for n in g.findall("g:node", ns):
        nid = n.get("id"); idx[nid] = len(idx)
        lab = 0
        for d in n.findall("g:data", ns):
            if d.get("key") in node_label_keys:
                try: lab = int(float(d.text))
                except Exception: pass
        labels.append(lab)
    adj = {i: set() for i in range(len(idx))}
    eprob, elabel = {}, {}
    for e in g.findall("g:edge", ns):
        a, b = idx[e.get("source")], idx[e.get("target")]
        adj[a].add(b); adj[b].add(a)
        p, el = 1.0, 0
        for d in e.findall("g:data", ns):
            kd = d.get("key")
            if kd in prob_keys:
                try: p = float(d.text)
                except Exception: pass
            elif kd in edge_label_keys:
                try: el = int(float(d.text))
                except Exception: pass
        eprob[(min(a, b), max(a, b))] = p
        elabel[(min(a, b), max(a, b))] = el
    return adj, labels, eprob, elabel, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphml", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", default="results/blp_baseline.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=30.0)
    a = ap.parse_args()

    Gadj, glab, gprob, gelab, idmap = load_graphml(a.graphml)
    qspec = json.load(open(a.queries))
    queries = qspec["queries"] if isinstance(qspec, dict) else qspec

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "query", "blp_ms", "recovered", "obj"])
        ds = Path(a.graphml).stem
        for qi, q in enumerate(queries):
            if a.limit and qi >= a.limit:
                break
            nodes = q["nodes"] if isinstance(q, dict) else q
            nodes = [idmap.get(str(n), n) if not isinstance(n, int) else n for n in nodes]
            pos = {n: i for i, n in enumerate(nodes)}
            Qadj = {i: set() for i in range(len(nodes))}
            Qelab = {}
            ql = [glab[n] for n in nodes]
            for n in nodes:
                for m in Gadj[n]:
                    if m in pos:
                        i, j = pos[n], pos[m]
                        Qadj[i].add(j)
                        if i < j:
                            Qelab[(i, j)] = gelab[(min(n, m), max(n, m))]
            t0 = time.perf_counter()
            obj, _, assign = solve_blp(Qadj, Qelab, Gadj, ql, glab, gprob, gelab, a.time_limit)
            ms = (time.perf_counter() - t0) * 1e3
            recovered = int(assign is not None and all(assign.get(i) == nodes[i] for i in range(len(nodes))))
            w.writerow([ds, qi, f"{ms:.1f}", recovered, "" if obj is None else f"{obj:.4f}"])
            print(f"  q{qi}: {ms:.0f}ms recovered={recovered} obj={obj}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
