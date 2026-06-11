# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
"""Resumable per-query experiment runner.

Runs one (cell, method, query) at a time, appends a JSONL row immediately, and
skips rows already present. Honours a wall-clock budget so it can be invoked
repeatedly under a short per-command limit. Reuses run_experiments' exact
engine wrappers/metrics so numbers match the paper's pipeline.

Usage:
  python3 grid.py <tag[,tag...]|all> [wall_budget_s] [runs]
"""
import sys, os, json, time
sys.path.insert(0, ".")
import networkx as nx
import run_experiments as R
from vf2_prob import ProbSGI, AStarProb

RAW_DIR = "../results_raw"

# tag -> dict(kind, ...params..., timeout)
CELLS = {
    # synthetic (generated reproducibly); Makefile params per regime
    "synth_S": dict(kind="synth", qmin=5,  qmax=8,  nq=16, pmin=0.80, flip=0.00, timeout=12),
    "synth_M": dict(kind="synth", qmin=9,  qmax=12, nq=16, pmin=0.70, flip=0.05, timeout=12),
    "synth_L": dict(kind="synth", qmin=13, qmax=20, nq=12, pmin=0.60, flip=0.10, timeout=12),
    # molecular (real-ish, 70 nodes)
    "mol_S": dict(kind="file", graphml="../data/mol/mol.graphml", qjson="../queries/mol/mol_S.json", timeout=12),
    "mol_M": dict(kind="file", graphml="../data/mol/mol.graphml", qjson="../queries/mol/mol_M.json", timeout=12),
    "mol_L": dict(kind="file", graphml="../data/mol/mol.graphml", qjson="../queries/mol/mol_L.json", timeout=12),
    # gMark-like (2100 nodes) -- heavy; bounded timeout (compute budget)
    "gmark_S": dict(kind="file", graphml="../data/gmark/gmark.graphml", qjson="../queries/gmark/gmark_S.json", timeout=10, maxq=8),
    "gmark_M": dict(kind="file", graphml="../data/gmark/gmark.graphml", qjson="../queries/gmark/gmark_M.json", timeout=10, maxq=8),
    "gmark_L": dict(kind="file", graphml="../data/gmark/gmark.graphml", qjson="../queries/gmark/gmark_L.json", timeout=10, maxq=8),
}

BASELINES = ["vf2", "vf2pp", "vf2bin"]
PROB = [  # (method_label, ub_mode, search)
    ("vf2prob",              "node",   "dfs"),
    ("vf2prob-assign",       "assign", "dfs"),
    ("vf2prob-astar",        "node",   "astar"),
    ("vf2prob-astar-assign", "assign", "astar"),
]
SEED = 324
BIN_TAU = 0.7

def _coerce(nid, nodes_set):
    for c in (nid, str(nid)):
        if c in nodes_set:
            return c
    try:
        ix = int(nid)
        if ix in nodes_set: return ix
    except Exception:
        pass
    return None

def build_cell(tag):
    c = CELLS[tag]
    R._seed_everything(SEED)
    if c["kind"] == "synth":
        G = R._synth_graph(num_nodes=120, p=0.03, seed=SEED)
        R._inject_uncertainty(G, pmin=c["pmin"], flip=c["flip"])
        Q = R._sample_connected_queries(G, size_min=c["qmin"], size_max=c["qmax"], num=c["nq"])
        dataset = "synth"
    else:
        G = nx.read_graphml(c["graphml"])
        R._inject_uncertainty(G, pmin=1.0, flip=0.0)
        payload = json.loads(open(c["qjson"]).read())
        nodes_set = set(G.nodes())
        Q = []
        for item in payload.get("queries", []):
            mp = [_coerce(n, nodes_set) for n in item.get("nodes", [])]
            mp = [m for m in mp if m is not None]
            if len(mp) >= 2:
                sub = G.subgraph(mp).copy()
                if sub.number_of_nodes() >= 2:
                    Q.append(sub)
        dataset = os.path.splitext(os.path.basename(c["graphml"]))[0]
    return G, Q, dataset, c

def run_one(G, Q, dataset, c, method, ub, search):
    T = c["timeout"]
    if method in BASELINES:
        t0 = time.time()
        if method == "vf2":
            raw = R.run_vf2_wrapper(G, Q, timeout=T)
        elif method == "vf2pp":
            raw = R.run_vf2pp_wrapper(G, Q, timeout=T)
        else:
            raw = R.run_vf2bin_wrapper(G, Q, timeout=T, bin_threshold=BIN_TAU)
        ms = (time.time() - t0) * 1000.0
        nm = R._norm_metrics(raw)
        to = int(nm["timed_out"])
        success = 1.0 if (nm.get("solutions_found", 0) >= 1 and not to) else 0.0
    else:
        # Single fast pass (no expensive per-state report-UB). Tightness is the
        # ROOT gap of the actual admissible pruning bound: root_ub - optimum >= 0.
        nl_G, nl_Q = R._build_shared_node_label_ids(G, Q)
        el_G = R._build_edge_label_ids(G); ep = R._edge_probabilities(G)
        Cls = AStarProb if search == "astar" else ProbSGI
        e = Cls(G, Q, nl_G, nl_Q, el_G, ep, timeout_s=T, use_bound=True,
                use_node=True, use_edge=True, collect_metrics=False, compute_report_ub=False)
        e.ub_mode = ub
        t0 = time.time()
        bm, cnt, toR = e.run()
        ms = (time.time() - t0) * 1000.0
        states = int(cnt["states"]); pruned = int(cnt["pruned"])
        prune_rate = float(pruned) / float(max(1, states + pruned))
        bll = float(cnt["best_loglik"])
        root_ub = e.root_ub_clamped
        root_gap = (root_ub - bll) if (bll > float("-inf") and root_ub is not None) else float("nan")
        to = int(bool(toR))
        success = 1.0 if (bll > float("-inf") and not to) else 0.0
        return {
            "states": states, "prune_rate": prune_rate, "best_loglik": bll,
            "timed_out": to, "success": success, "elapsed_ms": ms,
            "ub_gap_p50": root_gap, "ub_gap_p90": root_gap,
        }
    return {
        "states": nm["states"], "prune_rate": nm["prune_rate"], "best_loglik": nm["best_loglik"],
        "timed_out": to, "success": success, "elapsed_ms": ms,
        "ub_gap_p50": nm.get("ub_gap_p50", float("nan")), "ub_gap_p90": nm.get("ub_gap_p90", float("nan")),
    }

def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    r = json.loads(line)
                    done.add((r["method"], r["qidx"], r["run"]))
                except Exception:
                    pass
    return done

def main():
    tags = sys.argv[1] if len(sys.argv) > 1 else "all"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 38.0
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if tags == "all":
        taglist = list(CELLS.keys())
    else:
        taglist = [t.strip() for t in tags.split(",") if t.strip()]
    os.makedirs(RAW_DIR, exist_ok=True)
    t_start = time.time()
    n_new = 0
    for tag in taglist:
        G, Q, dataset, c = build_cell(tag)
        path = os.path.join(RAW_DIR, f"{tag}.jsonl")
        done = load_done(path)
        methods = [(m, None, None) for m in BASELINES] + [(m, u, s) for (m, u, s) in PROB]
        maxq = c.get("maxq", len(Q))
        for (method, ub, search) in methods:
            for qidx, Qg in enumerate(Q):
                if qidx >= maxq:
                    break
                for run in range(runs):
                    if (method, qidx, run) in done:
                        continue
                    if time.time() - t_start > budget:
                        print(f"[budget hit] new={n_new}; stopping in {tag} at {method} q{qidx}", flush=True)
                        return
                    res = run_one(G, Qg, dataset, c, method, ub, search)
                    row = dict(tag=tag, dataset=dataset, method=method, qidx=qidx, run=run,
                               qn=Qg.number_of_nodes(), qe=Qg.number_of_edges(), **res)
                    with open(path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    n_new += 1
        print(f"[done cell {tag}] methods x queries complete", flush=True)
    print(f"[finished] new rows this call = {n_new}; elapsed {time.time()-t_start:.1f}s", flush=True)

if __name__ == "__main__":
    main()
