# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except Exception:
    psutil = None

# engines
try:
    from vf2_engine import vf2_search
except Exception:
    vf2_search = None

try:
    from vf2pp_engine import VF2PP
except Exception:
    VF2PP = None

try:
    from vf2_prob import ProbSGI
except Exception:
    ProbSGI = None

# dataset loaders
bundle_lsqb = None
bundle_ldbc = None
bundle_qmark = None
bundle_mol = None
for _mod in ("loader_lsqb", "loader_ldbc", "loader_qmark", "loader_mol"):
    try:
        globals()[_mod] = __import__(_mod)
    except Exception:
        pass

try:
    import networkx as nx
except Exception:
    nx = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _percentile(xs: List[float], p: float) -> float:
    if not xs: return float("nan")
    vs = sorted(xs)
    if p <= 0: return vs[0]
    if p >= 100: return vs[-1]
    k = (len(vs) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c: return vs[int(k)]
    return vs[f] * (c - k) + vs[c] * (k - f)


def _median(xs: List[float]) -> float:
    return _percentile(xs, 50.0)


def _iqr(xs: List[float]) -> float:
    if not xs: return float("nan")
    return _percentile(xs, 75.0) - _percentile(xs, 25.0)


def _node_label_value(attrs: Dict[str, Any]) -> Any:
    for k in ("label", "type", "label_id"):
        if k in attrs: return attrs[k]
    return None


def _build_shared_node_label_ids(G, Q) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    vocab: List[Any] = []
    seen = set()
    for n in G.nodes():
        v = _node_label_value(G.nodes[n]);
        if v is not None and v not in seen: seen.add(v); vocab.append(v)
    for n in Q.nodes():
        v = _node_label_value(Q.nodes[n]);
        if v is not None and v not in seen: seen.add(v); vocab.append(v)
    index = {v: i for i, v in enumerate(vocab)}
    nl_G = {n: index.get(_node_label_value(G.nodes[n]), -1) for n in G.nodes()}
    nl_Q = {n: index.get(_node_label_value(Q.nodes[n]), -1) for n in Q.nodes()}
    return nl_G, nl_Q


def _build_edge_label_ids(G) -> Dict[Tuple[Any, Any], int]:
    vocab: List[Any] = []
    seen = set()
    for _, _, d in G.edges(data=True):
        v = d.get("label", d.get("type", d.get("label_id", None)))
        if v is not None and v not in seen: seen.add(v); vocab.append(v)
    index = {v: i for i, v in enumerate(vocab)}
    el_G: Dict[Tuple[Any, Any], int] = {}
    for u, v, d in G.edges(data=True):
        val = d.get("label", d.get("type", d.get("label_id", None)))
        key = (u, v) if u <= v else (v, u)
        el_G[key] = index.get(val, -1)
    return el_G


def _edge_probabilities(G) -> Dict[Tuple[Any, Any], float]:
    ep: Dict[Tuple[Any, Any], float] = {}
    for u, v, d in G.edges(data=True):
        key = (u, v) if u <= v else (v, u)
        try:
            ep[key] = float(d.get("prob", 1.0))
        except Exception:
            ep[key] = 1.0
    return ep


def _synth_graph(num_nodes: int = 120, p: float = 0.03, seed=324):
    if nx is None: raise RuntimeError("networkx required for synthetic")
    G = nx.erdos_renyi_graph(num_nodes, p, seed=seed, directed=False)
    for n in G.nodes():
        G.nodes[n]["label"] = str(n % 5)
    for u, v in G.edges():
        G.edges[u, v]["label"] = str((u + v) % 3)
        G.edges[u, v]["prob"] = 1.0
    return G


def _sample_connected_queries(G, size_min: int, size_max: int, num: int):
    if nx is None: raise RuntimeError("networkx required for query sampling")
    if size_min > size_max: size_min, size_max = size_max, size_min
    sizes = [random.randint(size_min, size_max) for _ in range(num)]
    nodes = list(G.nodes())
    queries = []
    for k in sizes:
        sub = None
        for _ in range(50):
            start = random.choice(nodes)
            seen = {start};
            frontier = [start]
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
                sub = H;
                break
        if sub is None:
            sub = G.subgraph(random.sample(nodes, min(k, len(nodes)))).copy()
        queries.append(sub)
    return queries


def _inject_uncertainty(G, *, pmin: float = 1.0, flip: float = 0.0) -> None:
    try:
        pmin = float(pmin)
    except Exception:
        pmin = 1.0
    try:
        flip = float(flip)
    except Exception:
        flip = 0.0
    pmin = max(0.0, min(1.0, pmin))
    flip = max(0.0, min(1.0, flip))

    import random as _r

    if pmin < 1.0:
        for _, _, d in G.edges(data=True):
            d["prob"] = _r.uniform(pmin, 1.0)
    else:
        for _, _, d in G.edges(data=True):
            if "prob" not in d: d["prob"] = 1.0

    if flip <= 0.0: return

    def _pick_node_attr(Gx):
        for k in ("label", "type", "label_id"):
            if any(k in Gx.nodes[n] for n in Gx.nodes()): return k
        return None

    def _pick_edge_attr(Gx):
        for k in ("label", "type", "label_id"):
            if any(k in d for _, _, d in Gx.edges(data=True)): return k
        return None

    n_attr = _pick_node_attr(G)
    e_attr = _pick_edge_attr(G)

    def _vocab_nodes(Gx, key):
        vals = set()
        for n in Gx.nodes():
            if key in Gx.nodes[n]: vals.add(Gx.nodes[n][key])
        return list(vals)

    def _vocab_edges(Gx, key):
        vals = set()
        for _, _, d in Gx.edges(data=True):
            if key in d: vals.add(d[key])
        return list(vals)

    if n_attr:
        nv = _vocab_nodes(G, n_attr)
        if nv:
            for n in G.nodes():
                if n_attr in G.nodes[n] and _r.random() < flip:
                    cur = G.nodes[n][n_attr]
                    choices = [x for x in nv if x != cur] or nv
                    G.nodes[n][n_attr] = _r.choice(choices)
    if e_attr:
        ev = _vocab_edges(G, e_attr)
        if ev:
            for u, v, d in G.edges(data=True):
                if e_attr in d and _r.random() < flip:
                    cur = d[e_attr]
                    choices = [x for x in ev if x != cur] or ev
                    d[e_attr] = _r.choice(choices)


@dataclass
class Bundle:
    graph: Any
    queries: List[Any]
    meta: Dict[str, Any]


def _load_bundle_from_args(args: argparse.Namespace) -> Optional[Bundle]:
    ds = getattr(args, "dataset", "synth")
    if ds == "synth": return None

    if ds == "lsqb" and bundle_lsqb:
        try:
            b = bundle_lsqb.load(args.data_path, args.queries_path)  # type: ignore
            return Bundle(b.graph, b.queries, getattr(b, "meta", {}))
        except Exception:
            pass
    if ds == "ldbc" and bundle_ldbc:
        try:
            b = bundle_ldbc.load(args.data_path, args.queries_path)  # type: ignore
            return Bundle(b.graph, b.queries, getattr(b, "meta", {}))
        except Exception:
            pass
    if ds == "qmark" and bundle_qmark:
        try:
            b = bundle_qmark.load(args.data_path, args.queries_path)  # type: ignore
            return Bundle(b.graph, b.queries, getattr(b, "meta", {}))
        except Exception:
            pass
    if ds == "mol" and bundle_mol:
        try:
            b = bundle_mol.load(args.data_path, args.queries_path)  # type: ignore
            return Bundle(b.graph, b.queries, getattr(b, "meta", {}))
        except Exception:
            pass
    return None


def _norm_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["states"] = int(res.get("states", res.get("states_visited", 0)))
    out["pruned"] = int(res.get("pruned", res.get("prunes", 0)))
    denom = max(1, out["states"] + out["pruned"])
    out["prune_rate"] = float(out["pruned"]) / float(denom)
    out["timed_out"] = int(bool(res.get("timed_out", False)))
    out["best_loglik"] = float(res.get("best_loglik", float("-inf")))
    out["solutions_found"] = int(res.get("solutions_found", res.get("solutions", 0)))
    out["ub_gap_p50"] = float(res.get("ub_gap_p50", float("nan")))
    out["ub_gap_p90"] = float(res.get("ub_gap_p90", float("nan")))
    return out


def run_vf2_wrapper(G, Q, *, timeout: float) -> Dict[str, Any]:
    if vf2_search is None:
        raise RuntimeError("vf2_engine.vf2_search not importable")
    nl_G, nl_Q = _build_shared_node_label_ids(G, Q)
    best, counters, timed_out = vf2_search(G, Q, nl_G, nl_Q, timeout_s=timeout)
    return {
        "states": int(counters.get("states", 0)),
        "pruned": int(counters.get("pruned", 0)),
        "solutions_found": int(counters.get("solutions", counters.get("solutions_found", 0))),
        "timed_out": bool(timed_out),
        "best_loglik": float("-inf"),
        "ub_gap_p50": float("nan"),
        "ub_gap_p90": float("nan"),
    }


def run_vf2bin_wrapper(G, Q, *, timeout: float, bin_threshold: float) -> Dict[str, Any]:
    if vf2_search is None or nx is None:
        raise RuntimeError("vf2_engine or networkx not importable")
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        try:
            p = float(d.get("prob", 1.0))
        except Exception:
            p = 1.0
        if p >= bin_threshold:
            H.add_edge(u, v, **d)
    nl_G, nl_Q = _build_shared_node_label_ids(H, Q)
    best, counters, timed_out = vf2_search(H, Q, nl_G, nl_Q, timeout_s=timeout)
    return {
        "states": int(counters.get("states", 0)),
        "pruned": int(counters.get("pruned", 0)),
        "solutions_found": int(counters.get("solutions", counters.get("solutions_found", 0))),
        "timed_out": bool(timed_out),
        "best_loglik": float("-inf"),
        "ub_gap_p50": float("nan"),
        "ub_gap_p90": float("nan"),
    }


def run_vf2pp_wrapper(G, Q, *, timeout: float) -> Dict[str, Any]:
    if VF2PP is None:
        raise RuntimeError("vf2pp_engine.VF2PP not importable")

    def _pick_node_attr(Gx):
        if any("label" in Gx.nodes[n] for n in Gx.nodes()): return "label"
        if any("type" in Gx.nodes[n] for n in Gx.nodes()): return "type"
        if any("label_id" in Gx.nodes[n] for n in Gx.nodes()): return "label_id"
        return None

    def _pick_edge_attr(Gx):
        if any("label" in d for _, _, d in Gx.edges(data=True)): return "label"
        if any("type" in d for _, _, d in Gx.edges(data=True)): return "type"
        if any("label_id" in d for _, _, d in Gx.edges(data=True)): return "label_id"
        return None

    node_attr = _pick_node_attr(G)
    edge_attr = _pick_edge_attr(G)

    engine = VF2PP(Q, G, node_label_attr=node_attr, edge_label_attr=edge_attr)

    import time as _t
    t0 = _t.time()
    sols = 0
    for _m in engine.match():
        sols += 1
        if (_t.time() - t0) > timeout:
            break

    return {
        "states": int(getattr(engine, "stats").states_visited) if hasattr(engine, "stats") else 0,
        "pruned": int(getattr(engine, "stats").prunes) if hasattr(engine, "stats") else 0,
        "solutions_found": int(sols),
        "timed_out": int((_t.time() - t0) > timeout),
        "best_loglik": float("-inf"),
        "ub_gap_p50": float("nan"),
        "ub_gap_p90": float("nan"),
    }


def run_vf2prob_once(G, Q, *, timeout: float, time_only: bool, sample_every: int = 300, ub_mode: str = "node",
                     search: str = "dfs"):
    if ProbSGI is None:
        raise RuntimeError("vf2_prob.ProbSGI not importable")
    try:
        from vf2_prob import AStarProb  # type: ignore
        _HAS_AST = True
    except Exception:
        _HAS_AST = False
    nl_G, nl_Q = _build_shared_node_label_ids(G, Q)
    el_G = _build_edge_label_ids(G)
    edge_p = _edge_probabilities(G)
    EngineCls = ProbSGI
    if search == "astar" and _HAS_AST:
        EngineCls = AStarProb
    engine = EngineCls(
        G, Q, nl_G, nl_Q, el_G, edge_p,
        timeout_s=timeout, use_bound=True, use_node=True, use_edge=True,
        sample_every=(10 ** 9 if time_only else sample_every),
        collect_metrics=(not time_only),
        compute_report_ub=(not time_only),
    )
    if hasattr(engine, "ub_mode"):
        setattr(engine, "ub_mode", ub_mode)
    t0 = time.time()
    best_map, counters, timed_out = engine.run()
    elapsed_ms = (time.time() - t0) * 1000.0
    return elapsed_ms, counters, bool(timed_out)


def run_vf2prob_wrapper(G, Q, *, timeout: float, two_pass: bool, sample_every: int = 300, ub_mode: str = "node",
                        search: str = "dfs"):
    if not two_pass:
        ms, cnt, to = run_vf2prob_once(G, Q, timeout=timeout, time_only=False, sample_every=sample_every,
                                       ub_mode=ub_mode, search=search)
        merged = dict(cnt)
        merged["timed_out"] = int(bool(to))
        merged["elapsed_ms"] = float(ms)
        return merged
    # two-pass: A=time-only, B=metrics; merge into one dict
    msA, cntA, toA = run_vf2prob_once(G, Q, timeout=timeout, time_only=True, sample_every=sample_every, ub_mode=ub_mode,
                                      search=search)
    msB, cntB, toB = run_vf2prob_once(G, Q, timeout=timeout, time_only=False, sample_every=sample_every,
                                      ub_mode=ub_mode, search=search)
    merged = dict(cntB)
    merged["timed_out"] = int(bool(toA or toB))
    merged["elapsed_ms"] = float(msA)  # take time from pass-A
    return merged


import csv, tempfile, shutil


def _csv_read_rows(path, expected_header):
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames != list(expected_header):
            return []
        for r in rdr:
            rows.append(r)
    return rows


def _csv_write_rows_atomic(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8"), \
            open(0) as _:
        tmpname = _
    import uuid
    tmpfile = path.parent / f".tmp_{uuid.uuid4().hex}.csv"
    try:
        with tmpfile.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=header)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
        shutil.move(str(tmpfile), str(path))
    finally:
        try:
            tmpfile.unlink(missing_ok=True)
        except Exception:
            pass


def _csv_append_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        if not exists:
            wr.writeheader()
        for r in rows:
            wr.writerow(r)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-mode", type=str, choices=("overwrite", "append", "update"), default="update",
                        help="How to write results.csv: overwrite = replace file; append = append rows; update = upsert by (dataset,method).")
    parser.add_argument("--seed", type=int, default=324)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", type=str, default="results.csv")
    parser.add_argument("--bin-threshold", type=float, default=0.5,
                        help="Edge probability threshold τ for VF2-Bin.")
    parser.add_argument("--pmin", type=float, default=1.0,
                        help="If <1.0, resample ALL edge probabilities to Uniform[pmin,1].")
    parser.add_argument("--flip", type=float, default=0.0,
                        help="Label flip rate in [0,1] for node/edge categorical labels.")
    parser.add_argument("--dataset", type=str, choices=("synth", "lsqb", "ldbc", "qmark", "mol"), default="synth")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--queries-path", type=str, default="")
    parser.add_argument("--qmin", type=int, default=6)
    parser.add_argument("--qmax", type=int, default=10)
    parser.add_argument("--nqueries", type=int, default=8)
    parser.add_argument("--methods", type=str, default="vf2,vf2pp,vf2bin,vf2prob",
                        help="comma-separated among: vf2,vf2pp,vf2bin,vf2prob")
    parser.add_argument("--vf2prob-two-pass", action="store_true",
                        help="Run VF2-Prob in two passes (time-only + metrics) and merge as ONE row.")
    parser.add_argument("--vf2prob-ub", type=str, choices=("node", "assign"), default="node",
                        help="Upper bound mode for VF2-Prob: 'node' (default) or 'assign' (assignment-based).")
    parser.add_argument("--vf2prob-search", type=str, choices=("dfs", "astar"), default="dfs",
                        help="Search strategy for VF2-Prob: 'dfs' (branch-and-bound) or 'astar' (best-first).")
    parser.add_argument("--vf2prob-sample-every", type=int, default=300,
                        help="Sampling period for VF2-Prob metrics (pass-B).")
    parser.add_argument("--tag", type=str, default="", help="Optional short tag appended to method label.")

    args = parser.parse_args(argv)
    _seed_everything(args.seed)

    from pathlib import Path
    import json
    import networkx as nx

    def _load_graph_from_path(p: str):
        P = Path(p)
        if not P.exists():
            return None
        ext = P.suffix.lower()
        if ext == ".graphml":
            return nx.read_graphml(P)
        if ext in (".gpickle", ".pickle"):
            return nx.read_gpickle(P)
        if ext in (".edgelist", ".txt"):
            return nx.read_edgelist(P, nodetype=str)
        return None

    def _coerce_node_id(x, nodes_set):
        if x in nodes_set:
            return x
        sx = str(x)
        if sx in nodes_set:
            return sx
        try:
            ix = int(x)
            if ix in nodes_set:
                return ix
        except Exception:
            pass
        try:
            fx = float(x)
            if fx.is_integer():
                ix = int(fx)
                if ix in nodes_set:
                    return ix
        except Exception:
            pass
        return None

    def _load_queries_from_json(p: str, G):
        P = Path(p)
        if not P.exists():
            return None
        payload = json.loads(P.read_text(encoding="utf-8"))
        nodes_set = set(G.nodes())
        out = []
        for item in payload.get("queries", []):
            raw_nodes = item.get("nodes", [])
            mapped = []
            for nid in raw_nodes:
                m = _coerce_node_id(nid, nodes_set)
                if m is not None:
                    mapped.append(m)
            if len(mapped) >= 2:
                sub = G.subgraph(mapped).copy()
                if len(sub) >= 2:
                    out.append(sub)
        return out

    G = None;
    queries = None;
    dataset_name = getattr(args, "dataset", "synth")

    maybe = _load_graph_from_path(args.data_path) if args.data_path else None
    if maybe is not None:
        G = maybe
        dataset_name = Path(args.data_path).stem

    if G is None:
        bundle = _load_bundle_from_args(args)
        if bundle is None:
            G = _synth_graph(num_nodes=120, p=0.03, seed=args.seed)
            _inject_uncertainty(G, pmin=args.pmin, flip=args.flip)
            queries = _sample_connected_queries(G, size_min=args.qmin, size_max=args.qmax, num=args.nqueries)
            dataset_name = "synth"
        else:
            G = bundle.graph
            _inject_uncertainty(G, pmin=args.pmin, flip=args.flip)
            queries = bundle.queries if bundle.queries else _sample_connected_queries(
                G, size_min=args.qmin, size_max=args.qmax, num=args.nqueries
            )
            dataset_name = args.dataset

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.queries_path:
        q_from_file = _load_queries_from_json(args.queries_path, G)
        if q_from_file: queries = q_from_file

    # CSV
    header = ["dataset", "method", "median_ms", "iqr_ms", "states_med", "prune_rate",
              "success_rate", "best_loglik_med", "mem_mb", "timeout", "ub_gap_p50", "ub_gap_p90"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = []

    for method in methods:
        per_run_times, per_run_states, per_run_prune = [], [], []
        per_run_bestll, per_run_success, per_run_timeout = [], [], []
        per_run_mem, per_run_ub50, per_run_ub90 = [], [], []

        # method dispatcher
        if method == "vf2":
            def run_engine(G, Q, timeout):
                return run_vf2_wrapper(G, Q, timeout=timeout)
        elif method == "vf2pp":
            def run_engine(G, Q, timeout):
                return run_vf2pp_wrapper(G, Q, timeout=timeout)
        elif method == "vf2bin":
            def run_engine(G, Q, timeout):
                return run_vf2bin_wrapper(G, Q, timeout=timeout, bin_threshold=args.bin_threshold)
        elif method == "vf2prob":
            def run_engine(G, Q, timeout):
                return run_vf2prob_wrapper(G, Q, timeout=timeout,
                                           two_pass=args.vf2prob_two_pass,
                                           sample_every=args.vf2prob_sample_every,
                                           ub_mode=args.vf2prob_ub,
                                           search=args.vf2prob_search)
        else:
            raise ValueError(f"Unknown method: {method}")

        for Q in queries:
            for _ in range(args.runs):
                if method == "vf2prob":
                    res = run_engine(G, Q, timeout=args.timeout)
                    elapsed_ms = float(res.pop("elapsed_ms", float("nan")))
                    nm = _norm_metrics(res)
                    timed_out = int(nm["timed_out"])
                else:
                    t0 = time.time()
                    raw = run_engine(G, Q, timeout=args.timeout)
                    elapsed_ms = (time.time() - t0) * 1000.0
                    nm = _norm_metrics(raw)
                    timed_out = int(nm["timed_out"])

                if method in ("vf2", "vf2pp", "vf2bin"):
                    success = 1.0 if (nm.get("solutions_found", 1) >= 1 and not timed_out) else 0.0
                else:
                    success = 1.0 if (
                                (nm.get("solutions_found", 0) >= 1 or nm.get("best_loglik", float("-inf")) > float(
                                    "-inf")) and not timed_out) else 0.0

                mem_mb = float("nan")
                if psutil is not None:
                    try:
                        mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    except Exception:
                        mem_mb = float("nan")

                per_run_times.append(float(elapsed_ms))
                per_run_states.append(float(nm["states"]))
                per_run_prune.append(float(nm["prune_rate"]))
                per_run_bestll.append(float(nm["best_loglik"]))
                per_run_success.append(float(success))
                per_run_timeout.append(int(timed_out))
                per_run_mem.append(float(mem_mb))
                per_run_ub50.append(float(nm.get("ub_gap_p50", float("nan"))))
                per_run_ub90.append(float(nm.get("ub_gap_p90", float("nan"))))

            method_label = method
            if method == "vf2prob":
                ub_suf = {"node": "node", "assign": "assign"}.get(args.vf2prob_ub, args.vf2prob_ub)
                srch_suf = {"dfs": "dfs", "astar": "astar"}.get(args.vf2prob_search, args.vf2prob_search)
                # -> "vf2prob", "vf2prob-astar", "vf2prob-assign", "vf2prob-astar-assign"
                method_label = "vf2prob"
                if srch_suf == "astar":
                    method_label += "-astar"
                if ub_suf == "assign":
                    method_label += "-assign"

            if method == "vf2prob" and args.tag:
                method_label += f"+{args.tag}"

        row = {
            "dataset": dataset_name,
            "method": method_label,
            "median_ms": _median(per_run_times),
            "iqr_ms": _iqr(per_run_times),
            "states_med": _median(per_run_states),
            "prune_rate": _median(per_run_prune),
            "success_rate": _median(per_run_success),
            "best_loglik_med": _median(per_run_bestll),
            "mem_mb": _median(per_run_mem),
            "timeout": int(any(per_run_timeout)),
            "ub_gap_p50": _median(per_run_ub50),
            "ub_gap_p90": _median(per_run_ub90),
        }
        row = {k: _fmt_cell(k, v) for k, v in row.items()}
        new_rows.append(row)

    mode = args.write_mode
    if mode == "overwrite":
        _csv_write_rows_atomic(out_path, header, new_rows)
    elif mode == "append":
        _csv_append_rows(out_path, header, new_rows)
    else:
        old = _csv_read_rows(out_path, header)
        bykey = {}
        for r in old:
            bykey[(r.get("dataset", ""), r.get("method", ""))] = r
        for r in new_rows:
            bykey[(r.get("dataset", ""), r.get("method", ""))] = r
        merged = list(bykey.values())

        merged.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("method", ""))))
        _csv_write_rows_atomic(out_path, header, merged)

    print(f"Wrote results to {out_path} (mode={mode})")
    return 0


import math


def _fmt_cell(col: str, v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    if isinstance(v, float):
        c = col.lower()
        if c.endswith("_ms") or c in {"median_ms", "iqr_ms"}:
            return f"{v:.3f}"
        if c.endswith("_mb") or ("mem" in c) or ("_rate" in c) or ("states_med" in c):
            return f"{v:.2f}"
        return f"{v:.6f}"
    return v


if __name__ == "__main__":
    raise SystemExit(main())
