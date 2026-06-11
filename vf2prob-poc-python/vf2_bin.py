# VF2-Prob, Jan Hladěna, FIM UHK
# vf2_bin.py
from __future__ import annotations

from typing import Any, Dict

import networkx as nx

# VF2 baseline
from vf2_engine import vf2_search  # (G, Q, nl_G, nl_Q, timeout_s) -> (best_map, counters, timed_out)


def _node_label_value(attrs: Dict[str, Any]) -> Any:
    if "label" in attrs: return attrs["label"]
    if "type" in attrs: return attrs["type"]
    if "label_id" in attrs: return attrs["label_id"]
    return None


def _build_shared_node_label_ids(G: nx.Graph, Q: nx.Graph):
    vocab = []
    seen = set()
    for n in G.nodes():
        v = _node_label_value(G.nodes[n])
        if v is not None and v not in seen:
            vocab.append(v);
            seen.add(v)
    for n in Q.nodes():
        v = _node_label_value(Q.nodes[n])
        if v is not None and v not in seen:
            vocab.append(v);
            seen.add(v)
    index = {v: i for i, v in enumerate(vocab)}
    nl_G = {n: index.get(_node_label_value(G.nodes[n]), -1) for n in G.nodes()}
    nl_Q = {n: index.get(_node_label_value(Q.nodes[n]), -1) for n in Q.nodes()}
    return nl_G, nl_Q


def binarize_graph(G: nx.Graph, tau: float = 0.5) -> nx.Graph:
    H = G.__class__()  # Graph/DiGraph

    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    for u, v, d in G.edges(data=True):
        p = d.get("prob", 1.0)
        try:
            p = float(p)
        except Exception:
            p = 1.0
        if p >= tau:
            H.add_edge(u, v, **d)
    return H


def run_vf2_bin(G: nx.Graph, Q: nx.Graph, *, tau: float = 0.5, timeout: float = 60.0) -> Dict[str, Any]:
    H = binarize_graph(G, tau=tau)
    nl_H, nl_Q = _build_shared_node_label_ids(H, Q)

    best_map, counters, timed_out = vf2_search(H, Q, nl_H, nl_Q, timeout_s=timeout)

    res = {
        "states": int(counters.get("states", counters.get("states_visited", 0))),
        "pruned": int(counters.get("pruned", counters.get("prunes", 0))),
        "solutions_found": 1 if best_map else 0,
        "timed_out": bool(timed_out),
        "best_loglik": float("-inf"),  # VF2-Bin -inf
        "ub_gap_p50": float("nan"),
        "ub_gap_p90": float("nan"),
    }
    return res
