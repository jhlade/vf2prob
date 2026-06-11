# VF2-Prob, Jan Hladěna, FIM UHK
from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx
import numpy as np


@dataclass
class GraphBundle:
    G: nx.Graph
    node_labels: Dict[int, int]  # node -> label id
    edge_labels: Dict[Tuple[int, int], int]  # (u<=v) -> label id (optional)
    edge_p: Dict[Tuple[int, int], float]  # (u<=v) -> existence probability


def _canon(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def er_graph(n: int, avg_deg: int, seed: int = 0) -> nx.Graph:
    p = max(min(avg_deg / max(n - 1, 1), 1.0), 0.0)
    return nx.fast_gnp_random_graph(n, p, seed=seed)


def ba_graph(n: int, m_attach: int, seed: int = 0) -> nx.Graph:
    rng = np.random.default_rng(seed)
    return nx.barabasi_albert_graph(n, max(1, m_attach), seed=int(rng.integers(1 << 31)))


def add_labels(G: nx.Graph, nlabels: int = 8, elabels: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)
    node_labels = {u: int(rng.integers(nlabels)) for u in G.nodes()}
    edge_labels: Dict[Tuple[int, int], int] = {}
    if elabels > 0:
        for u, v in G.edges():
            edge_labels[_canon(u, v)] = int(rng.integers(elabels))
    return node_labels, edge_labels


def apply_noise(node_labels: Dict[int, int], rv: float, nlabels: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    noisy = dict(node_labels)
    if rv <= 0:
        return noisy
    nodes = list(noisy.keys())
    k = int(round(rv * len(nodes)))
    if k <= 0:
        return noisy
    flip = set(rng.choice(nodes, size=k, replace=False))
    for u in flip:
        add = 1 + int(rng.integers(max(1, nlabels - 1)))
        noisy[u] = (noisy[u] + add) % nlabels
    return noisy


def uncertain_edge_p(G: nx.Graph, base_p: float = 0.8):
    return {_canon(u, v): float(base_p) for u, v in G.edges()}


def sample_query(G: nx.Graph, node_labels: Dict[int, int], k: int, seed: int = 0):
    """
    Sample a connected induced subgraph Q of size k from G (guaranteeing at least one embedding).
    Nodes of Q are relabeled to 0..k-1 for convenience.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if len(nodes) < k:
        raise ValueError("G too small for requested k")

    for _ in range(50):
        start = int(rng.choice(nodes))
        visited = [start]
        frontier = list(G.neighbors(start))
        seen = set(visited)
        while len(visited) < k and frontier:
            u = int(rng.choice(frontier))
            if u not in seen:
                visited.append(u)
                seen.add(u)
                for w in G.neighbors(u):
                    if w not in seen:
                        frontier.append(w)
            frontier = [w for w in frontier if w not in seen]
        if len(visited) >= k:
            pick = visited[:k]
            break
    else:
        # fallback: take largest component and BFS from a random node inside
        comp = max(nx.connected_components(G), key=len)
        sub = G.subgraph(comp).copy()
        # BFS until k nodes
        s = int(rng.choice(list(sub.nodes())))
        pick = []
        for u in nx.bfs_tree(sub, s):
            pick.append(u)
            if len(pick) == k:
                break

    Q = G.subgraph(pick).copy()
    qlabels = {i: node_labels[i] for i in Q.nodes()}
    # relabel Q nodes to 0..k-1
    mapping = {u: i for i, u in enumerate(Q.nodes())}
    Q = nx.relabel_nodes(Q, mapping, copy=True)
    qlabels = {mapping[u]: lab for u, lab in qlabels.items()}
    return Q, qlabels


def build_bundle(kind: str, n: int, avg_deg: int, nlabels: int, elabels: int,
                 rv: float, edge_p: float, seed: int) -> GraphBundle:
    if kind.upper() == "ER":
        G = er_graph(n, avg_deg, seed)
    elif kind.upper() == "BA":
        G = ba_graph(n, max(1, avg_deg // 2), seed)
    else:
        raise ValueError("kind must be 'ER' or 'BA'")
    nl, el = add_labels(G, nlabels, elabels, seed)
    nl_noisy = apply_noise(nl, rv, nlabels, seed + 1)
    ep = uncertain_edge_p(G, edge_p)
    return GraphBundle(G=G, node_labels=nl_noisy, edge_labels=el, edge_p=ep)
