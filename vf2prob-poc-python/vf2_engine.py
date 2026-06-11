# VF2-Prob, Jan Hladěna, FIM UHK
import time
from typing import Dict, Set, Tuple, Optional

import networkx as nx


def _feasible(G: nx.Graph, Q: nx.Graph, nl_G: Dict[int, int], nl_Q: Dict[int, int],
              mapping: Dict[int, int], q: int, u: int) -> bool:
    if u in mapping.values(): return False
    if nl_Q[q] != nl_G[u]: return False
    if Q.degree[q] > G.degree[u]: return False

    for qn in Q.neighbors(q):
        if qn in mapping and not G.has_edge(u, mapping[qn]): return False
    return True


def vf2_search(G: nx.Graph, Q: nx.Graph, nl_G: Dict[int, int], nl_Q: Dict[int, int],
               timeout_s: float = 60.0) -> Tuple[Optional[Dict[int, int]], Dict[str, int], bool]:
    """
    Deterministic VF2-style baseline.
    Returns (mapping or None, counters, timed_out).
    counters: {'states': int, 'pruned': int}
    """
    start = time.time()
    counters = {"states": 0, "pruned": 0}
    best: Optional[Dict[int, int]] = None

    Q_nodes = list(Q.nodes())
    Q_nodes.sort(key=lambda x: (-Q.degree[x], nl_Q[x]))

    used_u: Set[int] = set()
    mapping: Dict[int, int] = {}

    def dfs(i: int) -> bool:
        if time.time() - start > timeout_s:
            return False
        if i == len(Q_nodes):
            nonlocal best

            counters["solutions"] = counters.get("solutions", 0) + 1
            if best is None:
                best = dict(mapping)

            return True
        q = Q_nodes[i]

        cand = [u for u in G.nodes()
                if (u not in used_u) and (nl_G[u] == nl_Q[q]) and (G.degree[u] >= Q.degree[q])]
        cand.sort(key=lambda u: G.degree[u])
        for u in cand:
            counters["states"] += 1
            if _feasible(G, Q, nl_G, nl_Q, mapping, q, u):
                mapping[q] = u;
                used_u.add(u)
                ok = dfs(i + 1)
                if ok:
                    return True
                used_u.remove(u);
                del mapping[q]
            else:
                counters["pruned"] += 1
        return False

    _ = dfs(0)
    timed_out = (time.time() - start > timeout_s)
    return best, counters, timed_out
