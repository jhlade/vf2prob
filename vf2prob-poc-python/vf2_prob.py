# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
"""
VF2-Prob: branch-and-bound SGI with probabilistic (nonlinear) compatibilities.

- Candidate generation: purely structural (no hard label filter); attributes are soft.
- Pruning UB (admissible): node bound (precomputed per-node maxima) or assignment
  bound (Hungarian over structurally feasible pairs, with the bound-preserving
  top-n column reduction; deterministic fallback to the node bound on dead or
  oversized states — never on wall-clock time, so explored-state counts are
  machine-independent).
- Report UB (informative): node bonus + edge bonus to already mapped neighbors.
- Streaming gap metrics: record on EVERY UB evaluation (reservoir sampling cap), plus root sample.
- 2-pass friendly: when collect_metrics=False/compute_report_ub=False, metrics overhead ~0.
"""

import random
import time
from typing import Dict, Set, Tuple, Optional, List

import networkx as nx

from compat import node_compat_label, edge_compat_prob, safe_log

try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _canon(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


class ProbSGI:
    def __init__(
            self,
            G: nx.Graph,
            Q: nx.Graph,
            nl_G: Dict[int, int],
            nl_Q: Dict[int, int],
            el_G: Dict[Tuple[int, int], int],
            edge_p: Dict[Tuple[int, int], float],
            el_Q: Optional[Dict[Tuple[int, int], int]] = None,
            timeout_s: float = 120.0,
            use_bound: bool = True,
            use_node: bool = True,
            use_edge: bool = True,
            sample_every: int = 300,  # TODO stride pro UB sampling; efektivně na 1
            k_mismatch: int = 16,
            collect_metrics: bool = True,
            compute_report_ub: bool = True,
    ) -> None:
        self.G, self.Q = G, Q
        self.nl_G, self.nl_Q = nl_G, nl_Q
        self.el_G, self.edge_p = el_G, edge_p
        # Query edge labels. When omitted, fall back to looking the query edge
        # up in el_G — correct only for queries that inherit the data graph's
        # node ids (the harness's sampled motifs); external queries should pass
        # their own el_Q.
        self.el_Q = el_Q if el_Q is not None else el_G
        self.timeout_s = float(timeout_s)
        self.use_bound = bool(use_bound)
        self.use_node = bool(use_node)
        self.use_edge = bool(use_edge)
        # optional modes (non-breaking): can be changed from outside
        self.ub_mode = getattr(self, 'ub_mode', 'node')  # 'node' or 'assign'
        self.search_mode = getattr(self, 'search_mode', 'dfs')  # 'dfs' | 'astar'


        self.sample_every = 1
        self._gap_cap = 4096  # cap

        self.k_mismatch = int(k_mismatch)
        self.topk_screen = None  # None => exact (all structurally feasible candidates)
        # ordering ablation: when False, candidates are tried in a fixed id order
        # (no gain-based ordering) -- used to isolate the effect of the heuristic.
        self.order_by_gain = True
        self.collect_metrics = bool(collect_metrics)
        self.compute_report_ub = bool(compute_report_ub)

        # stats
        self.states = 0
        self.pruned = 0

        # fail-first (deg, label id)
        self.Q_nodes: List[int] = list(Q.nodes())
        self.Q_nodes.sort(key=lambda q: (-Q.degree[q], nl_Q[q], str(q)))

        # cache
        self._nc_cache: Dict[Tuple[int, int], float] = {}
        self._ec_cache: Dict[Tuple[int, int, int, int], float] = {}

        # max. node-compat (UB)
        self._max_node_compat: Dict[int, float] = {}
        for q in self.Q_nodes:
            dq = self.Q.degree[q]
            best = 0.0
            for u in self.G.nodes():
                if self.G.degree[u] < dq:
                    continue
                v = node_compat_label(self.nl_Q[q], self.nl_G[u]) if self.use_node else 1.0
                if v > best: best = v
            self._max_node_compat[q] = best

        # UB / tightness tracking
        self.root_ub_clamped: Optional[float] = None
        self.root_ub_report: Optional[float] = None

        # streaming gap metrics
        self._gap_buf: List[float] = []
        self._gap_seen = 0
        self._ub_calls = 0


    def node_compat(self, q: int, u: int) -> float:
        if not self.use_node:
            return 1.0
        k = (q, u)
        v = self._nc_cache.get(k)
        if v is None:
            v = node_compat_label(self.nl_Q[q], self.nl_G[u])
            self._nc_cache[k] = v
        return v

    def edge_compat(self, qi: int, qj: int, ui: int, uj: int) -> float:
        if not self.use_edge:
            return 1.0
        k = (qi, qj, ui, uj)
        v = self._ec_cache.get(k)
        if v is None:
            lm = (self.el_Q.get(_canon(qi, qj)) == self.el_G.get(_canon(ui, uj)))
            v = edge_compat_prob(lm, self.edge_p.get(_canon(ui, uj), 1.0))
            self._ec_cache[k] = v
        return v

    def _feasible_candidates(self, q, deg_q, mapped_neighbors, used_u):
        """All VF2-feasible candidates for q (no hard label filter; attributes
        are soft). Edges to already-mapped neighbours must be preserved.
        Ordered by descending immediate log-gain so branch-and-bound finds a
        strong incumbent early. Exact unless self.topk_screen is set."""
        if mapped_neighbors:
            imgs = [un for _, un in mapped_neighbors]
            common = set(self.G.neighbors(imgs[0]))
            for un in imgs[1:]:
                common &= set(self.G.neighbors(un))
            pool = [u for u in common if (u not in used_u) and (self.G.degree[u] >= deg_q)]
        else:
            pool = [u for u in self.G.nodes() if (u not in used_u) and (self.G.degree[u] >= deg_q)]

        def gain(u):
            g = safe_log(self.node_compat(q, u)) if self.use_node else 0.0
            if self.use_edge:
                for qn, un in mapped_neighbors:
                    g += safe_log(self.edge_compat(q, qn, u, un))
            return g

        # Deterministic order: strongest immediate gain first, with a stable
        # node-id tiebreak so results are reproducible regardless of set/hash
        # iteration order (otherwise the incumbent found first — and thus, under
        # a timeout, the reported result — would vary run to run). With
        # order_by_gain disabled, fall back to a pure id order (ablation).
        if self.order_by_gain:
            pool.sort(key=lambda u: (-gain(u), str(u)))
        else:
            pool.sort(key=lambda u: str(u))
        if self.topk_screen is not None and len(pool) > self.topk_screen:
            pool = pool[:self.topk_screen]
        return pool

    # UB: pruning vs report
    def _UB_pruning_nodes_only(self, remaining: List[int]) -> float:
        ub_c = 0.0
        for q in remaining:
            best_nv = self._max_node_compat.get(q, 1e-12)
            ub_c += safe_log(best_nv if best_nv > 0 else 1e-12)
        return ub_c

    def _UB_report_nodes_edges(self, i: int, mapping: Dict[int, int], used_u: Set[int]) -> float:
        remaining = self.Q_nodes[i:]
        ub_r = 0.0
        for q in remaining:
            best_n = 0.0
            for u in self.G.nodes():
                if u in used_u:
                    continue
                if self.G.degree[u] < self.Q.degree[q]:
                    continue
                val = self.node_compat(q, u) if self.use_node else 1.0
                if val > best_n:
                    best_n = val
            ub_r += safe_log(best_n if best_n > 0 else 1e-12)
        for q in remaining:
            for qn in self.Q.neighbors(q):
                if qn not in mapping:
                    continue
                un = mapping[qn]
                best_e = 0.0
                for u in self.G.nodes():
                    if u in used_u:
                        continue
                    if self.G.degree[u] < self.Q.degree[q]:
                        continue
                    if self.G.has_edge(u, un):
                        val = self.edge_compat(q, qn, u, un)
                        if val > best_e:
                            best_e = val
                ub_r += safe_log(best_e if best_e > 0 else 1e-12)
        return ub_r

    _ASSIGN_MAX_FRONTIER = 64  # deterministic guard, mirrors the C++ implementation

    def _UB_assign(self, mapping: Dict[int, int], used_u: Set[int]) -> Tuple[float, float]:
        """Assignment-based upper bound (admissible).
        Uses candidates that are structurally consistent w.r.t. already mapped neighbors
        (edges to mapped neighbors must exist) and degree filter; no label filtering is applied.

        Fallback to the node-only UB is deterministic — a dead state (a row with no
        candidates), no injective assignment (n > m), an unavailable scipy solver, or a
        frontier beyond the guard size — never a function of wall-clock time, so
        explored-state counts do not depend on machine speed. The top-n column
        reduction keeps the matching small without changing the bound: in an optimal
        injective assignment a row is outranked on at most n-1 of its columns, so one
        of its top-n columns is always free.
        Returns (ub_prune, ub_report).
        """
        unmapped_q = [q for q in self.Q_nodes if q not in mapping]
        n = len(unmapped_q)
        if n == 0:
            return 0.0, 0.0

        def _node_fallback():
            ub_c = self._UB_pruning_nodes_only(unmapped_q)
            ub_r = self._UB_report_nodes_edges(len(mapping), mapping, used_u) if self.compute_report_ub else 0.0
            return ub_c, ub_r

        if n > self._ASSIGN_MAX_FRONTIER or not _HAS_SCIPY:
            return _node_fallback()

        rows: List[List[Tuple[float, int]]] = []  # per row: [(weight, candidate u)]
        cand_index: Dict[int, int] = {}
        for q in unmapped_q:
            deg_q = self.Q.degree[q]
            mapped_neighbors = [(qn, mapping[qn]) for qn in self.Q.neighbors(q) if qn in mapping]
            if mapped_neighbors:
                nbrs = set(self.G.neighbors(mapped_neighbors[0][1]))
                for _, un in mapped_neighbors[1:]:
                    nbrs &= set(self.G.neighbors(un))
                pool = [u for u in nbrs if (u not in used_u) and (self.G.degree[u] >= deg_q)]
            else:
                pool = [u for u in self.G.nodes() if (u not in used_u) and (self.G.degree[u] >= deg_q)]
            if not pool:
                return _node_fallback()  # dead state -> node bound is safe

            weighted = []
            for u in pool:
                w = safe_log(self.node_compat(q, u)) if self.use_node else 0.0
                if self.use_edge:
                    # edges to mapped neighbors exist by candidate construction
                    for qn, un in mapped_neighbors:
                        w += safe_log(self.edge_compat(q, qn, u, un))
                weighted.append((w, u))
            # top-n column reduction (bound-preserving; deterministic tie-break by id)
            if len(weighted) > n:
                weighted.sort(key=lambda t: (-t[0], str(t[1])))
                weighted = weighted[:n]
            for _, u in weighted:
                if u not in cand_index:
                    cand_index[u] = len(cand_index)
            rows.append(weighted)

        m = len(cand_index)
        if n > m:
            return _node_fallback()  # no injective assignment exists; dead state

        BIG = 1e9
        C = [[BIG] * m for _ in range(n)]
        for i in range(n):
            for w, u in rows[i]:
                C[i][cand_index[u]] = -w  # maximize w <=> minimize -w

        row_ind, col_ind = linear_sum_assignment(C)
        ub = 0.0
        for i, j in zip(row_ind, col_ind):
            c = C[i][j]
            if c >= BIG * 0.5:
                return _node_fallback()  # a row was forced onto a forbidden column
            ub += -c

        return ub, ub

    def UB(self, i: int, mapping: Dict[int, int], used_u: Set[int]) -> Tuple[float, float]:
        if getattr(self, 'ub_mode', 'node') == 'assign':
            # pruning UB
            ub_c, _ = self._UB_assign(mapping, used_u)
            # report UB
            ub_r = self._UB_report_nodes_edges(i, mapping, used_u) if self.compute_report_ub else 0.0
            return ub_c, ub_r
        # node-only
        remaining = self.Q_nodes[i:]
        ub_c = self._UB_pruning_nodes_only(remaining)
        ub_r = self._UB_report_nodes_edges(i, mapping, used_u)
        return ub_c, ub_r


    def _maybe_take_gap_sample(self, deficit: float) -> None:

        self._gap_seen += 1
        if len(self._gap_buf) < self._gap_cap:
            self._gap_buf.append(deficit)
        else:
            j = random.randint(0, self._gap_seen - 1)
            if j < self._gap_cap:
                self._gap_buf[j] = deficit


    def run(self):
        self.start = time.time()

        mapping: Dict[int, int] = {}
        used_u: Set[int] = set()

        best_map: Optional[Dict[int, int]] = None
        best_score: float = float("-inf")

        # root UB
        ub0_c, ub0_r = self.UB(0, mapping, used_u)
        self.root_ub_clamped = ub0_c
        self.root_ub_report = (ub0_r if self.compute_report_ub else 0.0)

        # root sample
        if self.collect_metrics and self.compute_report_ub:
            root_deficit = max(0.0, - (self.root_ub_report or 0.0))
            self._maybe_take_gap_sample(root_deficit)

        def dfs(i: int, cur_score: float) -> bool:
            """Returns True iff the time budget was exhausted (propagated up so
            the caller can stop and the run is honestly flagged as timed out)."""
            nonlocal best_score, best_map

            if self.timeout_s and (time.time() - self.start) > self.timeout_s:
                return True  # timeout

            if i == len(self.Q_nodes):
                if cur_score > best_score:
                    best_score = cur_score
                    best_map = dict(mapping)
                return False

            if self.use_bound:
                # UB evaluation
                ub_c, ub_r = self.UB(i, mapping, used_u)
                self._ub_calls += 1

                ub_val_prune = cur_score + ub_c
                ub_val_report = cur_score + ub_r


                if self.collect_metrics and self.compute_report_ub:
                    deficit = max(0.0, (max(best_score, cur_score) - ub_val_report))
                    self._maybe_take_gap_sample(deficit)

                if ub_val_prune <= best_score:
                    self.pruned += 1
                    return False

            q = self.Q_nodes[i]
            deg_q = self.Q.degree[q]
            mapped_neighbors = [(qn, mapping[qn]) for qn in self.Q.neighbors(q) if qn in mapping]

            cands = self._feasible_candidates(q, deg_q, mapped_neighbors, used_u)

            for u in cands:
                self.states += 1

                ok = True
                edge_gain = 0.0
                for qn, un in mapped_neighbors:
                    if not self.G.has_edge(u, un):
                        ok = False
                        break
                    edge_gain += safe_log(self.edge_compat(q, qn, u, un)) if self.use_edge else 0.0
                if not ok:
                    self.pruned += 1
                    continue

                node_gain = safe_log(self.node_compat(q, u)) if self.use_node else 0.0
                mapping[q] = u
                used_u.add(u)

                timed_out_deep = dfs(i + 1, cur_score + node_gain + edge_gain)

                used_u.remove(u)
                del mapping[q]

                if timed_out_deep:
                    return True  # propagate timeout up the recursion

            return False

        timed_out = bool(dfs(0, 0.0))

        ub_p50 = float('nan')
        ub_p90 = float('nan')
        if self._gap_buf:
            gsorted = sorted(self._gap_buf)
            k = len(gsorted)
            ub_p50 = gsorted[int(0.5 * (k - 1))]
            ub_p90 = gsorted[int(0.9 * (k - 1))]

        root_deficit = max(0.0, - (self.root_ub_report or 0.0)) if self.compute_report_ub else 0.0
        mid_deficit = self._gap_buf[len(self._gap_buf) // 2] if self._gap_buf else 0.0
        final_deficit = self._gap_buf[-1] if self._gap_buf else 0.0

        counters = {
            "states": int(self.states),
            "pruned": int(self.pruned),
            "best_loglik": float(best_score if isinstance(best_score, float) and best_score > float('-inf') else float('-inf')),
            "root_gap": float(root_deficit),
            "mid_gap": float(mid_deficit),
            "final_gap": float(final_deficit),
            "ub_gap_p50": float(ub_p50),
            "ub_gap_p90": float(ub_p90),
        }
        return best_map, counters, timed_out


class AStarProb(ProbSGI):
    """Best-first (A*) variant over the same admissible UB."""

    def _select_query_node(self, mapping: Dict[int, int], used_u: Set[int]) -> int:
        remaining = [q for q in self.Q_nodes if q not in mapping]
        # MRV: number of structurally feasible candidates (no label filter)
        best = None
        for q in remaining:
            deg_q = self.Q.degree[q]
            mapped_neighbors = [mapping[qn] for qn in self.Q.neighbors(q) if qn in mapping]
            if mapped_neighbors:
                nbrs = set(self.G.neighbors(mapped_neighbors[0]))
                for un in mapped_neighbors[1:]:
                    nbrs &= set(self.G.neighbors(un))
                k = sum(1 for u in nbrs if (u not in used_u) and (self.G.degree[u] >= deg_q))
            else:
                k = sum(1 for u in self.G.nodes() if (u not in used_u) and (self.G.degree[u] >= deg_q))
            key = (k, -self.Q.degree[q], q)
            if best is None or key < best[0]:
                best = (key, q)
        return best[1]

    def run(self):
        import heapq
        self.start = time.time()
        self.states = 0
        self.pruned = 0
        timed_out = False

        mapping: Dict[int, int] = {}
        used_u: Set[int] = set()

        # root UB
        ub0_c, ub0_r = self.UB(0, mapping, used_u)
        self.root_ub_clamped = ub0_c
        self.root_ub_report  = (ub0_r if self.compute_report_ub else 0.0)
        if self.collect_metrics and self.compute_report_ub:
            root_deficit = max(0.0, -(self.root_ub_report or 0.0))
            self._maybe_take_gap_sample(root_deficit)


        pq = []
        seq = 0
        def _push(f: float, g: float, mapping: Dict[int,int], used_u: Set[int]):
            nonlocal seq
            heapq.heappush(pq, ( -f, -g, seq, mapping, used_u ))
            seq += 1

        # f = g + h; start g=0
        _push(ub0_c + 0.0, 0.0, dict(mapping), set(used_u))

        best_score = float("-inf")
        best_map: Optional[Dict[int,int]] = None

        while pq:
            if self.timeout_s and (time.time() - self.start) > self.timeout_s:
                timed_out = True
                break

            neg_f, neg_g, _, mapping, used_u = heapq.heappop(pq)
            g = -neg_g
            # f = -neg_f

            if len(mapping) == len(self.Q_nodes):
                best_score = g
                best_map = dict(mapping)
                break

            q = self._select_query_node(mapping, used_u)
            deg_q = self.Q.degree[q]
            mapped_neighbors = [(qn, mapping[qn]) for qn in self.Q.neighbors(q) if qn in mapping]

            cands = self._feasible_candidates(q, deg_q, mapped_neighbors, used_u)

            for u in cands:
                ok = True
                edge_gain = 0.0
                for qn, un in mapped_neighbors:
                    if not self.G.has_edge(u, un):
                        ok = False
                        break
                    edge_gain += safe_log(self.edge_compat(q, qn, u, un)) if self.use_edge else 0.0
                if not ok:
                    self.pruned += 1
                    continue

                node_gain = safe_log(self.node_compat(q, u)) if self.use_node else 0.0
                g_new = g + node_gain + edge_gain

                mapping_new = dict(mapping); mapping_new[q] = u
                used_u_new = set(used_u); used_u_new.add(u)

                ub_c, ub_r = self.UB(len(mapping_new), mapping_new, used_u_new)
                self._ub_calls += 1

                f = g_new + ub_c

                if self.collect_metrics and self.compute_report_ub:
                    deficit = max(0.0, (max(best_score, g_new) - (g_new + ub_r)))
                    self._maybe_take_gap_sample(deficit)

                if f <= best_score:
                    self.pruned += 1
                    continue

                _push(f, g_new, mapping_new, used_u_new)
                self.states += 1

        ub_p50 = ub_p90 = float('nan')
        if self._gap_buf:
            gsorted = sorted(self._gap_buf)
            k = len(gsorted)
            ub_p50 = gsorted[int(0.5*(k-1))]
            ub_p90 = gsorted[int(0.9*(k-1))]

        root_deficit = max(0.0, -(self.root_ub_report or 0.0)) if self.compute_report_ub else 0.0
        mid_deficit = self._gap_buf[len(self._gap_buf)//2] if self._gap_buf else 0.0
        final_deficit = self._gap_buf[-1] if self._gap_buf else 0.0

        counters = {
            "states": int(self.states),
            "pruned": int(self.pruned),
            "best_loglik": float(best_score if isinstance(best_score, float) and best_score > float('-inf') else float('-inf')),
            "root_gap": float(root_deficit),
            "mid_gap": float(mid_deficit),
            "final_gap": float(final_deficit),
            "ub_gap_p50": float(ub_p50),
            "ub_gap_p90": float(ub_p90),
        }
        return best_map, counters, timed_out


__all__ = ["ProbSGI", "AStarProb"]