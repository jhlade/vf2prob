# VF2-Prob, Jan Hladěna, FIM UHK
# vf2pp_engine.py
# VF2++-like subgraph isomorphism engine (Python reimplementation).
# Focus: refined node ordering + forward-checking feasibility + Tin/Tout frontier.
# Works with NetworkX >= 3.5 (undirected or directed graphs).
#
# Public API:
#   - VF2PP(Gq, Gt, node_label_attr=None, edge_label_attr=None).match() -> generator of dicts (u in Gq -> v in Gt)
#   - vf2pp_all_isomorphisms(Gq, Gt, **kwargs) -> list of mappings
#
# Statistics:
#   engine.stats.states_visited
#   engine.stats.prunes
#   engine.stats.nodes_mapped_max
#   engine.stats.runtime_seconds()

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Hashable

import networkx as nx

Node = Hashable


def _get_node_label(G: nx.Graph, u: Node, node_label_attr: Optional[str], default: str = "_") -> str:
    if node_label_attr is None:
        return default
    return G.nodes[u].get(node_label_attr, default)


def _get_edge_label(G: nx.Graph, u: Node, v: Node, edge_label_attr: Optional[str], default: str = "_") -> str:
    if edge_label_attr is None:
        return default
    data = G.get_edge_data(u, v, default={})
    if isinstance(data, dict) and edge_label_attr in data:
        return data[edge_label_attr]
    # For multigraphs, NetworkX returns a dict of dicts; take any (all must match anyway in exact mode)
    if isinstance(data, dict) and len(data) and isinstance(next(iter(data.values())), dict):
        any_key = next(iter(data))
        return data[any_key].get(edge_label_attr, default)
    return default


def _degree_tuple(G: nx.Graph, u: Node) -> Tuple[int, int]:
    if G.is_directed():
        return (G.out_degree(u), G.in_degree(u))
    else:
        d = G.degree(u)
        return (d, d)


@dataclass
class Stats:
    start_ts: float = field(default_factory=time.time)
    states_visited: int = 0
    prunes: int = 0
    nodes_mapped_max: int = 0

    def bump_state(self) -> None:
        self.states_visited += 1

    def bump_prune(self, k: int = 1) -> None:
        self.prunes += k

    def note_depth(self, d: int) -> None:
        if d > self.nodes_mapped_max:
            self.nodes_mapped_max = d

    def runtime_seconds(self) -> float:
        return time.time() - self.start_ts


@dataclass
class GraphParams:
    Gq: nx.Graph
    Gt: nx.Graph
    node_label_attr: Optional[str] = None
    edge_label_attr: Optional[str] = None
    default_label: str = "_"

    # Precomputed properties
    q_labels: Dict[Node, str] = field(default_factory=dict)
    t_labels: Dict[Node, str] = field(default_factory=dict)
    q_deg: Dict[Node, Tuple[int, int]] = field(default_factory=dict)
    t_deg: Dict[Node, Tuple[int, int]] = field(default_factory=dict)
    label_counts_q: Dict[str, int] = field(default_factory=dict)
    label_counts_t: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.q_labels = {u: _get_node_label(self.Gq, u, self.node_label_attr, self.default_label) for u in self.Gq}
        self.t_labels = {v: _get_node_label(self.Gt, v, self.node_label_attr, self.default_label) for v in self.Gt}
        self.q_deg = {u: _degree_tuple(self.Gq, u) for u in self.Gq}
        self.t_deg = {v: _degree_tuple(self.Gt, v) for v in self.Gt}

        for lab in self.q_labels.values():
            self.label_counts_q[lab] = self.label_counts_q.get(lab, 0) + 1
        for lab in self.t_labels.values():
            self.label_counts_t[lab] = self.label_counts_t.get(lab, 0) + 1


@dataclass
class State:
    mapping: Dict[Node, Node] = field(default_factory=dict)  # u in Gq -> v in Gt
    rev_mapping: Dict[Node, Node] = field(default_factory=dict)  # v in Gt -> u in Gq

    Tin_q: Set[Node] = field(default_factory=set)  # frontier nodes in Gq (adjacent to mapped)
    Tout_q: Set[Node] = field(default_factory=set)
    Tin_t: Set[Node] = field(default_factory=set)  # frontier nodes in Gt (adjacent to mapped)
    Tout_t: Set[Node] = field(default_factory=set)

    free_q: Set[Node] = field(default_factory=set)
    free_t: Set[Node] = field(default_factory=set)

    _stack_T_changes: List[Tuple[Set[Node], Set[Node], Set[Node], Set[Node]]] = field(default_factory=list)

    def push_T_snapshot(self) -> None:
        self._stack_T_changes.append((
            set(self.Tin_q), set(self.Tout_q), set(self.Tin_t), set(self.Tout_t)
        ))

    def pop_T_snapshot(self) -> None:
        Tin_q, Tout_q, Tin_t, Tout_t = self._stack_T_changes.pop()
        self.Tin_q, self.Tout_q, self.Tin_t, self.Tout_t = Tin_q, Tout_q, Tin_t, Tout_t


class VF2PP:
    """
    VF2++-like subgraph isomorphism engine.

    Parameters
    ----------
    Gq : nx.Graph
        Query (pattern) graph. Will be mapped into Gt.
    Gt : nx.Graph
        Target (world) graph.
    node_label_attr : Optional[str]
        Node attribute name for labels. If None, all nodes share a default label.
    edge_label_attr : Optional[str]
        Edge attribute name for labels. If None, edge labels are ignored (exact existence check only).
    """

    def __init__(
            self,
            Gq: nx.Graph,
            Gt: nx.Graph,
            node_label_attr: Optional[str] = None,
            edge_label_attr: Optional[str] = None,
            default_label: str = "_",
    ):
        self.params = GraphParams(Gq, Gt, node_label_attr=node_label_attr, edge_label_attr=edge_label_attr,
                                  default_label=default_label)
        self.stats = Stats()

        # Optional wall-clock deadline (absolute time.time() timestamp). When
        # set, match() aborts between states even if no solution has been
        # yielded yet — without it a barren stretch of the search could run
        # arbitrarily past the caller's time limit.
        self.deadline_ts: Optional[float] = None
        self.timed_out: bool = False

        self.order_q: List[Node] = self._compute_static_order()

    # public API

    def match(self) -> Iterator[Dict[Node, Node]]:
        """ u_in_Gq -> v_in_Gt"""
        gp = self.params
        Gq, Gt = gp.Gq, gp.Gt

        # quick fail
        if Gq.number_of_nodes() == 0:
            yield {}
            return
        if Gt.number_of_nodes() == 0 or Gt.number_of_nodes() < Gq.number_of_nodes():
            return

        for lab, cnt_q in gp.label_counts_q.items():
            if gp.label_counts_t.get(lab, 0) < cnt_q:
                return

        if not self._degree_multiset_dominates():
            return

        st = State()
        st.free_q = set(Gq.nodes())
        st.free_t = set(Gt.nodes())
        st.Tin_q.clear();
        st.Tout_q.clear();
        st.Tin_t.clear();
        st.Tout_t.clear()

        stack: List[Tuple[Node, Iterator[Node]]] = []

        u0 = self._select_next_q(st)
        cand0 = self._find_candidates_for(u0, st)
        stack.append((u0, iter(cand0)))

        while stack:
            if (self.deadline_ts is not None and (self.stats.states_visited & 0xFF) == 0
                    and time.time() > self.deadline_ts):
                self.timed_out = True
                return
            u, cand_iter = stack[-1]
            try:
                v = next(cand_iter)
            except StopIteration:
                stack.pop()
                if stack:
                    self._pop_mapping_and_restore(st)
                continue

            self.stats.bump_state()

            if self._feasible(u, v, st):
                self._push_mapping_and_update(u, v, st)
                self.stats.note_depth(len(st.mapping))

                if len(st.mapping) == gp.Gq.number_of_nodes():
                    yield dict(st.mapping)
                    self._pop_mapping_and_restore(st)
                else:
                    u_next = self._select_next_q(st)
                    cands = self._find_candidates_for(u_next, st)
                    stack.append((u_next, iter(cands)))
            else:
                self.stats.bump_prune()

    def all_isomorphisms(self) -> List[Dict[Node, Node]]:
        return list(self.match())

    def _compute_static_order(self) -> List[Node]:
        """
        Heuristic static order:
        sort by:
          - descending degree sum (out+in for DiGraph, 2*deg for Graph),
          - descending 'label rarity' (rarer label first),
          - tie-breaker: node id order.
        """
        gp = self.params
        rarity = {lab: 1.0 / (1 + gp.label_counts_q.get(lab, 0)) for lab in gp.label_counts_q}

        def key(u: Node) -> Tuple[int, float, int]:
            dout, din = gp.q_deg[u]
            degsum = dout + din
            lab = gp.q_labels[u]
            return (degsum, rarity.get(lab, 0.0), -hash(u))

        return sorted(gp.Gq.nodes(), key=key, reverse=True)

    def _select_next_q(self, st: State) -> Node:
        """
        VF2++ strategy: pick from frontier (Tin_q) if available; otherwise use static order among free_q.
        Frontier choice: pick the frontier node with max (#mapped neighbors, degree, label rarity).
        """
        gp = self.params
        if st.Tin_q:
            rarity = {lab: 1.0 / (1 + gp.label_counts_q.get(lab, 0)) for lab in gp.label_counts_q}

            def score(u: Node) -> Tuple[int, int, float, int]:
                mapped_nbrs = 0
                for w in gp.Gq.predecessors(u) if gp.Gq.is_directed() else gp.Gq.neighbors(u):
                    if w in st.mapping:
                        mapped_nbrs += 1
                dout, din = gp.q_deg[u]
                lab = gp.q_labels[u]
                return (mapped_nbrs, dout + din, rarity.get(lab, 0.0), -hash(u))

            u_best = max(st.Tin_q, key=score)
            return u_best
        for u in self.order_q:
            if u in st.free_q:
                return u
        return next(iter(st.free_q))

    def _find_candidates_for(self, u: Node, st: State) -> List[Node]:
        """
        Candidate set for u under VF2++ rules:
          - If u in Tin_q: candidates from Tin_t that match label/degree and not mapped.
          - Else: candidates from free_t matching label/degree.
        Further filtering: cheap degree domination checks and (optional) edge-label prefilters on already mapped neighbors.
        """
        gp = self.params
        lab_u = gp.q_labels[u]
        dout_u, din_u = gp.q_deg[u]

        pool: Iterable[Node]
        if u in st.Tin_q and st.Tin_t:
            pool = (v for v in st.Tin_t if v not in st.rev_mapping)
        else:
            pool = (v for v in st.free_t)

        cands: List[Node] = []
        for v in pool:
            if gp.t_labels[v] != lab_u:
                continue
            dout_v, din_v = gp.t_deg[v]
            if dout_v < dout_u or din_v < din_u:
                continue
            cands.append(v)

        # heuristics ordering of candidates: prefer those preserving neighborhood structure
        def ckey(v: Node) -> Tuple[int, int, int]:
            # how many mapped neighbors in T coincide with mapped neighbors of u in Q
            concord = 0
            mapped_nbrs_q = self._mapped_neighbors_in_Q(u, st)
            for wq in mapped_nbrs_q:
                vt = st.mapping.get(wq, None)
                if vt is None:
                    continue
                if gp.Gt.has_edge(v, vt) or gp.Gt.has_edge(vt, v):
                    concord += 1
            # higher total degree first; tie-breaker by how many Tin_t neighbors (more connectivity)
            dv = gp.t_deg[v][0] + gp.t_deg[v][1]
            frontier_nbrs = 0
            for w in gp.Gt.neighbors(v) if not gp.Gt.is_directed() else list(gp.Gt.predecessors(v)) + list(
                    gp.Gt.successors(v)):
                if w in st.Tin_t:
                    frontier_nbrs += 1
            return (concord, dv, frontier_nbrs)

        cands.sort(key=ckey, reverse=True)
        return cands

    def _push_mapping_and_update(self, u: Node, v: Node, st: State) -> None:
        """ mapping by (u->v) and update Tin/Tout/frontiers"""
        gp = self.params
        Gq, Gt = gp.Gq, gp.Gt

        st.push_T_snapshot()

        st.mapping[u] = v
        st.rev_mapping[v] = u
        st.free_q.discard(u)
        st.free_t.discard(v)

        def iter_q_succpred(x: Node) -> Iterable[Node]:
            if Gq.is_directed():
                for a in Gq.predecessors(x):
                    yield a
                for b in Gq.successors(x):
                    yield b
            else:
                for a in Gq.neighbors(x):
                    yield a

        def iter_t_succpred(y: Node) -> Iterable[Node]:
            if Gt.is_directed():
                for a in Gt.predecessors(y):
                    yield a
                for b in Gt.successors(y):
                    yield b
            else:
                for a in Gt.neighbors(y):
                    yield a

        st.Tin_q.discard(u)
        st.Tout_q.discard(u)
        for w in iter_q_succpred(u):
            if w in st.free_q:
                st.Tin_q.add(w)

        st.Tin_t.discard(v)
        st.Tout_t.discard(v)
        for z in iter_t_succpred(v):
            if z in st.free_t:
                st.Tin_t.add(z)

    def _pop_mapping_and_restore(self, st: State) -> None:
        if not st.mapping:
            return

        u_last, v_last = next(reversed(st.mapping.items()))
        st.mapping.pop(u_last)
        st.rev_mapping.pop(v_last)

        st.free_q.add(u_last)
        st.free_t.add(v_last)

        st.pop_T_snapshot()

    def _feasible(self, u: Node, v: Node, st: State) -> bool:
        """
        Feasibility rules:
          1) Injectivity: v not already used.
          2) Label equality on nodes; degree domination (checked earlier in candidates).
          3) Neighbor consistency: for every mapped neighbor u' of u, edges must exist consistently to v and mapping[u'] (edge labels if enabled).
          4) Frontier size constraints (Tin-constraints): do not exceed capacity of available T frontier.
          5) Forward checking (lightweight): each yet-unmapped neighbor of u must have at least one potential host in T (quick filter).
        """
        gp = self.params
        Gq, Gt = gp.Gq, gp.Gt

        # 1) Injectivity
        if v in st.rev_mapping:
            return False

        # 2) Node label equality (redundant with candidate filter but keep for safety)
        if gp.t_labels[v] != gp.q_labels[u]:
            return False

        # 3) Neighbor consistency against already mapped neighbors
        # For directed graphs: check both dir; for undirected: single check
        def edges_match(x1: Node, x2: Node) -> bool:
            if not gp.edge_label_attr:
                if Gq.is_directed() and Gt.is_directed():
                    cond1 = (not Gq.has_edge(u, x1)) or Gt.has_edge(v, x2)
                    cond2 = (not Gq.has_edge(x1, u)) or Gt.has_edge(x2, v)
                    return cond1 and cond2
                else:
                    return (not Gq.has_edge(u, x1)) or Gt.has_edge(v, x2)
            else:
                if Gq.is_directed() and Gt.is_directed():
                    if Gq.has_edge(u, x1):
                        if not Gt.has_edge(v, x2):
                            return False
                        if _get_edge_label(Gq, u, x1, gp.edge_label_attr) != _get_edge_label(Gt, v, x2,
                                                                                             gp.edge_label_attr):
                            return False
                    if Gq.has_edge(x1, u):
                        if not Gt.has_edge(x2, v):
                            return False
                        if _get_edge_label(Gq, x1, u, gp.edge_label_attr) != _get_edge_label(Gt, x2, v,
                                                                                             gp.edge_label_attr):
                            return False
                    return True
                else:
                    if Gq.has_edge(u, x1):
                        if not Gt.has_edge(v, x2):
                            return False
                        if _get_edge_label(Gq, u, x1, gp.edge_label_attr) != _get_edge_label(Gt, v, x2,
                                                                                             gp.edge_label_attr):
                            return False
                    return True

        for um in self._mapped_neighbors_in_Q(u, st):
            vm = st.mapping[um]
            if not edges_match(um, vm):
                return False

        # 4) Tin frontier consistency (VF2 rule): counts of new frontier nodes introduced should not exceed available T capacity
        # Compute number of NEW Tin nodes if we map (u->v)
        new_Tin_q = 0
        for w in self._unmapped_neighbors_Q(u, st):
            if w not in st.Tin_q:
                new_Tin_q += 1

        # Available T frontier after mapping v
        # Approximated by current |Tin_t| plus free neighbors of v not already in Tin_t.
        # If capacity clearly insufficient, prune.
        avail_Tin_t = len(st.Tin_t)
        for z in self._free_neighbors_T(v, st):
            if z not in st.Tin_t:
                avail_Tin_t += 1
        if new_Tin_q > avail_Tin_t:
            return False

        # 5) Lightweight forward checking: each unmapped neighbor of u must have at least one feasible T host
        # Check label/degree and quick edge-consistency with already mapped neighbors.
        for w in self._unmapped_neighbors_Q(u, st):
            if not self._has_quick_host_for(w, st):
                return False

        return True

    def _has_quick_host_for(self, w: Node, st: State) -> bool:
        gp = self.params
        lab = gp.q_labels[w]
        dw_out, dw_in = gp.q_deg[w]

        for v in st.free_t:
            if gp.t_labels[v] != lab:
                continue
            dv_out, dv_in = gp.t_deg[v]
            if dv_out < dw_out or dv_in < dw_in:
                continue
            ok = True
            for um in self._mapped_neighbors_in_Q(w, st):
                vm = st.mapping[um]
                if gp.Gq.has_edge(w, um) and not gp.Gt.has_edge(v, vm):
                    ok = False;
                    break
                if gp.Gq.is_directed() and gp.Gt.is_directed():
                    if gp.Gq.has_edge(um, w) and not gp.Gt.has_edge(vm, v):
                        ok = False;
                        break
            if ok:
                return True
        return False

    def _mapped_neighbors_in_Q(self, u: Node, st: State) -> Iterable[Node]:
        gp = self.params
        if gp.Gq.is_directed():
            nbrs = set(gp.Gq.predecessors(u)) | set(gp.Gq.successors(u))
        else:
            nbrs = gp.Gq.neighbors(u)
        return (w for w in nbrs if w in st.mapping)

    def _unmapped_neighbors_Q(self, u: Node, st: State) -> Iterable[Node]:
        gp = self.params
        if gp.Gq.is_directed():
            nbrs = set(gp.Gq.predecessors(u)) | set(gp.Gq.successors(u))
        else:
            nbrs = gp.Gq.neighbors(u)
        return (w for w in nbrs if w in st.free_q)

    def _free_neighbors_T(self, v: Node, st: State) -> Iterable[Node]:
        gp = self.params
        if gp.Gt.is_directed():
            nbrs = set(gp.Gt.predecessors(v)) | set(gp.Gt.successors(v))
        else:
            nbrs = gp.Gt.neighbors(v)
        return (z for z in nbrs if z in st.free_t)

    def _degree_multiset_dominates(self) -> bool:
        """
        For undirected graphs, the sorted list of degrees in Q must be component-wise <= the top-|Q| degrees in T.
        For directed graps, check both in and out separately.
        """
        gp = self.params
        if not gp.Gq.is_directed() and not gp.Gt.is_directed():
            dq = sorted((d for (_, d) in gp.Gq.degree()), reverse=True)
            dt = sorted((d for (_, d) in gp.Gt.degree()), reverse=True)
            if len(dt) < len(dq):
                return False
            dt_top = dt[:len(dq)]
            return all(dq[i] <= dt_top[i] for i in range(len(dq)))
        else:
            q_out = sorted((gp.Gq.out_degree(u) for u in gp.Gq), reverse=True)
            t_out = sorted((gp.Gt.out_degree(v) for v in gp.Gt), reverse=True)
            q_in = sorted((gp.Gq.in_degree(u) for u in gp.Gq), reverse=True)
            t_in = sorted((gp.Gt.in_degree(v) for v in gp.Gt), reverse=True)
            if len(t_out) < len(q_out) or len(t_in) < len(q_in):
                return False
            return all(q_out[i] <= t_out[i] for i in range(len(q_out))) and \
                all(q_in[i] <= t_in[i] for i in range(len(q_in)))


def vf2pp_all_isomorphisms(
        Gq: nx.Graph,
        Gt: nx.Graph,
        node_label_attr: Optional[str] = None,
        edge_label_attr: Optional[str] = None,
) -> List[Dict[Node, Node]]:
    return VF2PP(Gq, Gt, node_label_attr=node_label_attr, edge_label_attr=edge_label_attr).all_isomorphisms()
