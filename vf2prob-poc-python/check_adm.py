# VF2-Prob, Jan Hladěna, FIM UHK
"""Definitive admissibility check for assign-UB, independent of search timeouts.

For each query:
  1. Run node-mode to completion -> true optimum V, optimal map M*.
  2. Walk the prefix of M* in self.Q_nodes order. At each partial mapping,
     compute assign-UB (ub_c) and check cur_score + ub_c >= V (admissible).
  3. Report the first prefix where it fails (assign-UB pruned the optimum).
"""
import sys, json
sys.path.insert(0, ".")
import networkx as nx
import run_experiments as R
from vf2_prob import ProbSGI
from compat import safe_log

def queries_from_json(G, jpath):
    payload = json.loads(open(jpath).read())
    nodes_set = set(G.nodes())
    def coerce(nid):
        for c in (nid, str(nid)):
            if c in nodes_set:
                return c
        return None
    out = []
    for item in payload["queries"]:
        mp = [coerce(n) for n in item["nodes"]]
        mp = [m for m in mp if m is not None]
        if len(mp) < 2:
            continue
        out.append(G.subgraph(mp).copy())
    return out

def make_engine(G, Q, ub, timeout):
    nl_G, nl_Q = R._build_shared_node_label_ids(G, Q)
    el_G = R._build_edge_label_ids(G)
    ep = R._edge_probabilities(G)
    e = ProbSGI(G, Q, nl_G, nl_Q, el_G, ep, timeout_s=timeout, use_bound=True,
                use_node=True, use_edge=True, collect_metrics=False, compute_report_ub=False)
    e.ub_mode = ub
    return e

def path_score(e, Mstar, order):
    """incremental score adding query nodes in 'order' per dfs accounting."""
    scores = [0.0]
    mapping = {}
    cur = 0.0
    for q in order:
        u = Mstar[q]
        ng = safe_log(e.node_compat(q, u))
        eg = 0.0
        for qn in e.Q.neighbors(q):
            if qn in mapping:
                eg += safe_log(e.edge_compat(q, qn, u, mapping[qn]))
        cur += ng + eg
        mapping[q] = u
        scores.append(cur)
    return scores

def check_query(G, Q, timeout, idx):
    en = make_engine(G, Q, "node", timeout)
    bm, c, to = en.run()
    V = c["best_loglik"]
    if to or bm is None:
        return f"q{idx}: node-mode TIMEOUT/none (V={V:.4f}), skip", None
    order = en.Q_nodes
    # build assign engine sharing same graph
    ea = make_engine(G, Q, "assign", timeout)
    # incremental cur scores along optimal path
    scores = path_score(ea, bm, order)
    used = set()
    mapping = {}
    worst = None
    n = len(order)
    for i in range(n):  # partial mapping of first i nodes; UB over remaining
        ub_c, _ = ea._UB_assign(mapping, set(used), budget_ms=1e9)
        cur = scores[i]
        slack = (cur + ub_c) - V
        if worst is None or slack < worst[0]:
            worst = (slack, i, cur, ub_c)
        # also node-UB for reference
        if slack < -1e-9:
            # found violation
            unb = en._UB_pruning_nodes_only(en.Q_nodes[i:])
            return (f"q{idx} |Q|={Q.number_of_nodes()}E{Q.number_of_edges()} VIOLATION at prefix i={i}: "
                    f"cur={cur:.4f} assignUB={ub_c:.4f} sum={cur+ub_c:.4f} < V={V:.4f} "
                    f"(slack={slack:.4f}); nodeUB={unb:.4f} (sum={cur+unb:.4f})"), (G,Q,bm,order,i)
        # advance
        q = order[i]
        mapping[q] = bm[q]
        used.add(bm[q])
    return (f"q{idx} |Q|={Q.number_of_nodes()}E{Q.number_of_edges()} OK  V={V:.4f} "
            f"min-slack={worst[0]:.4f} at i={worst[1]}"), None

def main():
    which = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    maxq = int(sys.argv[3]) if len(sys.argv) > 3 else 999
    if which == "mol":
        G = nx.read_graphml("../data/mol/mol.graphml")
        Qs = queries_from_json(G, "../queries/mol/mol_S.json")
    else:
        import random
        G = R._synth_graph(num_nodes=120, p=0.06, seed=324)
        random.seed(324)
        Qs = R._sample_connected_queries(G, 5, 8, 16)
    print(f"== {which}: |G|={G.number_of_nodes()} edges={G.number_of_edges()} nq={len(Qs)} timeout={timeout} ==", flush=True)
    found = []
    for i, Q in enumerate(Qs[:maxq]):
        msg, hit = check_query(G, Q, timeout, i)
        print(msg, flush=True)
        if hit:
            found.append(hit)
    print(f"== violations found: {len(found)} ==", flush=True)

if __name__ == "__main__":
    main()
