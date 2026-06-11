# VF2-Prob, Jan Hladěna, FIM UHK
import sys, json
sys.path.insert(0, ".")
import networkx as nx
import run_experiments as R
from vf2_prob import ProbSGI, AStarProb
from compat import safe_log

def queries_from_json(G, jpath):
    payload=json.loads(open(jpath).read()); nodes_set=set(G.nodes())
    def coerce(nid):
        for c in (nid,str(nid)):
            if c in nodes_set: return c
        return None
    out=[]
    for item in payload["queries"]:
        mp=[coerce(n) for n in item["nodes"]]; mp=[m for m in mp if m is not None]
        if len(mp)<2: continue
        out.append(G.subgraph(mp).copy())
    return out

def mk(G,Q,ub,Cls=ProbSGI):
    nl_G,nl_Q=R._build_shared_node_label_ids(G,Q)
    el_G=R._build_edge_label_ids(G); ep=R._edge_probabilities(G)
    e=Cls(G,Q,nl_G,nl_Q,el_G,ep,timeout_s=30,use_bound=True,use_node=True,use_edge=True,
          collect_metrics=False,compute_report_ub=False)
    e.ub_mode=ub
    return e

G=nx.read_graphml("../data/mol/mol.graphml")
Qs=queries_from_json(G,"../queries/mol/mol_S.json")
Q=Qs[11]
# true optimum via astar-assign
ea=mk(G,Q,"assign",AStarProb)
bm,c,to=ea.run()
V=c["best_loglik"]
print(f"q11 |Q|={Q.number_of_nodes()}E{Q.number_of_edges()} true V={V:.4f} to={to}")
print("Q nodes order:", ea.Q_nodes)
print("Q edges:", list(Q.edges()))

en=mk(G,Q,"node")
order=en.Q_nodes
# incremental cur scores along M*
mapping={}; used=set(); cur=0.0
print("\nprefix  q->u           cur       nodeUB   cur+nUB   assignUB cur+aUB  (V=%.4f)"%V)
for i in range(len(order)+1):
    nUB=en._UB_pruning_nodes_only(en.Q_nodes[i:])
    aUB,_=en._UB_assign(dict(mapping), set(used), budget_ms=1e9)
    flag_n = " nUB<V!" if cur+nUB < V-1e-9 else ""
    flag_a = " aUB<V!" if cur+aUB < V-1e-9 else ""
    qd = order[i] if i<len(order) else "-"
    ud = bm.get(order[i]) if i<len(order) else "-"
    print(f"i={i:2d}  {str(qd):>4}->{str(ud):>4}   cur={cur:8.4f}  {nUB:7.3f} {cur+nUB:8.4f}  {aUB:7.3f} {cur+aUB:8.4f}{flag_n}{flag_a}")
    if i<len(order):
        q=order[i]; u=bm[q]
        ng=safe_log(en.node_compat(q,u)); eg=0.0
        for qn in en.Q.neighbors(q):
            if qn in mapping: eg+=safe_log(en.edge_compat(q,qn,u,mapping[qn]))
        cur+=ng+eg; mapping[q]=u; used.add(u)
print("final cur (should==V):", cur)
