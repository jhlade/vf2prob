# VF2-Prob, Jan Hladěna, FIM UHK
import sys, json, time
sys.path.insert(0, ".")
import networkx as nx
import run_experiments as R
from vf2_prob import ProbSGI, AStarProb

def queries_from_json(G, jpath):
    payload = json.loads(open(jpath).read())
    nodes_set = set(G.nodes())
    def coerce(nid):
        for c in (nid, str(nid)):
            if c in nodes_set: return c
        return None
    out=[]
    for item in payload["queries"]:
        mp=[coerce(n) for n in item["nodes"]]; mp=[m for m in mp if m is not None]
        if len(mp)<2: continue
        out.append(G.subgraph(mp).copy())
    return out

def run(G, Q, ub, search, timeout):
    nl_G,nl_Q=R._build_shared_node_label_ids(G,Q)
    el_G=R._build_edge_label_ids(G); ep=R._edge_probabilities(G)
    Cls = AStarProb if search=="astar" else ProbSGI
    e=Cls(G,Q,nl_G,nl_Q,el_G,ep,timeout_s=timeout,use_bound=True,use_node=True,use_edge=True,
          collect_metrics=False,compute_report_ub=False)
    e.ub_mode=ub
    t0=time.time(); _,c,to=e.run(); dt=time.time()-t0
    return c["best_loglik"], c["states"], to, dt

G=nx.read_graphml("../data/mol/mol.graphml")
Qs=queries_from_json(G,"../queries/mol/mol_S.json")
to=float(sys.argv[1]) if len(sys.argv)>1 else 8.0
lim=int(sys.argv[2]) if len(sys.argv)>2 else 5
print(f"timeout={to}s")
for i,Q in enumerate(Qs[:lim]):
    row=[f"q{i} |Q|={Q.number_of_nodes()}E{Q.number_of_edges()}"]
    for ub,search in [("node","dfs"),("assign","dfs"),("node","astar"),("assign","astar")]:
        ll,st,t,dt=run(G,Q,ub,search,to)
        row.append(f"{ub[:1]}{search[:1]}:ll={ll:.4f},st={st},{'TO' if t else 'ok'},{dt:.2f}s")
    print("  ".join(row), flush=True)
