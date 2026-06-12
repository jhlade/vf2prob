# VF2-Prob, Jan Hladěna, FIM UHK
import sys, json
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

def run(G,Q,ub,search,timeout):
    nl_G,nl_Q=R._build_shared_node_label_ids(G,Q)
    el_G=R._build_edge_label_ids(G); ep=R._edge_probabilities(G)
    Cls=AStarProb if search=="astar" else ProbSGI
    e=Cls(G,Q,nl_G,nl_Q,el_G,ep,timeout_s=timeout,use_bound=True,use_node=True,use_edge=True,
          collect_metrics=False,compute_report_ub=False)
    e.ub_mode=ub
    _,c,to=e.run()
    return c["best_loglik"],c["states"],to

graphml=sys.argv[1]; jpath=sys.argv[2]; to=float(sys.argv[3]);
lo=int(sys.argv[4]) if len(sys.argv)>4 else 0
hi=int(sys.argv[5]) if len(sys.argv)>5 else 999
G=nx.read_graphml(graphml)
Qs=queries_from_json(G,jpath)
variants=[("node","dfs"),("assign","dfs"),("node","astar"),("assign","astar")]
mism=0; tos=0
for i,Q in enumerate(Qs):
    if i<lo or i>=hi: continue
    res=[run(G,Q,u,s,to) for (u,s) in variants]
    lls=[r[0] for r in res]; anyto=any(r[2] for r in res)
    base=lls[0]
    bad=any(abs(x-base)>1e-6 for x in lls)
    if anyto: tos+=1
    if bad: mism+=1
    tag="  <<<MISMATCH" if bad else ""
    if anyto: tag+=" [TO]"
    print(f"q{i} |Q|={Q.number_of_nodes()}E{Q.number_of_edges()} "
          f"nd={lls[0]:.4f}({res[0][1]}) ad={lls[1]:.4f}({res[1][1]}) "
          f"na={lls[2]:.4f}({res[2][1]}) aa={lls[3]:.4f}({res[3][1]}){tag}", flush=True)
print(f"== range[{lo}:{hi}] mismatches={mism} timeouts={tos} ==", flush=True)
