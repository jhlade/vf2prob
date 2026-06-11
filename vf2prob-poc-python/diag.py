# VF2-Prob, Jan Hladěna, FIM UHK
import sys, json
sys.path.insert(0, ".")
import networkx as nx
import run_experiments as R
from vf2_prob import ProbSGI

def load_graph(path):
    return nx.read_graphml(path)

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

def run(G, Q, ub):
    nl_G, nl_Q = R._build_shared_node_label_ids(G, Q)
    el_G = R._build_edge_label_ids(G)
    ep = R._edge_probabilities(G)
    e = ProbSGI(G, Q, nl_G, nl_Q, el_G, ep, timeout_s=60, use_bound=True,
                use_node=True, use_edge=True, collect_metrics=False, compute_report_ub=False)
    e.ub_mode = ub
    _, c, to = e.run()
    return c["best_loglik"], c["states"], to

def main(graphml, jpath, tag):
    G = load_graph(graphml)
    Qs = queries_from_json(G, jpath)
    print(f"== {tag}: |G|={G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(Qs)} queries ==")
    mism = 0
    for i, Q in enumerate(Qs):
        a = run(G, Q, "node")
        b = run(G, Q, "assign")
        flag = ""
        if abs(a[0] - b[0]) > 1e-6:
            mism += 1
            flag = "  <<< MISMATCH"
            print(f"q{i} |Q|={Q.number_of_nodes()}E{Q.number_of_edges()} node=({a[0]:.4f},{a[1]}st,to{int(a[2])}) assign=({b[0]:.4f},{b[1]}st,to{int(b[2])}){flag}")
    print(f"TOTAL mismatches {tag}: {mism}/{len(Qs)}")
    return mism

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "mol"
    if which == "mol":
        main("../data/mol/mol.graphml", "../queries/mol/mol_S.json", "mol_S")
    elif which == "synth":
        # build synth graph + queries reproducibly
        import random
        G = R._synth_graph(num_nodes=120, p=0.06, seed=324)
        random.seed(324)
        Qs = R._sample_connected_queries(G, 5, 8, 16)
        print(f"== synth: |G|={G.number_of_nodes()} {len(Qs)} queries ==")
        mism = 0
        for i, Q in enumerate(Qs):
            a = run(G, Q, "node"); b = run(G, Q, "assign")
            if abs(a[0]-b[0])>1e-6:
                mism+=1
                print(f"q{i} |Q|={Q.number_of_nodes()}E{Q.number_of_edges()} node={a} assign={b}  <<< MISMATCH")
        print(f"TOTAL mismatches synth: {mism}/{len(Qs)}")
