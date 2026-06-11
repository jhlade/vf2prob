# VF2-Prob, Jan Hladěna, FIM UHK
import sys, statistics
sys.path.insert(0, ".")
import grid, run_experiments as R
from vf2_prob import ProbSGI

def run(G, Q, T, order):
    nl_G, nl_Q = R._build_shared_node_label_ids(G, Q)
    el_G = R._build_edge_label_ids(G); ep = R._edge_probabilities(G)
    e = ProbSGI(G, Q, nl_G, nl_Q, el_G, ep, timeout_s=T, use_bound=True, use_node=True,
                use_edge=True, collect_metrics=False, compute_report_ub=False)
    e.ub_mode = "assign"; e.order_by_gain = order
    _, c, to = e.run()
    return c["states"], int(bool(to))

med = lambda xs: statistics.median(xs) if xs else float("nan")
for tag in ["synth_S","synth_M","synth_L","mol_S","mol_M","mol_L"]:
    G, Q, ds, c = grid.build_cell(tag)
    on, off = [], []
    for Qg in Q:
        s1,t1 = run(G,Qg,c["timeout"],True)
        s2,t2 = run(G,Qg,c["timeout"],False)
        if not t1 and not t2:
            on.append(s1); off.append(s2)
    ratio = med([o/g for g,o in zip(on,off) if g>0]) if on else float("nan")
    print(f"{tag:8s} n={len(on):2d}  states_ordered={med(on):7.1f}  states_unordered={med(off):8.1f}  median(off/on)={ratio:.2f}", flush=True)
