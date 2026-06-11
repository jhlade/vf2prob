# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
"""Compute summary statistics for the write-up: REI, completion, state medians,
tightness (root UB gap), and paired Wilcoxon tests on per-query states."""
import json, glob, os, statistics, collections
from scipy.stats import wilcoxon

RAW = "../results_raw"
PROB = ["vf2prob", "vf2prob-assign", "vf2prob-astar", "vf2prob-astar-assign"]
ALL = ["vf2", "vf2pp", "vf2bin"] + PROB

def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
        tag = os.path.splitext(os.path.basename(p))[0]
        fam = tag.split("_")[0]; size = tag.split("_")[1]
        for l in open(p):
            l = l.strip()
            if l:
                r = json.loads(l); r["fam"] = fam; r["size"] = size; r["tag"] = tag
                rows.append(r)
    return rows

rows = load()
def med(xs): return statistics.median(xs) if xs else float("nan")

# index by (tag,qidx,method)
idx = {}
for r in rows:
    idx[(r["tag"], r["qidx"], r["method"])] = r

print("=== median states by family x method (completed runs only) ===")
for fam in ["synth", "mol", "gmark"]:
    print(f"-- {fam} --")
    for m in ALL:
        st = [r["states"] for r in rows if r["fam"] == fam and r["method"] == m and int(r["timed_out"]) == 0]
        print(f"  {m:24s} med_states(completed)={med(st):10.1f}  n={len(st)}")

print("\n=== REI (median per-query states_baseline/states_method), paired, both completed ===")
for fam in ["synth", "mol", "gmark"]:
    tags = sorted({r["tag"] for r in rows if r["fam"] == fam})
    for base in ["vf2", "vf2pp"]:
        for m in PROB:
            ratios = []
            for tag in tags:
                qs = {r["qidx"] for r in rows if r["tag"] == tag}
                for q in qs:
                    rb = idx.get((tag, q, base)); rm = idx.get((tag, q, m))
                    if rb and rm and int(rb["timed_out"]) == 0 and int(rm["timed_out"]) == 0 and rm["states"] > 0:
                        ratios.append(rb["states"] / rm["states"])
            if ratios:
                print(f"  {fam:6s} REI({m} | {base}) = {med(ratios):6.2f}  (n={len(ratios)})")

print("\n=== completion (fraction solved within limit) by family x method ===")
for fam in ["synth", "mol", "gmark"]:
    print(f"-- {fam} --")
    for m in ALL:
        rs = [r for r in rows if r["fam"] == fam and r["method"] == m]
        comp = sum(1 for r in rs if int(r["timed_out"]) == 0 and r["best_loglik"] > float("-inf") or (m in ("vf2","vf2pp","vf2bin") and int(r["timed_out"])==0 and r["success"]==1.0))
        compf = sum(r["success"] for r in rs)/len(rs) if rs else float("nan")
        print(f"  {m:24s} completion={compf:4.2f}  n={len(rs)}")

print("\n=== tightness: median root UB gap (ub_gap_p50) node vs assign (completed) ===")
for fam in ["synth", "mol", "gmark"]:
    g_node = [r["ub_gap_p50"] for r in rows if r["fam"]==fam and r["method"]=="vf2prob" and int(r["timed_out"])==0 and r["ub_gap_p50"]==r["ub_gap_p50"]]
    g_assign = [r["ub_gap_p50"] for r in rows if r["fam"]==fam and r["method"]=="vf2prob-assign" and int(r["timed_out"])==0 and r["ub_gap_p50"]==r["ub_gap_p50"]]
    print(f"  {fam:6s} root-gap node={med(g_node):.3f}  assign={med(g_assign):.3f}")

print("\n=== Wilcoxon signed-rank on per-query states (paired, both completed) ===")
def paired(mA, mB, fams):
    a, b = [], []
    for r in rows:
        if r["fam"] in fams and r["method"] == mA and int(r["timed_out"]) == 0:
            rb = idx.get((r["tag"], r["qidx"], mB))
            if rb and int(rb["timed_out"]) == 0:
                a.append(r["states"]); b.append(rb["states"])
    return a, b
for (mA, mB, fams, label) in [
    ("vf2prob-astar-assign", "vf2pp", ["synth","mol"], "A*+assign vs VF2++ (synth+mol)"),
    ("vf2prob-assign", "vf2prob", ["synth","mol","gmark"], "assign-UB vs node-UB DFS (all)"),
    ("vf2prob-astar-assign", "vf2prob", ["synth","mol"], "A*+assign vs DFS node-UB (synth+mol)"),
]:
    a, b = paired(mA, mB, fams)
    if len(a) >= 6 and any(x != y for x, y in zip(a, b)):
        try:
            W, p = wilcoxon(a, b)
            print(f"  {label}: n={len(a)} median({mA})={med(a):.0f} median({mB})={med(b):.0f}  W={W:.1f} p={p:.2e}")
        except Exception as e:
            print(f"  {label}: wilcoxon error {e}")
    else:
        print(f"  {label}: n={len(a)} (insufficient/ties)")
