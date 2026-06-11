# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
import json, glob, os, collections

RAW = "../results_raw"
PROB = {"vf2prob", "vf2prob-assign", "vf2prob-astar", "vf2prob-astar-assign"}
# cell timeouts actually used (s)
TO = {"synth": 12, "mol": 12, "gmark": 10}

def cell_to(tag):
    for k, v in TO.items():
        if tag.startswith(k):
            return v
    return 1e9

bad_c1 = bad_c2 = bad_c3 = 0
summary = []
for path in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
    tag = os.path.splitext(os.path.basename(path))[0]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    tlim = cell_to(tag)
    byq = collections.defaultdict(dict)  # qidx -> method -> row
    for r in rows:
        if r["method"] in PROB:
            byq[r["qidx"]][r["method"]] = r
    # C1: agreement among non-timed-out prob variants per query
    c1 = 0
    for qidx, mm in byq.items():
        vals = [(m, rr["best_loglik"]) for m, rr in mm.items() if int(rr["timed_out"]) == 0 and rr["best_loglik"] > float("-inf")]
        if len(vals) >= 2:
            base = vals[0][1]
            if any(abs(v - base) > 1e-6 for _, v in vals):
                c1 += 1
                print(f"  C1 MISMATCH {tag} q{qidx}: " + ", ".join(f"{m}={v:.4f}" for m, v in vals))
    # C2 + C3 across all prob rows
    c2 = c3 = 0
    for r in rows:
        if r["method"] in PROB:
            if int(r["timed_out"]) == 0 and float(r["success"]) != 1.0 and r["best_loglik"] > float("-inf"):
                c2 += 1
                print(f"  C2 {tag} {r['method']} q{r['qidx']}: timeout=0 but success={r['success']}")
        # C3 for all methods: elapsed shouldn't exceed limit by much (timeouts ~ limit)
        if float(r["elapsed_ms"]) > (tlim * 1000.0) * 1.5:
            c3 += 1
            print(f"  C3 {tag} {r['method']} q{r['qidx']}: elapsed_ms={r['elapsed_ms']:.0f} > 1.5*{tlim}s")
    # per-cell timeout counts per prob variant
    tos = collections.Counter()
    tot = collections.Counter()
    for r in rows:
        if r["method"] in PROB:
            tot[r["method"]] += 1
            tos[r["method"]] += int(r["timed_out"])
    summary.append((tag, dict(tos), dict(tot), c1, c2, c3))
    bad_c1 += c1; bad_c2 += c2; bad_c3 += c3

print("\n=== per-cell prob timeouts (timed_out / total) ===")
for tag, tos, tot, c1, c2, c3 in summary:
    s = "  ".join(f"{m.replace('vf2prob','vp')}:{tos.get(m,0)}/{tot.get(m,0)}" for m in
                  ["vf2prob","vf2prob-assign","vf2prob-astar","vf2prob-astar-assign"])
    print(f"{tag:9s}  {s}")

print(f"\n=== TOTALS: C1 mismatches={bad_c1}  C2 violations={bad_c2}  C3 anomalies={bad_c3} ===")
print("PASS" if (bad_c1 == bad_c2 == bad_c3 == 0) else "CHECK ABOVE")
