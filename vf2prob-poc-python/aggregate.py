# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
"""Aggregate per-query JSONL (results_raw/<tag>.jsonl) into the standard
results CSV (one row per dataset,method) matching run_experiments' schema."""
import sys, os, json, glob
sys.path.insert(0, ".")
import run_experiments as R  # reuse _median, _iqr, _fmt_cell

RAW_DIR = "../results_raw"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "../results"

HEADER = ["dataset", "method", "median_ms", "iqr_ms", "states_med", "prune_rate",
          "success_rate", "best_loglik_med", "mem_mb", "timeout", "ub_gap_p50", "ub_gap_p90"]

# canonical method order for stable output
ORDER = ["vf2", "vf2bin", "vf2pp", "vf2prob", "vf2prob-assign", "vf2prob-astar", "vf2prob-astar-assign"]

def agg_tag(tag):
    path = os.path.join(RAW_DIR, f"{tag}.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by = {}
    dataset = rows[0]["dataset"] if rows else tag
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    out = []
    for method in sorted(by.keys(), key=lambda m: (ORDER.index(m) if m in ORDER else 99, m)):
        rs = by[method]
        times = [x["elapsed_ms"] for x in rs]
        states = [x["states"] for x in rs]
        prune = [x["prune_rate"] for x in rs]
        succ = [x["success"] for x in rs]
        bll = [x["best_loglik"] for x in rs if x["best_loglik"] > float("-inf")]
        ub50 = [x["ub_gap_p50"] for x in rs if x["ub_gap_p50"] == x["ub_gap_p50"]]  # drop nan
        ub90 = [x["ub_gap_p90"] for x in rs if x["ub_gap_p90"] == x["ub_gap_p90"]]
        n_to = sum(int(x["timed_out"]) for x in rs)
        row = {
            "dataset": dataset, "method": method,
            "median_ms": R._median(times), "iqr_ms": R._iqr(times),
            "states_med": R._median(states), "prune_rate": R._median(prune),
            # success_rate = FRACTION of queries solved (mean), the completion metric
            "success_rate": (sum(succ) / len(succ) if succ else 0.0),
            "best_loglik_med": (R._median(bll) if bll else float("nan")),
            "mem_mb": float("nan"), "timeout": n_to,
            "ub_gap_p50": (R._median(ub50) if ub50 else float("nan")),
            "ub_gap_p90": (R._median(ub90) if ub90 else float("nan")),
        }
        out.append({k: R._fmt_cell(k, v) for k, v in row.items()})
    return dataset, out

def write_csv(path, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    tags = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(RAW_DIR, "*.jsonl")))]
    if len(sys.argv) > 2:
        tags = sys.argv[2].split(",")
    for tag in tags:
        path = os.path.join(RAW_DIR, f"{tag}.jsonl")
        if (not os.path.exists(path)) or os.path.getsize(path) == 0:
            print(f"skip {tag} (no data yet)")
            continue
        dataset, rows = agg_tag(tag)
        if not rows:
            print(f"skip {tag} (empty)")
            continue
        out = os.path.join(OUT_DIR, f"results_{tag}.csv")
        write_csv(out, rows)
        print(f"wrote {out}  ({len(rows)} methods)")

if __name__ == "__main__":
    main()
