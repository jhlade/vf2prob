# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
"""Parallel (per-cell) resumable runner. Runs pending (method,query,run) tasks
of ONE cell across worker processes, appending each result as it finishes.
Honours a wall budget; in-flight tasks at budget time are simply retried next
call (idempotent via the done-set). Usage:

  python3 gridp.py <tag[,tag...]> [wall_budget_s] [runs] [nproc]
"""
import sys, os, json, time
import multiprocessing as mp
sys.path.insert(0, ".")
import grid

_G = _Q = _DS = _C = _TAG = None

def _init(tag):
    global _G, _Q, _DS, _C, _TAG
    _G, _Q, _DS, _C = grid.build_cell(tag)
    _TAG = tag

def _task(args):
    method, ub, search, qidx, run = args
    Qg = _Q[qidx]
    res = grid.run_one(_G, Qg, _DS, _C, method, ub, search)
    return dict(tag=_TAG, dataset=_DS, method=method, qidx=qidx, run=run,
                qn=Qg.number_of_nodes(), qe=Qg.number_of_edges(), **res)

def run_cell(tag, budget, runs, nproc):
    grid_path = os.path.join(grid.RAW_DIR, f"{tag}.jsonl")
    os.makedirs(grid.RAW_DIR, exist_ok=True)
    done = grid.load_done(grid_path)
    G, Q, dataset, c = grid.build_cell(tag)
    maxq = c.get("maxq", len(Q))
    methods = [(m, None, None) for m in grid.BASELINES] + [(m, u, s) for (m, u, s) in grid.PROB]
    pending = []
    for (method, ub, search) in methods:
        for qidx in range(min(maxq, len(Q))):
            for run in range(runs):
                if (method, qidx, run) not in done:
                    pending.append((method, ub, search, qidx, run))
    if not pending:
        print(f"[{tag}] already complete ({len(done)} rows)")
        return 0
    n = 0
    t0 = time.time()
    pool = mp.Pool(processes=nproc, initializer=_init, initargs=(tag,))
    try:
        for row in pool.imap_unordered(_task, pending):
            with open(grid_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            n += 1
            if time.time() - t0 > budget:
                print(f"[{tag}] budget hit; appended {n} rows ({len(pending)-n}+ left)", flush=True)
                break
        else:
            print(f"[{tag}] cell complete; appended {n} rows", flush=True)
    finally:
        pool.terminate()
        pool.join()
    return n

def main():
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else list(grid.CELLS.keys())
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 38.0
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    t0 = time.time()
    for tag in tags:
        if time.time() - t0 > budget:
            break
        run_cell(tag, budget - (time.time() - t0), runs, nproc)

if __name__ == "__main__":
    main()
