# 🫧 vf2prob-cpp

C++ implementation of VF2-Prob: exact probabilistic subgraph isomorphism with
logistic node/edge compatibilities and an admissible branch-and-bound. Includes
the baselines (VF2, VF2++, VF2-Bin) and the four VF2-Prob variants —
`(DFS | A*) × (node-UB | assign-UB)` — behind one interface, a measuring harness,
and SNAP data tooling.

## 🏗️ Build

Requires CMake ≥ 3.20 and a C++20 compiler. No external dependencies (SNAP
download shells out to `curl`/`gzip`).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## 🚀 Run

```bash
# Synthetic data (generated in memory):
./build/vf2prob_run --dataset synth --runs 3 --timeout 30 --pmin 0.7 --flip 0.05 \
  --out results/synth.csv

# Real data (SNAP): prepare graphs + queries, then run:
./build/prep_snap --datasets facebook,enron,hep-th --pmin 0.6
./build/vf2prob_run --dataset graphml --data-path data/snap/facebook_combined.graphml \
  --queries-path queries/snap/facebook_S.json --out results/facebook_S.csv

# Aggregate + paired statistics (Wilcoxon, REI):
./build/aggregate --in results/synth.csv --compare vf2prob:vf2prob-astar-assign

# Learn compatibility weights, then run with them:
./build/train_weights --flip 0.15 --out results/weights.txt
./build/vf2prob_run --dataset synth --weights results/weights.txt --out results/learned.csv

# Full grid (synthetic S/M/L; add --snap for SNAP):
scripts/run_grid.sh
```

Methods: `vf2`, `vf2pp`, `vf2bin`, `vf2prob`, `vf2prob-assign`, `vf2prob-astar`,
`vf2prob-astar-assign`.
