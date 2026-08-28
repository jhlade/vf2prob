# 🫧 vf2prob-cpp

C++ implementation of VF2-Prob: exact probabilistic subgraph isomorphism with
logistic node/edge compatibilities and an admissible branch-and-bound. Includes
the baselines (VF2, VF2++, VF2-Bin), the uncertain-graph competitor MPM, and the
four VF2-Prob variants — `(DFS | A*) × (node-UB | assign-UB)` — behind one
`IMatcher` interface, a measuring harness (wall-clock, peak RSS, explored
states), and apps for training, recovery sweeps, and aggregation.

## 🏗️ Build

Requires CMake ≥ 3.20 and a C++20 compiler; no external dependencies.

```bash
make build          # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && build
make test           # ctest (includes the bound-admissibility property tests)
```

## 🚀 Run

Input data (SNAP downloads, mol/gmark/STRING/IAM graphs, query JSONs) are
prepared by the **top-level** Makefile: run `make fetch-data` in the repository
root first. Then, from this directory:

```bash
# Synthetic benchmark (generated in memory), one regime per target:
make bench-synth-S     # or -M / -L; bench-synth runs all three
make bench-synth-S SEED=324 METHODS=vf2prob,vf2prob-astar-assign

# Real graphs (after `make fetch-data` in the repo root):
make bench-snap        # facebook / ca-HepTh / email-Enron, sizes S/M/L
make bench-string      # STRING PPI scalability (feeds tab:snap)

# Aggregate per-method medians/IQRs from the per-query CSVs:
make aggregate         # results/cpp_*.csv -> results/cpp_*_agg.csv
```

Direct app invocations (see `--help` of each binary in `build/`):

```bash
# Harness: one dataset x methods grid -> per-(query,run) CSV rows
./build/vf2prob_run --dataset synth --runs 3 --timeout 30 --pmin 0.7 --flip 0.05 \
  --out ../results/synth.csv

# Learn compatibility weights, then run with them:
./build/train_weights --flip 0.15 --out ../results/weights.txt
./build/vf2prob_run --dataset synth --weights ../results/weights.txt \
  --out ../results/learned.csv

# Recovery-vs-noise sweep (fixed + learned weights on identical seeded instances);
# the full set of sweeps is `make bench-recovery` in the repo root:
./build/recovery --instances 200 --structured 1 --seed 324 \
  --out ../results/recovery_synth.csv
```

Methods: `vf2`, `vf2pp` (first-match decision semantics), `vf2bin`, `mpm`,
`vf2prob`, `vf2prob-assign`, `vf2prob-astar`, `vf2prob-astar-assign`.
