# VF2-Prob: Probabilistic Subgraph Isomorphism

> **TBD**

---

## Requirements

- Python 3.10+, pip
- CMake 3.20+, C++20 compiler (GCC 12+ or Clang 15+)
- curl, gunzip (for SNAP dataset download)

---

## Quick start

```
make run            # fetch data, build both implementations, run all benchmarks
make run SEED=324   # same with a custom seed
```

---

## Step by step

```
make build          # Python venv + C++ compilation
make fetch-data     # download SNAP; generate mol/gmark datasets
make bench-python   # Python PoC benchmarks  ->  results/results_*.csv
make bench-cpp      # C++ benchmarks         ->  results/cpp_*.csv
```

---

## Individual suites

```
# Python PoC
make -C vf2prob-poc-python synth_S
make -C vf2prob-poc-python synth_S SEED=324 RUNS=5 PROB_UB=assign
make -C vf2prob-poc-python real_all
make -C vf2prob-poc-python snap_all

# C++ system
make -C vf2prob-sys-cpp bench-synth-S
make -C vf2prob-sys-cpp bench-synth-S SEED=324 METHODS=vf2prob,vf2prob-astar-assign
make -C vf2prob-sys-cpp bench-snap
make -C vf2prob-sys-cpp test
```

---

## Results

All outputs land in `results/` as CSV files.

- Python PoC produces one aggregated row per method: `results/results_<dataset>_<size>.csv`
- C++ produces per-query/run rows: `results/cpp_<dataset>_<size>.csv`; run `make -C vf2prob-sys-cpp aggregate` for per-method summaries.

---

## Repository layout

```
vf2prob/
├── Makefile                   global orchestration
├── scripts/                   data preparation and paper-data helpers
├── data/                      graph GraphML files (gitignored)
├── queries/                   query JSON files (gitignored)
├── results/                   CSV outputs (gitignored)
├── vf2prob-poc-python/        Python proof-of-concept implementation
└── vf2prob-sys-cpp/           C++ system implementation
```
