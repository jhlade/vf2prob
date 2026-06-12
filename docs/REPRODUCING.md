# Reproducing the paper

This guide reproduces the numbers, tables, and figures of *"Probabilistic
Subgraph Isomorphism with Learned Nonlinear Compatibility and Admissible Branch
and Bound."*

The repository holds two implementations that share one seeded benchmark
protocol:

- **`vf2prob-poc-python/`** — NetworkX proof of concept. Establishes the
  implementation-independent **explored-states** metric and confirms that every
  variant that terminates returns the same provably-optimal optimum.
- **`vf2prob-sys-cpp/`** — modular C++ system. Adds **measured** wall-clock time
  and peak memory and scales to the large SNAP graphs.

Every stochastic step is seeded (`SEED`, default **324**); explored-states and
the returned optima are bit-reproducible. Wall-clock and memory are secondary,
machine-dependent context.

## Requirements

- Python 3.10+ and `pip`
- CMake 3.20+ and a C++20 compiler (GCC 12+ or Clang 15+)
- `curl` and `gunzip` (for the SNAP download)

The Python proof of concept needs only `networkx`, `scipy`, and `numpy`; CSV
output uses the standard-library `csv` module (no `pandas`, and nothing is
plotted in Python — the paper's figures are pgfplots built from the CSVs).

## Quick start

From the repository root:

```sh
make run             # fetch data, build both implementations, run all benchmarks
make run SEED=324    # same, with an explicit seed (324 is the default)
```

`make run` = `fetch-data` + `build` + `bench-python` + `bench-cpp`. All CSV
outputs land in `results/`.

## Step by step

```sh
make build           # Python venv + C++ compilation
make fetch-data      # download SNAP; generate the mol/gMark datasets (seeded)
make bench-python    # Python PoC: synth + mol + gMark
make bench-cpp       # C++: synth + SNAP
make test            # C++ unit tests (ctest)
```

Finer sub-targets exist: `build-python`, `build-cpp`, `bench-python-synth`,
`bench-python-real`, `bench-cpp-synth`, `bench-cpp-snap`.

## Individual suites

**Python PoC** (`vf2prob-poc-python/`). Per-query time limits match the paper's
protocol — 12 s synthetic/molecular, 10 s gMark, one run per query:

```sh
make -C vf2prob-poc-python synth_all     # synthetic S/M/L, all four variants
make -C vf2prob-poc-python real_all      # mol + gMark S/M/L, all four variants
make -C vf2prob-poc-python synth_S PROB_UB=assign PROB_SEARCH=astar SEED=324
```

The four VF2-Prob variants are selected by `PROB_UB=node|assign` and
`PROB_SEARCH=dfs|astar`; the `*_assign`, `*_astar`, `*_astar_assign` targets
sweep them.

**C++ system** (`vf2prob-sys-cpp/`):

```sh
make -C vf2prob-sys-cpp bench-synth      # synthetic S/M/L
make -C vf2prob-sys-cpp bench-snap       # SNAP facebook / email-Enron / ca-HepTh, S/M/L
make -C vf2prob-sys-cpp aggregate        # per-method summaries from per-query CSVs
make -C vf2prob-sys-cpp test
```

## Outputs

All results are written to `results/` as CSV:

- Python PoC → `results/results_<dataset>_<size>.csv`, one aggregated row per
  method with columns `states_med`, `prune_rate`, `success_rate`,
  `best_loglik_med`, `median_ms`, `iqr_ms`, `mem_mb`, `timeout`,
  `ub_gap_p50/p90`.
- C++ → `results/cpp_<dataset>_<size>.csv` (one row per query/run); run
  `make -C vf2prob-sys-cpp aggregate` for per-method summaries.

## Mapping to the paper's tables and figures

| Paper element | Produced by | Key column(s) |
|---|---|---|
| Fig. 2 — PoC explored states + prune rate | `bench-python` | `states_med`, `prune_rate` |
| Fig. 3 + Table 3 — PoC wall-clock runtime | `bench-python` | `median_ms`, `iqr_ms` |
| Fig. 4 — bound×search ablation | `bench-python` variant sweep (`PROB_UB`/`PROB_SEARCH`) | `states_med` |
| Fig. 5 — best log-likelihood | `bench-python` | `best_loglik_med` |
| Table 4 — completion rate | `bench-python` | `success_rate` |
| Fig. 6 + Table 5 — C++ synthetic benchmark, REI | `bench-cpp-synth` | states, wall-clock |
| Table 6 — SNAP scalability | `bench-cpp-snap` (state budget B = 2×10⁵) | completion, states |

The explored-states numbers (Fig. 2, Fig. 4, Tables 5–6) are the primary metric
and are identical across machines for a fixed seed; the wall-clock panels
(Fig. 3, Fig. 6b) are secondary, machine-dependent context.

### Ordering ablation, noise sensitivity, and learning

These studies reuse the same seeded engine and live in `vf2prob-poc-python/`:

- **Expansion-order ablation** (RQ2, "Ordering"): `python3 ablate_order.py`
  toggles the gain-sorted heuristic on/off per family.
- **Attribute-noise sensitivity** (Fig. 9): the runner injects controlled noise
  via `--flip <rate>` (with `--pmin`); sweeping the flip rate reproduces the
  recovery-vs-noise curves. `grid.py` / `gridp.py` run these as resumable
  (serial / parallel) per-cell jobs under a wall-clock budget into
  `results_raw/`, which `aggregate.py` folds into the standard CSV schema.
- **Learned & calibrated compatibilities** (Figs 7–8): the fixed/calibrated
  scorers are defined in `compat.py`; the learning-and-calibration study fits and
  evaluates them on seeded train/calibration/test splits and reports the exact
  embedding-recovery accuracy and the expected calibration error (ECE).

Run `make help` in each sub-directory, and `python3 grid.py` (no arguments) or
`python3 <script>.py --help`, for the available cells and parameters.

## Determinism

Every stochastic step — graph generation, query sampling, attribute-noise
injection, the train/calibration/test splits, and the bound-tightness reservoir
sampling — is driven by explicit seeds (`SEED`, default 324). The reproducible
completion criterion is a deterministic **state budget** (B = 2×10⁵ for SNAP),
not the wall-clock limit, so completion and explored-states are bit-identical
across runs and machines; only wall-clock and peak memory vary with hardware.
