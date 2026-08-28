# VF2-Prob, Jan Hladěna, FIM UHK

SEED ?= 324

.PHONY: help \
        build build-python build-cpp \
        fetch-data \
        bench-python bench-python-real bench-python-synth \
        bench-cpp bench-cpp-snap bench-cpp-synth \
        fetch-iam bench-iam iam bench-recovery \
        run run-python run-cpp run-all \
        test \
        clean clean-python clean-cpp clean-results

help:
	@printf "Usage: make [target] [SEED=N]\n\n"
	@printf "Data:\n"
	@printf "  fetch-data          download SNAP; generate mol/gmark data\n"
	@printf "\nBuild:\n"
	@printf "  build               build-python + build-cpp\n"
	@printf "  build-python        create Python venv, install dependencies\n"
	@printf "  build-cpp           cmake configure + compile C++\n"
	@printf "\nBenchmarks (Python PoC):\n"
	@printf "  bench-python        synth + mol + gmark\n"
	@printf "  bench-python-synth  synthetic S/M/L only\n"
	@printf "  bench-python-real   mol + gmark S/M/L\n"
	@printf "\nBenchmarks (C++ system):\n"
	@printf "  bench-cpp           synth + SNAP\n"
	@printf "  bench-cpp-synth     synthetic S/M/L only\n"
	@printf "  bench-cpp-snap      SNAP facebook/enron/hepth\n"
	@printf "  iam                 IAM Letter geometric recovery (fetch + run)\n"
	@printf "  bench-recovery      recovery vs noise: synth, ca-HepTh, STRING, IAM\n"
	@printf "  paper-data          regenerate paper figure/table data from results (PAPER_DIR)\n"
	@printf "\nFull pipelines:\n"
	@printf "  run                 fetch-data + build + bench-python + bench-cpp\n"
	@printf "  run-python          fetch-data + build-python + bench-python\n"
	@printf "  run-cpp             fetch-data + build-cpp + bench-cpp\n"
	@printf "\nTest:\n"
	@printf "  test                C++ unit tests via ctest\n"
	@printf "\nClean:\n"
	@printf "  clean               remove venv + C++ build\n"
	@printf "  clean-results       remove all CSVs in results/\n"
	@printf "\nSEED defaults to 324.\n"


# =========================
# Build
# =========================
build: build-python build-cpp

build-python:
	$(MAKE) -C vf2prob-poc-python build

build-cpp:
	$(MAKE) -C vf2prob-sys-cpp build


# =========================
# Data acquisition
# =========================
fetch-data: build-python
	vf2prob-poc-python/.venv/bin/python3 scripts/fetch_data.py --seed $(SEED)


# =========================
# Python PoC benchmarks
# =========================
# SNAP is a C++-only scalability track in the paper (the Python PoC covers only
# synth/mol/gmark; see DTEI2 experiments.tex). Hence no bench-python-snap here.
bench-python: bench-python-synth bench-python-real

bench-python-synth:
	$(MAKE) -C vf2prob-poc-python synth_all SEED=$(SEED)

bench-python-real:
	$(MAKE) -C vf2prob-poc-python real_all SEED=$(SEED)


# =========================
# C++ system benchmarks
# =========================
bench-cpp: bench-cpp-synth bench-cpp-snap

bench-cpp-synth:
	$(MAKE) -C vf2prob-sys-cpp bench-synth SEED=$(SEED)

bench-cpp-snap:
	$(MAKE) -C vf2prob-sys-cpp bench-snap SEED=$(SEED)


# =========================
# IAM Letter geometric recovery (computer-vision pattern recognition)
# =========================
# A scene of 15 isomorphic same-class letter drawings overlaid in one coordinate
# frame: structure alone cannot localise the query, so recovery measures what the
# RBF kernel on (x, y) buys over a structure-only matcher under coordinate noise.
fetch-iam: build-python
	vf2prob-poc-python/.venv/bin/python3 scripts/fetch_data.py --datasets iam --seed $(SEED)

bench-iam: build-cpp
	@mkdir -p results
	vf2prob-sys-cpp/build/recovery \
	  --data-path data/iam/letter.graphml --node-kernel rbf --kernel-h 0.15 \
	  --qsize 6 --instances 100 --methods vf2pp,vf2prob-astar-assign \
	  --flips 0,0.02,0.05,0.1,0.15,0.2,0.25,0.35,0.5 \
	  --learned 0 --decoy-frac 0 --true-pmin 1 --seed $(SEED) \
	  --out results/iam_recovery.csv

iam: fetch-iam bench-iam

# All recovery-vs-noise sweeps that feed the robustness figure (Fig. recovery).
# Real-graph sweeps are skipped (not fatal) if their data is absent.
bench-recovery: build-cpp
	@mkdir -p results
	vf2prob-sys-cpp/build/recovery --instances 200 --structured 1 \
	  --methods mpm,vf2bin,vf2prob-astar-assign --seed $(SEED) --out results/recovery_synth.csv
	@[ -f data/snap/ca_hepth.graphml ] \
	  && vf2prob-sys-cpp/build/recovery --instances 50 --data-path data/snap/ca_hepth.graphml \
	     --relabel 12 --methods mpm,vf2bin,vf2prob-astar-assign --seed $(SEED) --out results/recovery_hepth.csv \
	  || echo "  (skip ca-HepTh recovery: run 'make fetch-data' first)"
	@[ -f data/string/scerevisiae.graphml ] \
	  && vf2prob-sys-cpp/build/recovery --instances 50 --data-path data/string/scerevisiae.graphml \
	     --methods mpm,vf2bin,vf2prob-astar-assign --seed $(SEED) --out results/string_recovery.csv \
	  || echo "  (skip STRING recovery: prep the STRING graph first)"
	@[ -f data/mol/mol.graphml ] \
	  && vf2prob-sys-cpp/build/recovery --instances 50 --data-path data/mol/mol.graphml \
	     --relabel 6 --methods mpm,vf2bin,vf2prob-astar-assign --seed $(SEED) --out results/recovery_mol.csv \
	  || echo "  (skip mol recovery: run 'make fetch-data' first)"
	$(MAKE) bench-iam SEED=$(SEED)


# =========================
# Full pipelines
# =========================
run: fetch-data build bench-python bench-cpp

run-python: fetch-data build-python bench-python

run-cpp: fetch-data build-cpp bench-cpp

run-all: run bench-recovery

# Regenerate every data-driven figure/table input of the paper from results/*.csv.
# PAPER_DIR must point at the paper's TeX directory (defaults to ./paper).
paper-data:
	python3 scripts/make_paper_data.py
	python3 scripts/make_poc_figures.py
	python3 scripts/paper_fallback.py results/*.log


# =========================
# Tests
# =========================
test:
	$(MAKE) -C vf2prob-sys-cpp test


# =========================
# Housekeeping
# =========================
clean: clean-python clean-cpp

clean-python:
	rm -rf vf2prob-poc-python/.venv

clean-cpp:
	$(MAKE) -C vf2prob-sys-cpp clean

clean-results:
	rm -f results/*.csv
