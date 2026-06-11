# VF2-Prob, Jan Hladěna, FIM UHK

SEED ?= 324

.PHONY: help \
        build build-python build-cpp \
        fetch-data \
        bench-python bench-python-snap bench-python-real bench-python-synth \
        bench-cpp bench-cpp-snap bench-cpp-synth \
        run run-python run-cpp \
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
	@printf "  bench-python        synth + mol + gmark + SNAP\n"
	@printf "  bench-python-synth  synthetic S/M/L only\n"
	@printf "  bench-python-real   mol + gmark S/M/L\n"
	@printf "  bench-python-snap   SNAP facebook/enron/hepth\n"
	@printf "\nBenchmarks (C++ system):\n"
	@printf "  bench-cpp           synth + SNAP\n"
	@printf "  bench-cpp-synth     synthetic S/M/L only\n"
	@printf "  bench-cpp-snap      SNAP facebook/enron/hepth\n"
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
bench-python: bench-python-synth bench-python-real bench-python-snap

bench-python-synth:
	$(MAKE) -C vf2prob-poc-python synth_all SEED=$(SEED)

bench-python-real:
	$(MAKE) -C vf2prob-poc-python real_all SEED=$(SEED)

bench-python-snap:
	$(MAKE) -C vf2prob-poc-python snap_all SEED=$(SEED)


# =========================
# C++ system benchmarks
# =========================
bench-cpp: bench-cpp-synth bench-cpp-snap

bench-cpp-synth:
	$(MAKE) -C vf2prob-sys-cpp bench-synth SEED=$(SEED)

bench-cpp-snap:
	$(MAKE) -C vf2prob-sys-cpp bench-snap SEED=$(SEED)


# =========================
# Full pipelines
# =========================
run: fetch-data build bench-python bench-cpp

run-python: fetch-data build-python bench-python

run-cpp: fetch-data build-cpp bench-cpp


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
