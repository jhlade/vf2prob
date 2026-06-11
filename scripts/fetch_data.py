# VF2-Prob, Jan Hladěna, FIM UHK
"""
Fetch and generate all datasets into data/ and queries/.
Single entry point for all data acquisition; calls prep_*.py co-located here.
"""

import argparse
import os
import subprocess
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS)
VENV_PY = os.path.join(ROOT, "vf2prob-poc-python", ".venv", "bin", "python3")
DATA = os.path.join(ROOT, "data")
QDIR = os.path.join(ROOT, "queries")


def _python():
    if os.path.isfile(VENV_PY):
        return VENV_PY
    sys.exit(
        f"error: venv not found at {VENV_PY}\n"
        "Run 'make build-python' (or 'make build') first."
    )


def _run(script_name, extra_args):
    script = os.path.join(SCRIPTS, script_name)
    cmd = [_python(), script] + extra_args
    print(f"  running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"error: {script_name} exited with code {result.returncode}")


def fetch_snap(seed):
    print("==> SNAP datasets (facebook, enron, hep-th)", flush=True)
    os.makedirs(os.path.join(DATA, "snap"), exist_ok=True)
    os.makedirs(os.path.join(QDIR, "snap"), exist_ok=True)
    _run("prep_datasets.py", [
        "--out-data-dir", os.path.join(DATA, "snap"),
        "--out-queries-dir", os.path.join(QDIR, "snap"),
        "--datasets", "facebook,enron,hep-th",
        "--seed", str(seed),
        "--pmin", "0.6",
    ])


def fetch_mol(seed):
    print("==> Molecular graphs", flush=True)
    os.makedirs(os.path.join(DATA, "mol"), exist_ok=True)
    os.makedirs(os.path.join(QDIR, "mol"), exist_ok=True)
    _run("prep_mol.py", [
        "--out-data-dir", os.path.join(DATA, "mol"),
        "--out-queries-dir", os.path.join(QDIR, "mol"),
        "--seed", str(seed),
        "--pmin", "0.7",
        "--connect-components",
    ])


def fetch_gmark(seed):
    print("==> gMark graphs", flush=True)
    os.makedirs(os.path.join(DATA, "gmark"), exist_ok=True)
    os.makedirs(os.path.join(QDIR, "gmark"), exist_ok=True)
    _run("prep_gmark.py", [
        "--out-data-dir", os.path.join(DATA, "gmark"),
        "--out-queries-dir", os.path.join(QDIR, "gmark"),
        "--seed", str(seed),
        "--pmin", "0.7",
        "--scale", "0.5",
    ])


DATASET_HANDLERS = {
    "snap": fetch_snap,
    "mol": fetch_mol,
    "gmark": fetch_gmark,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch/generate all datasets.")
    parser.add_argument("--seed", type=int, default=324)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated list: snap,mol,gmark or 'all'",
    )
    args = parser.parse_args()

    chosen = list(DATASET_HANDLERS.keys()) if args.datasets == "all" \
        else [d.strip() for d in args.datasets.split(",")]

    unknown = [d for d in chosen if d not in DATASET_HANDLERS]
    if unknown:
        sys.exit(f"error: unknown dataset(s): {', '.join(unknown)}")

    for name in chosen:
        DATASET_HANDLERS[name](args.seed)

    print("done.", flush=True)
