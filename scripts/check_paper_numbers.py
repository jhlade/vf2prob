#!/usr/bin/env python3
"""Audit headline numbers in the DTEI2 prose against results/*.csv.

For every claim below, the value is recomputed from its source CSV and then
searched for in the named .tex file. Output per claim: OK (found), STALE
(computed value not found in the file — prose needs updating), or NO-SOURCE
(source CSV missing/incomplete — re-run the bench first).

Run after make_paper_data.py whenever benches are re-run. Exit code 1 if any
claim is STALE, 2 if only NO-SOURCE issues remain.
"""
import csv
import os
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = Path(os.environ.get("PAPER_DIR", REPO / "paper"))


def per_query_states(path, method):
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(open(path)) if r["method"] == method]
    if not rows:
        return None
    q = {}
    for r in rows:
        q.setdefault(r["query_id"], []).append(float(r["states"]))
    return {k: st.median(v) for k, v in q.items()}


def fmt(x):
    return f"{x:g}"


def main():
    results = REPO / "results"
    claims = []  # (description, tex file, list of expected strings, source ok?)

    # --- tab:rei (results-systems.tex): per-regime medians + REI ---
    for size, regime in [("S", "S"), ("M", "M"), ("L", "L")]:
        node = per_query_states(results / f"cpp_synth_{size}.csv", "vf2prob")
        aub = per_query_states(results / f"cpp_synth_{size}.csv", "vf2prob-astar-assign")
        if node is None or aub is None:
            claims.append((f"tab:rei {regime}", "sections/results-systems.tex", None, False))
            continue
        nm, am = st.median(node.values()), st.median(aub.values())
        rei = st.median(node[q] / aub[q] for q in node if q in aub and aub[q] > 0)
        claims.append(
            (
                f"tab:rei {regime} (median {fmt(nm)}->{fmt(am)}, REI {rei:.1f}x)",
                "sections/results-systems.tex",
                [f"{fmt(nm)}\\to{fmt(am)}", f"{rei:.1f}"],
                True,
            )
        )

    # --- synth-bench prose (results-systems + introduction): M-regime node vs AUB ---
    node = per_query_states(results / "cpp_synth_M.csv", "vf2prob")
    aub = per_query_states(results / "cpp_synth_M.csv", "vf2prob-astar-assign")
    if node and aub:
        nm, am = st.median(node.values()), st.median(aub.values())
        claims.append(
            (
                f"synth-bench M medians ({fmt(am)} vs {fmt(nm)})",
                "sections/results-systems.tex",
                [fmt(nm), fmt(am)],
                True,
            )
        )
        claims.append(
            (
                f"intro contribution 3 ({fmt(am)} vs {fmt(nm)})",
                "sections/introduction.tex",
                [fmt(nm), fmt(am)],
                True,
            )
        )
    else:
        claims.append(("synth-bench M medians", "sections/results-systems.tex", None, False))

    # --- competitor sentence: pooled S+M+L paired MPM vs AUB ---
    pairs = []
    for size in ["S", "M", "L"]:
        mpm = per_query_states(results / f"cpp_synth_{size}.csv", "mpm")
        a = per_query_states(results / f"cpp_synth_{size}.csv", "vf2prob-astar-assign")
        if mpm is None or a is None:
            pairs = None
            break
        pairs += [(mpm[q], a[q]) for q in mpm if q in a]
    if pairs:
        mm = st.median(x for x, _ in pairs)
        am = st.median(y for _, y in pairs)
        rei = st.median(x / y for x, y in pairs if y > 0)
        claims.append(
            (
                f"competitor pooled (MPM {fmt(mm)} vs AUB {fmt(am)}, REI {rei:.1f})",
                "sections/results-systems.tex",
                [fmt(mm), fmt(am)],
                True,
            )
        )
    else:
        claims.append(("competitor pooled", "sections/results-systems.tex", None, False))

    # --- abstract/conclusion "up to NNx" from tab:rei L REI ---
    node = per_query_states(results / "cpp_synth_L.csv", "vf2prob")
    aub = per_query_states(results / "cpp_synth_L.csv", "vf2prob-astar-assign")
    if node and aub:
        rei = st.median(node[q] / aub[q] for q in node if q in aub and aub[q] > 0)
        up = f"{round(rei):d}"
        for tex in ["paper-cas.tex", "sections/conclusion.tex"]:
            claims.append((f"'up to ~{up}x' states reduction", tex, [up], True))

    # --- recovery/ML prose: sources not yet in results/ (regenerated on weekend) ---
    for name in ["recovery_synth", "recovery_hepth", "sweep_weights"]:
        ok = (results / f"{name}.csv").exists()
        claims.append(
            (f"{name}.csv present (recovery/sweep prose verifiable)", "-", [], ok)
        )

    stale = nosrc = 0
    for desc, tex, expected, ok in claims:
        if not ok:
            print(f"NO-SOURCE  {desc}")
            nosrc += 1
            continue
        if not expected:
            print(f"OK         {desc}")
            continue
        text = (PAPER / tex).read_text()
        missing = [e for e in expected if e not in text]
        if missing:
            print(f"STALE      {desc} — not found in {tex}: {missing}")
            stale += 1
        else:
            print(f"OK         {desc}")
    print(f"\n{stale} stale, {nosrc} without source")
    return 1 if stale else (2 if nosrc else 0)


if __name__ == "__main__":
    sys.exit(main())
