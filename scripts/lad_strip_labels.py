#!/usr/bin/env python3
"""Convert Glasgow vertex-labelled LAD -> plain (unlabelled) LAD for PathLAD+.

Glasgow LAD (what vf2prob_run --export-lad writes):
    line 1: n
    vertex i: <label> <degree> <neighbor ids...>
Plain LAD (what PathLAD+ createGraph reads, graph.c):
    line 1: n
    vertex i: <degree> <neighbor ids...>

So we drop the leading label token of every vertex line. Usage:
    lad_strip_labels.py in.lad out.lad
"""
import sys


def strip(src, dst):
    with open(src) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    out = [str(n)]
    for _ in range(n):
        next(it)  # drop label
        deg = int(next(it))
        nbrs = [next(it) for _ in range(deg)]
        out.append(" ".join([str(deg), *nbrs]))
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: lad_strip_labels.py in.lad out.lad")
    strip(sys.argv[1], sys.argv[2])
