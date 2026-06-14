#!/usr/bin/env python3
# VF2-Prob, Jan Hladěna, FIM UHK
# Build a composite "scene" GraphML from the IAM Letter database (TU text export)
# for the geometric / pattern-recognition recovery experiment.
#
# Each IAM Letter instance is a small drawing of a capital letter: nodes carry
# 2-D coordinates (x, y), edges are strokes. We overlay K *structurally
# isomorphic* instances of one class (same node count and degree sequence) in a
# shared, normalised coordinate frame, with a constant node label. A query
# sub-pattern then embeds structurally in every component, so structure alone
# cannot localise the true correspondence -- only the geometric (RBF on x, y)
# node compatibility can. The recovery driver samples queries from this graph,
# perturbs coordinates (--coord-noise), and measures ground-truth recovery.
import argparse
import os
import random
from collections import defaultdict

CLASS_NAME = {0: "Z", 1: "N", 2: "X", 3: "T", 4: "W", 5: "A", 6: "M", 7: "K",
              8: "L", 9: "E", 10: "H", 11: "F", 12: "V", 13: "I", 14: "Y"}
NAME_CLASS = {v: k for k, v in CLASS_NAME.items()}


def _read_lines(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_tu(d, name="Letter-med"):
    """Return graphs: list of (class_int, nodes[(x,y)], edges[(u,v)] local)."""
    attrs = [tuple(float(x) for x in ln.split(","))
             for ln in _read_lines(os.path.join(d, f"{name}_node_attributes.txt"))]
    gind = [int(x) for x in _read_lines(os.path.join(d, f"{name}_graph_indicator.txt"))]
    glab = [int(x) for x in _read_lines(os.path.join(d, f"{name}_graph_labels.txt"))]
    # Global (1-indexed) node -> its graph id; group nodes per graph in order.
    nodes_of = defaultdict(list)
    for gnode, g in enumerate(gind, start=1):
        nodes_of[g].append(gnode)
    edges_of = defaultdict(set)
    for ln in _read_lines(os.path.join(d, f"{name}_A.txt")):
        u, v = (int(x) for x in ln.split(","))
        g = gind[u - 1]
        a, b = (u, v) if u < v else (v, u)
        if a != b:
            edges_of[g].add((a, b))
    graphs = []
    for g in sorted(nodes_of):
        gnodes = nodes_of[g]                       # global ids, in file order
        loc = {gn: i for i, gn in enumerate(gnodes)}
        nodes = [attrs[gn - 1] for gn in gnodes]
        edges = sorted((loc[a], loc[b]) for (a, b) in edges_of[g]
                       if a in loc and b in loc)
        graphs.append((glab[g - 1], nodes, edges))
    return graphs


def signature(nodes, edges):
    """Structural fingerprint: (#nodes, #edges, sorted degree sequence)."""
    deg = [0] * len(nodes)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return (len(nodes), len(edges), tuple(sorted(deg)))


def is_connected(nodes, edges):
    if len(nodes) <= 1:
        return True
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen, stack = {0}, [0]
    while stack:
        for y in adj[stack.pop()]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == len(nodes)


def pick_group(graphs, want_class, min_nodes):
    """Largest set of isomorphic-skeleton instances (optionally of want_class)."""
    buckets = defaultdict(list)  # (class, signature) -> [instance index]
    for idx, (cls, nodes, edges) in enumerate(graphs):
        if len(nodes) < min_nodes or not is_connected(nodes, edges):
            continue
        buckets[(cls, signature(nodes, edges))].append(idx)
    if not buckets:
        raise SystemExit(f"no instances with >= {min_nodes} nodes")
    if want_class is not None:
        buckets = {k: v for k, v in buckets.items() if k[0] == want_class}
        if not buckets:
            raise SystemExit(f"class {CLASS_NAME.get(want_class)} has no group "
                             f">= {min_nodes} nodes")
    (cls, sig), members = max(buckets.items(), key=lambda kv: len(kv[1]))
    return cls, sig, members


def write_graphml(path, nodes, edges, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        f.write("<graphml xmlns='http://graphml.graphdrawing.org/xmlns'>\n")
        f.write("<key id='d0' for='node' attr.name='label' attr.type='long'/>\n")
        f.write("<key id='d1' for='node' attr.name='x' attr.type='double'/>\n")
        f.write("<key id='d2' for='node' attr.name='y' attr.type='double'/>\n")
        f.write("<graph edgedefault='undirected'>\n")
        for i, (x, y) in enumerate(nodes):
            f.write(f"<node id='{i}'><data key='d0'>{labels[i]}</data>"
                    f"<data key='d1'>{x:.6f}</data>"
                    f"<data key='d2'>{y:.6f}</data></node>\n")
        for u, v in edges:
            f.write(f"<edge source='{u}' target='{v}'/>\n")
        f.write("</graph>\n</graphml>\n")


def build_composite(graphs, members, k, seed, label_mode):
    rng = random.Random(seed)
    chosen = list(members)
    rng.shuffle(chosen)
    chosen = chosen[:k]
    nodes, edges, labels, manifest = [], [], [], []
    for comp, idx in enumerate(chosen):
        cls, gnodes, gedges = graphs[idx]
        off = len(nodes)
        for (x, y) in gnodes:
            nodes.append((x, y))
            labels.append(comp if label_mode == "component" else 0)
        for (u, v) in gedges:
            edges.append((u + off, v + off))
        manifest.append((idx, cls, len(gnodes)))
    # Normalise coordinates jointly to a unit box (uniform scale preserves shape),
    # so the RBF bandwidth h and the jitter sigma are in interpretable units.
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    lo_x, lo_y = min(xs), min(ys)
    span = max(max(xs) - lo_x, max(ys) - lo_y, 1e-9)
    nodes = [((x - lo_x) / span, (y - lo_y) / span) for (x, y) in nodes]
    return nodes, edges, labels, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/iam")
    ap.add_argument("--name", default="Letter-med")
    ap.add_argument("--class", dest="cls", default="A",
                    help="letter class to overlay, or 'auto' for the largest group")
    ap.add_argument("--k", type=int, default=12, help="instances in the scene")
    ap.add_argument("--min-nodes", type=int, default=6)
    ap.add_argument("--label-mode", choices=["constant", "component"],
                    default="constant",
                    help="constant => structure is ambiguous across components")
    ap.add_argument("--seed", type=int, default=324)
    ap.add_argument("--out", default="data/iam/letter.graphml")
    a = ap.parse_args()

    graphs = load_tu(a.in_dir, a.name)
    want = None if a.cls.lower() == "auto" else NAME_CLASS[a.cls.upper()]
    cls, sig, members = pick_group(graphs, want, a.min_nodes)
    nodes, edges, labels, manifest = build_composite(
        graphs, members, a.k, a.seed, a.label_mode)
    write_graphml(a.out, nodes, edges, labels)

    n_per = sig[0]
    print(f"class={CLASS_NAME[cls]} signature(nodes,edges,deg)={sig}")
    print(f"group has {len(members)} isomorphic instances; used {len(manifest)}")
    print(f"composite: {len(nodes)} nodes ({n_per}/instance), {len(edges)} edges, "
          f"labels={a.label_mode}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
