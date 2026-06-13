#!/usr/bin/env python3
"""Generate a small uncertain ER GraphML + induced-subgraph queries for the BLP
baseline. Same attribute scheme as the C++ harness (node 'label', edge 'label',
edge 'prob'), small enough that the BLP is tractable.
"""
import argparse, json, math, random
from pathlib import Path
import networkx as nx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--p", type=float, default=0.06)
    ap.add_argument("--nlabels", type=int, default=5)
    ap.add_argument("--elabels", type=int, default=3)
    ap.add_argument("--pmin", type=float, default=0.7)
    ap.add_argument("--qsize", type=int, default=7)
    ap.add_argument("--nqueries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=324)
    ap.add_argument("--out-graphml", default="data/small/er60.graphml")
    ap.add_argument("--out-queries", default="queries/small/er60.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    G = nx.gnp_random_graph(a.n, a.p, seed=a.seed)
    # ensure connected-ish: keep largest component
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    for v in G.nodes():
        G.nodes[v]["label"] = rng.randrange(a.nlabels)
    for u, v in G.edges():
        G[u][v]["label"] = rng.randrange(a.elabels)
        G[u][v]["prob"] = round(rng.uniform(a.pmin, 1.0), 4)
    Path(a.out_graphml).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, a.out_graphml)

    # queries: connected induced subgraphs
    queries = []
    nodes = list(G.nodes())
    for qi in range(a.nqueries):
        start = rng.choice(nodes)
        seen = {start}
        frontier = [start]
        while len(seen) < a.qsize and frontier:
            x = frontier.pop(rng.randrange(len(frontier)))
            nbrs = [w for w in G.neighbors(x) if w not in seen]
            rng.shuffle(nbrs)
            for w in nbrs:
                if len(seen) >= a.qsize:
                    break
                seen.add(w)
                frontier.append(w)
        if len(seen) >= 3:
            queries.append({"id": qi, "nodes": sorted(seen)})
    Path(a.out_queries).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"n": a.n, "qsize": a.qsize, "seed": a.seed},
               "queries": queries}, open(a.out_queries, "w"), indent=2)
    print(f"graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges; "
          f"{len(queries)} queries -> {a.out_graphml}, {a.out_queries}")


if __name__ == "__main__":
    main()
