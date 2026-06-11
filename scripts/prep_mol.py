# VF2-Prob, Jan Hladěna, FIM UHK
# -*- coding: utf-8 -*-
"""
prep_mol.py
-----------
Build an "uncertain molecular" property graph and S/M/L queries.

- If RDKit is available, parse a small set of SMILES into a single (weakly) connected graph:
  nodes: atoms, node['label']=element symbol ("C","O",...)
  edges: bonds,  edge['label'] in {"single","double","triple","aromatic"}
         edge['prob'] from bond type (with floor pmin)
- If RDKit is NOT available, fall back to synthetic molecules (chains/rings) with the same labeling.

Optionally connect molecules with low-probability "sim" edges to allow larger connected queries (L).

Outputs:
  data/mol/mol.graphml
  queries/mol/mol_S.json
  queries/mol/mol_M.json
  queries/mol/mol_L.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import networkx as nx


# helpers
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _percentile(vs, p):
    if not vs: return 0
    vs = sorted(vs)
    if p <= 0: return vs[0]
    if p >= 100: return vs[-1]
    k = (len(vs) - 1) * (p / 100.0)
    import math
    f, c = math.floor(k), math.ceil(k)
    if f == c: return vs[int(k)]
    return vs[f] * (c - k) + vs[c] * (k - f)


def _sample_connected_queries(G: nx.Graph, size_min: int, size_max: int, nqueries: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    out = []
    for qi in range(nqueries):
        k = rng.randint(size_min, size_max)
        for _ in range(128):
            s = rng.choice(nodes)
            # BFS growth
            seen = {s}
            frontier = [s]
            while len(seen) < k and frontier:
                u = frontier.pop(0)
                for w in G.neighbors(u):
                    if w not in seen:
                        seen.add(w);
                        frontier.append(w)
                    if len(seen) >= k: break
            if len(seen) < k:
                continue
            H = G.subgraph(seen).copy()
            if nx.is_connected(H):
                out.append({"id": qi, "nodes": sorted(seen)})
                break
        else:
            # fallback: biggest connected chunk we reached
            out.append({"id": qi, "nodes": sorted(seen)})
    return out


def _write_queries_json(path: Path, *, regime, size_range, nqueries, seed, queries):
    # queries je list slovníků {"id": ..., "nodes": [...]}, kde nodes jsou aktuálně int
    norm = []
    for q in queries:
        norm.append({"id": q["id"], "nodes": [str(x) for x in q["nodes"]]})
    payload = {
        "meta": {"regime": regime, "size_range": list(size_range), "nqueries": nqueries, "seed": seed},
        "queries": norm,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# RDKit path
def _try_build_from_rdkit(smiles_list: List[str], pmin: float, connect_components: bool, seed: int) -> nx.Graph | None:
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return None

    rng = random.Random(seed)
    G = nx.Graph()
    off = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)  # explicit H for richer structure
        Chem.Kekulize(mol, clearAromaticFlags=False)
        # add atoms
        for a in mol.GetAtoms():
            G.add_node(off + a.GetIdx(), label=a.GetSymbol())
        # add bonds
        for b in mol.GetBonds():
            u = off + b.GetBeginAtomIdx()
            v = off + b.GetEndAtomIdx()
            bt = b.GetBondType()
            if b.GetIsAromatic():
                lab = "aromatic";
                base = 0.85
            elif bt.name == "SINGLE":
                lab = "single";
                base = 0.90
            elif bt.name == "DOUBLE":
                lab = "double";
                base = 0.95
            elif bt.name == "TRIPLE":
                lab = "triple";
                base = 0.98
            else:
                lab = bt.name.lower();
                base = 0.88
            p = max(pmin, base - rng.uniform(0.0, 0.05))
            G.add_edge(u, v, label=lab, prob=float(p))
        off += mol.GetNumAtoms()

    comps = list(nx.connected_components(G))
    if connect_components and len(comps) > 1:
        reps = [sorted(list(c))[0] for c in comps]
        for i in range(len(reps) - 1):
            u, v = reps[i], reps[i + 1]
            G.add_edge(u, v, label="sim", prob=0.20)  # weak bridge

    return G


# Fallback synthetic molecules
def _fallback_molecules(pmin: float, connect_components: bool, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.Graph()

    def add_chain(n, base_label="C"):
        start = len(G)
        for i in range(n):
            G.add_node(start + i, label=base_label)
            if i > 0:
                G.add_edge(start + i - 1, start + i, label="single", prob=max(pmin, 0.90 - rng.uniform(0, 0.05)))
        return list(range(start, start + n))

    def add_ring(n=6, base_label="C"):
        start = len(G)
        for i in range(n):
            G.add_node(start + i, label=base_label)
        for i in range(n):
            u = start + i;
            v = start + ((i + 1) % n)
            lab = "aromatic" if n in (5, 6) else "single"
            base = 0.85 if lab == "aromatic" else 0.90
            G.add_edge(u, v, label=lab, prob=max(pmin, base - rng.uniform(0, 0.05)))
        return list(range(start, start + n))

    # build a palette of components
    comps = []
    for _ in range(6):
        comps.append(add_chain(rng.randint(5, 10)))
    for _ in range(5):
        comps.append(add_ring(rng.choice([5, 6])))
    # attach some hetero atoms
    for nid in list(G.nodes())[: max(1, len(G) // 10)]:
        if rng.random() < 0.3:
            G.nodes[nid]["label"] = rng.choice(["O", "N", "S", "F"])
    # bridge components to allow big queries
    if connect_components:
        reps = []
        seen = set()
        for c in nx.connected_components(G):
            c = list(c);
            c.sort()
            reps.append(c[0])
        for i in range(len(reps) - 1):
            u, v = reps[i], reps[i + 1]
            G.add_edge(u, v, label="sim", prob=0.20)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-data-dir", type=str, default="data/mol")
    ap.add_argument("--out-queries-dir", type=str, default="queries/mol")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pmin", type=float, default=0.7)
    ap.add_argument("--connect-components", action="store_true", help="Bridge components with low-prob edges.")
    args = ap.parse_args()

    _ensure_dir(Path(args.out_data_dir))
    _ensure_dir(Path(args.out_queries_dir))

    # small, diverse SMILES palette
    smiles = [
        "CC",  # ethane
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
        "Cc1ccccc1",  # toluene
        "O=c1ccccc1",  # benzoquinone (approx)
        "CC(=O)C",  # acetone
        "NCCO",  # ethanolamine-ish
        "c1ncccc1",  # pyridine
        "c1ccncc1",  # pyrimidine
    ]

    G = _try_build_from_rdkit(smiles, pmin=args.pmin, connect_components=args.connect_components, seed=args.seed)
    if G is None:
        print("[prep_mol] RDKit not found, using fallback synthetic molecules.")
        G = _fallback_molecules(pmin=args.pmin, connect_components=args.connect_components, seed=args.seed)

    print(f"[prep_mol] graph: |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
    out_graph = Path(args.out_data_dir) / "mol.graphml"
    nx.write_graphml(G, out_graph)
    print(f"[prep_mol] saved {out_graph}")

    regimes = {"S": ((5, 8), 16), "M": ((9, 12), 16), "L": ((13, 20), 12)}
    for reg, (rng_sz, nq) in regimes.items():
        qs = _sample_connected_queries(G, rng_sz[0], rng_sz[1], nq, seed=args.seed + hash(("mol", reg)) % 1_000_000)
        out_q = Path(args.out_queries_dir) / f"mol_{reg}.json"
        _write_queries_json(out_q, regime=reg, size_range=rng_sz, nqueries=nq, seed=args.seed, queries=qs)
        print(f"[prep_mol] saved {out_q} (n={len(qs)})")


if __name__ == "__main__":
    main()
