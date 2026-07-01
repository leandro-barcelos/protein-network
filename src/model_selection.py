import json
import os

import networkx as nx
import pandas as pd

from pdb_parser import PDB
from protein_network import (
    ProteinNetwork,
    AlphaCarbonNetwork,
    BetaCarbonNetwork,
    ChainNetwork,
    ResidueNetwork,
)
from parser import model_selection_parser


# graph_type -> (builder(pdb, cutoff, weighted), weighted variants to sweep)
GRAPH_BUILDERS = {
    "a-carbon": (
        lambda pdb, cutoff, weighted: AlphaCarbonNetwork(
            pdb, cutoff, weighted=weighted
        ),
        [False, True],
    ),
    "b-carbon": (
        lambda pdb, cutoff, weighted: BetaCarbonNetwork(pdb, cutoff, weighted=weighted),
        [False, True],
    ),
    "residue": (
        lambda pdb, cutoff, weighted: ResidueNetwork(pdb, cutoff),
        [True],
    ),
    "chain": (
        lambda pdb, cutoff, weighted: ChainNetwork(pdb, cutoff),
        [True],
    ),
}


def cutoff_grid(start: float, stop: float, step: float) -> list[float]:
    values, cur = [], start
    while cur <= stop + 1e-9:
        values.append(round(cur, 4))
        cur += step
    return values


def spectral_k_values(n_families: int | None, window: int) -> list[int]:
    if n_families is None:
        return list(range(2, 7))
    ks = {max(2, n_families + d) for d in range(-window, window + 1)}
    return sorted(ks)


def modularity(graph: nx.Graph, communities: list[set]) -> float:
    try:
        return nx.algorithms.community.modularity(graph, communities, weight="weight")
    except Exception:
        return float("nan")


def graph_stats(graph: nx.Graph) -> dict:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    return {
        "num_nodes": n,
        "num_edges": m,
        "avg_degree": (2 * m / n) if n > 0 else 0.0,
        "density": nx.density(graph),
    }


def run_algorithm(
    network: ProteinNetwork, algorithm: str, k: int, weighted: bool
) -> list[set] | None:
    if algorithm == "louvain":
        return network.louvain()
    if algorithm == "infomap":
        return network.infomap()
    if algorithm == "spectral":
        return network.spectral_bipartition(k, weighted)
    return None


def main():
    args = model_selection_parser().parse_args()

    chain_to_family: dict[str, str] = {}
    n_families: int | None = None
    if args.validate:
        with open(args.validate) as f:
            family_to_chains: dict[str, list[str]] = json.load(f)
        chain_to_family = {
            ch: fam for fam, chains in family_to_chains.items() for ch in chains
        }
        n_families = len(family_to_chains)

    cutoffs = cutoff_grid(args.cutoff_start, args.cutoff_stop, args.cutoff_step)
    k_values = spectral_k_values(n_families, args.k_window)

    pdb = PDB(args.filename)

    rows: list[dict] = []
    for graph_type in args.graphs:
        builder, weighted_variants = GRAPH_BUILDERS[graph_type]
        for cutoff in cutoffs:
            for weighted in weighted_variants:
                print(f"\n>>> {graph_type} | cutoff={cutoff} | weighted={weighted}")
                try:
                    network = builder(pdb, cutoff, weighted)
                except Exception as exc:  # skip degenerate configurations
                    print(f"    skipped ({type(exc).__name__}: {exc})")
                    continue

                stats = graph_stats(network.graph)
                if stats["num_edges"] == 0:
                    print("    skipped (no edges)")
                    continue

                base = {
                    "graph_type": graph_type,
                    "cutoff": cutoff,
                    "weighted": weighted,
                    **stats,
                }

                runs: list[tuple[str, int | None]] = [
                    ("louvain", None),
                    ("infomap", None),
                ]
                runs += [("spectral", k) for k in k_values]

                for algorithm, k in runs:
                    try:
                        communities = run_algorithm(
                            network, algorithm, k or 2, weighted
                        )
                    except Exception as exc:
                        print(f"    {algorithm} failed ({exc})")
                        continue
                    if not communities:
                        continue

                    row = {
                        **base,
                        "algorithm": algorithm,
                        "k": k,
                        "num_communities": len(communities),
                        "modularity": modularity(network.graph, communities),
                    }
                    row.update(
                        network.evaluate_communities(communities, chain_to_family)
                        if chain_to_family
                        else {
                            "ari": float("nan"),
                            "nmi": float("nan"),
                            "n_annotated": 0,
                        }
                    )
                    rows.append(row)

    if not rows:
        print("No results produced.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(args.out or ".", exist_ok=True)
    df.to_csv(os.path.join(args.out, "model_selection.csv"), index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
