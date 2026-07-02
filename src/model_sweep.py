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
    ChainSimilarityNetwork,
    ResidueNetwork,
)
from arg_parser import model_selection_parser


# graph_type -> (builder(pdb, cutoff, weighted), weighted variants to sweep)
GRAPH_BUILDERS = {
    "a-carbon": (
        lambda pdb, cutoff, weighted, kmer, sim_method: AlphaCarbonNetwork(
            pdb, cutoff, weighted=weighted
        ),
        [False, True],
    ),
    "b-carbon": (
        lambda pdb, cutoff, weighted, kmer, sim_method: BetaCarbonNetwork(pdb, cutoff, weighted=weighted),
        [False, True],
    ),
    "residue": (
        lambda pdb, cutoff, weighted, kmer, sim_method: ResidueNetwork(pdb, cutoff),
        [True],
    ),
    "chain": (
        lambda pdb, cutoff, weighted, kmer, sim_method: ChainNetwork(pdb, cutoff),
        [True],
    ),
    "chain-sim": (
        lambda pdb, cutoff, weighted, kmer, sim_method: ChainSimilarityNetwork(
            pdb, threshold=cutoff, k=kmer, method=sim_method
        ),
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
    if algorithm == "greedy":
        return network.greedy()
    if algorithm == "labelprop":
        return network.labelprop()
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
    sim_thresholds = cutoff_grid(
        args.sim_threshold_start, args.sim_threshold_stop, args.sim_threshold_step
    )
    k_values = spectral_k_values(n_families, args.k_window)

    pdb = PDB(args.filename)

    rows: list[dict] = []
    for graph_type in args.graphs:
        builder, weighted_variants = GRAPH_BUILDERS[graph_type]
        grid = sim_thresholds if graph_type == "chain-sim" else cutoffs
        for cutoff in grid:
            for weighted in weighted_variants:
                print(f"\n>>> {graph_type} | cutoff={cutoff} | weighted={weighted}")
                try:
                    network = builder(pdb, cutoff, weighted, args.kmer, args.sim_method)
                except Exception as exc:
                    print(f"    skipped ({type(exc).__name__}: {exc})")
                    continue

                stats = graph_stats(network.graph)
                if stats["num_edges"] == 0:
                    print("    skipped (no edges)")
                    continue

                is_chain_sim = graph_type == "chain-sim"
                base = {
                    "graph_type": graph_type,
                    "cutoff": cutoff,
                    "weighted": weighted,
                    "sim_method": args.sim_method if is_chain_sim else None,
                    "kmer_k": (
                        args.kmer
                        if is_chain_sim and args.sim_method == "kmer"
                        else None
                    ),
                    **stats,
                }

                runs: list[tuple[str, int | None]] = [
                    ("louvain", None),
                    ("infomap", None),
                    ("greedy", None),
                    ("labelprop", None),
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
                            "purity": float("nan"),
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
