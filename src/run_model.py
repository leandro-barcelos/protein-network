import json
import os

from pdb_parser import PDB
from protein_network import (
    ProteinNetwork,
    AlphaCarbonNetwork,
    BetaCarbonNetwork,
    ChainNetwork,
    ChainSimilarityNetwork,
    ResidueNetwork,
)
from arg_parser import run_model_parser
from utils import create_dir


def main():
    args = run_model_parser().parse_args()

    pdb = PDB(args.filename)

    network: ProteinNetwork = None
    if args.graph == "a-carbon" and args.cutoff:
        network = AlphaCarbonNetwork(
            pdb,
            args.cutoff,
            weighted=args.weighted,
        )
    elif args.graph == "b-carbon" and args.cutoff:
        network = BetaCarbonNetwork(
            pdb,
            args.cutoff,
            weighted=args.weighted,
        )
    elif args.graph == "chain" and args.cutoff:
        network = ChainNetwork(pdb, args.cutoff)
    elif args.graph == "residue" and args.cutoff:
        network = ResidueNetwork(pdb, args.cutoff)
    elif args.graph == "chain-sim":
        network = ChainSimilarityNetwork(
            pdb,
            threshold=args.sim_threshold,
            k=args.kmer,
            method=args.sim_method,
        )

    if network is None:
        print("Failed to create network")
        return

    if args.statistics:
        print("\n--- Network statistics ---")
        stats = network.compute_network_stats()
        for key, val in stats.items():
            print(
                f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}"
            )

    if not os.path.isdir("exports"):
        os.mkdir("exports")

    any_community = (
        args.louvain or args.infomap or args.greedy or args.labelprop or args.spectral
    )

    cent = network.compute_centralities() if (args.plot or any_community) else None

    if args.plot:
        network.plot_degree_distribution(args.out)
        create_dir(args.out)
        cent.to_csv(os.path.join(args.out, "centralities.csv"))
        network.plot_centralities(args.out, cent=cent)
        for measure in ("degree", "betweenness", "closeness", "eigenvector", "strength"):
            network.plot_centrality(measure, args.out)

    if args.validate:
        with open(args.validate) as f:
            family_to_chains = json.load(f)
        chain_to_family = {
            ch: fam for fam, chains in family_to_chains.items() for ch in chains
        }
        validation_label = os.path.basename(args.validate)
    else:
        chains = {network.graph.nodes[n]["chain_id"] for n in network.graph.nodes()}
        chain_to_family = {c: c for c in chains}
        validation_label = "cadeias"

    def handle_communities(name: str, communities: list[set]):
        subdir = os.path.join(args.out, name)
        create_dir(subdir)
        network.log_communities(communities)
        network.community_composition(communities).to_csv(
            os.path.join(subdir, "composition.csv"), index=False
        )
        network.export_membership_csv(communities, subdir)
        if args.plot:
            network.plot_communities(communities, subdir)
            network.plot_structure_3d(communities, subdir)
            network.plot_graph(communities, subdir, cent=cent)
        if args.validate:
            network.validate_communities(args.validate, communities, subdir)
        network.generate_report(
            subdir,
            {**vars(args), "method": name},
            communities,
            chain_to_family,
            validation_label,
            cent=cent,
        )
        if args.chimerax:
            network.export_chimerax_script(communities, subdir)

    if any_community:
        print("\n===== Community detection =====")

        if args.louvain:
            handle_communities("louvain", network.louvain())
        if args.infomap:
            handle_communities("infomap", network.infomap())
        if args.greedy:
            handle_communities("greedy", network.greedy())
        if args.labelprop:
            handle_communities("labelprop", network.labelprop())
        if args.spectral and args.groups:
            weighted = args.weighted or args.graph in ("chain", "residue", "chain-sim")
            handle_communities(
                "spectral", network.spectral_bipartition(args.groups, weighted)
            )


if __name__ == "__main__":
    main()
