import os

from pdb_parser import PDB
from protein_network import (
    ProteinNetwork,
    AlphaCarbonNetwork,
    BetaCarbonNetwork,
    ChainNetwork,
    ResidueNetwork,
)
from parser import run_model_parser


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

    if args.html and args.out:
        network.generate_interative_network(args.out)

    if args.plot:
        network.plot_degree_distribution(args.out)
        network.plot_centrality("degree", args.out)
        network.plot_centrality("betweenness", args.out)
        network.plot_centrality("closeness", args.out)

    if args.louvain or args.infomap or args.spectral:
        print("\n===== Community detection =====")

        if args.louvain:
            print("\n=== Louvain ===")
            communities = network.louvain()
            network.log_communities(communities)
            if args.plot:
                network.plot_communities(communities, os.path.join(args.out, "louvain"))
            if args.validate:
                network.validate_communities(
                    args.validate, communities, os.path.join(args.out, "louvain")
                )

        if args.infomap:
            print("\n=== InfoMap ===")
            communities = network.infomap()
            network.log_communities(communities)
            if args.plot:
                network.plot_communities(communities, os.path.join(args.out, "infomap"))
            if args.validate:
                network.validate_communities(
                    args.validate, communities, os.path.join(args.out, "infomap")
                )
                
        if args.spectral and args.groups:
            print("\n=== Spectral Bipartition ===")
            weighted = args.weighted or args.graph == "chain" or args.graph == "residue"
            communities = network.spectral_bipartition(args.groups, weighted)
            network.log_communities(communities)
            if args.plot:
                network.plot_communities(communities, os.path.join(args.out, "spectral"))
            if args.validate:
                network.validate_communities(
                    args.validate, communities, os.path.join(args.out, "spectral")
                )
            
    if args.chimerax:
            network.export_chimerax_script(communities, args.out)


if __name__ == "__main__":
    main()
