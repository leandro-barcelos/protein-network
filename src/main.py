import argparse
import os

from pdb_parser import PDB
from network_builder import (
    ProteinNetwork,
    AlphaCarbonNetwork,
    BetaCarbonNetwork,
    ChainNetwork,
    ResidueNetwork,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the .pdb or .tar.gz file")

    parser.add_argument(
        "-g",
        "--graph",
        choices=["a-carbon", "b-carbon", "residue", "chain"],
        required=True,
        help="Type of graph to create",
    )

    parser.add_argument("--cutoff", help="Distance cutoff", type=float, default=8.0)
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Weight edges by the inverse of the distance between the two nodes in a-carbon or b-carbon graphs",
    )

    parser.add_argument(
        "--scaling", help="Parameter s for residue networks", type=float, default=4.0
    )

    comm_algo = parser.add_mutually_exclusive_group()
    comm_algo.add_argument("-l", "--louvain", action="store_true")
    comm_algo.add_argument("-i", "--infomap", action="store_true")

    parser.add_argument(
        "--validate",
        metavar="JSON",
        help="Validate communities against a JSON{family_name: [chain_id, ...]}",
    )

    export = parser.add_argument_group("Exports")
    export.add_argument("-s", "--statistics", action="store_true")
    export.add_argument(
        "--html", action="store_true", help="Generate interactive graph"
    )
    export.add_argument("-p", "--plot", action="store_true", help="Plot graphs")
    export.add_argument(
        "--chimerax",
        action="store_true",
        help="Export ChimeraX script for 3D community visualization",
    )
    export.add_argument(
        "-o", "--out", help="Directory for saving network exports", default="exports"
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

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
        network.generate_interative_network(network, args.out)

    if args.plot:
        network.plot_degree_distribution(network, args.out)
        network.plot_centrality(network, "degree", args.out)
        network.plot_centrality(network, "betweenness", args.out)
        network.plot_centrality(network, "closeness", args.out)

    if args.louvain or args.infomap:
        print("\n--- Community detection ---")
        if args.louvain:
            communities = network.louvain()
        else:
            communities = network.infomap()

        print(f"Communities found: {len(communities)}")
        network.plot_communities(communities, args.out)

        if args.chimerax:
            network.export_chimerax_script(communities, args.out)

        if args.validate:
            network.validate_communities(args.validate, communities)


if __name__ == "__main__":
    main()
