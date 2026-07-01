import argparse


def run_model_parser() -> argparse.ArgumentParser:
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

    comm = parser.add_argument_group("Community detection")
    comm.add_argument("-l", "--louvain", action="store_true")
    comm.add_argument("-i", "--infomap", action="store_true")
    comm.add_argument("-b", "--spectral", action="store_true")
    comm.add_argument(
        "-c", "--greedy", action="store_true", help="Greedy modularity (CNM)"
    )
    comm.add_argument(
        "--labelprop", action="store_true", help="Asynchronous label propagation"
    )
    comm.add_argument(
        "--validate",
        metavar="JSON",
        help="Validate communities against a JSON{family_name: [chain_id, ...]}",
    )
    comm.add_argument(
        "-k",
        "--groups",
        type=int,
        default=2,
        help="Target number of groups for spectral bipartition",
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

def model_selection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", help="Path to the .pdb or .tar.gz file")
    parser.add_argument(
        "--validate",
        metavar="JSON",
        help="JSON {family_name: [chain_id, ...]} used for ARI/NMI and spectral k",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=["a-carbon", "b-carbon", "residue", "chain"],
        default=["a-carbon", "b-carbon", "residue", "chain"],
        help="Graph types to sweep (default: all)",
    )
    parser.add_argument("--cutoff-start", type=float, default=4.0)
    parser.add_argument("--cutoff-stop", type=float, default=12.0)
    parser.add_argument("--cutoff-step", type=float, default=1.0)
    parser.add_argument(
        "--k-window",
        type=int,
        default=1,
        help="Spectral k is swept as n_families +/- k-window (clamped to >= 2)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="exports",
        help="Directory where model_selection.csv is saved",
    )
    return parser