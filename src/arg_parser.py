"""Command-line interfaces for building and analysing protein-structure networks.

Two entry points share these parsers:
  * run_model.py         - build one network and run detection / exports
  * model_selection.py   - sweep graphs, cutoffs and algorithms into a CSV
"""

import argparse
from textwrap import dedent

GRAPH_TYPES = ("a-carbon", "b-carbon", "residue", "chain", "chain-sim")
SIM_METHODS = ("kmer", "identity")
DEFAULT_OUT = "exports"


def _add_similarity_args(group: argparse._ArgumentGroup) -> None:
    """Options shared by both parsers to configure chain-similarity graphs."""
    group.add_argument(
        "--sim-method",
        choices=SIM_METHODS,
        default="kmer",
        help="Chain similarity measure: k-mer cosine (fast) or alignment identity",
    )
    group.add_argument(
        "--kmer",
        type=int,
        default=3,
        metavar="K",
        help="k for k-mer profiles, used when --sim-method=kmer (default: %(default)s)",
    )


def run_model_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a single protein network from a structure file, then run "
        "centralities, community detection and exports.",
        epilog=dedent(
            """\
            Examples:
              # alpha-carbon contact network at 8 A, Louvain + statistics
              python src/run_model.py structure.pdb -g a-carbon --cutoff 8 -l -s

              # chain-similarity network validated against known families, with plots
              python src/run_model.py bundle.tar.gz -g chain-sim --sim-threshold 0.5 \\
                  -l -p --validate families.json
            """
        ),
    )
    parser.add_argument("filename", help="Path to a .pdb or .tar.gz structure file")
    parser.add_argument(
        "-g",
        "--graph",
        choices=GRAPH_TYPES,
        required=True,
        help="Type of network to build",
    )

    contact = parser.add_argument_group("Contact graphs (a-carbon, b-carbon, residue, chain)")
    contact.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        metavar="ANGSTROM",
        help="Distance cutoff for contacts (default: %(default)s)",
    )
    contact.add_argument(
        "--weighted",
        action="store_true",
        help="Weight a-carbon/b-carbon edges by the inverse distance between nodes",
    )

    sim = parser.add_argument_group("Chain similarity graph (chain-sim)")
    sim.add_argument(
        "--sim-threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="Minimum sequence similarity to connect two chains (default: %(default)s)",
    )
    _add_similarity_args(sim)

    comm = parser.add_argument_group("Community detection")
    comm.add_argument("-l", "--louvain", action="store_true", help="Louvain modularity")
    comm.add_argument("-i", "--infomap", action="store_true", help="Infomap")
    comm.add_argument("-c", "--greedy", action="store_true", help="Greedy modularity (CNM)")
    comm.add_argument(
        "--labelprop", action="store_true", help="Asynchronous label propagation"
    )
    comm.add_argument("-b", "--spectral", action="store_true", help="Spectral bipartition")
    comm.add_argument(
        "-k",
        "--groups",
        type=int,
        default=2,
        metavar="N",
        help="Number of groups for spectral bipartition (default: %(default)s)",
    )
    comm.add_argument(
        "--validate",
        metavar="JSON",
        help="Validate communities against a JSON {family_name: [chain_id, ...]}",
    )

    export = parser.add_argument_group("Exports")
    export.add_argument(
        "-s", "--statistics", action="store_true", help="Print network statistics"
    )
    export.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Save plots (degree distribution, centralities, communities)",
    )
    export.add_argument(
        "--html", action="store_true", help="Save an interactive HTML network"
    )
    export.add_argument(
        "--chimerax",
        action="store_true",
        help="Export a ChimeraX script for 3D community visualization",
    )
    export.add_argument(
        "-o",
        "--out",
        default=DEFAULT_OUT,
        metavar="DIR",
        help="Directory for saving network exports (default: %(default)s)",
    )

    return parser


def model_selection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Sweep graph types, cutoffs/thresholds and community algorithms, "
        "scoring each combination against known families (ARI/NMI) into a CSV.",
        epilog=dedent(
            """\
            Examples:
              # full sweep scored against known families
              python src/model_selection.py bundle.tar.gz --validate families.json

              # only chain-sim, with a finer threshold grid
              python src/model_selection.py bundle.tar.gz --graphs chain-sim \\
                  --sim-threshold-start 0.3 --sim-threshold-stop 0.7 --sim-threshold-step 0.05
            """
        ),
    )
    parser.add_argument("filename", help="Path to a .pdb or .tar.gz structure file")
    parser.add_argument(
        "--validate",
        metavar="JSON",
        help="JSON {family_name: [chain_id, ...]} used for ARI/NMI and spectral k",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=GRAPH_TYPES,
        default=list(GRAPH_TYPES),
        help="Graph types to sweep (default: all). Contact graphs use the --cutoff-* "
        "grid (angstroms); chain-sim uses the --sim-threshold-* grid.",
    )

    sim = parser.add_argument_group("Chain similarity (chain-sim)")
    _add_similarity_args(sim)

    grid = parser.add_argument_group("Sweep grids")
    grid.add_argument(
        "--cutoff-start", type=float, default=4.0, metavar="ANGSTROM"
    )
    grid.add_argument("--cutoff-stop", type=float, default=12.0, metavar="ANGSTROM")
    grid.add_argument("--cutoff-step", type=float, default=1.0, metavar="ANGSTROM")
    grid.add_argument(
        "--sim-threshold-start",
        type=float,
        default=0.1,
        metavar="T",
        help="Start of the chain-sim similarity threshold grid (0..1)",
    )
    grid.add_argument("--sim-threshold-stop", type=float, default=0.9, metavar="T")
    grid.add_argument("--sim-threshold-step", type=float, default=0.1, metavar="T")
    grid.add_argument(
        "--k-window",
        type=int,
        default=1,
        metavar="W",
        help="Spectral k is swept as n_families +/- k-window (clamped to >= 2)",
    )

    parser.add_argument(
        "-o",
        "--out",
        default=DEFAULT_OUT,
        metavar="DIR",
        help="Directory where model_selection.csv is saved (default: %(default)s)",
    )
    return parser
