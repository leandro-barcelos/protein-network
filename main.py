import argparse
import json
import logging
import os
import tarfile
from typing import Optional

import infomap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from ase.data import atomic_numbers, covalent_radii
from biopandas.pdb import PandasPdb
from pyvis.network import Network
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

LOGGER = logging.getLogger(__name__)
TAR_OUT_DIR = "extracted_pdbs"


def pdb_to_dataframe(
    pdb_path: str,
    model_index: int = 1,
) -> pd.DataFrame:
    LOGGER.info(f"Loading PDB file ({pdb_path})")

    pdbb = PandasPdb().read_pdb(pdb_path)
    model = pdbb.get_model(model_index)
    atom_df = model.df["ATOM"]
    if len(atom_df) == 0:
        LOGGER.fatal(f"No model found for index: {model_index}")
        exit(-1)

    return atom_df


def load_tar_pdb(filepath: str) -> pd.DataFrame:
    LOGGER.info(f"Extracting {filepath} to {TAR_OUT_DIR}")

    if os.path.isdir(TAR_OUT_DIR):
        for name in os.listdir(TAR_OUT_DIR):
            if not name.endswith(".pdb"):
                continue
            os.remove(os.path.join(TAR_OUT_DIR, name))

    with tarfile.open(filepath) as file:
        file.extractall(TAR_OUT_DIR)

    bundles = []
    for name in os.listdir(TAR_OUT_DIR):
        if not name.endswith(".pdb"):
            continue

        df = pdb_to_dataframe(os.path.join(TAR_OUT_DIR, name))
        bundles.append(df)

    if not bundles:
        LOGGER.fatal(f"No .pdb found in {filepath}")
        exit(-1)

    return pd.concat(bundles, ignore_index=True)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Processing dataframe")

    df_copy = df.copy()
    df_copy = df_copy.drop(
        columns=[
            "record_name",
            "atom_number",
            "blank_1",
            "blank_2",
            "blank_3",
            "occupancy",
            "b_factor",
            "blank_4",
            "segment_id",
            "charge",
            "line_idx",
            "model_id",
        ]
    )

    if "alt_loc" in df_copy.columns:
        df_copy = df_copy[df_copy["alt_loc"].isin(["", "A"])]

    df_copy["node_id"] = (
        df_copy["chain_id"].astype(str)
        + ":"
        + df_copy["residue_number"].astype(str)
        + df_copy["insertion"].astype(str).str.strip()
    )

    return df_copy  # pyright: ignore[reportReturnType]


def load_file(filepath: str) -> pd.DataFrame:
    if not filepath.endswith(".pdb") and not filepath.endswith(".tar.gz"):
        raise ValueError("Formats accepted: .pdb, .tar.gz")

    df: pd.DataFrame | None = None

    if filepath.endswith(".pdb"):
        df = pdb_to_dataframe(filepath)

    if filepath.endswith(".tar.gz"):
        df = load_tar_pdb(filepath)

    if df is None:
        raise Exception("failed to load PDB file")

    df = process_dataframe(df)

    LOGGER.info(f"Dataframe columns {df.columns}")
    LOGGER.info(f"Dataframe shape {df.shape}")

    return df


def filter_atoms(df: pd.DataFrame, atom: str = "CA") -> pd.DataFrame:
    df_copy = df.copy()
    if atom == "CB":
        mask_cb = df_copy["atom_name"] == "CB"
        mask_gly_ca = (df_copy["residue_name"] == "GLY") & (
            df_copy["atom_name"] == "CA"
        )
        df_copy = df_copy.loc[mask_cb | mask_gly_ca]
    else:
        df_copy = df_copy.loc[df_copy["atom_name"] == atom]
    df_copy = df_copy.drop_duplicates(
        subset=["chain_id", "residue_number", "insertion"]
    )
    return df_copy.reset_index(drop=True)


def build_carbon_graph(
    df: pd.DataFrame,
    cutoff: float,
    atom: str = "CA",
    lower_cutoff: Optional[float] = None,
    weighted: bool = False,
) -> nx.Graph:
    filtered_df = filter_atoms(df, atom)
    LOGGER.info(f"Filtered to {len(filtered_df)} {atom} atoms")

    coords = filtered_df[["x_coord", "y_coord", "z_coord"]].to_numpy()

    tree = cKDTree(coords)
    pairs = tree.query_pairs(cutoff, output_type="set")
    if lower_cutoff is not None:
        pairs -= tree.query_pairs(lower_cutoff, output_type="set")

    G = nx.Graph()
    G.add_nodes_from(range(len(filtered_df)))
    if weighted:
        for i, j in pairs:
            distance = np.linalg.norm(coords[i] - coords[j])
            G.add_edge(i, j, weight=1 / distance)
    else:
        G.add_edges_from(pairs, weight=1)
    for i, (node_id, chain_id, x, y, z, resnum, insertion) in enumerate(
        zip(
            filtered_df["node_id"],
            filtered_df["chain_id"],
            filtered_df["x_coord"],
            filtered_df["y_coord"],
            filtered_df["z_coord"],
            filtered_df["residue_number"],
            filtered_df["insertion"].str.strip(),
        )
    ):
        G.nodes[i]["node_id"] = node_id
        G.nodes[i]["chain_id"] = chain_id
        G.nodes[i]["x_coord"] = x
        G.nodes[i]["y_coord"] = y
        G.nodes[i]["z_coord"] = z
        G.nodes[i]["residue_number"] = resnum
        G.nodes[i]["insertion"] = insertion
    LOGGER.info(
        f"Alpha-carbon network: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def build_alpha_carbon_graph(
    df: pd.DataFrame,
    cutoff: float,
    lower_cutoff: Optional[float] = None,
    weighted: bool = False,
) -> nx.Graph:
    LOGGER.info(f"Building alpha-carbon network (cutoff={cutoff})")
    return build_carbon_graph(df, cutoff, lower_cutoff=lower_cutoff, weighted=weighted)


def build_beta_carbon_graph(
    df: pd.DataFrame,
    cutoff: float,
    lower_cutoff: Optional[float] = None,
    weighted: bool = False,
) -> nx.Graph:
    LOGGER.info(f"Building beta-carbon network (cutoff={cutoff})")
    return build_carbon_graph(
        df, cutoff, atom="CB", lower_cutoff=lower_cutoff, weighted=weighted
    )


def build_grant_ahnert_graph(df: pd.DataFrame, s: float) -> nx.Graph:
    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()
    distance = cdist(coords, coords)

    radii = np.array(
        [covalent_radii[atomic_numbers[atom]] for atom in df["element_symbol"]]
    )
    cutoff = s * (radii[:, None] + radii[None, :])

    adjacency_matrix = distance <= cutoff
    np.fill_diagonal(adjacency_matrix, 0)

    G = nx.from_numpy_array(adjacency_matrix)
    for i, (node_id, chain_id, x, y, z, resnum, insertion) in enumerate(
        zip(
            df["node_id"],
            df["chain_id"],
            df["x_coord"],
            df["y_coord"],
            df["z_coord"],
            df["residue_number"],
            df["insertion"].str.strip(),
        )
    ):
        G.nodes[i]["node_id"] = node_id
        G.nodes[i]["chain_id"] = chain_id
        G.nodes[i]["x_coord"] = x
        G.nodes[i]["y_coord"] = y
        G.nodes[i]["z_coord"] = z
        G.nodes[i]["residue_number"] = resnum
        G.nodes[i]["insertion"] = insertion
    return G


def build_residue_graph(df: pd.DataFrame, s: float = 4.0) -> nx.Graph:
    LOGGER.info(f"Building Grant-Ahnert residue network (s={s}, {len(df)} atoms)")
    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()
    distance = cdist(coords, coords)

    radii = np.array(
        [covalent_radii[atomic_numbers[atom]] for atom in df["element_symbol"]]
    )
    cutoff = s * (radii[:, None] + radii[None, :])
    atomic_adj = distance <= cutoff
    np.fill_diagonal(atomic_adj, 0)
    LOGGER.info(f"Atomic network: {int(atomic_adj.sum() // 2)} contacts")

    residues, atom_to_res = np.unique(df["node_id"].to_numpy(), return_inverse=True)
    n_atoms, n_res = len(df), len(residues)
    membership = sparse.csr_matrix(
        (np.ones(n_atoms), (np.arange(n_atoms), atom_to_res)),
        shape=(n_atoms, n_res),
    )
    weights = membership.T @ sparse.csr_matrix(atomic_adj) @ membership
    weights = sparse.triu(weights, k=1).tocoo()

    reps: dict[str, dict] = {}
    for node_id, group in df.groupby("node_id", sort=False):
        ca = group[group["atom_name"] == "CA"]
        ref = ca.iloc[0] if len(ca) else group.iloc[0]
        if len(ca):
            x, y, z = ref["x_coord"], ref["y_coord"], ref["z_coord"]
        else:
            x, y, z = (
                group["x_coord"].mean(),
                group["y_coord"].mean(),
                group["z_coord"].mean(),
            )
        reps[node_id] = {
            "node_id": str(node_id),
            "chain_id": str(ref["chain_id"]),
            "residue_number": int(ref["residue_number"]),
            "insertion": str(ref["insertion"]).strip(),
            "x_coord": float(x),
            "y_coord": float(y),
            "z_coord": float(z),
        }

    G = nx.Graph()
    for i, node_id in enumerate(residues):
        G.add_node(i, **reps[node_id])
    for a, b, w in zip(weights.row, weights.col, weights.data):
        G.add_edge(int(a), int(b), weight=int(w))
    LOGGER.info(
        f"Residue network: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} weighted edges"
    )
    return G


def build_residue_graph_per_chain(df: pd.DataFrame, s: float = 4.0) -> nx.Graph:
    subgraphs = []
    for chain_id, chain_df in df.groupby("chain_id", sort=False):
        LOGGER.info(f"Building residue network for chain {chain_id}")
        subgraphs.append(build_residue_graph(chain_df.reset_index(drop=True), s))
    return nx.disjoint_union_all(subgraphs)


def generate_interative_network(G: nx.Graph, path: str):
    net = Network(height="800px", width="100%", notebook=False)
    net.from_nx(G)
    net.write_html(path, notebook=False)


def compute_network_stats(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()

    stats: dict = {
        "num_nodes": n,
        "num_edges": m,
        "avg_degree": (2 * m / n) if n > 0 else 0.0,
        "density": nx.density(G),
        "clustering_coefficient": nx.average_clustering(G),
        "assortativity": (
            nx.degree_assortativity_coefficient(G) if m > 0 else float("nan")
        ),
    }

    largest_cc = max(nx.connected_components(G), key=len, default=set())
    G_lcc = G.subgraph(largest_cc)
    stats["largest_component_size"] = len(largest_cc)
    stats["largest_component_fraction"] = len(largest_cc) / n if n > 0 else 0.0
    stats["avg_shortest_path"] = (
        nx.average_shortest_path_length(G_lcc) if len(largest_cc) > 1 else 0.0
    )

    return stats


def plot_degree_distribution(G: nx.Graph, path: str):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)

    plt.figure()
    sns.histplot(degree_sequence, kde=True)
    plt.title("Degree histogram")
    plt.xlabel("Degree")
    plt.ylabel("# of Nodes")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_centrality(G: nx.Graph, measure_name: str, path: str):
    if measure_name == "betweenness":
        cent = nx.betweenness_centrality(G)
    elif measure_name == "degree":
        cent = nx.degree_centrality(G)
    elif measure_name == "closeness":
        cent = nx.closeness_centrality(G)
    else:
        return

    nodes = list(G.nodes())
    values = np.array([cent[n] for n in nodes])
    vmax = values.max() if values.max() > 0 else 1.0
    sizes = 20 + 400 * (values / vmax)

    pos = {n: (G.nodes[n]["x_coord"], G.nodes[n]["y_coord"]) for n in G.nodes()}

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.4)
    nodes_drawn = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_size=sizes,
        node_color=values,
        cmap=plt.cm.plasma,
    )
    plt.colorbar(nodes_drawn, label=measure_name.capitalize())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_communities(G: nx.Graph, communities: list[set], path: str):
    for i, comm in enumerate(communities):
        chains = {G.nodes[n]["node_id"].split(":")[0] for n in comm}
        print(f"Comm {i}: {chains}")

    node_to_community = {n: i for i, comm in enumerate(communities) for n in comm}
    colors = sns.color_palette(None, len(communities))
    node_colors = [colors[node_to_community[node]] for node in G.nodes()]

    pos = {n: (G.nodes[n]["x_coord"], G.nodes[n]["y_coord"]) for n in G.nodes()}

    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=30,
        width=0.3,
        with_labels=False,
    )
    plt.title("Communities (real spatial coordinates)", fontsize=18)
    plt.savefig(path, dpi=150)
    plt.close()


def biological_validation(
    validation_path: str, G: nx.Graph, communities: list[set]
) -> dict:
    with open(validation_path) as f:
        family_to_chains: dict[str, list[str]] = json.load(f)
    chain_to_family = {
        ch: family for family, chains in family_to_chains.items() for ch in chains
    }

    node_to_comm = {n: i for i, comm in enumerate(communities) for n in comm}

    comm_labels: list[int] = []
    truth_labels: list[str] = []
    skipped = 0
    for n in G.nodes():
        truth = chain_to_family.get(G.nodes[n]["chain_id"])
        if truth is None:
            skipped += 1
            continue
        comm_labels.append(node_to_comm[n])
        truth_labels.append(truth)

    if not comm_labels:
        LOGGER.warning("No annotated nodes found; cannot validate")
        return {}

    contingency = pd.crosstab(
        pd.Series(comm_labels, name="community"),
        pd.Series(truth_labels, name="family"),
    )
    ari = adjusted_rand_score(truth_labels, comm_labels)
    nmi = normalized_mutual_info_score(truth_labels, comm_labels)

    print("\n--- Biological validation ---")
    print(f"Ground truth: {validation_path} ({len(set(truth_labels))} families)")
    print(f"Annotated nodes: {len(comm_labels)} ({skipped} skipped)")
    print("\nContingency table (communities x family):")
    print(contingency.to_string())
    print(f"\nARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")

    return {"contingency": contingency, "ari": ari, "nmi": nmi}


def _to_residue_ranges(resnums: list[int]) -> str:
    sorted_nums = sorted(resnums)
    ranges, start, end = [], sorted_nums[0], sorted_nums[0]
    for n in sorted_nums[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = n
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def export_chimerax_script(
    filepath: str,
    G: nx.Graph,
    communities: list[set],
    path: str,
):
    palette = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "pink",
        "brown",
    ]

    pdb_paths = (
        [
            os.path.join(TAR_OUT_DIR, name)
            for name in os.listdir(TAR_OUT_DIR)
            if name.endswith(".pdb")
        ]
        if filepath.endswith(".tar.gz")
        else [filepath]
    )

    lines = [f"open {p}" for p in pdb_paths]
    lines += ["cartoon", "color gray", "set bgColor white", "lighting simple"]

    for comm_idx, comm in enumerate(communities):
        chain_residues: dict[str, list[int]] = {}
        for n in comm:
            chain_residues.setdefault(G.nodes[n]["chain_id"], []).append(
                G.nodes[n]["residue_number"]
            )
        color = palette[comm_idx % len(palette)]
        for chain, resnums in chain_residues.items():
            lines.append(f"color /{chain}:{_to_residue_ranges(resnums)} {color}")

    lines.append("save exports/communities_3d.png width 2000 height 2000 supersample 3")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    LOGGER.info(f"ChimeraX script saved to {path}")
    LOGGER.info(f"Run: chimerax --nogui {path}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the .pdb or .tar.gz file")

    graph = parser.add_mutually_exclusive_group()
    graph.add_argument(
        "--alpha-carbon",
        action="store_true",
        help="Generate a network using only alpha-carbon atoms, that are connected if their distance is less than a set cutoff",
    )
    graph.add_argument(
        "--beta-carbon",
        action="store_true",
        help="Generate a network using only beta-carbon atoms, that are connected if their distance is less than a set cutoff",
    )
    graph.add_argument(
        "--residue",
        action="store_true",
        help="Generate a network using the method discribed by Grant and Ahnert",
    )

    carbon_graph = parser.add_argument_group("Carbon network")
    carbon_graph.add_argument("--cutoff", help="Distance cutoff", type=float)
    carbon_graph.add_argument(
        "--lower-cutoff", help="[Optional] Lower distance cutoff", type=float
    )
    carbon_graph.add_argument(
        "--weighted",
        action="store_true",
        help="Weight edges by the inverse of the distance between the two nodes",
    )

    residue_network = parser.add_argument_group("Residue network")
    residue_network.add_argument(
        "--scaling", help="Parameter s", type=float, default=4.0
    )
    residue_network.add_argument(
        "--per-chain",
        action="store_true",
        help="Build the residue network chain-by-chain (sub-quaternary level), excluding inter-chain contacts",
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

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = load_file(args.filename)
    graph = None
    if args.alpha_carbon and args.cutoff:
        graph = build_alpha_carbon_graph(
            df,
            args.cutoff,
            args.lower_cutoff if args.lower_cutoff else None,
            weighted=args.weighted,
        )
    elif args.beta_carbon and args.cutoff:
        graph = build_beta_carbon_graph(
            df,
            args.cutoff,
            args.lower_cutoff if args.lower_cutoff else None,
            weighted=args.weighted,
        )
    elif args.residue and args.scaling:
        graph = (
            build_residue_graph_per_chain(df, args.scaling)
            if args.per_chain
            else build_residue_graph(df, args.scaling)
        )

    if graph is None:
        LOGGER.fatal("No network was created")
        return

    if args.statistics:
        print("\n--- Network statistics ---")
        stats = compute_network_stats(graph)
        for key, val in stats.items():
            print(
                f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}"
            )

    if not os.path.isdir("exports"):
        os.mkdir("exports")

    if args.html:
        LOGGER.info("Generating interactive network (exports/graph.html)")
        generate_interative_network(graph, "exports/graph.html")

    if args.plot:
        LOGGER.info("Plotting degree distribution and centralities")
        plot_degree_distribution(graph, "exports/degree_distribution.jpg")
        plot_centrality(graph, "degree", "exports/degree_centrality.jpg")
        plot_centrality(graph, "betweenness", "exports/betweenness_centrality.jpg")
        plot_centrality(graph, "closeness", "exports/closeness_centrality.jpg")

    if args.louvain or args.infomap:
        print("\n--- Community detection ---")
        if args.louvain:
            LOGGER.info("Detecting communities with Louvain")
            communities = nx.algorithms.community.louvain_communities(
                graph, weight="weight"
            )
        else:
            LOGGER.info("Detecting communities with Infomap")
            communities = infomap.find_communities(
                graph, weight="weight", seed=42, num_trials=20
            )

        print(f"Communities found: {len(communities)}")
        plot_communities(graph, communities, "exports/communities.jpg")

        if args.chimerax:
            export_chimerax_script(args.filename, graph, communities, "communities.cxc")

        if args.validate:
            biological_validation(args.validate, graph, communities)


if __name__ == "__main__":
    main()
