import argparse
from itertools import combinations
import json
import logging
import os
import tarfile

import infomap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from ase.data import atomic_numbers, covalent_radii
from biopandas.pdb import PandasPdb
from pyvis.network import Network
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

LOGGER = logging.getLogger(__name__)
TAR_OUT_DIR = "pdb/extracted"


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
    weighted: bool = False,
) -> nx.Graph:
    filtered_df = filter_atoms(df, atom)
    LOGGER.info(f"Filtered to {len(filtered_df)} {atom} atoms")

    coords = filtered_df[["x_coord", "y_coord", "z_coord"]].to_numpy()

    tree = cKDTree(coords)
    pairs = tree.query_pairs(cutoff, output_type="set")

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
    weighted: bool = False,
) -> nx.Graph:
    LOGGER.info(f"Building alpha-carbon network (cutoff={cutoff})")
    return build_carbon_graph(df, cutoff, weighted=weighted)


def build_beta_carbon_graph(
    df: pd.DataFrame,
    cutoff: float,
    weighted: bool = False,
) -> nx.Graph:
    LOGGER.info(f"Building beta-carbon network (cutoff={cutoff})")
    return build_carbon_graph(df, cutoff, atom="CB", weighted=weighted)


def build_residue_graph(df: pd.DataFrame, s: float = 4.0) -> nx.Graph:
    LOGGER.info(f"Building Grant-Ahnert residue network (s={s}, {len(df)} atoms)")
    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()

    radii = np.array(
        [covalent_radii[atomic_numbers[atom.strip()]] for atom in df["element_symbol"]]
    )

    node_ids = df["node_id"].to_numpy()
    residues = list(dict.fromkeys(node_ids))
    residue_to_idx = {node_id: i for i, node_id in enumerate(residues)}
    atom_to_residue = np.array([residue_to_idx[node_id] for node_id in node_ids])

    reps: dict[str, dict] = {}
    for node_id, group in df.groupby("node_id", sort=False):
        ca = group[group["atom_name"] == "CA"]
        ref = ca.iloc[0] if len(ca) else group.iloc[0]
        x, y, z = (
            (ref["x_coord"], ref["y_coord"], ref["z_coord"])
            if len(ca)
            else (
                group["x_coord"].mean(),
                group["y_coord"].mean(),
                group["z_coord"].mean(),
            )
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

    max_cutoff = s * 2.0 * radii.max()
    tree = cKDTree(coords)
    distances = tree.sparse_distance_matrix(tree, max_cutoff, output_type="coo_matrix")

    atom_i = distances.row
    atom_j = distances.col
    pair_distances = distances.data
    contact_mask = (atom_i < atom_j) & (
        pair_distances <= s * (radii[atom_i] + radii[atom_j])
    )

    atom_i = atom_i[contact_mask]
    atom_j = atom_j[contact_mask]
    pair_distances = pair_distances[contact_mask]
    atomic_contacts = len(pair_distances)

    res_i = atom_to_residue[atom_i]
    res_j = atom_to_residue[atom_j]
    inter_residue_mask = res_i != res_j
    res_i = res_i[inter_residue_mask]
    res_j = res_j[inter_residue_mask]
    pair_distances = pair_distances[inter_residue_mask]

    residue_contacts = pd.DataFrame(
        {
            "source": np.minimum(res_i, res_j),
            "target": np.maximum(res_i, res_j),
            "distance": pair_distances,
        }
    )
    residue_edges = residue_contacts.groupby(["source", "target"]).agg(
        weight=("distance", "size"),
        min_distance=("distance", "min"),
    )
    for (res_i, res_j), edge_data in residue_edges.iterrows():
        G.add_edge(
            int(res_i),
            int(res_j),
            weight=int(edge_data["weight"]),
            min_distance=float(edge_data["min_distance"]),
        )

    LOGGER.info(f"Atomic network: {atomic_contacts} contacts")

    LOGGER.info(
        f"Residue network: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} weighted edges"
    )
    return G


def build_chain_graph(df: pd.DataFrame, cutoff: float) -> nx.Graph:
    LOGGER.info(f"Building chain network (cutoff={cutoff}, {len(df)} atoms)")

    df = df.reset_index(drop=True)

    G = nx.Graph()
    chains: dict[str, dict] = {}
    for chain, sub in df.groupby("chain_id", sort=False):
        coords = sub[["x_coord", "y_coord", "z_coord"]].to_numpy()
        chains[chain] = {
            "tree": cKDTree(coords),
            "mins": coords.min(axis=0),
            "maxs": coords.max(axis=0),
        }
        G.add_node(
            chain,
            chain_id=chain,
            size=len(sub),
            x_coord=float(sub["x_coord"].mean()),
            y_coord=float(sub["y_coord"].mean()),
            z_coord=float(sub["z_coord"].mean()),
        )

    for chain_a, chain_b in combinations(chains, 2):
        data_a = chains[chain_a]
        data_b = chains[chain_b]
        gap = np.maximum(
            0,
            np.maximum(data_a["mins"] - data_b["maxs"], data_b["mins"] - data_a["maxs"]),
        )
        if np.linalg.norm(gap) > cutoff:
            continue

        distances = data_a["tree"].sparse_distance_matrix(
            data_b["tree"], cutoff, output_type="coo_matrix"
        )
        if distances.nnz == 0:
            continue

        G.add_edge(
            chain_a,
            chain_b,
            weight=int(distances.nnz),
            min_distance=float(distances.data.min()),
        )

    LOGGER.info(
        f"Chain network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    return G


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
        try:
            chains = {G.nodes[n]["node_id"].split(":")[0] for n in comm}
        except KeyError:
            chains = {G.nodes[n]["chain_id"] for n in comm}

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

    sample_node = next(iter(G.nodes()))
    is_chain_graph = "residue_number" not in G.nodes[sample_node]

    for comm_idx, comm in enumerate(communities):
        color = palette[comm_idx % len(palette)]
        if is_chain_graph:
            for n in comm:
                chain = G.nodes[n].get("chain_id", n)
                lines.append(f"color /{chain} {color}")
        else:
            chain_residues: dict[str, list[int]] = {}
            for n in comm:
                chain_residues.setdefault(G.nodes[n]["chain_id"], []).append(
                    G.nodes[n]["residue_number"]
                )
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

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = load_file(args.filename)
    graph = None
    if args.graph == "a-carbon" and args.cutoff:
        graph = build_alpha_carbon_graph(
            df,
            args.cutoff,
            weighted=args.weighted,
        )
    elif args.graph == "b-carbon" and args.cutoff:
        graph = build_beta_carbon_graph(
            df,
            args.cutoff,
            weighted=args.weighted,
        )
    elif args.graph == "chain" and args.cutoff:
        graph = build_chain_graph(df, args.cutoff)
    elif args.graph == "residue" and args.scaling:
        graph = build_residue_graph(df, args.scaling)

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
            export_chimerax_script(args.filename, graph, communities, "exports/communities.cxc")

        if args.validate:
            biological_validation(args.validate, graph, communities)


if __name__ == "__main__":
    main()
