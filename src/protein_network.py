from abc import ABC, abstractmethod
from itertools import combinations
import json
import os

import pandas as pd
from pdb_parser import PDB, EXTRACTED_PDB_PATH
import networkx as nx
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import seaborn as sns
from utils import create_dir
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import infomap


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


def _fiedler_split(G: nx.Graph, weight: float):
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return set(nodes), set()
    L = nx.laplacian_matrix(G, nodelist=nodes, weight=weight).astype(float)
    try:
        from scipy.sparse.linalg import eigsh

        k = min(2, L.shape[0] - 1)
        vals, vecs = eigsh(L, k=k + 1, which="SM")
        fiedler = vecs[:, 1]
    except Exception:
        L_dense = L.toarray()
        vals, vecs = np.linalg.eigh(L_dense)
        fiedler = vecs[:, 1]
    a = {nodes[i] for i in range(len(nodes)) if fiedler[i] >= 0}
    b = {nodes[i] for i in range(len(nodes)) if fiedler[i] < 0}
    return a, b


def _safe(fn, G) -> float:
    try:
        return float(fn(G))
    except Exception:
        return float("nan")


def _purity(comm_labels: list[int], truth_labels: list[str]) -> float:
    contingency = pd.crosstab(pd.Series(comm_labels), pd.Series(truth_labels))
    return float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())


def _sampled_closeness(G: nx.Graph, k: int, weight, seed: int) -> dict:
    import random

    rng = random.Random(seed)
    nodes = list(G.nodes())
    sources = rng.sample(nodes, min(k, len(nodes)))

    dist_sum = {v: 0.0 for v in nodes}
    reach = {v: 0 for v in nodes}
    for s in sources:
        if weight:
            lengths = nx.single_source_dijkstra_path_length(G, s, weight=weight)
        else:
            lengths = nx.single_source_shortest_path_length(G, s)
        for v, d in lengths.items():
            dist_sum[v] += d
            reach[v] += 1

    closeness = {}
    for v in nodes:
        if reach[v] > 1 and dist_sum[v] > 0:
            avg = dist_sum[v] / reach[v]
            closeness[v] = (1.0 / avg) * ((reach[v] - 1) / (len(nodes) - 1))
        else:
            closeness[v] = 0.0
    return closeness


class ProteinNetwork(ABC):
    def __init__(self, pdb: PDB):
        self.pdb = pdb
        self.graph: nx.Graph
        self._build_network()

    @abstractmethod
    def _build_network(self) -> None:
        pass

    def louvain(self) -> list[set]:
        print("Detecting communities with Louvain")
        return nx.algorithms.community.louvain_communities(self.graph, weight="weight")

    def infomap(self) -> list[set]:
        print("Detecting communities with Infomap")
        return infomap.find_communities(
            self.graph, weight="weight", seed=42, num_trials=20
        )

    def greedy(self) -> list[set]:
        print("Detecting communities with greedy modularity (Clauset-Newman-Moore)")
        return list(
            nx.algorithms.community.greedy_modularity_communities(
                self.graph, weight="weight"
            )
        )

    def labelprop(self, seed: int = 42) -> list[set]:
        print("Detecting communities with label propagation")
        return list(
            nx.algorithms.community.asyn_lpa_communities(
                self.graph, weight="weight", seed=seed
            )
        )

    def spectral_bipartition(self, k: int, weighted: bool):
        weight = "weight" if weighted else None
        comps = [set(c) for c in nx.connected_components(self.graph)]
        while len(comps) < k:
            comps.sort(key=len, reverse=True)
            biggest = comps.pop(0)
            if len(biggest) < 2:
                comps.append(biggest)
                break
            sub = self.graph.subgraph(biggest)
            part_a, part_b = _fiedler_split(sub, weight)
            if not part_a or not part_b:
                comps.append(biggest)
                break
            comps.extend([part_a, part_b])
        return comps

    def _compare_to_annotations(
        self, communities: list[set], chain_to_family: dict[str, str]
    ) -> tuple[list[int], list[str], int]:
        node_to_comm = {n: i for i, comm in enumerate(communities) for n in comm}

        comm_labels: list[int] = []
        truth_labels: list[str] = []
        skipped = 0
        for n in self.graph.nodes():
            truth = chain_to_family.get(self.graph.nodes[n]["chain_id"])
            if truth is None:
                skipped += 1
                continue
            comm_labels.append(node_to_comm[n])
            truth_labels.append(truth)

        return comm_labels, truth_labels, skipped

    def evaluate_communities(
        self, communities: list[set], chain_to_family: dict[str, str]
    ) -> dict:
        comm_labels, truth_labels, _ = self._compare_to_annotations(
            communities, chain_to_family
        )
        if not comm_labels:
            return {
                "ari": float("nan"),
                "nmi": float("nan"),
                "purity": float("nan"),
                "n_annotated": 0,
            }
        return {
            "ari": adjusted_rand_score(truth_labels, comm_labels),
            "nmi": normalized_mutual_info_score(truth_labels, comm_labels),
            "purity": _purity(comm_labels, truth_labels),
            "n_annotated": len(comm_labels),
        }

    def validate_communities(
        self, validation_filepath: str, communities: list[set], out_dir: str
    ) -> dict:
        with open(validation_filepath) as f:
            family_to_chains: dict[str, list[str]] = json.load(f)
        chain_to_family = {
            ch: family for family, chains in family_to_chains.items() for ch in chains
        }

        comm_labels, truth_labels, skipped = self._compare_to_annotations(
            communities, chain_to_family
        )

        if not comm_labels:
            return {}

        contingency = pd.crosstab(
            pd.Series(comm_labels, name="community"),
            pd.Series(truth_labels, name="family"),
        )
        ari = adjusted_rand_score(truth_labels, comm_labels)
        nmi = normalized_mutual_info_score(truth_labels, comm_labels)
        purity = _purity(comm_labels, truth_labels)

        create_dir(out_dir)
        contingency_filepath = os.path.join(out_dir, "contingency.csv")
        contingency.to_csv(contingency_filepath)

        print("\n= Communities Validation =")
        print(
            f"Annotations: {validation_filepath} ({len(set(truth_labels))} families)"
        )
        print(f"Annotated nodes: {len(comm_labels)} ({skipped} skipped)")
        print(f"Contingency table (communities x family) saved to {contingency_filepath}")
        print(f"\nARI: {ari:.4f}")
        print(f"NMI: {nmi:.4f}")
        print(f"Purity: {purity:.4f}")

        return {"contingency": contingency, "ari": ari, "nmi": nmi, "purity": purity}

    def generate_interative_network(self, out_dir: str):
        filepath = os.path.join(out_dir, "graph.html")
        print(f"Generating interactive network ({filepath})")

        net = Network(height="800px", width="100%", notebook=False)
        net.from_nx(self.graph)

        create_dir(out_dir)
        net.write_html(filepath, notebook=False)

    def compute_network_stats(self) -> dict:
        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()

        stats: dict = {
            "num_nodes": n,
            "num_edges": m,
            "avg_degree": (2 * m / n) if n > 0 else 0.0,
            "density": nx.density(self.graph),
            "clustering_coefficient": nx.average_clustering(self.graph),
            "assortativity": (
                nx.degree_assortativity_coefficient(self.graph)
                if m > 0
                else float("nan")
            ),
        }

        largest_cc = max(nx.connected_components(self.graph), key=len, default=set())
        G_lcc = self.graph.subgraph(largest_cc)
        stats["largest_component_size"] = len(largest_cc)
        stats["largest_component_fraction"] = len(largest_cc) / n if n > 0 else 0.0
        stats["avg_shortest_path"] = (
            nx.average_shortest_path_length(G_lcc) if len(largest_cc) > 1 else 0.0
        )

        return stats

    def degree_distribution(self) -> pd.DataFrame:
        degrees = np.array([d for _, d in self.graph.degree()])
        values, counts = np.unique(degrees, return_counts=True)
        return pd.DataFrame(
            {"degree": values, "count": counts, "prob": counts / counts.sum()}
        )

    def compute_centralities(
        self, weighted: bool = True, betweenness_k: int | None = 500, seed: int = 42
    ) -> pd.DataFrame:
        weight = "weight" if weighted else None
        n = self.graph.number_of_nodes()
        approximate = betweenness_k is not None and n > betweenness_k

        deg = dict(self.graph.degree())
        strength = dict(self.graph.degree(weight="weight"))

        if approximate:
            k = min(betweenness_k, n)
            print(f"Computing centralities (sampling {k}/{n} sources)")
            btw = nx.betweenness_centrality(
                self.graph, k=k, weight=weight, normalized=True, seed=seed
            )
            close = _sampled_closeness(self.graph, k=k, weight=weight, seed=seed)
        else:
            btw = nx.betweenness_centrality(
                self.graph, weight=weight, normalized=True
            )
            close = nx.closeness_centrality(self.graph, distance=weight)

        try:
            eig = nx.eigenvector_centrality_numpy(self.graph, weight=weight)
        except Exception:
            eig = {node: float("nan") for node in self.graph.nodes()}

        clust = nx.clustering(self.graph, weight=weight)

        df = pd.DataFrame(
            {
                "degree": pd.Series(deg),
                "strength": pd.Series(strength),
                "betweenness": pd.Series(btw),
                "closeness": pd.Series(close),
                "eigenvector": pd.Series(eig),
                "clustering": pd.Series(clust),
            }
        )
        meta = pd.DataFrame.from_dict(dict(self.graph.nodes(data=True)), orient="index")
        meta_cols = [
            c for c in ("node_id", "chain_id", "residue_number") if c in meta.columns
        ]
        df = df.join(meta[meta_cols])
        return df.sort_values("betweenness", ascending=False)

    def plot_degree_distribution(self, out_dir: str):
        filepath = os.path.join(out_dir, "degree_dist.jpg")
        print(f"Plotting degree distribution ({filepath})")

        dist = self.degree_distribution()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        sns.histplot(
            x=dist["degree"], weights=dist["count"], kde=True, ax=axes[0]
        )
        axes[0].set(xlabel="Degree k", ylabel="# of Nodes", title="Degree histogram")

        mask = (dist["degree"] > 0) & (dist["prob"] > 0)
        axes[1].loglog(dist["degree"][mask], dist["prob"][mask], "o", ms=4)
        axes[1].set(xlabel="Degree k (log)", ylabel="P(k) (log)", title="Log-log scale")

        fig.tight_layout()
        create_dir(out_dir)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

    def plot_centrality(self, measure_name: str, out_dir: str):
        if measure_name == "betweenness":
            cent = nx.betweenness_centrality(self.graph)
        elif measure_name == "degree":
            cent = nx.degree_centrality(self.graph)
        elif measure_name == "closeness":
            cent = nx.closeness_centrality(self.graph)
        elif measure_name == "eigenvector":
            try:
                cent = nx.eigenvector_centrality_numpy(self.graph, weight="weight")
            except Exception:
                return
        elif measure_name == "strength":
            cent = dict(self.graph.degree(weight="weight"))
        else:
            return

        filepath = os.path.join(out_dir, f"{measure_name}_centrality.jpg")
        print(f"Plotting {measure_name} centrality ({filepath})")

        nodes = list(self.graph.nodes())
        values = np.array([cent[n] for n in nodes])
        vmax = values.max() if values.max() > 0 else 1.0
        sizes = 20 + 400 * (values / vmax)

        pos = {
            n: (self.graph.nodes[n]["x_coord"], self.graph.nodes[n]["y_coord"])
            for n in self.graph.nodes()
        }

        plt.figure(figsize=(10, 10))
        nx.draw_networkx_edges(self.graph, pos, width=0.3, alpha=0.4)
        nodes_drawn = nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=nodes,
            node_size=sizes,
            node_color=values,
            cmap=plt.cm.plasma,
        )
        plt.colorbar(nodes_drawn, label=measure_name.capitalize())
        plt.axis("off")
        plt.tight_layout()

        create_dir(out_dir)
        plt.savefig(filepath, dpi=150)
        plt.close()

    def plot_centralities(
        self, out_dir: str, cent: pd.DataFrame | None = None, weighted: bool = True
    ):
        """Distribution (histogram) of each centrality measure in one figure."""
        if cent is None:
            cent = self.compute_centralities(weighted=weighted)

        filepath = os.path.join(out_dir, "centrality_distributions.jpg")
        print(f"Plotting centrality distributions ({filepath})")

        cols = [
            "degree",
            "strength",
            "betweenness",
            "closeness",
            "eigenvector",
            "clustering",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, col in zip(axes.flat, cols):
            sns.histplot(cent[col].dropna(), bins=40, ax=ax)
            ax.set(xlabel=col, ylabel="# of Nodes", title=f"Centrality: {col}")

        fig.tight_layout()
        create_dir(out_dir)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

    def _structure_summary(self) -> str:
        df = self.pdb.dataframe
        name = os.path.splitext(os.path.basename(self.pdb.filepath))[0]
        chains = sorted(df["chain_id"].unique())
        return (
            f"Estrutura '{name}': {len(df)} átomos, "
            f"{df['node_id'].nunique()} resíduos, "
            f"{len(chains)} cadeia(s) [{', '.join(chains)}]"
        )

    def network_summary(self) -> dict:
        G = self.graph
        n, m = G.number_of_nodes(), G.number_of_edges()
        comps = list(nx.connected_components(G))
        return {
            "n_nodes": n,
            "n_edges": m,
            "avg_degree": (2 * m / n) if n else 0.0,
            "density": nx.density(G),
            "n_components": len(comps),
            "giant_frac": (max(len(c) for c in comps) / n) if n else 0.0,
            "avg_clustering": nx.average_clustering(G) if n else 0.0,
        }

    def _topology_metrics(self, cent: pd.DataFrame, top: int = 10) -> dict:
        degrees = np.array([d for _, d in self.graph.degree()])

        def top_nodes(col: str) -> list:
            idx = cent.nlargest(top, col).index
            if "node_id" in cent.columns:
                return cent.loc[idx, "node_id"].tolist()
            return [int(i) for i in idx]

        return {
            "avg_degree": float(degrees.mean()) if len(degrees) else 0.0,
            "max_degree": int(degrees.max()) if len(degrees) else 0,
            "degree_assortativity": _safe(
                nx.degree_assortativity_coefficient, self.graph
            ),
            "avg_clustering": float(nx.average_clustering(self.graph)),
            "top_degree": top_nodes("degree"),
            "top_betweenness": top_nodes("betweenness"),
        }

    def _community_metrics(self, communities: list[set]) -> dict:
        sizes = sorted((len(c) for c in communities), reverse=True)
        try:
            mod = nx.algorithms.community.modularity(
                self.graph, communities, weight="weight"
            )
        except Exception:
            mod = float("nan")
        return {
            "n_communities": len(communities),
            "modularity": float(mod),
            "largest": sizes[0] if sizes else 0,
            "sizes_top10": sizes[:10],
        }

    def _validation_metrics(
        self, communities: list[set], chain_to_family: dict[str, str], label: str
    ) -> dict:
        metrics = self.evaluate_communities(communities, chain_to_family)
        return {
            "label": label,
            "n_communities": len(communities),
            "n_truth_classes": len(set(chain_to_family.values())),
            "purity": metrics["purity"],
            "nmi": metrics["nmi"],
            "ari": metrics["ari"],
        }

    def generate_report(
        self,
        out_dir: str,
        params: dict,
        communities: list[set],
        chain_to_family: dict[str, str],
        validation_label: str,
        cent: pd.DataFrame | None = None,
        top: int = 10,
    ) -> dict:
        """Consolidated run report (structure/params/network/topology/communities/
        validation), saved as report.json."""
        if cent is None:
            cent = self.compute_centralities()

        report = {
            "input": os.path.basename(self.pdb.filepath),
            "params": params,
            "structure": self._structure_summary(),
            "network": self.network_summary(),
            "topology": self._topology_metrics(cent, top),
            "communities": self._community_metrics(communities),
            "validation": self._validation_metrics(
                communities, chain_to_family, validation_label
            ),
        }

        filepath = os.path.join(out_dir, "report.json")
        print(f"Saving report ({filepath})")
        create_dir(out_dir)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def log_communities(self, communities: list[set]):
        print(f"Communities found: {len(communities)}")
        for i, comm in enumerate(communities):
            try:
                chains = {self.graph.nodes[n]["node_id"].split(":")[0] for n in comm}
            except KeyError:
                chains = {self.graph.nodes[n]["chain_id"] for n in comm}

            print(f"Comm {i}: {chains}")

    def community_composition(self, communities: list[set]) -> pd.DataFrame:
        rows = []
        for i, comm in enumerate(sorted(communities, key=len, reverse=True)):
            chain_counts = (
                pd.Series([self.graph.nodes[n]["chain_id"] for n in comm])
                .value_counts()
            )
            rows.append(
                {
                    "community": i,
                    "size": len(comm),
                    "n_chains": len(chain_counts),
                    "top_chains": ", ".join(
                        f"{c}({v})" for c, v in chain_counts.head(5).items()
                    ),
                }
            )
        return pd.DataFrame(rows)

    def plot_communities(self, communities: list[set], out_dir: str):
        filepath = os.path.join(out_dir, "communities.jpg")
        print(f"Plotting communities ({filepath})")

        node_to_community = {n: i for i, comm in enumerate(communities) for n in comm}
        colors = sns.color_palette(None, len(communities))
        node_colors = [colors[node_to_community[node]] for node in self.graph.nodes()]

        pos = {
            n: (self.graph.nodes[n]["x_coord"], self.graph.nodes[n]["y_coord"])
            for n in self.graph.nodes()
        }

        plt.figure(figsize=(12, 8))
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=30,
            width=0.3,
            with_labels=False,
        )
        plt.title("Communities (real spatial coordinates)", fontsize=18)

        create_dir(out_dir)
        plt.savefig(filepath, dpi=150)
        plt.close()

    def plot_structure_3d(
        self, communities: list[set], out_dir: str, by: str = "community"
    ):
        filepath = os.path.join(out_dir, f"structure_3d_{by}.jpg")
        print(f"Plotting 3D structure ({filepath})")

        nodes = list(self.graph.nodes())
        xyz = np.array(
            [
                (
                    self.graph.nodes[n]["x_coord"],
                    self.graph.nodes[n]["y_coord"],
                    self.graph.nodes[n]["z_coord"],
                )
                for n in nodes
            ]
        )

        if by == "chain":
            chains = sorted({self.graph.nodes[n]["chain_id"] for n in nodes})
            chain_to_idx = {c: i for i, c in enumerate(chains)}
            palette = sns.color_palette(None, len(chains))
            node_colors = [
                palette[chain_to_idx[self.graph.nodes[n]["chain_id"]]] for n in nodes
            ]
        else:
            node_to_community = {
                n: i for i, comm in enumerate(communities) for n in comm
            }
            palette = sns.color_palette(None, len(communities))
            node_colors = [palette[node_to_community[n]] for n in nodes]

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=node_colors, s=8, alpha=0.8, linewidths=0
        )
        ax.set(
            xlabel="x (Å)",
            ylabel="y (Å)",
            zlabel="z (Å)",
            title=f"3D structure colored by {by}",
        )
        fig.tight_layout()

        create_dir(out_dir)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

    def export_membership_csv(self, communities: list[set], out_dir: str):
        """Export the node -> community mapping (with metadata) to a CSV."""
        filepath = os.path.join(out_dir, "membership.csv")
        print(f"Exporting community membership ({filepath})")

        node_to_community = {n: i for i, comm in enumerate(communities) for n in comm}
        rows = []
        for n in self.graph.nodes():
            attrs = self.graph.nodes[n]
            rows.append(
                {
                    "node": n,
                    "node_id": attrs.get("node_id", n),
                    "chain_id": attrs.get("chain_id"),
                    "residue_number": attrs.get("residue_number"),
                    "x_coord": attrs.get("x_coord"),
                    "y_coord": attrs.get("y_coord"),
                    "z_coord": attrs.get("z_coord"),
                    "community": node_to_community.get(n),
                }
            )

        create_dir(out_dir)
        pd.DataFrame(rows).to_csv(filepath, index=False)

    def export_chimerax_script(
        self,
        communities: list[set],
        out_dir: str,
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
                os.path.join(EXTRACTED_PDB_PATH, name)
                for name in os.listdir(EXTRACTED_PDB_PATH)
                if name.endswith(".pdb")
            ]
            if self.pdb.filepath.endswith(".tar.gz")
            else [self.pdb.filepath]
        )

        lines = [f"open {p}" for p in pdb_paths]
        lines += ["cartoon", "color gray", "set bgColor white", "lighting simple"]

        sample_node = next(iter(self.graph.nodes()))
        is_chain_graph = "residue_number" not in self.graph.nodes[sample_node]

        for comm_idx, comm in enumerate(communities):
            color = palette[comm_idx % len(palette)]
            if is_chain_graph:
                for n in comm:
                    chain = self.graph.nodes[n].get("chain_id", n)
                    lines.append(f"color /{chain} {color}")
            else:
                chain_residues: dict[str, list[int]] = {}
                for n in comm:
                    chain_residues.setdefault(
                        self.graph.nodes[n]["chain_id"], []
                    ).append(self.graph.nodes[n]["residue_number"])
                for chain, resnums in chain_residues.items():
                    lines.append(
                        f"color /{chain}:{_to_residue_ranges(resnums)} {color}"
                    )

        lines.append(
            "save exports/communities_3d.png width 2000 height 2000 supersample 3"
        )

        create_dir(out_dir)
        filepath = os.path.join(out_dir, "chimerax.cxc")
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        print(f"ChimeraX script saved to {filepath}")


class AlphaCarbonNetwork(ProteinNetwork):
    def __init__(self, pdb: PDB, cutoff: float, weighted: bool = False):
        self.cutoff = cutoff
        self.weighted = weighted
        super().__init__(pdb)

    def _build_network(self) -> None:
        filtered_df = self.pdb.filter_atoms("CA")

        coords = filtered_df[["x_coord", "y_coord", "z_coord"]].to_numpy()

        tree = cKDTree(coords)
        pairs = tree.query_pairs(self.cutoff, output_type="set")

        G = nx.Graph()
        G.add_nodes_from(range(len(filtered_df)))
        if self.weighted:
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

        print(
            f"Alpha-carbon network: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )

        self.graph = G


class BetaCarbonNetwork(ProteinNetwork):
    def __init__(self, pdb: PDB, cutoff: float, weighted: bool = False):
        self.cutoff = cutoff
        self.weighted = weighted
        super().__init__(pdb)

    def _build_network(self) -> None:
        filtered_df = self.pdb.filter_atoms("CB")

        coords = filtered_df[["x_coord", "y_coord", "z_coord"]].to_numpy()

        tree = cKDTree(coords)
        pairs = tree.query_pairs(self.cutoff, output_type="set")

        G = nx.Graph()
        G.add_nodes_from(range(len(filtered_df)))
        if self.weighted:
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

        print(
            f"Beta-carbon network: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )

        self.graph = G


class ResidueNetwork(ProteinNetwork):
    def __init__(self, pdb: PDB, cutoff: float):
        self.cutoff = cutoff
        super().__init__(pdb)

    def _build_network(self) -> None:
        coords = self.pdb.dataframe[["x_coord", "y_coord", "z_coord"]].to_numpy()

        node_ids = self.pdb.dataframe["node_id"].to_numpy()
        residues = list(dict.fromkeys(node_ids))
        residue_to_idx = {node_id: i for i, node_id in enumerate(residues)}
        atom_to_residue = np.array([residue_to_idx[node_id] for node_id in node_ids])

        reps: dict[str, dict] = {}
        for node_id, group in self.pdb.dataframe.groupby("node_id", sort=False):
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

        tree = cKDTree(coords)
        distances = tree.sparse_distance_matrix(
            tree, self.cutoff, output_type="coo_matrix"
        )

        atom_i = distances.row
        atom_j = distances.col
        pair_distances = distances.data
        contact_mask = atom_i < atom_j

        atom_i = atom_i[contact_mask]
        atom_j = atom_j[contact_mask]
        pair_distances = pair_distances[contact_mask]

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

        print(
            f"Residue network: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} weighted edges"
        )

        self.graph = G


class ChainNetwork(ProteinNetwork):
    def __init__(self, pdb: PDB, cutoff: float):
        self.cutoff = cutoff
        super().__init__(pdb)

    def _build_network(self) -> None:
        df = self.pdb.dataframe.reset_index(drop=True)

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
                np.maximum(
                    data_a["mins"] - data_b["maxs"], data_b["mins"] - data_a["maxs"]
                ),
            )
            if np.linalg.norm(gap) > self.cutoff:
                continue

            distances = data_a["tree"].sparse_distance_matrix(
                data_b["tree"], self.cutoff, output_type="coo_matrix"
            )
            if distances.nnz == 0:
                continue

            G.add_edge(
                chain_a,
                chain_b,
                weight=int(distances.nnz),
                min_distance=float(distances.data.min()),
            )

        print(
            f"Chain network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        self.graph = G
