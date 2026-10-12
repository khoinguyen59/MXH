"""Synthetic BA-Shapes Dataset Loader with Ground-Truth Motifs.

Generates a Barabási-Albert base graph with attached 5-node house motifs,
providing exact ground-truth subgraphs for Motif-AUC explanation validation.
"""

from typing import List, Tuple
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.datasets.anti_leakage_splitter import AntiLeakageSplitter
from src.datasets.base_loader import BaseDatasetLoader


class BAShapesLoader(BaseDatasetLoader):
    """Generates the canonical BA-Shapes synthetic benchmark for XAI evaluation."""

    def __init__(
        self,
        data_dir: str,
        num_base_nodes: int = 300,
        num_motifs: int = 80,
        random_seed: int = 42,
    ) -> None:
        """Initialize the BA-Shapes generator.

        Args:
            data_dir: Storage directory.
            num_base_nodes: Number of nodes in the Barabási-Albert base graph (default: 300).
            num_motifs: Number of 5-node house motifs attached to base nodes (default: 80).
            random_seed: Seed for deterministic generation.
        """
        super().__init__(data_dir=data_dir, random_seed=random_seed)
        self.num_base_nodes = num_base_nodes
        self.num_motifs = num_motifs

    def _build_house_motif(self) -> List[Tuple[int, int]]:
        """Construct the 5-node house motif edge list (3-node triangle roof + 4-node square base).

        Returns:
            List of undirected edge tuples relative to motif local indices [0, 1, 2, 3, 4].
        """
        # Node 0, 1, 2, 3 form base cycle; 0 and 1 connect to roof node 4
        return [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)]

    def load(self) -> Data:
        """Generate the synthetic graph, assign ground-truth labels, and apply anti-leakage split.

        Returns:
            PyG Data object containing ground-truth edge masks and class labels.
        """
        rng = np.random.RandomState(self.random_seed)

        # 1. Generate Barabási-Albert base graph
        base_graph = nx.barabasi_albert_graph(
            n=self.num_base_nodes, m=1, seed=self.random_seed
        )

        edge_list = list(base_graph.edges())
        total_nodes = self.num_base_nodes + self.num_motifs * 5
        labels = np.zeros(total_nodes, dtype=np.int64)  # 0: background BA node
        motif_edge_mask = []

        # Current total edge list
        all_edges = [(u, v) for u, v in edge_list] + [(v, u) for u, v in edge_list]
        is_motif_edge = [False] * len(all_edges)

        current_node_id = self.num_base_nodes
        house_edges_rel = self._build_house_motif()

        for _ in range(self.num_motifs):
            # Pick a random base node to attach the motif
            attach_node = rng.randint(0, self.num_base_nodes)

            # Local motif node IDs
            motif_node_ids = [current_node_id + i for i in range(5)]

            # Label nodes: 1 for base of house, 2 for roof
            labels[motif_node_ids[0]] = 1
            labels[motif_node_ids[1]] = 1
            labels[motif_node_ids[2]] = 1
            labels[motif_node_ids[3]] = 1
            labels[motif_node_ids[4]] = 2

            # Connect attach_node to motif_node_ids[0]
            all_edges.append((attach_node, motif_node_ids[0]))
            all_edges.append((motif_node_ids[0], attach_node))
            is_motif_edge.append(False)
            is_motif_edge.append(False)

            # Add internal house motif edges (undirected)
            for u_rel, v_rel in house_edges_rel:
                u = motif_node_ids[u_rel]
                v = motif_node_ids[v_rel]
                all_edges.append((u, v))
                all_edges.append((v, u))
                is_motif_edge.append(True)
                is_motif_edge.append(True)

            current_node_id += 5

        # Feature matrix: One-hot degree distribution + synthetic perturbation
        degrees = [0] * total_nodes
        for u, _ in all_edges:
            degrees[u] += 1
        features = np.column_stack([
            np.log1p(degrees),
            rng.normal(loc=0.0, scale=0.1, size=(total_nodes, 9)),
        ])

        # Convert to PyG tensors
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        ground_truth_mask = torch.tensor(is_motif_edge, dtype=torch.bool)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            ground_truth_mask=ground_truth_mask,
            num_nodes=total_nodes,
        )

        # Apply strict Anti-Leakage Split
        splitter = AntiLeakageSplitter(random_seed=self.random_seed)
        train_mask, val_mask, test_mask = splitter.split_nodes(data, stratify=True)
        data = splitter.fit_transform_features(data, train_mask)

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        self.data = data
        return data
