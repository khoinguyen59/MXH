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
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        use_rwpe: bool = False,
        rwpe_steps: int = 4,
    ) -> None:
        """Initialize the BA-Shapes generator.

        Args:
            data_dir: Storage directory.
            num_base_nodes: Number of nodes in the Barabási-Albert base graph (default: 300).
            num_motifs: Number of 5-node house motifs attached to base nodes (default: 80).
            random_seed: Seed for deterministic generation.
            train_ratio: Proportion of nodes for training (default: 0.60).
            val_ratio: Proportion of nodes for validation (default: 0.20).
            test_ratio: Proportion of nodes for testing (default: 0.20).
            use_rwpe: Whether to append Random Walk Positional Encoding (RWPE) to break topological symmetry.
            rwpe_steps: Number of random walk return steps (default: 4).
        """
        super().__init__(data_dir=data_dir, random_seed=random_seed)
        self.num_base_nodes = num_base_nodes
        self.num_motifs = num_motifs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.use_rwpe = use_rwpe
        self.rwpe_steps = rwpe_steps

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

            # Label nodes: 1 for bottom base, 2 for middle walls, 3 for top roof
            labels[motif_node_ids[0]] = 1
            labels[motif_node_ids[1]] = 1
            labels[motif_node_ids[2]] = 2
            labels[motif_node_ids[3]] = 2
            labels[motif_node_ids[4]] = 3

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

        # Feature matrix: All-ones constant vector of dimension 10 (Canonical Benchmark Standard)
        # Feature matrix: All-ones constant vector of dimension 10 (Canonical Benchmark Standard)
        features = np.ones((total_nodes, 10), dtype=np.float32)

        # Convert to PyG tensors
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float32)

        if self.use_rwpe:
            from torch_geometric.utils import to_dense_adj
            adj = to_dense_adj(edge_index, max_num_nodes=total_nodes)[0]
            deg = adj.sum(dim=-1).clamp(min=1)
            p_matrix = torch.matmul(torch.diag(1.0 / deg), adj)
            rwpe_list = []
            current_p = p_matrix
            for _ in range(self.rwpe_steps):
                rwpe_list.append(current_p.diag())
                current_p = torch.matmul(current_p, p_matrix)
            rwpe_tensor = torch.stack(rwpe_list, dim=-1)
            x = torch.cat([x, rwpe_tensor], dim=-1)

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
        splitter = AntiLeakageSplitter(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
        )
        train_mask, val_mask, test_mask = splitter.split_nodes(data, stratify=True)
        data = splitter.fit_transform_features(data, train_mask)

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        self.data = data
        return data

    def get_clean_val_nodes(self, data: Data, count: int = 100) -> List[int]:
        """Extract a clean, balanced list of validation nodes strictly inside val_mask.

        Args:
            data: PyG Data object containing val_mask and y labels.
            count: Number of validation nodes to select (default: 100).

        Returns:
            Sorted list of node indices strictly within val_mask.
        """
        val_indices = torch.where(data.val_mask)[0].tolist()
        val_labels = data.y[val_indices].tolist()

        by_class: dict = {c: [] for c in range(4)}
        for idx, lbl in zip(val_indices, val_labels):
            by_class[lbl].append(idx)

        selected = []
        # Class 3 (roof) has fewer nodes, so prioritize taking all available motif validation nodes
        for c in [3, 1, 2, 0]:
            target_per_class = count // 4
            take = min(len(by_class[c]), target_per_class if c != 3 else len(by_class[3]))
            selected.extend(by_class[c][:take])

        # Fill remaining slots up to count from available val_indices
        selected_set = set(selected)
        remaining = [idx for idx in val_indices if idx not in selected_set]
        needed = count - len(selected)
        if needed > 0:
            selected.extend(remaining[:needed])

        return sorted(selected[:count])

