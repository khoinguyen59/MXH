"""ogbn-arxiv Dataset Loader.

Tier-2 directed graph benchmark from the Open Graph Benchmark (OGB).
Supports both directed message passing and symmetrized (undirected) modes.
"""

from typing import Optional
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from src.datasets.base_loader import BaseDatasetLoader


class OgbnArxivLoader(BaseDatasetLoader):
    """Dataset loader for the ogbn-arxiv directed citation network."""

    def __init__(self, data_dir: str = "data/raw", random_seed: int = 42) -> None:
        """Initialize the ogbn-arxiv loader.

        Args:
            data_dir: Path to directory where datasets are stored.
            random_seed: Random seed for deterministic reproducibility.
        """
        super().__init__(data_dir=data_dir, random_seed=random_seed)

    def load(self, directed: bool = True) -> Data:
        """Download and load ogbn-arxiv dataset with temporal train/val/test splits.

        Args:
            directed: If True, keep directed edges. If False, symmetrize via to_undirected.

        Returns:
            PyG Data object with temporal train_mask, val_mask, and test_mask.
        """
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError:
            raise ImportError(
                "Package 'ogb' is required to load ogbn-arxiv. "
                "Install it using: pip install ogb"
            )

        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=self.data_dir)
        split_idx = dataset.get_idx_split()
        raw_data = dataset[0]

        num_nodes = raw_data.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[split_idx["train"]] = True
        val_mask[split_idx["valid"]] = True
        test_mask[split_idx["test"]] = True

        edge_index = raw_data.edge_index
        if not directed:
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        # Flatten 2D target labels if needed
        y = raw_data.y.squeeze(1) if raw_data.y.dim() > 1 else raw_data.y

        self.data = Data(
            x=raw_data.x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        return self.data
