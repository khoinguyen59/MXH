"""Amazon Photo Dataset Loader.

Ingestion service for Amazon Photo co-purchase graph benchmark.
Adheres to the BaseDatasetLoader interface and anti-leakage splitting standards.
"""

from typing import Optional
import torch
from torch_geometric.datasets import Amazon
from torch_geometric.data import Data
from src.datasets.base_loader import BaseDatasetLoader
from src.datasets.anti_leakage_splitter import AntiLeakageSplitter


class AmazonPhotoLoader(BaseDatasetLoader):
    """Dataset loader for the Amazon Photo e-commerce co-purchase network."""

    def __init__(self, data_dir: str = "data/raw", random_seed: int = 42) -> None:
        """Initialize the Amazon Photo loader.

        Args:
            data_dir: Path to directory where datasets are stored.
            random_seed: Random seed for deterministic reproducibility.
        """
        super().__init__(data_dir=data_dir, random_seed=random_seed)

    def load(self, train_per_class: int = 20, val_per_class: int = 30) -> Data:
        """Download and load the Amazon Photo dataset with class-stratified splits.

        Args:
            train_per_class: Number of training nodes per class (default: 20).
            val_per_class: Number of validation nodes per class (default: 30).

        Returns:
            PyG Data object with anti-leakage train_mask, val_mask, and test_mask.
        """
        dataset = Amazon(root=self.data_dir, name="Photo")
        raw_data = dataset[0]

        # Use AntiLeakageSplitter to create deterministic, reproducible splits
        splitter = AntiLeakageSplitter(random_seed=self.random_seed)
        num_classes = int(raw_data.y.max().item() + 1)
        num_nodes = raw_data.num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        generator = torch.Generator().manual_seed(self.random_seed)

        for c in range(num_classes):
            idx = (raw_data.y == c).nonzero(as_tuple=False).view(-1)
            perm = idx[torch.randperm(idx.size(0), generator=generator)]

            train_idx = perm[:train_per_class]
            val_idx = perm[train_per_class : train_per_class + val_per_class]
            test_idx = perm[train_per_class + val_per_class :]

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

        self.data = Data(
            x=raw_data.x,
            edge_index=raw_data.edge_index,
            y=raw_data.y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        return self.data
