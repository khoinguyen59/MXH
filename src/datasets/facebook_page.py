"""Facebook Page-Page Dataset Loader.

Ingestion service for Facebook Page-Page verified network.
Core social network dataset for Social Network Analysis (SNA) course profile.
Adheres to the BaseDatasetLoader interface and anti-leakage splitting standards.
"""

import torch
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.data import Data
from src.datasets.base_loader import BaseDatasetLoader
from src.datasets.anti_leakage_splitter import AntiLeakageSplitter


class FacebookPageLoader(BaseDatasetLoader):
    """Dataset loader for the Facebook Page-Page social network benchmark."""

    def __init__(self, data_dir: str = "data/raw", random_seed: int = 42) -> None:
        """Initialize the Facebook Page-Page loader.

        Args:
            data_dir: Path to directory where datasets are stored.
            random_seed: Random seed for deterministic reproducibility.
        """
        super().__init__(data_dir=data_dir, random_seed=random_seed)

    def load(
        self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2
    ) -> Data:
        """Download and load Facebook Page-Page with stratified anti-leakage splits.

        Args:
            train_ratio: Proportion of training nodes (default: 0.6).
            val_ratio: Proportion of validation nodes (default: 0.2).
            test_ratio: Proportion of test nodes (default: 0.2).

        Returns:
            PyG Data object with anti-leakage train_mask, val_mask, and test_mask.
        """
        dataset = FacebookPagePage(root=self.data_dir)
        raw_data = dataset[0]

        splitter = AntiLeakageSplitter(random_seed=self.random_seed)
        split_data = splitter.split(
            data=raw_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratified=True,
        )

        self.data = split_data
        return self.data
