"""Twitch Gamers Social Network Dataset Loader.

Loads the Twitch social network (e.g., DE or EN subgraphs) for real-world social
graph node classification benchmarking and applies strict anti-leakage splitting.
"""

from torch_geometric.datasets import Twitch
from torch_geometric.data import Data

from src.datasets.anti_leakage_splitter import AntiLeakageSplitter
from src.datasets.base_loader import BaseDatasetLoader


class TwitchLoader(BaseDatasetLoader):
    """Loads and partitions the Twitch social network benchmark."""

    def __init__(self, data_dir: str, name: str = "DE", random_seed: int = 42) -> None:
        """Initialize Twitch loader.

        Args:
            data_dir: Storage directory.
            name: Twitch sub-community name ('DE', 'EN', 'ES', 'FR', 'PT', 'RU').
            random_seed: Random seed for reproducibility.
        """
        super().__init__(data_dir=data_dir, random_seed=random_seed)
        self.name = name

    def load(self) -> Data:
        """Download or load Twitch sub-community graph and apply anti-leakage split.

        Returns:
            PyG Data object with standardized features and stratified masks.
        """
        dataset = Twitch(root=str(self.data_dir), name=self.name)
        raw_data = dataset[0]

        # Apply strict Anti-Leakage Split
        splitter = AntiLeakageSplitter(random_seed=self.random_seed)
        train_mask, val_mask, test_mask = splitter.split_nodes(raw_data, stratify=True)
        data = splitter.fit_transform_features(raw_data, train_mask)

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        self.data = data
        return data
