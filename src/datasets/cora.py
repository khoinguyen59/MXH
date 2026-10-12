"""Cora Citation Network Dataset Loader.

Loads the standard Cora academic network for node classification benchmarking
and applies strict anti-leakage splitting.
"""

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

from src.datasets.anti_leakage_splitter import AntiLeakageSplitter
from src.datasets.base_loader import BaseDatasetLoader


class CoraLoader(BaseDatasetLoader):
    """Loads and preprocesses the Cora citation graph."""

    def load(self) -> Data:
        """Download or load Cora, override default masks with Anti-Leakage split.

        Returns:
            PyG Data object with standardized features and stratified masks.
        """
        dataset = Planetoid(root=str(self.data_dir), name="Cora")
        raw_data = dataset[0]

        # Apply strict Anti-Leakage Split (overriding default 20-per-class split)
        splitter = AntiLeakageSplitter(random_seed=self.random_seed)
        train_mask, val_mask, test_mask = splitter.split_nodes(raw_data, stratify=True)
        data = splitter.fit_transform_features(raw_data, train_mask)

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        self.data = data
        return data
