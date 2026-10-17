from typing import List, Optional
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

from src.datasets.anti_leakage_splitter import AntiLeakageSplitter
from src.datasets.base_loader import BaseDatasetLoader


class CoraLoader(BaseDatasetLoader):
    """Loads and preprocesses the Cora citation graph supporting Dual-Reporting Protocol."""

    def __init__(
        self,
        data_dir: str,
        split_mode: str = "planetoid",
        random_seed: int = 42,
    ) -> None:
        """Initialize Cora loader.

        Args:
            data_dir: Directory where dataset is stored.
            split_mode: 'planetoid' (140/500/1000 public benchmark) or 'full_dense' (70/15/15 full coverage).
            random_seed: Random seed for deterministic reproducibility.
        """
        super().__init__(data_dir=data_dir, random_seed=random_seed)
        self.split_mode = split_mode

    def load(self) -> Data:
        """Download or load Cora according to selected split mode.

        Returns:
            PyG Data object with standardized features and appropriate masks.
        """
        dataset = Planetoid(root=str(self.data_dir), name="Cora", split="public")
        raw_data = dataset[0]

        if self.split_mode == "planetoid":
            # Canonical Planetoid split (Yang et al., ICML 2016)
            unlabeled_mask = ~(raw_data.train_mask | raw_data.val_mask | raw_data.test_mask)
            raw_data.unlabeled_mask = unlabeled_mask
            raw_data.unlabeled_count = int(unlabeled_mask.sum())
            self.data = raw_data
            return raw_data

        # Modern Full Dense Anti-Leakage Split (70/15/15) covering 100% of 2,708 nodes
        splitter = AntiLeakageSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=self.random_seed,
        )
        train_mask, val_mask, test_mask = splitter.split_nodes(raw_data, stratify=True)
        data = splitter.fit_transform_features(raw_data, train_mask)

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.unlabeled_count = 0

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

        num_classes = int(data.y.max().item()) + 1
        by_class: dict = {c: [] for c in range(num_classes)}
        for idx, lbl in zip(val_indices, val_labels):
            by_class[lbl].append(idx)

        selected = []
        target_per_class = max(1, count // num_classes)
        for c in range(num_classes):
            take = min(len(by_class[c]), target_per_class)
            selected.extend(by_class[c][:take])

        # Fill remaining slots up to count
        selected_set = set(selected)
        remaining = [idx for idx in val_indices if idx not in selected_set]
        needed = count - len(selected)
        if needed > 0:
            selected.extend(remaining[:needed])

        return sorted(selected[:count])

