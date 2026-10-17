"""Anti-Leakage Data Partitioning and Scaling Module.

Implements strict stratified train/val/test masking and train-only feature
standardization to prevent transductive and inductive data leakage in compliance
with course governance standards (Special Lecture 01 - TS. Tran Hung Nghiep).
"""

from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data


class AntiLeakageSplitter:
    """Partitions graph node indices into mutually exclusive splits and standardizes features."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        """Initialize the anti-leakage splitter.

        Args:
            train_ratio: Proportion of nodes for training (default: 0.70).
            val_ratio: Proportion of nodes for validation (default: 0.15).
            test_ratio: Proportion of nodes for testing (default: 0.15).
            random_seed: Random seed for deterministic reproducibility.
        """
        assert (
            abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5
        ), "Split ratios must sum to 1.0."

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.scaler = StandardScaler()

    def split_nodes(
        self, data: Data, stratify: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate strictly non-overlapping train/val/test boolean masks.

        Args:
            data: PyG graph Data object containing `y` (target labels).
            stratify: Whether to preserve class proportions across splits.

        Returns:
            Tuple of (train_mask, val_mask, test_mask) as torch boolean tensors.
        """
        num_nodes = data.num_nodes
        indices = np.arange(num_nodes)
        labels = data.y.cpu().numpy() if (stratify and data.y is not None) else None

        # First split: Separate Train from (Val + Test)
        test_val_ratio = self.val_ratio + self.test_ratio
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=test_val_ratio,
            random_state=self.random_seed,
            stratify=labels if stratify else None,
        )

        # Second split: Separate Val from Test
        temp_labels = labels[temp_idx] if stratify else None
        relative_test_ratio = self.test_ratio / test_val_ratio
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_ratio,
            random_state=self.random_seed,
            stratify=temp_labels if stratify else None,
        )

        # Verification of mutual exclusivity (Anti-Leakage Check)
        assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train-Val overlap detected!"
        assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train-Test overlap detected!"
        assert len(set(val_idx).intersection(set(test_idx))) == 0, "Val-Test overlap detected!"
        assert len(train_idx) + len(val_idx) + len(test_idx) == num_nodes, "Total node count mismatch!"

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask

    def fit_transform_features(self, data: Data, train_mask: torch.Tensor) -> Data:
        """Standardize node features by fitting the scaler strictly on the training partition.

        Validation and test features are transformed using training parameters,
        preventing any information leakage.

        Args:
            data: PyG graph Data object containing `x` (node feature matrix).
            train_mask: Boolean mask indicating training nodes.

        Returns:
            PyG Data object with standardized node features.
        """
        if data.x is None or data.x.numel() == 0:
            return data

        x_np = data.x.cpu().numpy()
        train_x = x_np[train_mask.cpu().numpy()]

        # Fit strictly on training nodes
        self.scaler.fit(train_x)

        # Standardize columns with non-zero variance, preserve constant columns
        variances = np.var(train_x, axis=0)
        constant_cols = variances < 1e-6
        if np.all(constant_cols):
            # All columns are constant, keep original data
            return data

        # Transform all nodes using training statistics
        x_scaled = self.scaler.transform(x_np)
        if np.any(constant_cols):
            # Restore constant columns to their original values
            x_scaled[:, constant_cols] = x_np[:, constant_cols]

        data.x = torch.tensor(x_scaled, dtype=torch.float32)

        return data
