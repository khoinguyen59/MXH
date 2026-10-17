"""Unit Tests for Anti-Leakage Partitioning and Scaling.

Verifies strict non-overlapping partitions and absence of data leakage
(Special Lecture 01 - TS. Tran Hung Nghiep).
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.datasets.anti_leakage_splitter import AntiLeakageSplitter


def test_anti_leakage_partition_disjointness():
    """Verify train, val, and test splits have exactly zero overlap."""
    num_nodes = 500
    x = torch.randn((num_nodes, 16))
    y = torch.randint(0, 3, (num_nodes,))
    edge_index = torch.randint(0, num_nodes, (2, 1000))
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=num_nodes)

    splitter = AntiLeakageSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    train_mask, val_mask, test_mask = splitter.split_nodes(data, stratify=True)

    # Convert masks to indices
    train_idx = set(torch.where(train_mask)[0].tolist())
    val_idx = set(torch.where(val_mask)[0].tolist())
    test_idx = set(torch.where(test_mask)[0].tolist())

    # Mutual exclusivity checks
    assert len(train_idx.intersection(val_idx)) == 0, "Train and Val splits overlap!"
    assert len(train_idx.intersection(test_idx)) == 0, "Train and Test splits overlap!"
    assert len(val_idx.intersection(test_idx)) == 0, "Val and Test splits overlap!"

    # Total coverage check
    assert len(train_idx) + len(val_idx) + len(test_idx) == num_nodes, "Split size mismatch!"


def test_anti_leakage_feature_scaling_train_only():
    """Verify that feature scaler parameters are fitted strictly on train data."""
    num_nodes = 100
    x = torch.ones((num_nodes, 4))
    # Give test partition drastically different scale
    x[70:] = 100.0
    y = torch.zeros(num_nodes, dtype=torch.long)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:70] = True

    splitter = AntiLeakageSplitter(random_seed=42)
    transformed_data = splitter.fit_transform_features(data, train_mask)

    # Scaler mean must match the train mean (1.0), not the full dataset mean
    assert abs(splitter.scaler.mean_[0] - 1.0) < 1e-4, "Scaler leaked test data into mean computation!"
