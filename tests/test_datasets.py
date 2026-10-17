"""Unit Tests for Dataset Ingestion Service.

Verifies synthetic BA-Shapes graph generation, ground-truth motif attachment,
and feature tensor shapes.
"""

import pytest
import torch
from src.datasets.ba_shapes import BAShapesLoader


def test_ba_shapes_loader(tmp_path):
    """Verify BA-Shapes synthetic graph topology and motif ground truth."""
    loader = BAShapesLoader(
        data_dir=str(tmp_path),
        num_base_nodes=100,
        num_motifs=20,
        random_seed=42,
    )
    data = loader.load()

    # Total nodes = 100 base + 20 * 5 motif = 200 nodes
    assert data.num_nodes == 200
    assert data.x.shape == (200, 10)
    assert data.edge_index.size(0) == 2
    assert hasattr(data, "ground_truth_mask")
    # Ground truth edge mask should contain true edges
    assert (data.ground_truth_mask == 1.0).sum() > 0
    # Masks are set
    assert hasattr(data, "train_mask")
    assert hasattr(data, "val_mask")
    assert hasattr(data, "test_mask")
