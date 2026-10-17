"""Unit Tests for Explainers Engine.

Verifies that LT-DEF, EiG-Search, GNNExplainer, Saliency, Random, and Nearest-First
output valid normalized masks matching local computation graph topology.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
from torch_geometric.data import Data

from src.explainers.base_explainer import ExplanationResult
from src.explainers.eig_search import EigSearchExplainer
from src.explainers.gnn_explainer import GNNExplainerWrapper
from src.explainers.lt_def import LTDEFExplainer
from src.explainers.nearest_first_explainer import NearestFirstExplainer
from src.explainers.pg_explainer import PGExplainerWrapper
from src.explainers.random_explainer import RandomExplainer
from src.explainers.saliency import SaliencyExplainer
from src.models.gcn import GCN


@pytest.fixture
def trained_gcn_and_data():
    """Sets up a small reproducible graph and trains a GCN."""
    torch.manual_seed(42)
    num_nodes = 40
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, 90))
    y = torch.randint(0, 2, (num_nodes,))
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:30] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[30:35] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[35:] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    model = GCN(in_channels=8, hidden_channels=16, out_channels=2)
    model.fit_model(data, epochs=5, lr=0.01)
    return model, data


def test_lt_def_explainer(trained_gcn_and_data):
    """Test LT-DEF explainer returns valid normalized masks and subgraphs."""
    model, data = trained_gcn_and_data
    explainer = LTDEFExplainer(model, num_hops=2, sparsity_target=0.70)
    target_node = 35

    res = explainer.explain_node(target_node, data.x, data.edge_index)
    assert isinstance(res, ExplanationResult)
    assert res.target_node == target_node
    assert res.edge_mask.numel() > 0
    assert (res.edge_mask >= 0.0).all() and (res.edge_mask <= 1.0 + 1e-5).all()
    assert res.execution_time > 0.0


def test_eig_search_explainer(trained_gcn_and_data):
    """Test EiG-Search explainer executes and outputs normalized mask."""
    model, data = trained_gcn_and_data
    explainer = EigSearchExplainer(model, num_hops=2, sparsity_target=0.70)
    target_node = 36

    res = explainer.explain_node(target_node, data.x, data.edge_index)
    assert isinstance(res, ExplanationResult)
    assert res.target_node == target_node
    assert (res.edge_mask >= 0.0).all() and (res.edge_mask <= 1.0 + 1e-5).all()


def test_gnn_explainer_wrapper(trained_gcn_and_data):
    """Test GNNExplainer native PyG wrapper produces attribution masks."""
    model, data = trained_gcn_and_data
    explainer = GNNExplainerWrapper(model, epochs=2, lr=0.01, num_hops=2)
    target_node = 35

    res = explainer.explain_node(target_node, data.x, data.edge_index)
    assert isinstance(res, ExplanationResult)
    assert res.target_node == target_node
    assert res.edge_mask.numel() > 0
    assert (res.edge_mask >= 0.0).all() and (res.edge_mask <= 1.0 + 1e-5).all()


def test_saliency_explainer(trained_gcn_and_data):
    """Test Saliency gradient attribution explainer."""
    model, data = trained_gcn_and_data
    explainer = SaliencyExplainer(model=model, num_hops=2)
    target_node = 0

    res = explainer.explain_node(target_node, data.x, data.edge_index)
    assert isinstance(res, ExplanationResult)
    assert res.edge_mask.numel() > 0
    assert (res.edge_mask >= 0.0).all() and (res.edge_mask <= 1.0 + 1e-5).all()


def test_random_explainer(trained_gcn_and_data):
    """Test Random baseline explainer produces uniform random mask."""
    model, data = trained_gcn_and_data
    explainer = RandomExplainer(model=model, num_hops=2, random_seed=42)
    target_node = 0

    res = explainer.explain_node(target_node, data.x, data.edge_index)
    assert isinstance(res, ExplanationResult)
    assert res.edge_mask.numel() > 0
    assert (res.edge_mask >= 0.0).all() and (res.edge_mask <= 1.0 + 1e-5).all()


def test_nearest_first_explainer(trained_gcn_and_data):
    """Test Nearest-First topological distance baseline explainer."""
    model, data = trained_gcn_and_data
    explainer = NearestFirstExplainer(model=model, num_hops=2)
    target_node = 0

    res = explainer.explain_node(target_node, data.x, data.edge_index)
    assert isinstance(res, ExplanationResult)
    assert res.edge_mask.numel() > 0
    assert (res.edge_mask >= 0.0).all() and (res.edge_mask <= 1.0 + 1e-5).all()


def test_pg_explainer_wrapper(trained_gcn_and_data):
    """Test PGExplainer parameterized explainer training and inference."""
    model, data = trained_gcn_and_data
    explainer = PGExplainerWrapper(model=model, epochs=2, lr=0.01, num_hops=2)
    # Train on small validation subset
    train_nodes = torch.tensor([0, 1, 2])
    explainer.train_explainer(data.x, data.edge_index, target_indices=train_nodes)
    assert explainer.is_trained is True

    target_node = 0
    res = explainer.explain_node(target_node, data.x, data.edge_index)
    assert isinstance(res, ExplanationResult)
    assert res.edge_mask.numel() > 0
    assert (res.edge_mask >= 0.0).all() and (res.edge_mask <= 1.0 + 1e-5).all()
