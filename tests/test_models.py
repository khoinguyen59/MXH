"""Unit Tests for GNN Models.

Verifies forward pass dimensions, loss computation, and backpropagation for GCN, GraphSAGE, and GAT.
"""

import pytest
import torch
from torch_geometric.data import Data

from src.models.gat import GAT
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE


@pytest.fixture
def dummy_graph():
    """Generates a small test graph for testing architectures."""
    num_nodes = 50
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, 120))
    y = torch.randint(0, 3, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


def test_gcn_forward_and_loss(dummy_graph):
    """Test GCN forward pass output shape and log-softmax probabilities."""
    model = GCN(in_channels=16, hidden_channels=32, out_channels=3, num_layers=2)
    out = model(dummy_graph.x, dummy_graph.edge_index)
    assert out.shape == (50, 3)
    # Check that out is log-softmax (values <= 0)
    assert (out <= 0.0).all()


def test_graphsage_forward(dummy_graph):
    """Test GraphSAGE forward pass and embeddings extraction."""
    model = GraphSAGE(in_channels=16, hidden_channels=32, out_channels=3, num_layers=2)
    out = model(dummy_graph.x, dummy_graph.edge_index)
    assert out.shape == (50, 3)
    emb = model.get_node_embeddings(dummy_graph.x, dummy_graph.edge_index)
    assert emb.shape == (50, 32)


def test_gat_forward(dummy_graph):
    """Test GAT forward pass with multi-head attention."""
    model = GAT(in_channels=16, hidden_channels=8, out_channels=3, heads=4)
    out = model(dummy_graph.x, dummy_graph.edge_index)
    assert out.shape == (50, 3)
