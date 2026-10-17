"""Unit tests for BG04 (explain_edges) and BG08 (fidelity with smart tie-breaking).

Verifies compliance with Group 09 Handover Guidelines:
- BG04: explain_edges signature and 5-tuple return format.
- BG05 & BG07: Candidate edge extraction and undirected pair canonicalization.
- BG06: Max-aggregation across directions, finite scores.
- BG08: Smart tie-breaking (1 run for continuous floats, 20 runs for ties) and OOD tracking.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.explainers.edge_explainer_service import (
    explain_edges,
    get_candidate_edges,
    LAST_NODE_FEAT_MASK,
)
from src.evaluation.fidelity import compute_fidelity_bg08


class DummyGCN(nn.Module):
    """Tiny deterministic GCN for testing."""
    def __init__(self, in_dim=8, hidden_dim=8, out_dim=3):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        out = self.lin(x)
        if edge_index.size(1) > 0:
            agg = torch.zeros_like(out)
            weight = edge_weight.unsqueeze(-1) if edge_weight is not None else 1.0
            agg.index_add_(0, edge_index[0], out[edge_index[1]] * weight)
            out = out + agg * 0.1
        return out


@pytest.fixture
def sample_graph():
    """Create a small connected undirected graph."""
    torch.manual_seed(42)
    num_nodes = 6
    x = torch.randn(num_nodes, 8)
    # Undirected edges: (0-1), (0-2), (1-2), (1-3), (2-4), (3-5), (4-5) plus self-loop (0-0)
    directed = [
        (0, 1), (1, 0),
        (0, 2), (2, 0),
        (1, 2), (2, 1),
        (1, 3), (3, 1),
        (2, 4), (4, 2),
        (3, 5), (5, 3),
        (4, 5), (5, 4),
        (0, 0), # self-loop to test filtering
    ]
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    y = torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def test_get_candidate_edges_filtering(sample_graph):
    """Verify BG05 and BG07: self-loops filtered out and canonical undirected pairs."""
    candidates, sub_edges, edge_to_cand = get_candidate_edges(
        node_id=0,
        edge_index=sample_graph.edge_index,
        num_nodes=sample_graph.x.size(0),
        num_hops=2,
    )

    # Check no self loops in candidates
    for cand in candidates:
        assert cand["src"] != cand["dst"]
        assert cand["src"] < cand["dst"]

    # In 2-hop from node 0, nodes 0,1,2,3,4 are reachable
    assert len(candidates) > 0
    assert sub_edges.size(1) > 0
    assert edge_to_cand.size(0) == sub_edges.size(1)


def test_explain_edges_bg04_saliency(sample_graph):
    """Verify BG04 signature and 5-tuple return on Saliency."""
    model = DummyGCN()
    candidates, _, _ = get_candidate_edges(
        node_id=0,
        edge_index=sample_graph.edge_index,
        num_nodes=sample_graph.x.size(0),
        num_hops=2,
    )

    result = explain_edges(
        model=model,
        data=sample_graph,
        node_id=0,
        target_class=0,
        candidate_edges=candidates,
        config={"method": "saliency", "num_hops": 2},
        seed=42,
    )

    # Must be 5-tuple per BG04
    assert isinstance(result, tuple)
    assert len(result) == 5
    edge_ids, scores, runtime_sec, status, error_msg = result

    assert status == "success"
    assert error_msg == ""
    assert runtime_sec > 0.0
    assert len(edge_ids) == len(candidates)
    assert len(scores) == len(candidates)
    assert all(isinstance(s, float) for s in scores)
    assert all(not torch.isnan(torch.tensor(s)) for s in scores)


def test_explain_edges_bg04_baselines(sample_graph):
    """Verify BG04 signature on Random and Nearest-First."""
    model = DummyGCN()
    candidates, _, _ = get_candidate_edges(
        node_id=0,
        edge_index=sample_graph.edge_index,
        num_nodes=sample_graph.x.size(0),
        num_hops=2,
    )

    for method in ["random", "nearest_first"]:
        res = explain_edges(
            model=model,
            data=sample_graph,
            node_id=0,
            target_class=0,
            candidate_edges=candidates,
            config={"method": method, "num_hops": 2},
            seed=42,
        )
        edge_ids, scores, runtime_sec, status, error_msg = res
        assert status == "success"
        assert len(scores) == len(candidates)


def test_compute_fidelity_bg08_smart_tie_breaking(sample_graph):
    """Verify BG08 smart tie-breaking: 1 run for distinct scores, 20 runs for ties."""
    model = DummyGCN()
    candidates, sub_edges, _ = get_candidate_edges(
        node_id=0,
        edge_index=sample_graph.edge_index,
        num_nodes=sample_graph.x.size(0),
        num_hops=2,
    )

    # Case 1: Distinct float scores (e.g. Saliency, GNNExplainer)
    distinct_scores = [0.1 * (i + 1) for i in range(len(candidates))]
    res_distinct = compute_fidelity_bg08(
        model=model,
        x=sample_graph.x,
        sub_edge_index=sub_edges,
        target_node=0,
        target_class=0,
        candidate_edges=candidates,
        scores=distinct_scores,
        budgets_k=[1, 3],
        proportions_p=[0.2, 0.5],
        num_tie_runs=20,
        seed=42,
    )
    assert res_distinct["has_ties"] is False
    assert res_distinct["actual_runs_evaluated"] == 1  # Smart optimization: 1 run!
    assert "top_k_1_fid_plus" in res_distinct
    assert "prop_20pct_fid_plus" in res_distinct

    # Case 2: Tied scores (e.g. Nearest-First or all-zeros)
    tied_scores = [0.5] * len(candidates)
    res_tied = compute_fidelity_bg08(
        model=model,
        x=sample_graph.x,
        sub_edge_index=sub_edges,
        target_node=0,
        target_class=0,
        candidate_edges=candidates,
        scores=tied_scores,
        budgets_k=[1, 3],
        proportions_p=[0.2, 0.5],
        num_tie_runs=20,
        seed=42,
    )
    assert res_tied["has_ties"] is True
    assert res_tied["actual_runs_evaluated"] == 20  # Runs 20 times for actual ties!
    assert "isolated_at_budget" in res_tied
