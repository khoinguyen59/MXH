"""Unit & Regression Tests for T02 XAI-GNN Critical Fixes.

Covers:
1. CoraGCNModelWrapper return_type ('log_probs', 'probs', 'raw').
2. compute_fidelity_at_budget undirected symmetrical edge pruning.
3. EiG-Search, LT-DEF, and SubgraphX tensor dimension alignment (subgraph_edge_index matches edge_mask).
4. Edge case handling on isolated nodes (num_edges == 0) across all explainers.
5. End-to-end fidelity evaluation without IndexError or RuntimeError.
"""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.t02_explainers.adapters.cora_handover_adapter import CoraGCNModelWrapper
from src.t02_explainers.algorithms.eig_search import EigSearchExplainer
from src.t02_explainers.algorithms.lt_def import LTDEFExplainer
from src.t02_explainers.algorithms.subgraphx import SubgraphXExplainer
from src.t02_explainers.algorithms.saliency import SaliencyExplainer
from src.t02_explainers.algorithms.random_explainer import RandomExplainer
from src.t02_explainers.algorithms.nearest_first_explainer import NearestFirstExplainer
from src.t02_explainers.algorithms.gnn_explainer import GNNExplainerWrapper
from src.t02_explainers.algorithms.pg_explainer import PGExplainerWrapper
from src.t03_evaluation.fidelity import (
    compute_fidelity_at_budget,
    compute_fidelity_curve,
    compute_feature_fidelity_at_budget,
)


class Dummy2HopGCN(torch.nn.Module):
    """Simple 2-layer GCN for isolated and small graph testing."""

    def __init__(self, in_dim: int = 16, hidden: int = 8, n_classes: int = 3):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden, cached=False)
        self.c2 = GCNConv(hidden, n_classes, cached=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        return_type: str = "log_probs",
    ) -> torch.Tensor:
        h = F.relu(self.c1(x, edge_index, edge_weight=edge_weight))
        out = self.c2(h, edge_index, edge_weight=edge_weight)
        if return_type == "raw":
            return out
        elif return_type == "probs":
            return F.softmax(out, dim=-1)
        return F.log_softmax(out, dim=-1)


def test_cora_model_wrapper_return_types():
    """Verify CoraGCNModelWrapper supports log_probs, probs, and raw correctly."""
    torch.manual_seed(42)
    wrapper = CoraGCNModelWrapper(in_dim=10, hidden=8, n_classes=4)
    wrapper.eval()
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    # 1. Default / log_probs
    out_log = wrapper(x, edge_index, return_type="log_probs")
    assert out_log.shape == (5, 4)
    # Exponentiated log_probs must sum to 1 along class dimension
    probs_from_log = torch.exp(out_log)
    assert torch.allclose(probs_from_log.sum(dim=-1), torch.ones(5), atol=1e-5)

    # 2. probs
    out_probs = wrapper(x, edge_index, return_type="probs")
    assert out_probs.shape == (5, 4)
    assert torch.allclose(out_probs.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(probs_from_log, out_probs, atol=1e-5)

    # 3. raw & logits alias
    out_raw = wrapper(x, edge_index, return_type="raw")
    out_logits = wrapper(x, edge_index, return_type="logits")
    assert out_raw.shape == (5, 4)
    assert torch.allclose(out_raw, out_logits)
    expected_log = F.log_softmax(out_raw, dim=-1)
    assert torch.allclose(out_log, expected_log, atol=1e-5)

    # 4. Invalid return_type raises ValueError
    with pytest.raises(ValueError, match="Unsupported return_type"):
        wrapper(x, edge_index, return_type="invalid_type")


def test_symmetrical_edge_pruning_in_fidelity():
    """Verify compute_fidelity_at_budget removes undirected edge pairs symmetrically."""
    torch.manual_seed(42)
    model = Dummy2HopGCN(in_dim=4, hidden=4, n_classes=2)
    model.eval()

    # Graph with 3 nodes: 0 connected to 1 and 2
    # Directed edges: (0, 1), (1, 0), (0, 2), (2, 0)
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    x = torch.randn(3, 4)

    # Edge mask where edge 0 ((0, 1)) has highest score 1.0, edge 1 ((1, 0)) has 0.1
    # Without symmetrization, only edge 0 would be removed in top-1.
    # With symmetrization, both edge 0 (0, 1) and edge 1 (1, 0) must be removed!
    edge_mask = torch.tensor([1.0, 0.1, 0.5, 0.5], dtype=torch.float32)

    # Symmetrized run
    fid_plus_sym, fid_minus_sym = compute_fidelity_at_budget(
        model=model,
        x=x,
        sub_edge_index=edge_index,
        target_node=0,
        target_class=0,
        edge_mask=edge_mask,
        k=1,
        symmetrize=True,
    )

    # Asymmetric run
    fid_plus_asym, fid_minus_asym = compute_fidelity_at_budget(
        model=model,
        x=x,
        sub_edge_index=edge_index,
        target_node=0,
        target_class=0,
        edge_mask=edge_mask,
        k=1,
        symmetrize=False,
    )

    # Fidelity scores should be valid numbers
    assert isinstance(fid_plus_sym, float)
    assert isinstance(fid_minus_sym, float)
    assert not torch.isnan(torch.tensor(fid_plus_sym))
    assert not torch.isnan(torch.tensor(fid_minus_sym))

    # Test that budget progression on undirected symmetric mask removes additional edges at each k (no plateau)
    e_prog = torch.tensor([[0, 1, 0, 2, 0, 3], [1, 0, 2, 0, 3, 0]], dtype=torch.long)
    x_prog = torch.randn(4, 4)
    mask_prog = torch.tensor([1.0, 1.0, 0.8, 0.8, 0.6, 0.6], dtype=torch.float32)
    fp1, _ = compute_fidelity_at_budget(model, x_prog, e_prog, 0, 0, mask_prog, k=1, symmetrize=True)
    fp2, _ = compute_fidelity_at_budget(model, x_prog, e_prog, 0, 0, mask_prog, k=2, symmetrize=True)
    fp3, _ = compute_fidelity_at_budget(model, x_prog, e_prog, 0, 0, mask_prog, k=3, symmetrize=True)
    assert fp1 != fp2, "Fidelity at k=2 must not plateau identically to k=1"
    assert fp2 != fp3, "Fidelity at k=3 must not plateau identically to k=2"


def test_explainers_dimension_alignment():
    """Verify that subgraph_edge_index matches edge_mask dimensions exactly."""
    torch.manual_seed(42)
    model = Dummy2HopGCN(in_dim=8, hidden=8, n_classes=3)
    model.eval()

    # Graph with 6 nodes, 10 directed edges
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
         [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]],
        dtype=torch.long,
    )
    x = torch.randn(5, 8)

    explainers = [
        ("EiG-Search", EigSearchExplainer(model=model, num_hops=2)),
        ("LT-DEF", LTDEFExplainer(model=model, num_hops=2)),
        ("SubgraphX", SubgraphXExplainer(model=model, num_hops=2, rollout_steps=5)),
    ]

    for name, exp in explainers:
        res = exp.explain_node(node_idx=0, x=x, edge_index=edge_index)
        assert res.subgraph_edge_index.size(1) == res.edge_mask.numel(), (
            f"{name}: subgraph_edge_index size ({res.subgraph_edge_index.size(1)}) "
            f"does not match edge_mask size ({res.edge_mask.numel()})"
        )
        assert res.subgraph_nodes.numel() > 0

        # Run fidelity evaluation directly: must succeed without IndexError
        fid_plus, fid_minus = compute_fidelity_at_budget(
            model=model,
            x=x[res.subgraph_nodes],
            sub_edge_index=res.subgraph_edge_index,
            target_node=int(torch.where(res.subgraph_nodes == 0)[0].item()),
            target_class=res.predicted_class,
            edge_mask=res.edge_mask,
            k=3,
        )
        assert isinstance(fid_plus, float)
        assert isinstance(fid_minus, float)


def test_explainers_isolated_node_edge_case():
    """Verify all explainers handle isolated nodes (num_edges == 0) gracefully."""
    torch.manual_seed(42)
    model = Dummy2HopGCN(in_dim=8, hidden=8, n_classes=3)
    model.eval()

    # Graph with 3 disconnected nodes, 0 edges
    edge_index = torch.empty((2, 0), dtype=torch.long)
    x = torch.randn(3, 8)

    explainers = [
        ("Saliency", SaliencyExplainer(model=model, num_hops=2)),
        ("Random", RandomExplainer(model=model, num_hops=2, random_seed=42)),
        ("Nearest-First", NearestFirstExplainer(model=model, num_hops=2)),
        ("EiG-Search", EigSearchExplainer(model=model, num_hops=2)),
        ("LT-DEF", LTDEFExplainer(model=model, num_hops=2)),
        ("SubgraphX", SubgraphXExplainer(model=model, num_hops=2, rollout_steps=5)),
        ("GNNExplainer", GNNExplainerWrapper(model=model, epochs=2, num_hops=2)),
        ("PGExplainer", PGExplainerWrapper(model=model, epochs=2, num_hops=2)),
    ]

    for name, exp in explainers:
        res = exp.explain_node(node_idx=1, x=x, edge_index=edge_index)
        assert res.edge_mask.numel() == 0, f"{name} should return empty edge_mask on isolated node"
        assert res.subgraph_edge_index.size(1) == 0

        # Fidelity on isolated node should return 0.0, 0.0 without crash
        fp, fm = compute_fidelity_at_budget(
            model=model,
            x=x,
            sub_edge_index=res.subgraph_edge_index,
            target_node=1,
            target_class=res.predicted_class,
            edge_mask=res.edge_mask,
            k=5,
        )
        assert fp == 0.0
        assert fm == 0.0


def test_feature_fidelity_and_attribution():
    """Verify node feature attribution extraction and feature fidelity calculation (RQ2)."""
    torch.manual_seed(42)
    model = Dummy2HopGCN(in_dim=8, hidden=8, n_classes=3)
    model.eval()

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, 8)

    # 1. Test Saliency node_feat_mask
    sal = SaliencyExplainer(model=model, num_hops=2)
    res_sal = sal.explain_node(node_idx=0, x=x, edge_index=edge_index)
    assert res_sal.node_feat_mask is not None
    assert res_sal.node_feat_mask.shape == (8,)
    assert (res_sal.node_feat_mask >= 0.0).all()

    # 2. Test compute_feature_fidelity_at_budget
    feat_fp, feat_fm = compute_feature_fidelity_at_budget(
        model=model,
        x=x[res_sal.subgraph_nodes],
        sub_edge_index=res_sal.subgraph_edge_index,
        target_node=0,
        target_class=res_sal.predicted_class,
        node_feat_mask=res_sal.node_feat_mask,
        top_k=4,
    )
    assert isinstance(feat_fp, float)
    assert isinstance(feat_fm, float)
    assert not torch.isnan(torch.tensor(feat_fp))
    assert not torch.isnan(torch.tensor(feat_fm))
