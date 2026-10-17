"""Unit Tests for Cora Explainability Pipeline and Model Wrapper.

Verifies:
1. Complete loading of Cora artifacts via CoraHandoverAdapter.
2. Accuracy consistency of wrapped model matching benchmark (83.60% Test Acc).
3. Full support for `edge_weight` in `forward` without TypeError.
4. Correct generation of edge attributions using Saliency and Random explainers.
"""

import pytest
import torch
from src.datasets.cora_handover_adapter import CoraHandoverAdapter, CoraGCNModelWrapper
from src.explainers.saliency import SaliencyExplainer
from src.explainers.random_explainer import RandomExplainer


def test_cora_adapter_loading():
    """Verify loading and integrity of Cora bundle."""
    adapter = CoraHandoverAdapter()
    assert adapter.verify_files_exist(), "Handover files must exist in data/t01_handover/cora"

    bundle = adapter.load(device="cpu")
    assert bundle.data.num_nodes == 2708
    assert bundle.data.num_edges == 10556
    assert len(bundle.val_nodes) == 100
    assert len(bundle.multiseed_subset) == 30
    assert isinstance(bundle.model, CoraGCNModelWrapper)


def test_cora_model_accuracy_consistency():
    """Verify the wrapped model reproduces the exact test accuracy of benchmark."""
    adapter = CoraHandoverAdapter()
    bundle = adapter.load(device="cpu")

    model = bundle.model
    data = bundle.data

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=-1)

    test_mask = data.test_mask
    acc = (preds[test_mask] == data.y[test_mask]).float().mean().item()
    assert round(acc, 3) == 0.836, f"Expected 83.60% Test Acc, got {acc*100:.2f}%"


def test_cora_model_edge_weight_support():
    """Verify that forward pass supports continuous edge_weight without TypeError."""
    adapter = CoraHandoverAdapter()
    bundle = adapter.load(device="cpu")

    model = bundle.model
    data = bundle.data

    # Generate dummy continuous edge weights
    num_edges = data.edge_index.size(1)
    edge_weight = torch.rand(num_edges)

    # Calling forward with edge_weight must succeed
    out = model(data.x, data.edge_index, edge_weight=edge_weight)
    assert out.shape == (2708, 7)
    assert not torch.isnan(out).any()


def test_cora_fast_explanation():
    """Verify running XAI explainer on a target node from val_nodes."""
    adapter = CoraHandoverAdapter()
    bundle = adapter.load(device="cpu")

    target_node = bundle.val_nodes[0]
    explainer = SaliencyExplainer(model=bundle.model, num_hops=2)

    res = explainer.explain_node(target_node, bundle.data.x, bundle.data.edge_index)
    assert res.target_node == target_node
    assert res.edge_mask.numel() > 0
    assert (res.edge_mask >= 0.0).all() and (res.edge_mask <= 1.0 + 1e-5).all()

    # Also test random explainer baseline
    rand_explainer = RandomExplainer(model=bundle.model, num_hops=2)
    rand_res = rand_explainer.explain_node(target_node, bundle.data.x, bundle.data.edge_index)
    assert rand_res.edge_mask.numel() == res.edge_mask.numel()
