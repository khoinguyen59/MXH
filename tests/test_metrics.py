"""Unit Tests for Evaluation Metrics & Statistical Testing.

Verifies mathematical properties of Fidelity+, Fidelity-, Sparsity, Motif-AUC,
Hallucination rate, Fidelity curves, Chance-Corrected Kappa, k*,
Variance decomposition (ANOVA), and Holm-Bonferroni / TOST tests.
"""

import pytest
import torch
import numpy as np

from src.evaluation.agreement import (
    compute_chance_corrected_kappa,
    compute_jaccard_similarity,
    compute_pairwise_agreement_matrix,
)
from src.evaluation.fidelity import (
    compute_fidelity_at_budget,
    compute_fidelity_curve,
)
from src.evaluation.k_star import compute_k_star
from src.evaluation.metrics import (
    compute_fidelity_minus,
    compute_fidelity_plus,
    compute_hallucination_rate,
    compute_motif_roc_auc,
    compute_sparsity,
)
from src.evaluation.statistical_tests import (
    holm_bonferroni_correction,
    tost_equivalence_test,
)
from src.evaluation.variance_decomposition import decompose_explanation_variance
from src.models.gcn import GCN


@pytest.fixture
def dummy_graph_and_model():
    """Generates a small graph and model for metric evaluations."""
    torch.manual_seed(42)
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 8, 9, 0, 2],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 0, 6, 5, 7, 6, 8, 7, 9, 8, 2, 0],
    ], dtype=torch.long)
    model = GCN(in_channels=16, hidden_channels=8, out_channels=3, num_layers=2)
    model.eval()
    return x, edge_index, model


def test_sparsity_calculation():
    """Verify sparsity values against known mask ratios."""
    mask = torch.tensor([0.9, 0.8, 0.1, 0.2])  # 2 edges kept (>=0.5), 2 removed
    sp = compute_sparsity(mask, threshold=0.5)
    assert abs(sp - 0.5) < 1e-5

    # All edges kept
    mask_all = torch.tensor([0.9, 0.9, 0.9])
    assert abs(compute_sparsity(mask_all, threshold=0.5) - 0.0) < 1e-5

    # All edges removed
    mask_none = torch.tensor([0.1, 0.2, 0.3])
    assert abs(compute_sparsity(mask_none, threshold=0.5) - 1.0) < 1e-5


def test_motif_roc_auc():
    """Verify ROC-AUC returns 1.0 for perfect ground truth alignment."""
    y_true = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    y_scores = torch.tensor([0.95, 0.85, 0.10, 0.05])
    auc = compute_motif_roc_auc(y_scores, y_true)
    assert abs(auc - 1.0) < 1e-4


def test_hallucination_rate():
    """Verify hallucination detector flags nonexistent node references."""
    claims = ["Node_10 is connected to Node_20", "Node_999 is highly influential"]
    valid_subgraph_nodes = [10, 20, 30]
    rate = compute_hallucination_rate(claims, valid_subgraph_nodes)
    # 1 claim out of 2 has hallucinated Node_999
    assert abs(rate - 0.5) < 1e-5


def test_fidelity_plus_and_minus(dummy_graph_and_model):
    """Verify Fidelity+ and Fidelity- calculation on model predictions."""
    x, edge_index, model = dummy_graph_and_model
    edge_mask = torch.rand(edge_index.size(1))
    target_node = 0
    target_class = 1

    fid_plus = compute_fidelity_plus(
        model=model,
        x=x,
        sub_edge_index=edge_index,
        target_node=target_node,
        target_class=target_class,
        edge_mask=edge_mask,
        threshold=0.5,
    )
    assert isinstance(fid_plus, float)
    assert -1.0 <= fid_plus <= 1.0

    fid_minus = compute_fidelity_minus(
        model=model,
        x=x,
        sub_edge_index=edge_index,
        target_node=target_node,
        target_class=target_class,
        edge_mask=edge_mask,
        threshold=0.5,
    )
    assert isinstance(fid_minus, float)
    assert -1.0 <= fid_minus <= 1.0


def test_fidelity_curve_and_budget(dummy_graph_and_model):
    """Verify Fidelity budget and curve computation."""
    x, edge_index, model = dummy_graph_and_model
    edge_mask = torch.rand(edge_index.size(1))
    target_node = 0
    target_class = 1

    fid_plus, fid_minus = compute_fidelity_at_budget(
        model=model,
        x=x,
        sub_edge_index=edge_index,
        target_node=target_node,
        target_class=target_class,
        edge_mask=edge_mask,
        k=5,
    )
    assert isinstance(fid_plus, float)
    assert isinstance(fid_minus, float)

    curve_dict = compute_fidelity_curve(
        model=model,
        x=x,
        sub_edge_index=edge_index,
        target_node=target_node,
        target_class=target_class,
        edge_mask=edge_mask,
        budget_k_list=[2, 5, 10],
    )
    assert "auc_fid_plus" in curve_dict
    assert "auc_fid_minus" in curve_dict
    assert "fid_plus_k2" in curve_dict
    assert "fid_minus_k2" in curve_dict


def test_agreement_and_k_star(dummy_graph_and_model):
    """Verify Chance-Corrected Kappa, Jaccard similarity, and minimal budget k*."""
    m1 = torch.tensor([1.0, 0.8, 0.1, 0.0])
    m2 = torch.tensor([0.9, 0.7, 0.2, 0.0])
    m3 = torch.tensor([0.0, 0.1, 0.9, 0.8])

    jacc = compute_jaccard_similarity(m1, m2, k=2)
    assert jacc == 1.0

    kappa = compute_chance_corrected_kappa(m1, m2, k=2, candidate_m=4)
    assert kappa > 0.5

    matrix = compute_pairwise_agreement_matrix(
        {"exp1": m1, "exp2": m2, "exp3": m3}, k=2, candidate_m=4
    )
    assert "exp1" in matrix and "exp2" in matrix
    assert matrix["exp1"]["exp1"] == 1.0

    x, edge_index, model = dummy_graph_and_model
    edge_mask = torch.rand(edge_index.size(1))
    k_star_res = compute_k_star(
        model=model,
        x=x,
        sub_edge_index=edge_index,
        target_node=0,
        edge_mask=edge_mask,
        max_k=10,
    )
    assert "k_star" in k_star_res
    assert "is_zero_k_star" in k_star_res
    assert k_star_res["k_star"] >= 0


def test_variance_and_statistical_tests():
    """Verify ANOVA variance decomposition and Holm/TOST tests."""
    data = [
        {"explainer": "GNNExplainer", "gnn_seed": 42, "exp_seed": 1, "fidelity": 0.30},
        {"explainer": "GNNExplainer", "gnn_seed": 123, "exp_seed": 1, "fidelity": 0.32},
        {"explainer": "Random", "gnn_seed": 42, "exp_seed": 1, "fidelity": 0.05},
        {"explainer": "Random", "gnn_seed": 123, "exp_seed": 1, "fidelity": 0.06},
        {"explainer": "GNNExplainer", "gnn_seed": 42, "exp_seed": 2, "fidelity": 0.28},
        {"explainer": "GNNExplainer", "gnn_seed": 123, "exp_seed": 2, "fidelity": 0.29},
        {"explainer": "Random", "gnn_seed": 42, "exp_seed": 2, "fidelity": 0.04},
        {"explainer": "Random", "gnn_seed": 123, "exp_seed": 2, "fidelity": 0.05},
    ]

    decomp = decompose_explanation_variance(
        data,
        metric_key="fidelity",
        explainer_key="explainer",
        gnn_seed_key="gnn_seed",
        exp_seed_key="exp_seed",
    )
    assert "pct_algorithm" in decomp
    assert "pct_gnn_seed" in decomp
    assert decomp["pct_algorithm"] > 0.5

    p_vals = [0.001, 0.02, 0.04, 0.20]
    corr = holm_bonferroni_correction(p_vals, alpha=0.05)
    assert len(corr) == 4
    # Check that the first element has adjusted p-value and boolean significance
    assert isinstance(corr[0], tuple)
    assert corr[0][1] is True  # 0.001 should be significant

    g1 = [0.25, 0.26, 0.25, 0.27]
    g2 = [0.25, 0.25, 0.26, 0.26]
    tost = tost_equivalence_test(g1, g2, bound_delta=0.05)
    assert "is_equivalent" in tost
