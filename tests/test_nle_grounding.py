"""Unit Tests for Natural Language Explanation (NLE) Grounding Modules.

Verifies GraphSerializer serialization format, NLEVerbalizer explanation generation,
and RejectionFilter topological hallucination detection.
"""

import pytest
import torch

from src.explainers.base_explainer import ExplanationResult
from src.nle_grounding.graph_serializer import GraphSerializer
from src.nle_grounding.rejection_filter import RejectionFilter
from src.nle_grounding.verbalizer import NLEVerbalizer


@pytest.fixture
def dummy_explanation():
    """Build a dummy ExplanationResult for testing NLE modules."""
    target_node = 0
    predicted_class = 1
    edge_mask = torch.tensor([0.9, 0.8, 0.2, 0.1])
    subgraph_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    subgraph_nodes = torch.tensor([0, 1, 2], dtype=torch.long)
    node_feat_mask = torch.tensor([0.5, 0.8, 0.1, 0.0])

    return ExplanationResult(
        target_node=target_node,
        predicted_class=predicted_class,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        subgraph_nodes=subgraph_nodes,
        subgraph_edge_index=subgraph_edge_index,
        execution_time=0.01,
    )


def test_graph_serializer(dummy_explanation):
    """Test serializing explanation subgraph into structured markdown prompt."""
    x = torch.randn(3, 4)
    serializer = GraphSerializer(feature_names=["f0", "f1", "f2", "f3"])
    prompt = serializer.serialize(
        dummy_explanation, x, class_names={1: "Academic Community"}
    )

    assert "TARGET NODE: Node_0" in prompt
    assert "Academic Community" in prompt
    assert "SUBGRAPH TOPOLOGY" in prompt


def test_nle_verbalizer(dummy_explanation):
    """Test generating grounded natural language explanation string."""
    x = torch.randn(3, 4)
    verbalizer = NLEVerbalizer(domain_context="social network")
    text = verbalizer.generate_explanation(
        dummy_explanation, x, class_names={1: "Influential Leader"}
    )

    assert "Node_0" in text
    assert "Influential Leader" in text
    assert len(text) > 20


def test_rejection_filter_clean_and_hallucination(dummy_explanation):
    """Test RejectionFilter accepting valid claims and rejecting hallucinations."""
    filter_mod = RejectionFilter(max_hallucination_rate=0.02)

    # Clean text referencing only nodes {0, 1, 2}
    clean_text = "Node 0 is closely connected with Node 1 and Node 2 in the cluster."
    accepted, rate, violations = filter_mod.audit_explanation(clean_text, dummy_explanation)
    assert accepted is True
    assert rate == 0.0
    assert len(violations) == 0

    # Hallucinated text referencing nonexistent Node 99
    hallucinated_text = "Node 0 connects to Node 1 and also relies on Node 99."
    accepted_h, rate_h, violations_h = filter_mod.audit_explanation(hallucinated_text, dummy_explanation)
    assert accepted_h is False
    assert rate_h > 0.0
    assert len(violations_h) > 0
    assert "Node_99" in violations_h[0]
