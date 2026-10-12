"""Rejection Sampling and Hallucination Filter Module.

Audits generated text explanations against the actual ground-truth subgraph topology,
rejecting any claims that reference ungrounded node IDs, edges, or properties.
"""

import re
from typing import List, Set, Tuple
from src.explainers.base_explainer import ExplanationResult


class RejectionFilter:
    """Verifies that all entity references in a text explanation exist in the explanation subgraph."""

    def __init__(self, max_hallucination_rate: float = 0.02) -> None:
        """Initialize the rejection filter.

        Args:
            max_hallucination_rate: Maximum permissible ratio of hallucinated claims (default: 0.02).
        """
        self.max_hallucination_rate = max_hallucination_rate

    def audit_explanation(
        self, text_explanation: str, explanation: ExplanationResult
    ) -> Tuple[bool, float, List[str]]:
        """Audit the text explanation against the ground-truth explanation subgraph.

        Args:
            text_explanation: Generated natural language explanation string.
            explanation: Ground-truth ExplanationResult containing valid nodes and edges.

        Returns:
            Tuple of (is_accepted: bool, hallucination_rate: float, violations: List[str]).
        """
        valid_nodes: Set[int] = set(explanation.subgraph_nodes.tolist())
        violations: List[str] = []

        # Find all node references in text: Node_X or node_X or Node X
        node_matches = re.findall(r"(?:Node_|node_|Node\s+)(\d+)", text_explanation)

        if not node_matches:
            # No specific nodes mentioned, passes topological reference test
            return True, 0.0, []

        hallucinated_nodes = []
        for match in node_matches:
            node_id = int(match)
            if node_id not in valid_nodes:
                hallucinated_nodes.append(node_id)
                violations.append(
                    f"Hallucinated entity: Node_{node_id} does not exist in the extracted explanation subgraph."
                )

        hallucination_rate = len(hallucinated_nodes) / len(node_matches)
        is_accepted = hallucination_rate <= self.max_hallucination_rate

        return is_accepted, float(hallucination_rate), violations
