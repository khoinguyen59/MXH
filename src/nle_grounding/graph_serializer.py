"""Graph-to-Text Serialization Module.

Converts extracted explanation subgraphs, salient topology motifs, and top-ranking
node features into structured, hallucination-resistant textual representations.
"""

from typing import Dict, List, Optional
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

from src.explainers.base_explainer import ExplanationResult


class GraphSerializer:
    """Serializes explanation subgraphs into structured semantic prompts."""

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        """Initialize the serializer.

        Args:
            feature_names: Optional human-readable names for feature vector dimensions.
        """
        self.feature_names = feature_names

    def serialize(
        self,
        explanation: ExplanationResult,
        x: torch.Tensor,
        class_names: Optional[Dict[int, str]] = None,
    ) -> str:
        """Serialize an explanation result into a structured text context.

        Args:
            explanation: ExplanationResult object.
            x: Full graph node feature matrix.
            class_names: Optional mapping from class index to human label.

        Returns:
            Structured Markdown string describing the target node and explanation evidence.
        """
        target_node = explanation.target_node
        pred_class = explanation.predicted_class
        class_label = (
            class_names[pred_class]
            if class_names and pred_class in class_names
            else f"Class_{pred_class}"
        )

        sub_edges = explanation.subgraph_edge_index
        sub_nodes = explanation.subgraph_nodes.tolist()
        num_sub_nodes = len(sub_nodes)
        num_sub_edges = sub_edges.size(1) if sub_edges.numel() > 0 else 0

        # Construct NetworkX graph to detect motifs
        g = nx.Graph()
        g.add_nodes_from(sub_nodes)
        if num_sub_edges > 0:
            for i in range(num_sub_edges):
                u = int(sub_edges[0, i].item())
                v = int(sub_edges[1, i].item())
                g.add_edge(u, v)

        # Topological summary
        density = nx.density(g) if num_sub_nodes > 1 else 0.0
        degree_target = g.degree(target_node) if target_node in g else 0

        # Feature attribution summary
        top_features_str = "None"
        if explanation.node_feat_mask is not None and explanation.node_feat_mask.numel() > 0:
            top_k = min(3, explanation.node_feat_mask.numel())
            top_feat_indices = torch.topk(explanation.node_feat_mask, top_k).indices.tolist()
            feat_items = []
            for idx in top_feat_indices:
                name = (
                    self.feature_names[idx]
                    if self.feature_names and idx < len(self.feature_names)
                    else f"Feature_{idx}"
                )
                val = float(x[target_node, idx].item())
                weight = float(explanation.node_feat_mask[idx].item())
                feat_items.append(f"{name} (val: {val:.2f}, salience: {weight:.2f})")
            top_features_str = ", ".join(feat_items)

        # Connected neighbor list
        neighbors = list(g.neighbors(target_node)) if target_node in g else []
        neighbor_str = (
            ", ".join([f"Node_{n}" for n in neighbors[:5]])
            if neighbors
            else "isolated in pruned subgraph"
        )

        serialized_text = (
            f"TARGET NODE: Node_{target_node}\n"
            f"PREDICTED CLASS: {class_label}\n"
            f"SUBGRAPH TOPOLOGY: {num_sub_nodes} nodes, {num_sub_edges} edges, density: {density:.3f}\n"
            f"LOCAL CONNECTIVITY: Target node connected to [{neighbor_str}] with degree {degree_target}\n"
            f"SALIENT ATTRIBUTES: [{top_features_str}]\n"
            f"EXPLANATION RUNTIME: {explanation.execution_time:.4f}s\n"
        )

        return serialized_text
