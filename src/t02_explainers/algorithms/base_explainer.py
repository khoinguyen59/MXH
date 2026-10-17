"""Base Abstract Class for Graph Neural Network Explainers.

Defines the standardized interface, output dataclass, and subgraph extraction
utilities across all XAI methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch
from torch_geometric.utils import k_hop_subgraph


@dataclass
class ExplanationResult:
    """Standardized output container for GNN explanation results."""

    target_node: int
    predicted_class: int
    edge_mask: torch.Tensor
    node_feat_mask: Optional[torch.Tensor]
    subgraph_nodes: torch.Tensor
    subgraph_edge_index: torch.Tensor
    execution_time: float


class BaseExplainer(ABC):
    """Abstract base class for all GNN explanation algorithms."""

    def __init__(self, model: torch.nn.Module, num_hops: int = 2) -> None:
        """Initialize the base explainer.

        Args:
            model: Trained GNN model to explain.
            num_hops: Number of k-hop message-passing computational layers (default: 2).
        """
        self.model = model
        self.num_hops = num_hops
        self.model.eval()

    def get_k_hop_computation_graph(
        self, node_idx: int, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple:
        """Extract the computation subgraph centered around the target node.

        Args:
            node_idx: Index of target node.
            edge_index: Graph connectivity tensor.
            num_nodes: Total number of nodes in graph.

        Returns:
            Tuple of (subset_nodes, sub_edge_index, mapping, edge_mask).
        """
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=self.num_hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        return subset, sub_edge_index, mapping, edge_mask

    @abstractmethod
    def explain_node(
        self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor
    ) -> ExplanationResult:
        """Generate explanation (edge and feature masks) for a target node prediction.

        Args:
            node_idx: Index of target node to explain.
            x: Node feature matrix.
            edge_index: Graph edge index tensor.

        Returns:
            ExplanationResult containing edge importance, feature importance, and runtime.
        """
        pass
