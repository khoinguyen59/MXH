"""PGExplainer Implementation Wrapper.

Wraps PyTorch Geometric's Parameterized Explainer (Luo et al., NeurIPS 2020)
which trains an inductive neural network to predict edge inclusion probabilities.
"""

import time
import torch
from torch_geometric.explain import Explainer, PGExplainer as PyGPGExplainer

from src.explainers.base_explainer import BaseExplainer, ExplanationResult


class PGExplainerWrapper(BaseExplainer):
    """PGExplainer baseline wrapper conforming to BaseExplainer interface."""

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 30,
        lr: float = 0.005,
        num_hops: int = 2,
    ) -> None:
        """Initialize PGExplainer.

        Args:
            model: Trained GNN model to explain.
            epochs: Training epochs for parameterized mask network (default: 30).
            lr: Learning rate (default: 0.005).
            num_hops: Computational k-hop radius (default: 2).
        """
        super().__init__(model=model, num_hops=num_hops)
        self.epochs = epochs
        self.lr = lr
        self.is_trained = False
        self.pg_algorithm = PyGPGExplainer(epochs=epochs, lr=lr)
        self.explainer = Explainer(
            model=model,
            algorithm=self.pg_algorithm,
            explanation_type="model",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="log_probs",
            ),
        )

    def train_explainer(
        self, x: torch.Tensor, edge_index: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        """Train the parameterized neural network over a set of training node indices.

        Args:
            x: Node features.
            edge_index: Graph connectivity.
            target_indices: Node indices used to train the parameterized explainer.
        """
        for epoch in range(self.epochs):
            for idx in target_indices:
                _ = self.explainer(x=x, edge_index=edge_index, target=None, index=int(idx))
        self.is_trained = True

    def explain_node(
        self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor
    ) -> ExplanationResult:
        """Generate PGExplainer edge importance mask for target node."""
        start_time = time.perf_counter()

        with torch.no_grad():
            logits = self.model(x, edge_index)
            pred_class = int(logits[node_idx].argmax(dim=-1).item())

        explanation = self.explainer(
            x=x,
            edge_index=edge_index,
            index=node_idx,
        )

        elapsed = time.perf_counter() - start_time

        subset_nodes, sub_edge_index, mapping, edge_mask_bool = (
            self.get_k_hop_computation_graph(node_idx, edge_index, x.size(0))
        )

        edge_mask = explanation.edge_mask[edge_mask_bool]

        return ExplanationResult(
            target_node=node_idx,
            predicted_class=pred_class,
            edge_mask=edge_mask,
            node_feat_mask=None,
            subgraph_nodes=subset_nodes,
            subgraph_edge_index=sub_edge_index,
            execution_time=elapsed,
        )
