"""GNNExplainer Implementation Wrapper.

Wraps PyTorch Geometric's native GNNExplainer algorithm (Ying et al., NeurIPS 2019)
which optimizes soft edge and node feature masks via mutual information maximization.
"""

import time
import torch
from torch_geometric.explain import Explainer, GNNExplainer as PyGGNNExplainer

from src.explainers.base_explainer import BaseExplainer, ExplanationResult


class GNNExplainerWrapper(BaseExplainer):
    """GNNExplainer baseline wrapper conforming to BaseExplainer interface."""

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        num_hops: int = 2,
    ) -> None:
        """Initialize GNNExplainer.

        Args:
            model: Trained GNN model to explain.
            epochs: Optimization iterations per explanation (default: 100).
            lr: Learning rate for mask optimization (default: 0.01).
            num_hops: Computational k-hop radius (default: 2).
        """
        super().__init__(model=model, num_hops=num_hops)
        self.epochs = epochs
        self.lr = lr
        self.explainer = Explainer(
            model=model,
            algorithm=PyGGNNExplainer(epochs=epochs, lr=lr),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="log_probs",
            ),
        )

    def explain_node(
        self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor
    ) -> ExplanationResult:
        """Generate GNNExplainer attribution masks for target node.

        Args:
            node_idx: Target node index.
            x: Node feature matrix.
            edge_index: Graph connectivity tensor.

        Returns:
            Standardized ExplanationResult with continuous edge and node feature masks.
        """
        start_time = time.perf_counter()

        with torch.no_grad():
            logits = self.model(x, edge_index)
            pred_class = int(logits[node_idx].argmax(dim=-1).item())

        subset_nodes, sub_edge_index, mapping, edge_mask_bool = (
            self.get_k_hop_computation_graph(node_idx, edge_index, x.size(0))
        )

        if sub_edge_index.size(1) == 0:
            elapsed = time.perf_counter() - start_time
            return ExplanationResult(
                target_node=node_idx,
                predicted_class=pred_class,
                edge_mask=torch.empty(0, dtype=torch.float32, device=x.device),
                node_feat_mask=None,
                subgraph_nodes=subset_nodes,
                subgraph_edge_index=sub_edge_index,
                execution_time=elapsed,
            )

        # Generate explanation via PyG Explainer
        explanation = self.explainer(
            x=x,
            edge_index=edge_index,
            index=node_idx,
        )

        elapsed = time.perf_counter() - start_time

        # Extract edge mask corresponding to k-hop computation subgraph
        edge_mask = explanation.edge_mask[edge_mask_bool]
        node_feat_mask = explanation.node_mask[node_idx] if explanation.node_mask is not None else None

        return ExplanationResult(
            target_node=node_idx,
            predicted_class=pred_class,
            edge_mask=edge_mask,
            node_feat_mask=node_feat_mask,
            subgraph_nodes=subset_nodes,
            subgraph_edge_index=sub_edge_index,
            execution_time=elapsed,
        )
