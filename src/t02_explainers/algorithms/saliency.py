"""Saliency Explainer (First-Order Gradient Attribution).

Deterministic gradient-based edge explainer using PyTorch autograd or Captum.
Computes the gradient of the predicted class score with respect to edge mask weights.
"""

import copy
import time
import torch
import torch.nn as nn
from src.explainers.base_explainer import BaseExplainer, ExplanationResult


class SaliencyExplainer(BaseExplainer):
    """First-order gradient explainer for edge attribution on GNNs."""

    def __init__(self, model: nn.Module, num_hops: int = 2) -> None:
        """Initialize Saliency explainer.

        Args:
            model: Trained GNN model to explain.
            num_hops: Number of k-hop message-passing computational layers (default: 2).
        """
        # Deepcopy model to avoid Captum/Autograd state contamination across explainers
        cloned_model = copy.deepcopy(model)
        super().__init__(model=cloned_model, num_hops=num_hops)

    def explain_node(
        self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor, **kwargs
    ) -> ExplanationResult:
        """Generate gradient saliency edge explanation for the target node.

        Args:
            node_idx: Index of node to explain.
            x: Node feature matrix.
            edge_index: Graph connectivity tensor.

        Returns:
            ExplanationResult container with absolute gradient edge mask.
        """
        start_time = time.time()
        self.model.eval()

        # 1. Extract k-hop computation subgraph
        subset, sub_edge_index, mapping, _ = self.get_k_hop_computation_graph(
            node_idx=node_idx,
            edge_index=edge_index,
            num_nodes=x.size(0),
        )
        target_local_idx = mapping.item()
        sub_x = x[subset]

        # 2. Get baseline model prediction on clean subgraph
        with torch.no_grad():
            out = self.model(sub_x, sub_edge_index)
            pred_class = int(out[target_local_idx].argmax(dim=-1).item())

        num_edges = sub_edge_index.size(1)
        if num_edges == 0:
            sub_x_grad = sub_x.clone().detach().requires_grad_(True)
            empty_edge = torch.empty((2, 0), dtype=torch.long, device=x.device)
            node_feat_mask = None
            try:
                out_iso = self.model(sub_x_grad, empty_edge)
                target_logit = out_iso[target_local_idx, pred_class]
                target_logit.backward()
                if sub_x_grad.grad is not None:
                    feat_grad = sub_x_grad.grad[target_local_idx].abs().detach()
                    max_feat = feat_grad.max()
                    if max_feat > 1e-8:
                        feat_grad = feat_grad / max_feat
                    node_feat_mask = feat_grad
            except Exception:
                node_feat_mask = None

            return ExplanationResult(
                target_node=node_idx,
                predicted_class=pred_class,
                edge_mask=torch.empty(0),
                node_feat_mask=node_feat_mask,
                subgraph_nodes=subset,
                subgraph_edge_index=sub_edge_index,
                execution_time=time.time() - start_time,
            )

        # 3. Create continuous edge mask and enable gradients on node features
        edge_mask = torch.ones(num_edges, requires_grad=True, device=x.device)
        sub_x_grad = sub_x.clone().detach().requires_grad_(True)

        # 4. Forward pass weighting adjacency message passing
        out_masked = self.model(sub_x_grad, sub_edge_index, edge_weight=edge_mask)
        target_logit = out_masked[target_local_idx, pred_class]

        # 5. Backward pass to extract gradient saliency
        self.model.zero_grad()
        if edge_mask.grad is not None:
            edge_mask.grad.zero_()
        if sub_x_grad.grad is not None:
            sub_x_grad.grad.zero_()

        target_logit.backward()

        grad = edge_mask.grad
        if grad is None:
            grad = torch.zeros(num_edges, device=x.device)

        # Saliency magnitude: absolute value of gradient
        saliency_scores = grad.abs().detach()

        # Min-max normalize if non-zero
        max_val = saliency_scores.max()
        if max_val > 1e-8:
            saliency_scores = saliency_scores / max_val

        # Node Feature Saliency magnitude (First-order gradient w.r.t target node features)
        node_feat_mask = None
        if sub_x_grad.grad is not None:
            feat_grad = sub_x_grad.grad[target_local_idx].abs().detach()
            max_feat = feat_grad.max()
            if max_feat > 1e-8:
                feat_grad = feat_grad / max_feat
            node_feat_mask = feat_grad

        execution_time = time.time() - start_time

        return ExplanationResult(
            target_node=node_idx,
            predicted_class=pred_class,
            edge_mask=saliency_scores,
            node_feat_mask=node_feat_mask,
            subgraph_nodes=subset,
            subgraph_edge_index=sub_edge_index,
            execution_time=execution_time,
        )
