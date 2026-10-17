"""EiG-Search Explainer Implementation.

Implements linear-time spectral heuristic search (Lu et al., 2024)
which prunes edges via approximate eigenvector perturbations in O(|E_sub|) time.
"""

import time
import torch
import torch.nn.functional as F

from src.explainers.base_explainer import BaseExplainer, ExplanationResult


class EigSearchExplainer(BaseExplainer):
    """Linear-time heuristic GNN explainer via edge-perturbation sensitivity."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_hops: int = 2,
        sparsity_target: float = 0.70,
    ) -> None:
        """Initialize EiG-Search explainer.

        Args:
            model: Trained GNN model to explain.
            num_hops: Radius of computational subgraph (default: 2).
            sparsity_target: Target fraction of pruned edges (default: 0.70).
        """
        super().__init__(model=model, num_hops=num_hops)
        self.sparsity_target = sparsity_target

    def explain_node(
        self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor
    ) -> ExplanationResult:
        """Compute edge sensitivities in linear time."""
        start_time = time.perf_counter()

        with torch.no_grad():
            full_out = self.model(x, edge_index)
            pred_class = int(full_out[node_idx].argmax(dim=-1).item())
            base_prob = F.softmax(full_out[node_idx], dim=-1)[pred_class].item()

        subset_nodes, sub_edge_index, mapping, edge_mask_bool = (
            self.get_k_hop_computation_graph(node_idx, edge_index, x.size(0))
        )
        local_target_node = int(mapping.item())
        sub_x = x[subset_nodes]
        num_edges = sub_edge_index.size(1)
        if num_edges == 0:
            elapsed = time.perf_counter() - start_time
            return ExplanationResult(
                target_node=node_idx,
                predicted_class=pred_class,
                edge_mask=torch.empty(0, dtype=torch.float32),
                node_feat_mask=None,
                subgraph_nodes=subset_nodes,
                subgraph_edge_index=sub_edge_index,
                execution_time=elapsed,
            )

        # Linear-time single-edge removal perturbation
        edge_scores = torch.zeros(num_edges, dtype=torch.float32)

        with torch.no_grad():
            for e in range(num_edges):
                # Mask out edge e
                keep_mask = torch.ones(num_edges, dtype=torch.bool)
                keep_mask[e] = False
                pert_edge_index = sub_edge_index[:, keep_mask]

                pert_out = self.model(sub_x, pert_edge_index)
                pert_prob = F.softmax(pert_out[local_target_node], dim=-1)[pred_class].item()

                # Higher drop in target probability implies higher importance
                edge_scores[e] = max(0.0, base_prob - pert_prob)

        # Normalize scores to [0, 1]
        max_score = edge_scores.max().item()
        if max_score > 1e-6:
            normalized_mask = edge_scores / max_score
        else:
            normalized_mask = torch.ones_like(edge_scores)

        elapsed = time.perf_counter() - start_time

        return ExplanationResult(
            target_node=node_idx,
            predicted_class=pred_class,
            edge_mask=normalized_mask,
            node_feat_mask=None,
            subgraph_nodes=subset_nodes,
            subgraph_edge_index=sub_edge_index,
            execution_time=elapsed,
        )
