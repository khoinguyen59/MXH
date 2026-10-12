"""Random Baseline Explainer.

Generates uniform random edge attribution masks within the candidate L-hop subgraph.
Establishes the noise-floor anchor for all fidelity and agreement comparisons.
"""

import time
from typing import Optional
import torch
import torch.nn as nn
from src.explainers.base_explainer import BaseExplainer, ExplanationResult


class RandomExplainer(BaseExplainer):
    """Uniform random edge baseline explainer."""

    def __init__(
        self, model: nn.Module, num_hops: int = 2, random_seed: Optional[int] = None
    ) -> None:
        """Initialize Random baseline explainer.

        Args:
            model: Trained GNN model to explain.
            num_hops: Number of k-hop message-passing computational layers (default: 2).
            random_seed: Optional seed for deterministic reproducibility.
        """
        super().__init__(model=model, num_hops=num_hops)
        self.random_seed = random_seed

    def explain_node(
        self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor, **kwargs
    ) -> ExplanationResult:
        """Generate random edge attribution mask for the target node.

        Args:
            node_idx: Index of node to explain.
            x: Node feature matrix.
            edge_index: Graph connectivity tensor.

        Returns:
            ExplanationResult container with uniform random edge mask.
        """
        start_time = time.time()
        self.model.eval()

        # 1. Extract computation subgraph
        subset, sub_edge_index, mapping, _ = self.get_k_hop_computation_graph(
            node_idx=node_idx,
            edge_index=edge_index,
            num_nodes=x.size(0),
        )
        target_local_idx = mapping.item()
        sub_x = x[subset]

        # 2. Compute predicted class
        with torch.no_grad():
            out = self.model(sub_x, sub_edge_index)
            pred_class = int(out[target_local_idx].argmax(dim=-1).item())

        num_edges = sub_edge_index.size(1)
        if num_edges == 0:
            return ExplanationResult(
                target_node=node_idx,
                predicted_class=pred_class,
                edge_mask=torch.empty(0),
                node_feat_mask=None,
                subgraph_nodes=subset,
                subgraph_edge_index=sub_edge_index,
                execution_time=time.time() - start_time,
            )

        # 3. Generate uniform random mask
        generator = None
        if self.random_seed is not None:
            generator = torch.Generator(device=x.device).manual_seed(self.random_seed)

        edge_mask = torch.rand(num_edges, generator=generator, device=x.device)
        execution_time = time.time() - start_time

        return ExplanationResult(
            target_node=node_idx,
            predicted_class=pred_class,
            edge_mask=edge_mask,
            node_feat_mask=None,
            subgraph_nodes=subset,
            subgraph_edge_index=sub_edge_index,
            execution_time=execution_time,
        )
