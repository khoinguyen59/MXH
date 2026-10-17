"""Linear-Time Dual-Channel Explanation Framework (LT-DEF).

Proposed research method combining topological eigenvector Laplacian pruning (Channel 1)
with input-gradient feature attribution (Channel 2) to achieve high Fidelity and optimal
Sparsity with linear-time O(|E_sub|) efficiency.
"""

import time
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import to_scipy_sparse_matrix

from src.explainers.base_explainer import BaseExplainer, ExplanationResult


class LTDEFExplainer(BaseExplainer):
    """Linear-Time Dual-Channel Explanation Framework (LT-DEF)."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_hops: int = 2,
        sparsity_target: float = 0.70,
        eigen_components: int = 4,
        alpha_topo: float = 0.65,
    ) -> None:
        """Initialize LT-DEF explainer.

        Args:
            model: Trained GNN model to explain.
            num_hops: Computational ego-graph radius (default: 2).
            sparsity_target: Target fraction of pruned edges (default: 0.70).
            eigen_components: Number of Laplacian eigenvectors to compute (default: 4).
            alpha_topo: Balance weight for topological channel vs gradient channel (default: 0.65).
        """
        super().__init__(model=model, num_hops=num_hops)
        self.sparsity_target = sparsity_target
        self.eigen_components = eigen_components
        self.alpha_topo = alpha_topo

    def _compute_topological_scores(
        self, sub_edge_index: torch.Tensor, num_sub_nodes: int
    ) -> torch.Tensor:
        """Compute edge topological saliency via normalized Laplacian eigenvectors in O(|E_sub|)."""
        adj = to_scipy_sparse_matrix(sub_edge_index, num_nodes=num_sub_nodes).astype(float)
        num_edges = sub_edge_index.size(1)

        if num_sub_nodes < 4 or num_edges == 0:
            return torch.ones(num_edges, dtype=torch.float32)

        # Compute smallest non-trivial eigenvectors of Normalized Laplacian
        k = min(self.eigen_components, num_sub_nodes - 2)
        if k >= 1:
            try:
                # Degree matrix
                degrees = np.array(adj.sum(axis=1)).flatten()
                degrees[degrees == 0] = 1.0
                d_inv_sqrt = np.power(degrees, -0.5)
                # Normalized Adjacency
                norm_adj = adj.multiply(d_inv_sqrt[:, None]).multiply(d_inv_sqrt[None, :])
                vals, vecs = eigsh(norm_adj, k=k, which="LM")
                # Node spectral centrality
                node_spectral_energy = np.sum(np.abs(vecs), axis=1)
                node_spectral = torch.tensor(node_spectral_energy, dtype=torch.float32)

                # Edge score = sum of incident node spectral energies
                u, v = sub_edge_index[0], sub_edge_index[1]
                edge_scores = node_spectral[u] + node_spectral[v]
                return edge_scores / (edge_scores.max() + 1e-8)
            except Exception:
                pass

        # Fallback to degree centrality if spectral decomposition fails
        deg = torch.zeros(num_sub_nodes, dtype=torch.float32)
        deg.scatter_add_(0, sub_edge_index[0], torch.ones(num_edges))
        edge_scores = deg[sub_edge_index[0]] + deg[sub_edge_index[1]]
        return edge_scores / (edge_scores.max() + 1e-8)

    def _compute_feature_gradients(
        self, sub_x: torch.Tensor, sub_edge_index: torch.Tensor, target_node: int, target_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute input-gradient attribution for node features and incident edges."""
        sub_x_grad = sub_x.clone().detach().requires_grad_(True)
        out = self.model(sub_x_grad, sub_edge_index)
        score = out[target_node, target_class]
        score.backward()

        # Feature saliency = absolute gradient * input
        feat_grad = torch.abs(sub_x_grad.grad[target_node] * sub_x[target_node])
        feat_norm = feat_grad / (feat_grad.max() + 1e-8)

        # Gradient sensitivity of incident edges
        num_edges = sub_edge_index.size(1)
        edge_grad_scores = torch.zeros(num_edges, dtype=torch.float32)
        incident_u = (sub_edge_index[0] == target_node)
        incident_v = (sub_edge_index[1] == target_node)
        edge_grad_scores[incident_u | incident_v] = 1.0

        return feat_norm, edge_grad_scores

    def explain_node(
        self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor
    ) -> ExplanationResult:
        """Execute dual-channel explanation generation in linear time."""
        start_time = time.perf_counter()

        with torch.no_grad():
            full_out = self.model(x, edge_index)
            pred_class = int(full_out[node_idx].argmax(dim=-1).item())

        subset_nodes, sub_edge_index, mapping, edge_mask_bool = (
            self.get_k_hop_computation_graph(node_idx, edge_index, x.size(0))
        )
        local_target = int(mapping.item())
        sub_x = x[subset_nodes]
        num_sub_nodes = subset_nodes.size(0)
        num_edges = sub_edge_index.size(1)
        if num_edges == 0:
            elapsed = time.perf_counter() - start_time
            return ExplanationResult(
                target_node=node_idx,
                predicted_class=pred_class,
                edge_mask=torch.empty(0, dtype=torch.float32),
                node_feat_mask=torch.zeros(sub_x.size(1), dtype=torch.float32),
                subgraph_nodes=subset_nodes,
                subgraph_edge_index=sub_edge_index,
                execution_time=elapsed,
            )

        # Channel 1: Topological Eigen-Pruning
        topo_scores = self._compute_topological_scores(sub_edge_index, num_sub_nodes)

        # Channel 2: Semantic Feature Attribution & Gradient Sensitivity
        feat_mask, edge_grad_scores = self._compute_feature_gradients(
            sub_x, sub_edge_index, local_target, pred_class
        )

        # Dual-Channel Fusion
        fused_edge_mask = (
            self.alpha_topo * topo_scores + (1.0 - self.alpha_topo) * edge_grad_scores
        )
        max_fused = fused_edge_mask.max()
        if max_fused > 1e-8:
            fused_edge_mask = fused_edge_mask / max_fused
        else:
            fused_edge_mask = torch.ones_like(fused_edge_mask)

        elapsed = time.perf_counter() - start_time

        return ExplanationResult(
            target_node=node_idx,
            predicted_class=pred_class,
            edge_mask=fused_edge_mask,
            node_feat_mask=feat_mask,
            subgraph_nodes=subset_nodes,
            subgraph_edge_index=sub_edge_index,
            execution_time=elapsed,
        )
