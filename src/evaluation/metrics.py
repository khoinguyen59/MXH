"""Quantitative Evaluation Metrics for Graph Explainability.

Implements standard XAI-GNN metrics: Fidelity+, Fidelity-, Sparsity, Motif-AUC,
and Hallucination Rate in compliance with evaluation governance specifications.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


def compute_fidelity_plus(
    model: torch.nn.Module,
    x: torch.Tensor,
    sub_edge_index: torch.Tensor,
    target_node: int,
    target_class: int,
    edge_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute Fidelity+ (Probability drop when removing explanation edges).

    A higher Fidelity+ indicates that the pruned edges contained essential explanatory evidence.

    Args:
        model: Trained GNN model.
        x: Node feature matrix.
        sub_edge_index: Local computation graph edges.
        target_node: Target node index.
        target_class: Predicted class label.
        edge_mask: Importance weights of edges in sub_edge_index.
        threshold: Binarization threshold for identifying important edges.

    Returns:
        Fidelity+ score in [-1.0, 1.0].
    """
    model.eval()
    with torch.no_grad():
        # 1. Original probability
        orig_out = model(x, sub_edge_index)
        orig_prob = F.softmax(orig_out[target_node], dim=-1)[target_class].item()

        # 2. Probability after removing top edges (complement graph)
        keep_mask = edge_mask < threshold
        if not keep_mask.any():
            pert_edge_index = torch.empty((2, 0), dtype=torch.long, device=sub_edge_index.device)
        else:
            pert_edge_index = sub_edge_index[:, keep_mask]

        pert_out = model(x, pert_edge_index)
        pert_prob = F.softmax(pert_out[target_node], dim=-1)[target_class].item()

    return float(orig_prob - pert_prob)


def compute_fidelity_minus(
    model: torch.nn.Module,
    x: torch.Tensor,
    sub_edge_index: torch.Tensor,
    target_node: int,
    target_class: int,
    edge_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute Fidelity- (Probability drop when retaining ONLY explanation edges).

    A lower Fidelity- indicates that the explanation subgraph alone is sufficient to preserve prediction.

    Args:
        model: Trained GNN model.
        x: Node feature matrix.
        sub_edge_index: Local computation graph edges.
        target_node: Target node index.
        target_class: Predicted class label.
        edge_mask: Importance weights of edges in sub_edge_index.
        threshold: Binarization threshold for identifying important edges.

    Returns:
        Fidelity- score in [-1.0, 1.0].
    """
    model.eval()
    with torch.no_grad():
        # 1. Original probability
        orig_out = model(x, sub_edge_index)
        orig_prob = F.softmax(orig_out[target_node], dim=-1)[target_class].item()

        # 2. Probability keeping ONLY top edges
        keep_mask = edge_mask >= threshold
        if not keep_mask.any():
            pert_edge_index = torch.empty((2, 0), dtype=torch.long, device=sub_edge_index.device)
        else:
            pert_edge_index = sub_edge_index[:, keep_mask]

        pert_out = model(x, pert_edge_index)
        pert_prob = F.softmax(pert_out[target_node], dim=-1)[target_class].item()

    return float(orig_prob - pert_prob)


def compute_sparsity(edge_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute edge sparsity score (fraction of edges removed from computation graph).

    Args:
        edge_mask: Edge importance scores.
        threshold: Threshold above which an edge is considered part of the explanation.

    Returns:
        Sparsity value in [0.0, 1.0].
    """
    if edge_mask.numel() == 0:
        return 1.0
    num_selected = (edge_mask >= threshold).sum().item()
    total_edges = edge_mask.numel()
    return float(1.0 - (num_selected / total_edges))


def compute_motif_roc_auc(
    edge_mask: torch.Tensor, ground_truth_edges: torch.Tensor
) -> Optional[float]:
    """Compute ROC-AUC between continuous edge importance scores and ground-truth motif edges.

    Args:
        edge_mask: Continuous importance scores for edges in computation graph.
        ground_truth_edges: Binary ground-truth indicator (1 if motif edge, 0 otherwise).

    Returns:
        ROC-AUC score, or None if only one class is present.
    """
    y_true = ground_truth_edges.cpu().numpy()
    y_scores = edge_mask.detach().cpu().numpy()

    if len(np.unique(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, y_scores))


def compute_hallucination_rate(
    factual_claims: List[str], ground_truth_subgraph_nodes: List[int]
) -> float:
    """Compute NLE hallucination rate by verifying node/motif references in generated text.

    Args:
        factual_claims: List of factual claims extracted from natural language explanation.
        ground_truth_subgraph_nodes: Verifiable set of nodes in the explanation subgraph.

    Returns:
        Hallucination rate in [0.0, 1.0].
    """
    if not factual_claims:
        return 0.0

    valid_nodes_set = set(ground_truth_subgraph_nodes)
    hallucinated_count = 0

    for claim in factual_claims:
        # Check if claim references ungrounded node IDs or nonexistent topologies
        words = claim.split()
        for w in words:
            if w.startswith("Node_") or w.startswith("node_"):
                try:
                    node_id = int(w.split("_")[1])
                    if node_id not in valid_nodes_set:
                        hallucinated_count += 1
                        break
                except (ValueError, IndexError):
                    pass

    return float(hallucinated_count / len(factual_claims))
