"""Fidelity Evaluation & Budget Curve Metrics.

Computes Fidelity+ (unfaithfulness drop) and Fidelity- (preservation necessity)
across discrete edge budget grids (top-k and percentage-based), and calculates
the Area Under the Budget Curve (AUC) in compliance with international XAI benchmarks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def compute_fidelity_at_budget(
    model: torch.nn.Module,
    x: torch.Tensor,
    sub_edge_index: torch.Tensor,
    target_node: int,
    target_class: int,
    edge_mask: torch.Tensor,
    k: int,
    symmetrize: bool = True,
) -> Tuple[float, float]:
    """Compute Fidelity+ and Fidelity- at a specific top-k edge budget.

    Args:
        model: Trained GNN model.
        x: Node feature matrix.
        sub_edge_index: Local computation graph edges.
        target_node: Local target node index.
        target_class: Target predicted class.
        edge_mask: Continuous importance weights of edges.
        k: Number of top edges to select.
        symmetrize: Whether to symmetrize top edge pruning on undirected graphs.

    Returns:
        Tuple of (fid_plus, fid_minus) at budget k.
    """
    model.eval()
    num_edges = sub_edge_index.size(1)
    if num_edges == 0 or edge_mask.numel() == 0:
        return 0.0, 0.0

    k = min(max(1, k), num_edges)

    with torch.no_grad():
        # Baseline probability on clean graph
        orig_out = model(x, sub_edge_index)
        orig_prob = F.softmax(orig_out[target_node], dim=-1)[target_class].item()

        # Defensively sanitize edge_mask from NaNs or Infs
        edge_mask_clean = torch.nan_to_num(edge_mask, nan=0.0, posinf=1.0, neginf=0.0)

        if symmetrize and num_edges > 0:
            # Map directed edges to undirected canonical pairs (BG05, BG06)
            u = sub_edge_index[0]
            v = sub_edge_index[1]
            max_node = int(max(u.max().item(), v.max().item())) + 1
            undirected_u = torch.minimum(u, v)
            undirected_v = torch.maximum(u, v)
            edge_hash = undirected_u.long() * (max_node + 1) + undirected_v.long()

            # Aggregate scores per undirected edge using max-aggregation (BG06)
            unique_hashes, inverse_indices = torch.unique(edge_hash, return_inverse=True)
            num_unique = unique_hashes.size(0)
            undirected_scores = torch.full(
                (num_unique,), -float("inf"), dtype=edge_mask_clean.dtype, device=edge_mask_clean.device
            )
            undirected_scores.scatter_reduce_(
                0, inverse_indices, edge_mask_clean, reduce="amax", include_self=False
            )

            # Select top-k unique undirected edges
            effective_k = min(max(1, k), num_unique)
            _, top_undirected_idx = torch.topk(undirected_scores, k=effective_k, largest=True)
            selected_hashes = unique_hashes[top_undirected_idx]
            effective_mask = torch.isin(edge_hash, selected_hashes)
        else:
            k_eff = min(max(1, k), num_edges)
            _, top_indices = torch.topk(edge_mask_clean, k=k_eff, largest=True)
            effective_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_mask.device)
            effective_mask[top_indices] = True

        # Fidelity+: Remove top-k edges (test sufficiency)
        rem_edges = sub_edge_index[:, ~effective_mask]
        if rem_edges.size(1) == 0:
            try:
                pert_out = model(x, rem_edges)
                pert_prob = F.softmax(pert_out[target_node], dim=-1)[target_class].item()
            except Exception:
                pert_prob = 1.0 / orig_out.size(-1)  # uniform fallback
        else:
            pert_out = model(x, rem_edges)
            pert_prob = F.softmax(pert_out[target_node], dim=-1)[target_class].item()
        fid_plus = orig_prob - pert_prob

        # Fidelity-: Keep ONLY top-k edges (test necessity)
        keep_edges = sub_edge_index[:, effective_mask]
        if keep_edges.size(1) == 0:
            try:
                sub_out = model(x, keep_edges)
                sub_prob = F.softmax(sub_out[target_node], dim=-1)[target_class].item()
            except Exception:
                sub_prob = 1.0 / orig_out.size(-1)  # uniform fallback
        else:
            sub_out = model(x, keep_edges)
            sub_prob = F.softmax(sub_out[target_node], dim=-1)[target_class].item()
        fid_minus = orig_prob - sub_prob

    return float(fid_plus), float(fid_minus)


def compute_fidelity_curve(
    model: torch.nn.Module,
    x: torch.Tensor,
    sub_edge_index: torch.Tensor,
    target_node: int,
    target_class: int,
    edge_mask: torch.Tensor,
    budget_k_list: List[int] = [1, 2, 3, 5, 10, 15, 20],
    symmetrize: bool = True,
) -> Dict[str, float]:
    """Compute Fidelity curve across multiple budgets and its Area Under Curve (AUC).

    Args:
        model: Trained GNN model.
        x: Node feature matrix.
        sub_edge_index: Local computation graph edges.
        target_node: Local target node index.
        target_class: Target predicted class.
        edge_mask: Continuous importance weights of edges.
        budget_k_list: List of top-k edge budgets.
        symmetrize: Whether to symmetrize top edge pruning on undirected graphs.

    Returns:
        Dictionary containing fid_plus and fid_minus curves, along with AUC values.
    """
    fid_plus_list = []
    fid_minus_list = []

    for k in budget_k_list:
        fp, fm = compute_fidelity_at_budget(
            model=model,
            x=x,
            sub_edge_index=sub_edge_index,
            target_node=target_node,
            target_class=target_class,
            edge_mask=edge_mask,
            k=k,
            symmetrize=symmetrize,
        )
        fid_plus_list.append(fp)
        fid_minus_list.append(fm)

    # Normalize AUC using trapezoidal rule over budget range
    k_arr = np.array(budget_k_list, dtype=float)
    k_range = k_arr[-1] - k_arr[0] if len(k_arr) > 1 and k_arr[-1] > k_arr[0] else 1.0

    auc_fid_plus = float(np.trapezoid(fid_plus_list, k_arr) / k_range)
    auc_fid_minus = float(np.trapezoid(fid_minus_list, k_arr) / k_range)

    result = {
        "auc_fid_plus": auc_fid_plus,
        "auc_fid_minus": auc_fid_minus,
    }
    for k, fp, fm in zip(budget_k_list, fid_plus_list, fid_minus_list):
        result[f"fid_plus_k{k}"] = fp
        result[f"fid_minus_k{k}"] = fm

    return result


def compute_feature_fidelity_at_budget(
    model: torch.nn.Module,
    x: torch.Tensor,
    sub_edge_index: torch.Tensor,
    target_node: int,
    target_class: int,
    node_feat_mask: Optional[torch.Tensor],
    top_k: int = 50,
) -> Tuple[float, float]:
    """Compute Node Feature Fidelity+ and Fidelity- by perturbing node attributes.

    Measures the classification confidence drop when removing or retaining only top features
    of the target node, enabling direct empirical comparison with topological edge fidelity (RQ2).

    Args:
        model: Trained GNN model.
        x: Subgraph node feature matrix [num_sub_nodes, feat_dim].
        sub_edge_index: Subgraph edge index.
        target_node: Local index of the target node within subgraph.
        target_class: Predicted class.
        node_feat_mask: Feature attribution scores [feat_dim] or [num_sub_nodes, feat_dim].
        top_k: Number of top features to perturb / retain.

    Returns:
        Tuple of (feat_fid_plus, feat_fid_minus).
    """
    if node_feat_mask is None or node_feat_mask.numel() == 0:
        return 0.0, 0.0

    model.eval()
    with torch.no_grad():
        # Baseline probability on clean graph
        orig_out = model(x, sub_edge_index)
        orig_prob = F.softmax(orig_out[target_node], dim=-1)[target_class].item()

        # Handle mask shape
        if node_feat_mask.dim() == 2:
            feat_scores = node_feat_mask[target_node]
        else:
            feat_scores = node_feat_mask

        feat_scores_clean = torch.nan_to_num(feat_scores, nan=0.0, posinf=1.0, neginf=0.0)
        num_feats = feat_scores_clean.size(0)
        effective_k = min(max(1, top_k), num_feats)

        _, top_feat_idx = torch.topk(feat_scores_clean, k=effective_k, largest=True)

        # Feature Fidelity+: Mask top features (set to 0.0)
        x_plus = x.clone()
        x_plus[target_node, top_feat_idx] = 0.0
        out_plus = model(x_plus, sub_edge_index)
        prob_plus = F.softmax(out_plus[target_node], dim=-1)[target_class].item()
        feat_fid_plus = orig_prob - prob_plus

        # Feature Fidelity-: Keep ONLY top features (zero out non-top features)
        x_minus = x.clone()
        all_mask = torch.ones(num_feats, dtype=torch.bool, device=x.device)
        all_mask[top_feat_idx] = False
        x_minus[target_node, all_mask] = 0.0
        out_minus = model(x_minus, sub_edge_index)
        prob_minus = F.softmax(out_minus[target_node], dim=-1)[target_class].item()
        feat_fid_minus = orig_prob - prob_minus

    return float(feat_fid_plus), float(feat_fid_minus)
