"""Fidelity Evaluation & Budget Curve Metrics.

Computes Fidelity+ (unfaithfulness drop) and Fidelity- (preservation necessity)
across discrete edge budget grids (top-k and percentage-based), and calculates
the Area Under the Budget Curve (AUC) in compliance with international XAI benchmarks.
"""

from typing import Dict, List, Tuple
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

    Returns:
        Tuple of (fid_plus, fid_minus) at budget k.
    """
    model.eval()
    num_edges = sub_edge_index.size(1)
    if num_edges == 0:
        return 0.0, 0.0

    k = min(max(1, k), num_edges)

    with torch.no_grad():
        # Baseline probability on clean graph
        orig_out = model(x, sub_edge_index)
        orig_prob = F.softmax(orig_out[target_node], dim=-1)[target_class].item()

        # Identify top-k edges
        _, top_indices = torch.topk(edge_mask, k=k, largest=True)
        top_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_mask.device)
        top_mask[top_indices] = True

        # Fidelity+: Remove top-k edges (test sufficiency)
        rem_edges = sub_edge_index[:, ~top_mask]
        if rem_edges.size(1) == 0:
            pert_prob = 1.0 / orig_out.size(-1)  # uniform fallback
        else:
            pert_out = model(x, rem_edges)
            pert_prob = F.softmax(pert_out[target_node], dim=-1)[target_class].item()
        fid_plus = orig_prob - pert_prob

        # Fidelity-: Keep ONLY top-k edges (test necessity)
        keep_edges = sub_edge_index[:, top_mask]
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
