"""Fidelity Evaluation & Budget Curve Metrics.

Computes Fidelity+ (unfaithfulness drop) and Fidelity- (preservation necessity)
across discrete edge budget grids (top-k and percentage-based), and calculates
the Area Under the Budget Curve (AUC) in compliance with international XAI benchmarks.
"""

from typing import Any, Dict, List, Optional, Tuple
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


def compute_fidelity_bg08(
    model: torch.nn.Module,
    x: torch.Tensor,
    sub_edge_index: torch.Tensor,
    target_node: int,
    target_class: int,
    candidate_edges: List[Dict[str, int]],
    scores: List[float],
    budgets_k: List[int] = [1, 3, 5, 10],
    proportions_p: List[float] = [0.1, 0.2, 0.5],
    num_tie_runs: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Fidelity+ and Fidelity- across the full BG08 budget grid with smart tie-breaking.

    Implements Rule BG08:
    - Budgets: top-k in {1, 3, 5, 10} capped at m; proportion p in {10%, 20%, 50%} with ceil(p*m), min 1.
    - Tie-breaking: If scores contain duplicate values, ties are broken randomly with seed across
      num_tie_runs (default 20) and averaged. If all scores are distinct, only 1 run is executed (saving 95% CPU).
    - OOD Artifacts: Tracks whether target_node is isolated after pruning to document graph disconnection.

    Args:
        model: Trained GNN model in eval() mode.
        x: Subgraph node feature matrix.
        sub_edge_index: PyG subgraph edge index.
        target_node: Local target node index.
        target_class: Target class to explain.
        candidate_edges: List of canonical candidate edges with 'edge_id', 'src', 'dst'.
        scores: Float importance scores per candidate edge (matching candidate_edges).
        budgets_k: Top-k budget list (default: [1, 3, 5, 10]).
        proportions_p: Proportional budget list (default: [0.1, 0.2, 0.5]).
        num_tie_runs: Number of random tie-breaking iterations (default: 20).
        seed: Random seed for tie-breaking reproducibility.

    Returns:
        Dict with metrics for each budget type ('k_1', 'k_3', ..., 'p_10pct', ...),
        along with 'has_ties', 'num_candidates', and 'isolated_at_budget'.
    """
    m = len(candidate_edges)
    if m == 0 or sub_edge_index.size(1) == 0:
        return {
            "status": "empty_graph",
            "num_candidates": 0,
            "has_ties": False,
        }

    scores_arr = np.array(scores, dtype=np.float64)
    # Check for ties
    has_ties = len(np.unique(scores_arr)) < len(scores_arr)
    actual_tie_runs = num_tie_runs if has_ties else 1

    # Map candidate edges to directed columns in sub_edge_index
    sub_u = sub_edge_index[0].cpu().numpy()
    sub_v = sub_edge_index[1].cpu().numpy()
    num_sub_edges = sub_edge_index.size(1)

    cand_to_col_indices: Dict[int, List[int]] = {cand["edge_id"]: [] for cand in candidate_edges}
    canon_to_id = {(cand["src"], cand["dst"]): cand["edge_id"] for cand in candidate_edges}

    for col_idx in range(num_sub_edges):
        u_val = int(sub_u[col_idx])
        v_val = int(sub_v[col_idx])
        canon = (min(u_val, v_val), max(u_val, v_val))
        if canon in canon_to_id:
            c_id = canon_to_id[canon]
            cand_to_col_indices[c_id].append(col_idx)

    # Compute baseline prediction
    model.eval()
    with torch.no_grad():
        orig_out = model(x, sub_edge_index)
        orig_prob = F.softmax(orig_out[target_node], dim=-1)[target_class].item()

    # Build list of (budget_name, budget_size)
    budget_specs: List[Tuple[str, int]] = []
    for k in budgets_k:
        budget_specs.append((f"top_k_{k}", min(k, m)))
    for p in proportions_p:
        pct_label = f"prop_{int(round(p * 100))}pct"
        budget_size = max(1, int(np.ceil(p * m)))
        budget_specs.append((pct_label, min(budget_size, m)))

    results: Dict[str, Any] = {
        "num_candidates": m,
        "has_ties": has_ties,
        "actual_runs_evaluated": actual_tie_runs,
        "isolated_at_budget": {},
    }

    rng = np.random.RandomState(seed)

    for b_name, b_size in budget_specs:
        fp_runs = []
        fm_runs = []
        isolated_flags = []

        for run_idx in range(actual_tie_runs):
            if has_ties:
                # Add micro-noise for randomized tie-breaking with seed
                noise = rng.uniform(0, 1e-9, size=m)
                effective_scores = scores_arr + noise
            else:
                effective_scores = scores_arr

            # Select top b_size candidate edge IDs
            top_cand_indices = np.argsort(-effective_scores)[:b_size]
            selected_cand_ids = set([candidate_edges[idx]["edge_id"] for idx in top_cand_indices])

            # Build column mask for sub_edge_index
            col_mask = np.zeros(num_sub_edges, dtype=bool)
            for c_id in selected_cand_ids:
                for col_idx in cand_to_col_indices.get(c_id, []):
                    col_mask[col_idx] = True

            # Check if target_node is isolated when keeping ONLY top-k edges
            keep_edges_np = sub_edge_index[:, col_mask].cpu().numpy()
            target_deg = np.sum((keep_edges_np[0] == target_node) | (keep_edges_np[1] == target_node))
            isolated_flags.append(bool(target_deg == 0))

            # Fidelity+: remove top edges
            rem_edges = sub_edge_index[:, ~col_mask]
            with torch.no_grad():
                if rem_edges.size(1) == 0:
                    prob_plus = 1.0 / orig_out.size(-1)
                else:
                    out_plus = model(x, rem_edges)
                    prob_plus = F.softmax(out_plus[target_node], dim=-1)[target_class].item()
                fp = orig_prob - prob_plus

            # Fidelity-: keep ONLY top edges
            keep_edges = sub_edge_index[:, col_mask]
            with torch.no_grad():
                if keep_edges.size(1) == 0:
                    prob_minus = 1.0 / orig_out.size(-1)
                else:
                    out_minus = model(x, keep_edges)
                    prob_minus = F.softmax(out_minus[target_node], dim=-1)[target_class].item()
                fm = orig_prob - prob_minus

            fp_runs.append(fp)
            fm_runs.append(fm)

        results[f"{b_name}_fid_plus"] = float(np.mean(fp_runs))
        results[f"{b_name}_fid_minus"] = float(np.mean(fm_runs))
        results[f"{b_name}_fid_plus_std"] = float(np.std(fp_runs))
        results[f"{b_name}_fid_minus_std"] = float(np.std(fm_runs))
        results["isolated_at_budget"][b_name] = any(isolated_flags)

    return results

