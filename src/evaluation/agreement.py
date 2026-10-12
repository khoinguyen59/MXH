"""Inter-Explainer Agreement & Disagreement Quantification.

Implements the Disagreement Problem metric framework (Krishna et al., 2022) for Graph ML:
Chance-corrected Kappa overlap (kappa@k), Jaccard similarity, rank correlation (tau / rho),
and pairwise agreement matrices contrasted against theoretical reliability ceilings.
"""

from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats
import torch


def compute_top_k_indices(edge_mask: torch.Tensor, k: int) -> set:
    """Identify top-k edge indices, handling ties deterministically."""
    num_edges = edge_mask.numel()
    k = min(max(1, k), num_edges)
    _, top_idx = torch.topk(edge_mask, k=k, largest=True)
    return set(top_idx.cpu().tolist())


def compute_chance_corrected_kappa(
    mask_a: torch.Tensor, mask_b: torch.Tensor, k: int, candidate_m: int
) -> float:
    """Compute chance-corrected overlap kappa@k between two edge masks.

    Formula: kappa@k(A, B) = (|A ∩ B| - k^2 / m) / (k - k^2 / m)
    Subtracts the expected overlap under uniform random selection (k^2 / m).

    Args:
        mask_a: Importance weights from Explainer A.
        mask_b: Importance weights from Explainer B.
        k: Edge budget.
        candidate_m: Total candidate edge count in computational receptive field.

    Returns:
        Chance-corrected kappa score in [-1.0, 1.0].
    """
    if candidate_m <= 0 or k <= 0:
        return 0.0

    k = min(k, candidate_m)
    expected_overlap = (k * k) / float(candidate_m)
    denominator = k - expected_overlap

    if abs(denominator) < 1e-8:
        return 1.0  # Perfect agreement when k == candidate_m

    top_a = compute_top_k_indices(mask_a, k)
    top_b = compute_top_k_indices(mask_b, k)
    observed_overlap = len(top_a.intersection(top_b))

    kappa = (observed_overlap - expected_overlap) / denominator
    return float(np.clip(kappa, -1.0, 1.0))


def compute_jaccard_similarity(
    mask_a: torch.Tensor, mask_b: torch.Tensor, k: int
) -> float:
    """Compute raw Jaccard similarity between top-k edge selections."""
    top_a = compute_top_k_indices(mask_a, k)
    top_b = compute_top_k_indices(mask_b, k)

    intersection = len(top_a.intersection(top_b))
    union = len(top_a.union(top_b))

    return float(intersection / union) if union > 0 else 0.0


def compute_rank_correlation(
    mask_a: torch.Tensor, mask_b: torch.Tensor
) -> Tuple[float, float]:
    """Compute Spearman rho and Kendall tau rank correlations between continuous edge masks.

    Returns:
        Tuple of (spearman_rho, kendall_tau).
    """
    a_np = mask_a.detach().cpu().numpy()
    b_np = mask_b.detach().cpu().numpy()

    if len(a_np) < 2 or np.all(a_np == a_np[0]) or np.all(b_np == b_np[0]):
        return 0.0, 0.0

    rho, _ = stats.spearmanr(a_np, b_np)
    tau, _ = stats.kendalltau(a_np, b_np)

    rho_val = float(rho) if not np.isnan(rho) else 0.0
    tau_val = float(tau) if not np.isnan(tau) else 0.0
    return rho_val, tau_val


def compute_reliability_ceiling(rel_a: float, rel_b: float) -> float:
    """Compute the theoretical maximum agreement ceiling: sqrt(Rel_A * Rel_B)."""
    return float(np.sqrt(max(0.0, rel_a) * max(0.0, rel_b)))


def compute_pairwise_agreement_matrix(
    masks_dict: Dict[str, torch.Tensor], k: int, candidate_m: int
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise chance-corrected kappa agreement matrix across multiple explainers.

    Args:
        masks_dict: Mapping of explainer_name -> edge_mask tensor.
        k: Edge budget.
        candidate_m: Total candidate edge count in computational receptive field.

    Returns:
        Nested dictionary of pairwise kappa scores: matrix[exp_1][exp_2].
    """
    names = list(masks_dict.keys())
    matrix = {n1: {} for n1 in names}

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i == j:
                matrix[n1][n2] = 1.0
            else:
                kappa = compute_chance_corrected_kappa(
                    mask_a=masks_dict[n1],
                    mask_b=masks_dict[n2],
                    k=k,
                    candidate_m=candidate_m,
                )
                matrix[n1][n2] = kappa

    return matrix
