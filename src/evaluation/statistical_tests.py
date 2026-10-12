"""Statistical Testing & Hypothesis Evaluation Service.

Implements pre-registered statistical protocols:
1. Holm-Bonferroni step-down correction for multiple pairwise explainer comparisons.
2. Two One-Sided Tests (TOST) for asserting practical equivalence between explanations.
"""

from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats


def holm_bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni step-down family-wise error rate correction.

    Args:
        p_values: List of raw uncorrected p-values.
        alpha: Target significance level (default: 0.05).

    Returns:
        List of tuples: (adjusted_p_value, is_significant_after_correction).
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort with original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    significant = [False] * m

    cum_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj_p = min(1.0, max(cum_max, p_val * multiplier))
        cum_max = adj_p

        adjusted[orig_idx] = float(adj_p)
        significant[orig_idx] = bool(adj_p < alpha)

    return list(zip(adjusted, significant))


def tost_equivalence_test(
    sample_a: List[float],
    sample_b: List[float],
    bound_delta: float = 0.05,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Perform Two One-Sided Tests (TOST) for testing statistical equivalence.

    Tests whether the difference in means between two explainer metrics falls
    strictly within the pre-registered equivalence margin [-bound_delta, +bound_delta].

    Args:
        sample_a: Observations from Explainer A.
        sample_b: Observations from Explainer B.
        bound_delta: Equivalence boundary margin (default: 0.05).
        alpha: Significance level (default: 0.05).

    Returns:
        Dictionary with p_value_lower, p_value_upper, mean_diff, and is_equivalent flag.
    """
    a = np.array(sample_a, dtype=float)
    b = np.array(sample_b, dtype=float)

    diff = float(np.mean(a) - np.mean(b))
    se_diff = float(
        np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    )

    if se_diff < 1e-9:
        is_equiv = abs(diff) < bound_delta
        return {
            "mean_difference": diff,
            "t_lower": 0.0,
            "t_upper": 0.0,
            "p_lower": 0.0 if is_equiv else 1.0,
            "p_upper": 0.0 if is_equiv else 1.0,
            "is_equivalent": float(is_equiv),
        }

    # Degrees of freedom (Welch-Satterthwaite)
    v1 = np.var(a, ddof=1) / len(a)
    v2 = np.var(b, ddof=1) / len(b)
    df = (v1 + v2) ** 2 / (v1**2 / (len(a) - 1) + v2**2 / (len(b) - 1))

    # Test 1: H0: diff <= -bound_delta vs H1: diff > -bound_delta
    t_lower = (diff - (-bound_delta)) / se_diff
    p_lower = 1.0 - stats.t.cdf(t_lower, df=df)

    # Test 2: H0: diff >= bound_delta vs H1: diff < bound_delta
    t_upper = (diff - bound_delta) / se_diff
    p_upper = stats.t.cdf(t_upper, df=df)

    # Equivalence is established if BOTH one-sided null hypotheses are rejected
    p_tost = max(p_lower, p_upper)
    is_equivalent = bool(p_tost < alpha)

    return {
        "mean_difference": diff,
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_tost": float(p_tost),
        "is_equivalent": float(is_equivalent),
    }
