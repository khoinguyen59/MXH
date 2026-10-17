"""Variance Decomposition Service.

Decomposes the total observed variance of explanation attributions across three sources:
1. Explainer Algorithm choice (Method variance)
2. GNN training seed (Model sensitivity)
3. Explainer initialization seed (Retrain-stability)
"""

from typing import Dict, List
import numpy as np


def decompose_explanation_variance(
    data_records: List[Dict[str, float]],
    metric_key: str = "fidelity",
    explainer_key: str = "explainer",
    gnn_seed_key: str = "gnn_seed",
    exp_seed_key: str = "exp_seed",
) -> Dict[str, float]:
    """Perform ANOVA-style variance decomposition on explanation metric observations.

    Args:
        data_records: List of experiment observation dictionaries.
        metric_key: Name of target metric to decompose.
        explainer_key: Key identifying the explainer algorithm.
        gnn_seed_key: Key identifying the GNN model training seed.
        exp_seed_key: Key identifying the explainer run seed.

    Returns:
        Dictionary of variance proportions:
        {
            "pct_algorithm": float,
            "pct_gnn_seed": float,
            "pct_exp_seed": float,
            "pct_residual": float,
            "total_variance": float
        }
    """
    if not data_records:
        return {
            "pct_algorithm": 0.0,
            "pct_gnn_seed": 0.0,
            "pct_exp_seed": 0.0,
            "pct_residual": 0.0,
            "total_variance": 0.0,
        }

    values = np.array([r[metric_key] for r in data_records], dtype=float)
    total_var = float(np.var(values))

    if total_var < 1e-9:
        return {
            "pct_algorithm": 0.0,
            "pct_gnn_seed": 0.0,
            "pct_exp_seed": 0.0,
            "pct_residual": 0.0,
            "total_variance": 0.0,
        }

    grand_mean = np.mean(values)

    # 1. Variance explained by Explainer Algorithm
    explainer_groups = {}
    for r in data_records:
        explainer_groups.setdefault(r[explainer_key], []).append(r[metric_key])

    ss_explainer = sum(
        len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in explainer_groups.values()
    ) / len(values)

    # 2. Variance explained by GNN Model Seed
    gnn_seed_groups = {}
    for r in data_records:
        gnn_seed_groups.setdefault(r[gnn_seed_key], []).append(r[metric_key])

    ss_gnn_seed = sum(
        len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in gnn_seed_groups.values()
    ) / len(values)

    # 3. Variance explained by Explainer Seed
    exp_seed_groups = {}
    for r in data_records:
        exp_seed_groups.setdefault(r[exp_seed_key], []).append(r[metric_key])

    ss_exp_seed = sum(
        len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in exp_seed_groups.values()
    ) / len(values)

    # Normalization to proportions
    pct_alg = float(np.clip(ss_explainer / total_var, 0.0, 1.0))
    pct_gnn = float(np.clip(ss_gnn_seed / total_var, 0.0, 1.0))
    pct_exp = float(np.clip(ss_exp_seed / total_var, 0.0, 1.0))

    allocated = pct_alg + pct_gnn + pct_exp
    if allocated > 1.0:
        pct_alg /= allocated
        pct_gnn /= allocated
        pct_exp /= allocated
        pct_residual = 0.0
    else:
        pct_residual = 1.0 - allocated

    return {
        "pct_algorithm": pct_alg,
        "pct_gnn_seed": pct_gnn,
        "pct_exp_seed": pct_exp,
        "pct_residual": float(pct_residual),
        "total_variance": total_var,
    }
