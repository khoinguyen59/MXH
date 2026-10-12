"""Evaluation package for SNA Topic C.10."""

from src.evaluation.agreement import (
    compute_chance_corrected_kappa,
    compute_jaccard_similarity,
    compute_pairwise_agreement_matrix,
    compute_rank_correlation,
    compute_reliability_ceiling,
)
from src.evaluation.benchmark_evaluator import BenchmarkEvaluator
from src.evaluation.fidelity import (
    compute_fidelity_at_budget,
    compute_fidelity_curve,
)
from src.evaluation.k_star import compute_k_star
from src.evaluation.metrics import (
    compute_fidelity_minus,
    compute_fidelity_plus,
    compute_hallucination_rate,
    compute_motif_roc_auc,
    compute_sparsity,
)
from src.evaluation.statistical_tests import (
    holm_bonferroni_correction,
    tost_equivalence_test,
)
from src.evaluation.variance_decomposition import decompose_explanation_variance

__all__ = [
    "compute_fidelity_plus",
    "compute_fidelity_minus",
    "compute_fidelity_at_budget",
    "compute_fidelity_curve",
    "compute_sparsity",
    "compute_motif_roc_auc",
    "compute_hallucination_rate",
    "compute_chance_corrected_kappa",
    "compute_jaccard_similarity",
    "compute_rank_correlation",
    "compute_reliability_ceiling",
    "compute_pairwise_agreement_matrix",
    "compute_k_star",
    "decompose_explanation_variance",
    "holm_bonferroni_correction",
    "tost_equivalence_test",
    "BenchmarkEvaluator",
]

