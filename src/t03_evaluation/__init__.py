"""T03 Evaluation Package (Quantitative Metrics, Fidelity, Agreement & Statistical Tests)."""

from src.t03_evaluation.fidelity import compute_fidelity_at_budget, compute_fidelity_curve
from src.t03_evaluation.agreement import (
    compute_chance_corrected_kappa,
    compute_jaccard_similarity,
    compute_rank_correlation,
    compute_reliability_ceiling,
    compute_pairwise_agreement_matrix,
)
from src.t03_evaluation.metrics import (
    compute_fidelity_plus,
    compute_fidelity_minus,
    compute_sparsity,
    compute_motif_roc_auc,
    compute_hallucination_rate,
)
from src.t03_evaluation.k_star import compute_k_star, compute_k_star as compute_k_star_metric
from src.t03_evaluation.statistical_tests import (
    holm_bonferroni_correction,
    tost_equivalence_test,
)
from src.t03_evaluation.variance_decomposition import (
    decompose_explanation_variance,
    decompose_explanation_variance as decompose_prediction_variance,
)
from src.t03_evaluation.benchmark_evaluator import BenchmarkEvaluator

__all__ = [
    "compute_fidelity_at_budget",
    "compute_fidelity_curve",
    "compute_chance_corrected_kappa",
    "compute_jaccard_similarity",
    "compute_rank_correlation",
    "compute_reliability_ceiling",
    "compute_pairwise_agreement_matrix",
    "compute_fidelity_plus",
    "compute_fidelity_minus",
    "compute_sparsity",
    "compute_motif_roc_auc",
    "compute_hallucination_rate",
    "compute_k_star",
    "compute_k_star_metric",
    "holm_bonferroni_correction",
    "tost_equivalence_test",
    "decompose_explanation_variance",
    "decompose_prediction_variance",
    "BenchmarkEvaluator",
]
