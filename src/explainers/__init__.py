"""Explainability Engine package for SNA Topic C.10."""

from src.explainers.base_explainer import BaseExplainer, ExplanationResult
from src.explainers.eig_search import EigSearchExplainer
from src.explainers.gnn_explainer import GNNExplainerWrapper
from src.explainers.lt_def import LTDEFExplainer
from src.explainers.nearest_first_explainer import NearestFirstExplainer
from src.explainers.pg_explainer import PGExplainerWrapper
from src.explainers.random_explainer import RandomExplainer
from src.explainers.saliency import SaliencyExplainer
from src.explainers.subgraphx import SubgraphXExplainer

from src.explainers.edge_explainer_service import (
    explain_edges,
    get_candidate_edges,
    LAST_NODE_FEAT_MASK,
    export_explanation_results,
)

__all__ = [
    "BaseExplainer",
    "ExplanationResult",
    "GNNExplainerWrapper",
    "PGExplainerWrapper",
    "SaliencyExplainer",
    "RandomExplainer",
    "NearestFirstExplainer",
    "SubgraphXExplainer",
    "EigSearchExplainer",
    "LTDEFExplainer",
    "explain_edges",
    "get_candidate_edges",
    "LAST_NODE_FEAT_MASK",
    "export_explanation_results",
]


