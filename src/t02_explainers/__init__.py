"""T02 Explainers Package (Group 09 Explainability Engine & T01 Handover Adapters)."""

from src.t02_explainers.adapters.t01_model_adapter import T01GCNModelWrapper
from src.t02_explainers.adapters.cora_handover_adapter import CoraHandoverAdapter, CoraHandoverBundle
from src.t02_explainers.adapters.bashapes_handover_adapter import BAShapesHandoverAdapter, BAShapesHandoverBundle
from src.t02_explainers.algorithms.base_explainer import BaseExplainer, ExplanationResult
from src.t02_explainers.algorithms.gnn_explainer import GNNExplainerWrapper
from src.t02_explainers.algorithms.pg_explainer import PGExplainerWrapper
from src.t02_explainers.algorithms.saliency import SaliencyExplainer
from src.t02_explainers.algorithms.random_explainer import RandomExplainer
from src.t02_explainers.algorithms.nearest_first_explainer import NearestFirstExplainer
from src.t02_explainers.algorithms.eig_search import EigSearchExplainer
from src.t02_explainers.algorithms.lt_def import LTDEFExplainer
from src.t02_explainers.algorithms.subgraphx import SubgraphXExplainer
from src.t02_explainers.services.edge_explainer_service import (
    get_candidate_edges,
    explain_edges_gnn_explainer,
    explain_edges_saliency,
    export_explanation_results,
)

__all__ = [
    "T01GCNModelWrapper",
    "CoraHandoverAdapter",
    "CoraHandoverBundle",
    "BAShapesHandoverAdapter",
    "BAShapesHandoverBundle",
    "BaseExplainer",
    "ExplanationResult",
    "GNNExplainerWrapper",
    "PGExplainerWrapper",
    "SaliencyExplainer",
    "RandomExplainer",
    "NearestFirstExplainer",
    "EigSearchExplainer",
    "LTDEFExplainer",
    "SubgraphXExplainer",
    "get_candidate_edges",
    "explain_edges_gnn_explainer",
    "explain_edges_saliency",
    "export_explanation_results",
]
