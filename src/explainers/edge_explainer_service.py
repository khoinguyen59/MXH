"""Standardized Edge Explanation Service for GNNs (SNA Topic C.10).

Adheres strictly to Group 09 Unified Baseline Guidelines (BG04 - BG07, BG10, BG11):
- BG04: Unified function signature:
    explain_edges(model, data, node_id, target_class, candidate_edges, config, seed)
    Returns: (edge_ids, scores, runtime_sec, status, error_message)
- BG05: 2-hop candidate edges extraction, undirected canonical indexing (min, max), no self-loops.
- BG06: Single finite score per undirected edge, max-aggregation across directions, absolute gradients for Saliency.
- BG07: Undirected pair identification (min(u,v), max(u,v)).
- BG10 & BG11: Export to edge_scores.csv and run_metadata.json.
- Preservation of Node Feature Attribution: Stores feature masks to answer research question RQ2.
"""

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.explainers.base_explainer import BaseExplainer, ExplanationResult
from src.explainers.eig_search import EigSearchExplainer
from src.explainers.gnn_explainer import GNNExplainerWrapper
from src.explainers.lt_def import LTDEFExplainer
from src.explainers.nearest_first_explainer import NearestFirstExplainer
from src.explainers.pg_explainer import PGExplainerWrapper
from src.explainers.random_explainer import RandomExplainer
from src.explainers.saliency import SaliencyExplainer
from src.explainers.subgraphx import SubgraphXExplainer


# Global storage for last node feature mask to preserve RQ2 attribution data
LAST_NODE_FEAT_MASK: Optional[torch.Tensor] = None


def get_candidate_edges(
    node_id: int,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_hops: int = 2,
) -> Tuple[List[Dict[str, int]], torch.Tensor, torch.Tensor]:
    """Extract undirected candidate edges within k-hop computational subgraph (BG05, BG07).

    Returns:
        candidate_edges: List of dicts with keys {'edge_id', 'src', 'dst'} where src < dst (undirected).
        sub_edge_index: PyG edge index for the k-hop subgraph.
        edge_to_candidate_id: Mapping from each directed edge in sub_edge_index to candidate edge_id.
    """
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=node_id,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=False,  # Keep global node IDs for clean mapping
        num_nodes=num_nodes,
    )

    # Filter out self-loops (BG05)
    mask = sub_edge_index[0] != sub_edge_index[1]
    sub_edge_index = sub_edge_index[:, mask]

    # Canonical undirected edge set (BG07)
    undirected_dict: Dict[Tuple[int, int], int] = {}
    candidate_list: List[Dict[str, int]] = []

    num_sub_edges = sub_edge_index.size(1)
    edge_to_candidate = torch.full((num_sub_edges,), -1, dtype=torch.long)

    for i in range(num_sub_edges):
        u = int(sub_edge_index[0, i].item())
        v = int(sub_edge_index[1, i].item())
        canon = (min(u, v), max(u, v))

        if canon not in undirected_dict:
            edge_id = len(candidate_list)
            undirected_dict[canon] = edge_id
            candidate_list.append({"edge_id": edge_id, "src": canon[0], "dst": canon[1]})

        edge_to_candidate[i] = undirected_dict[canon]

    return candidate_list, sub_edge_index, edge_to_candidate


def build_explainer_instance(
    method: str,
    model: torch.nn.Module,
    config: Dict[str, Any],
    num_hops: int = 2,
) -> BaseExplainer:
    """Internal factory to create the underlying explainer instance."""
    method_lower = method.lower()
    if method_lower == "saliency":
        return SaliencyExplainer(model=model, num_hops=num_hops)
    elif method_lower == "gnn_explainer":
        return GNNExplainerWrapper(
            model=model,
            epochs=config.get("epochs", 50),
            lr=config.get("lr", 0.01),
            num_hops=num_hops,
        )
    elif method_lower == "random":
        return RandomExplainer(model=model, num_hops=num_hops)
    elif method_lower == "nearest_first":
        return NearestFirstExplainer(model=model, num_hops=num_hops)
    elif method_lower in ("eig_search", "eigsearch"):
        return EigSearchExplainer(model=model, num_hops=num_hops)
    elif method_lower in ("lt_def", "ltdef"):
        return LTDEFExplainer(model=model, num_hops=num_hops)
    elif method_lower in ("subgraphx", "subgraph_x"):
        return SubgraphXExplainer(
            model=model,
            num_hops=num_hops,
            rollout_steps=config.get("rollout_steps", 10),
        )
    elif method_lower in ("pg_explainer", "pgexplainer"):
        return PGExplainerWrapper(
            model=model,
            epochs=config.get("epochs", 20),
            lr=config.get("lr", 0.005),
            num_hops=num_hops,
        )
    else:
        raise ValueError(f"Unsupported explainer method: '{method}'.")


def explain_edges(
    model: torch.nn.Module,
    data: Data,
    node_id: int,
    target_class: int,
    candidate_edges: List[Dict[str, int]],
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[List[int], List[float], float, str, str]:
    """Unified standardized interface for GNN edge explanations (Rule BG04).

    Args:
        model: Trained predictor model in eval() mode.
        data: PyG graph data container.
        node_id: Target node to explain.
        target_class: Target prediction class to explain.
        candidate_edges: Candidate edges list with keys {'edge_id', 'src', 'dst'}.
        config: Dict of hyperparams. Must specify 'method' (default: 'saliency').
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (edge_ids, scores, runtime_sec, status, error_message):
            edge_ids: List of integer edge IDs matching candidate_edges.
            scores: List of float importance scores (finite, max-aggregated per BG06).
            runtime_sec: Total execution time in seconds.
            status: Execution status ('success', 'failed', 'skipped').
            error_message: String describing error if failed, else empty string.
    """
    global LAST_NODE_FEAT_MASK
    config = config or {}
    method = config.get("method", "saliency")
    num_hops = config.get("num_hops", 2)

    # Set seeds for deterministic output
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Enforce predictor model.eval() and clone weights defensively (BG02)
    cloned_model = copy.deepcopy(model)
    cloned_model.eval()
    for param in cloned_model.parameters():
        param.requires_grad = False

    edge_ids = [cand["edge_id"] for cand in candidate_edges]
    scores = [0.0] * len(candidate_edges)
    status = "success"
    error_message = ""
    start_time = time.perf_counter()

    if len(candidate_edges) == 0:
        runtime_sec = time.perf_counter() - start_time
        return edge_ids, scores, runtime_sec, "skipped", "No candidate edges in computation graph"

    try:
        # Build explainer
        explainer = build_explainer_instance(
            method=method,
            model=cloned_model,
            config=config,
            num_hops=num_hops,
        )

        # Run explanation
        res: ExplanationResult = explainer.explain_node(
            node_idx=node_id,
            x=data.x,
            edge_index=data.edge_index,
        )

        # Store node feature mask to protect RQ2 research data
        if res.node_feat_mask is not None:
            LAST_NODE_FEAT_MASK = res.node_feat_mask.detach().cpu()
        else:
            LAST_NODE_FEAT_MASK = None

        # Build lookup table for candidate canonical pairs
        canon_to_id = {(cand["src"], cand["dst"]): cand["edge_id"] for cand in candidate_edges}

        # Subgraph edges and raw edge scores
        sub_nodes = res.subgraph_nodes
        sub_edges = res.subgraph_edge_index
        raw_mask = res.edge_mask.detach().cpu().numpy() if res.edge_mask.numel() > 0 else np.array([])

        if sub_edges.numel() > 0 and len(raw_mask) > 0:
            for i in range(sub_edges.size(1)):
                local_u = int(sub_edges[0, i].item())
                local_v = int(sub_edges[1, i].item())
                global_u = int(sub_nodes[local_u].item())
                global_v = int(sub_nodes[local_v].item())
                canon = (min(global_u, global_v), max(global_u, global_v))

                if canon in canon_to_id:
                    c_id = canon_to_id[canon]
                    score_val = float(raw_mask[i])
                    # Finite sanity check
                    if np.isnan(score_val) or np.isinf(score_val):
                        score_val = 0.0
                    # For Saliency, gradient absolute value is already handled in explainer
                    # Aggregate max score across both directed edges (BG06)
                    scores[c_id] = max(scores[c_id], score_val)

    except Exception as e:
        status = "failed"
        error_message = str(e)
        scores = [0.0] * len(candidate_edges)

    runtime_sec = time.perf_counter() - start_time
    return edge_ids, scores, runtime_sec, status, error_message


def export_explanation_results(
    records: List[Dict[str, Any]],
    output_dir: str = "experiments/pilot_study/results",
) -> Tuple[str, str]:
    """Export edge scores and run metadata per BG10 & BG11."""
    os.makedirs(output_dir, exist_ok=True)

    scores_path = os.path.join(output_dir, "edge_scores.csv")
    meta_path = os.path.join(output_dir, "run_metadata.json")

    # 1. Export edge_scores.csv (BG10)
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write("run_id,edge_id,src,dst,score\n")
        for rec in records:
            run_id = rec["run_id"]
            for cand, score in zip(rec["candidate_edges"], rec["scores"]):
                f.write(f"{run_id},{cand['edge_id']},{cand['src']},{cand['dst']},{score:.6f}\n")

    # 2. Export run_metadata.json (BG11)
    metadata_list = []
    for rec in records:
        metadata_list.append(
            {
                "run_id": rec["run_id"],
                "dataset": rec.get("dataset", "cora"),
                "node_id": rec["node_id"],
                "target_class": rec["target_class"],
                "predicted_class": rec.get("predicted_class", rec["target_class"]),
                "original_prob": rec.get("original_prob", 1.0),
                "checkpoint_id": rec.get("checkpoint_id", "ckpt_gcn"),
                "model_architecture": rec.get("model_architecture", "GCN_2Layer"),
                "seed_gnn": rec.get("seed_gnn", 1000),
                "explainer_method": rec.get("explainer_method", "saliency"),
                "seed_explainer": rec.get("seed_explainer", 42),
                "config": rec.get("config", {}),
                "runtime_sec": rec["runtime_sec"],
                "num_candidates": len(rec["candidate_edges"]),
                "status": rec["status"],
                "error_message": rec["error_message"],
            }
        )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    return scores_path, meta_path
