"""Standardized Edge Explanation Service for GNNs (SNA Topic C.10).

Adheres strictly to Group 09 Unified Baseline Guidelines (BG04 - BG06, BG10, BG11):
- BG04: Standardized interface: explain_edges(model, data, node_id, target_class, candidate_edges, config, seed)
- BG05: 2-hop candidate edges extraction, undirected canonical indexing (min, max), no self-loops.
- BG06: Single finite score per undirected edge, max-aggregation across directions, absolute gradients for Saliency.
- BG10 & BG11: Export to edge_scores.csv and run_metadata.json.
"""

import time
import copy
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.explainers.gnn_explainer import GNNExplainerWrapper
from src.explainers.saliency import SaliencyExplainer


def get_candidate_edges(node_id: int, edge_index: torch.Tensor, num_nodes: int, num_hops: int = 2) -> Tuple[List[Dict[str, int]], torch.Tensor, torch.Tensor]:
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
        num_nodes=num_nodes
    )
    
    # Filter out self-loops
    mask = sub_edge_index[0] != sub_edge_index[1]
    sub_edge_index = sub_edge_index[:, mask]
    
    # Canonical undirected edge set
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
            candidate_list.append({
                "edge_id": edge_id,
                "src": canon[0],
                "dst": canon[1]
            })
        
        edge_to_candidate[i] = undirected_dict[canon]
        
    return candidate_list, sub_edge_index, edge_to_candidate


def explain_edges_gnn_explainer(
    model: torch.nn.Module,
    data: Data,
    node_id: int,
    target_class: int,
    candidate_edges: List[Dict[str, int]],
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Explain edges via GNNExplainer (Ying et al., 2019) with BG04/BG06 compliance."""
    config = config or {"epochs": 100, "lr": 0.01, "num_hops": 2}
    torch.manual_seed(seed)
    
    # Use isolated clone of model (BG02)
    cloned_model = copy.deepcopy(model)
    cloned_model.eval()
    
    start_time = time.perf_counter()
    status = "success"
    error_message = ""
    scores = [0.0] * len(candidate_edges)
    
    try:
        wrapper = GNNExplainerWrapper(
            model=cloned_model,
            epochs=config.get("epochs", 100),
            lr=config.get("lr", 0.01),
            num_hops=config.get("num_hops", 2)
        )
        res = wrapper.explain_node(node_idx=node_id, x=data.x, edge_index=data.edge_index)
        
        sub_nodes = res.subgraph_nodes
        sub_edges = res.subgraph_edge_index
        raw_mask = res.edge_mask.detach().cpu().numpy()
        
        # Build lookup for candidates
        canon_map = {(cand["src"], cand["dst"]): cand["edge_id"] for cand in candidate_edges}
        
        # Aggregate max score across both directions (BG06, BG07)
        for i in range(sub_edges.size(1)):
            local_u = int(sub_edges[0, i].item())
            local_v = int(sub_edges[1, i].item())
            global_u = int(sub_nodes[local_u].item())
            global_v = int(sub_nodes[local_v].item())
            canon = (min(global_u, global_v), max(global_u, global_v))
            if canon in canon_map:
                c_id = canon_map[canon]
                score_val = float(raw_mask[i])
                scores[c_id] = max(scores[c_id], score_val)
                
    except Exception as e:
        status = "failed"
        error_message = str(e)
        
    runtime_sec = time.perf_counter() - start_time
    
    return {
        "edge_ids": [c["edge_id"] for c in candidate_edges],
        "scores": scores,
        "runtime_sec": runtime_sec,
        "status": status,
        "error_message": error_message
    }


def explain_edges_saliency(
    model: torch.nn.Module,
    data: Data,
    node_id: int,
    target_class: int,
    candidate_edges: List[Dict[str, int]],
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Explain edges via Saliency (first-order gradient attribution) with BG04/BG06 compliance."""
    config = config or {"num_hops": 2}
    torch.manual_seed(seed)
    
    cloned_model = copy.deepcopy(model)
    cloned_model.eval()
    
    start_time = time.perf_counter()
    status = "success"
    error_message = ""
    scores = [0.0] * len(candidate_edges)
    
    try:
        wrapper = SaliencyExplainer(model=cloned_model, num_hops=config.get("num_hops", 2))
        res = wrapper.explain_node(node_idx=node_id, x=data.x, edge_index=data.edge_index)
        
        sub_nodes = res.subgraph_nodes
        sub_edges = res.subgraph_edge_index
        raw_mask = res.edge_mask.detach().cpu().numpy()
        
        canon_map = {(cand["src"], cand["dst"]): cand["edge_id"] for cand in candidate_edges}
        
        # Aggregate max score across both directions using absolute gradient (BG06)
        for i in range(sub_edges.size(1)):
            local_u = int(sub_edges[0, i].item())
            local_v = int(sub_edges[1, i].item())
            global_u = int(sub_nodes[local_u].item())
            global_v = int(sub_nodes[local_v].item())
            canon = (min(global_u, global_v), max(global_u, global_v))
            if canon in canon_map:
                c_id = canon_map[canon]
                score_val = float(raw_mask[i])
                scores[c_id] = max(scores[c_id], score_val)
                
    except Exception as e:
        status = "failed"
        error_message = str(e)
        
    runtime_sec = time.perf_counter() - start_time
    
    return {
        "edge_ids": [c["edge_id"] for c in candidate_edges],
        "scores": scores,
        "runtime_sec": runtime_sec,
        "status": status,
        "error_message": error_message
    }


def export_explanation_results(
    records: List[Dict[str, Any]],
    output_dir: str = "experiments/pilot_study/results"
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
        metadata_list.append({
            "run_id": rec["run_id"],
            "dataset": rec["dataset"],
            "node_id": rec["node_id"],
            "target_class": rec["target_class"],
            "predicted_class": rec["predicted_class"],
            "original_prob": rec["original_prob"],
            "checkpoint_id": rec["checkpoint_id"],
            "model_architecture": rec["model_architecture"],
            "seed_gnn": rec["seed_gnn"],
            "explainer_method": rec["explainer_method"],
            "seed_explainer": rec["seed_explainer"],
            "config": rec["config"],
            "runtime_sec": rec["runtime_sec"],
            "num_candidates": len(rec["candidate_edges"]),
            "status": rec["status"],
            "error_message": rec["error_message"]
        })
        
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        
    return scores_path, meta_path
