"""Cora Graph Explainability Benchmark Pipeline.

Executes explainability algorithms (GNNExplainer, Saliency, Random, Nearest-First)
on the Cora citation graph model checkpoint.
Calculates standardized XAI metrics: Fidelity+, Fidelity-, Sparsity, and Runtime.
Saves structured JSON and CSV reports in experiments/logs/cora_explainers/.
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.cora_handover_adapter import CoraHandoverAdapter, CoraHandoverBundle
from src.evaluation.fidelity import compute_fidelity_at_budget
from src.explainers.base_explainer import BaseExplainer, ExplanationResult
from src.explainers.gnn_explainer import GNNExplainerWrapper
from src.explainers.nearest_first_explainer import NearestFirstExplainer
from src.explainers.random_explainer import RandomExplainer
from src.explainers.saliency import SaliencyExplainer
from src.explainers.eig_search import EigSearchExplainer
from src.explainers.lt_def import LTDEFExplainer
from src.explainers.subgraphx import SubgraphXExplainer


def build_explainer(name: str, model: torch.nn.Module, num_hops: int = 2) -> BaseExplainer:
    """Factory creating an explainer instance by name."""
    name_lower = name.lower()
    if name_lower == "gnn_explainer":
        return GNNExplainerWrapper(model=model, epochs=50, lr=0.01, num_hops=num_hops)
    elif name_lower == "saliency":
        return SaliencyExplainer(model=model, num_hops=num_hops)
    elif name_lower == "random":
        return RandomExplainer(model=model, num_hops=num_hops)
    elif name_lower == "nearest_first":
        return NearestFirstExplainer(model=model, num_hops=num_hops)
    elif name_lower in ("eig_search", "eigsearch"):
        return EigSearchExplainer(model=model, num_hops=num_hops)
    elif name_lower in ("lt_def", "ltdef"):
        return LTDEFExplainer(model=model, num_hops=num_hops)
    elif name_lower in ("subgraphx", "subgraph_x"):
        return SubgraphXExplainer(model=model, num_hops=num_hops, rollout_steps=10)
    else:
        raise ValueError(
            f"Unknown explainer '{name}'. Supported: gnn_explainer, saliency, random, nearest_first, eig_search, lt_def, subgraphx."
        )


def run_cora_explainers_pipeline(
    num_nodes: int = 5,
    explainers: List[str] = None,
    output_dir: str = "experiments/logs/cora_explainers",
    budget_k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end XAI evaluation across target nodes in Cora.

    Args:
        num_nodes: Number of validation nodes to evaluate (default: 5 for local smoke test).
        explainers: List of explainer names to run.
        output_dir: Directory where JSON and CSV reports are written.
        budget_k: Edge budget for Fidelity calculation.
        seed: Random seed for reproducibility.

    Returns:
        Summary dict containing mean and std of all metrics per explainer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if explainers is None:
        explainers = ["saliency", "random", "nearest_first", "gnn_explainer"]

    print("=" * 70)
    print("CORA GNN EXPLAINABILITY BENCHMARK PIPELINE")
    print("=" * 70)
    print("Loading Cora artifacts via CoraHandoverAdapter...")

    adapter = CoraHandoverAdapter()
    bundle: CoraHandoverBundle = adapter.load(device="cpu")
    data = bundle.data
    model = bundle.model

    # Verify model accuracy
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=-1)
    acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    print(f"Baseline Model Test Accuracy: {acc*100:.2f}% (Verified)")

    # Select target nodes from designated val_nodes
    target_nodes = bundle.val_nodes[:num_nodes]
    print(f"Evaluating {len(target_nodes)} target nodes: {target_nodes}")
    print(f"Active Explainers: {explainers}\n")

    summary_results: Dict[str, Dict[str, Any]] = {}
    csv_rows: List[str] = [
        "node_id,true_label,predicted_label,explainer,runtime_ms,fid_plus,fid_minus,sparsity"
    ]

    for explainer_name in explainers:
        print(f"--- Running {explainer_name.upper()} ---")
        explainer = build_explainer(explainer_name, model, num_hops=2)

        node_runtimes = []
        node_fid_plus = []
        node_fid_minus = []
        node_sparsity = []

        for node_id in target_nodes:
            true_label = int(data.y[node_id].item())
            pred_label = int(preds[node_id].item())

            start_t = time.perf_counter()
            res: ExplanationResult = explainer.explain_node(node_id, data.x, data.edge_index)
            runtime_ms = (time.perf_counter() - start_t) * 1000.0

            # Calculate sparsity: fraction of edges with weight < 0.1
            if res.edge_mask.numel() > 0:
                sparsity = float((res.edge_mask < 0.1).float().mean().item())
            else:
                sparsity = 1.0

            # Calculate Fidelity+ and Fidelity-
            sub_x = data.x[res.subgraph_nodes]
            local_target = int(torch.where(res.subgraph_nodes == node_id)[0].item())
            fid_plus, fid_minus = compute_fidelity_at_budget(
                model=model,
                x=sub_x,
                sub_edge_index=res.subgraph_edge_index,
                target_node=local_target,
                target_class=pred_label,
                edge_mask=res.edge_mask,
                k=budget_k,
            )

            node_runtimes.append(runtime_ms)
            node_fid_plus.append(fid_plus)
            node_fid_minus.append(fid_minus)
            node_sparsity.append(sparsity)

            csv_rows.append(
                f"{node_id},{true_label},{pred_label},{explainer_name},"
                f"{runtime_ms:.2f},{fid_plus:.4f},{fid_minus:.4f},{sparsity:.4f}"
            )

        summary_results[explainer_name] = {
            "runtime_ms_mean": float(np.mean(node_runtimes)),
            "runtime_ms_std": float(np.std(node_runtimes)),
            "fidelity_plus_mean": float(np.mean(node_fid_plus)),
            "fidelity_plus_std": float(np.std(node_fid_plus)),
            "fidelity_minus_mean": float(np.mean(node_fid_minus)),
            "fidelity_minus_std": float(np.std(node_fid_minus)),
            "sparsity_mean": float(np.mean(node_sparsity)),
            "sparsity_std": float(np.std(node_sparsity)),
            "num_nodes_evaluated": len(target_nodes),
            "budget_k": budget_k,
        }

        print(
            f"  Fidelity+: {np.mean(node_fid_plus):.4f} +/- {np.std(node_fid_plus):.4f} | "
            f"Fidelity-: {np.mean(node_fid_minus):.4f} +/- {np.std(node_fid_minus):.4f} | "
            f"Sparsity: {np.mean(node_sparsity):.4f} | "
            f"Avg Time: {np.mean(node_runtimes):.1f}ms"
        )

    # Export CSV
    csv_file = output_path / "cora_node_evaluations.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"\nSaved CSV evaluation logs: {csv_file}")

    # Export JSON Summary
    json_file = output_path / "cora_xai_summary.json"
    full_output = {
        "metadata": {
            "dataset": "cora",
            "model": "gcn",
            "source_checkpoint": "data/t01_handover/cora/gcn_ckpt.pt",
            "test_accuracy": acc,
            "evaluated_nodes": target_nodes,
            "budget_k": budget_k,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "explainers": summary_results,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON summary: {json_file}")

    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cora GNN Explainability Pipeline")
    parser.add_argument(
        "--nodes", type=int, default=5, help="Number of validation nodes to explain (default: 5)"
    )
    parser.add_argument(
        "--budget-k", type=int, default=5, help="Top-k budget for fidelity calculation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/logs/cora_explainers",
        help="Output directory",
    )
    parser.add_argument(
        "--explainers",
        type=str,
        default=None,
        help="Comma-separated list of explainers to run (e.g. saliency,random,nearest_first,eig_search,lt_def,subgraphx,gnn_explainer)",
    )
    args = parser.parse_args()

    exp_list = [e.strip() for e in args.explainers.split(",")] if args.explainers else None

    run_cora_explainers_pipeline(
        num_nodes=args.nodes,
        budget_k=args.budget_k,
        output_dir=args.output_dir,
        explainers=exp_list,
    )
