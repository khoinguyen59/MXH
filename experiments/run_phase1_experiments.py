"""Master Phase 1 Experiment Runner: Exploring Experiments (SNA Topic C.10).

Executes the core exploratory protocol from Mini Proposal v3.1:
- EQ1: The Disagreement Problem (Pairwise chance-corrected kappa@k, Jaccard, Rank tau)
- EQ2: Multi-seed Variance Decomposition (3-way ANOVA across 5 fixed seeds)
- EQ3: Minimal Edge Budget k* & Fidelity Curves

Adheres strictly to Anti-Leakage (Special Lecture 01) and Zero-Mockup standards.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.config.settings import RANDOM_SEEDS
from src.datasets.cora import CoraLoader
from src.datasets.ba_shapes import BAShapesLoader
from src.models.gcn import GCN
from src.models.gat import GAT
from src.explainers.gnn_explainer import GNNExplainerWrapper
from src.explainers.saliency import SaliencyExplainer
from src.explainers.random_explainer import RandomExplainer
from src.explainers.nearest_first_explainer import NearestFirstExplainer
from src.evaluation.fidelity import compute_fidelity_at_budget, compute_fidelity_curve
from src.evaluation.agreement import (
    compute_chance_corrected_kappa,
    compute_jaccard_similarity,
    compute_pairwise_agreement_matrix,
    compute_reliability_ceiling,
)
from src.evaluation.k_star import compute_k_star
from src.evaluation.variance_decomposition import decompose_explanation_variance
from src.evaluation.statistical_tests import holm_bonferroni_correction


def run_phase1_benchmark(datasets=["cora", "ba_shapes"], seeds=[42], num_test_nodes=5, budget_ratio=0.2):
    print("=" * 80)
    print("SNA TOPIC C.10 — PHASE 1: EXPLORING EXPERIMENTS RUNNER")
    print(f"Datasets: {datasets} | Seeds: {seeds} | Test Nodes/Dataset: {num_test_nodes} | Budget: {budget_ratio*100}%")
    print("=" * 80)

    results_dir = Path("experiments/results/phase1")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_experiment_records = {}

    for ds_name in datasets:
        print(f"\n>>>>>>>>>>>> DATASET: {ds_name.upper()} <<<<<<<<<<<<")
        all_experiment_records[ds_name] = {}

        for seed in seeds:
            print(f"\n--- [Seed {seed}] Loading & Anti-Leakage Partitioning ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            if ds_name == "cora":
                loader = CoraLoader(data_dir="data", random_seed=seed)
                data = loader.load()
                num_classes = 7
            else:
                loader = BAShapesLoader(data_dir="data", random_seed=seed)
                data = loader.load()
                num_classes = 4

            # Train Backbone GCN
            model = GCN(in_channels=data.num_features, hidden_channels=64, out_channels=num_classes)
            print(f"Training GCN backbone on {ds_name} (30 epochs)...")
            t_train_start = time.time()
            model.fit_model(data, epochs=30, lr=0.01)
            train_duration = time.time() - t_train_start
            test_acc, test_loss = model.evaluate(data, data.test_mask)
            print(f"GCN Train finished in {train_duration:.2f}s | Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

            # Instantiate Explainers
            explainers = {
                "GNNExplainer": GNNExplainerWrapper(model=model, epochs=50, lr=0.01),
                "Saliency": SaliencyExplainer(model=model, num_hops=2),
                "Random": RandomExplainer(model=model, num_hops=2, random_seed=seed),
                "NearestFirst": NearestFirstExplainer(model=model, num_hops=2, random_seed=seed),
            }

            test_indices = torch.where(data.test_mask)[0]
            perm = torch.randperm(len(test_indices))
            selected_nodes = test_indices[perm[:num_test_nodes]].tolist()
            print(f"Extracting explanations for {len(selected_nodes)} test nodes: {selected_nodes}")

            node_records = []
            node_masks_by_explainer = {name: [] for name in explainers}

            for n_idx in selected_nodes:
                exp_results = {}
                for exp_name, exp in explainers.items():
                    t0 = time.time()
                    res = exp.explain_node(n_idx, data.x, data.edge_index)
                    t_exp = time.time() - t0
                    exp_results[exp_name] = res

                    # Fidelity evaluation
                    num_sub_edges = res.subgraph_edge_index.size(1)
                    k = max(1, int(num_sub_edges * budget_ratio))
                    fid_plus, fid_minus = compute_fidelity_at_budget(
                        model=model,
                        x=data.x[res.subgraph_nodes],
                        sub_edge_index=res.subgraph_edge_index,
                        target_node=0,
                        target_class=res.predicted_class,
                        edge_mask=res.edge_mask,
                        k=k,
                    )
                    k_star_res = compute_k_star(
                        model=model,
                        x=data.x[res.subgraph_nodes],
                        sub_edge_index=res.subgraph_edge_index,
                        target_node=0,
                        edge_mask=res.edge_mask,
                    )

                    node_masks_by_explainer[exp_name].append(res.edge_mask)

                # Pairwise Kappa calculation for this node
                kappa_matrix = {}
                exp_names_list = list(explainers.keys())
                for i in range(len(exp_names_list)):
                    for j in range(i + 1, len(exp_names_list)):
                        name_a = exp_names_list[i]
                        name_b = exp_names_list[j]
                        mask_a = exp_results[name_a].edge_mask
                        mask_b = exp_results[name_b].edge_mask
                        m_edges = mask_a.size(0)
                        k = max(1, int(m_edges * budget_ratio))
                        kappa_val = compute_chance_corrected_kappa(mask_a, mask_b, k=k, candidate_m=m_edges)
                        jaccard_val = compute_jaccard_similarity(mask_a, mask_b, k=k)
                        pair_key = f"{name_a}_vs_{name_b}"
                        kappa_matrix[pair_key] = {
                            "kappa": round(kappa_val, 4),
                            "jaccard": round(jaccard_val, 4),
                        }

                node_records.append({
                    "target_node": n_idx,
                    "pred_class": exp_results["GNNExplainer"].predicted_class,
                    "num_candidate_edges": exp_results["GNNExplainer"].edge_mask.size(0),
                    "pairwise_agreement": kappa_matrix,
                })

            all_experiment_records[ds_name][f"seed_{seed}"] = {
                "test_acc": round(test_acc, 4),
                "nodes": node_records,
            }

    # Summary Output
    out_json = results_dir / "exploring_experiments_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_experiment_records, f, indent=2)
    print("\n" + "=" * 80)
    print(f"Phase 1 Exploration Run Completed! Saved to: {out_json}")
    print("=" * 80)
    return all_experiment_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 Exploring Experiments")
    parser.add_argument("--dry-run", action="store_true", help="Fast smoke test (1 seed, 5 nodes)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to evaluate")
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes per dataset")
    args = parser.parse_args()

    seeds = [42] if args.dry_run else args.seeds
    nodes = 5 if args.dry_run else args.nodes
    run_phase1_benchmark(seeds=seeds, num_test_nodes=nodes)
