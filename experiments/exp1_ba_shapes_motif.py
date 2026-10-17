"""Experiment 1: Synthetic BA-Shapes Motif Ground-Truth Validation.

Evaluates Motif-AUC across 5 random seeds to test whether explainers can accurately
recover the ground-truth 5-node house motifs attached to Barabási-Albert graphs.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.config.settings import DATA_DIR, RANDOM_SEEDS, REPORTS_DIR
from src.datasets.ba_shapes import BAShapesLoader
from src.evaluation.benchmark_evaluator import BenchmarkEvaluator
from src.explainers.eig_search import EigSearchExplainer
from src.explainers.gnn_explainer import GNNExplainerWrapper
from src.explainers.lt_def import LTDEFExplainer
from src.explainers.subgraphx import SubgraphXExplainer
from src.models.gcn import GCN


def run_experiment_1(seeds=RANDOM_SEEDS, num_test_nodes=20):
    print("=" * 65)
    print("Starting Experiment 1: BA-Shapes Ground-Truth Motif AUC Validation")
    print(f"Seeds: {seeds} | Samples per seed: {num_test_nodes}")
    print("=" * 65)

    all_seed_results = []

    for seed in seeds:
        print(f"\n--- Running Seed: {seed} ---")
        torch.manual_seed(seed)
        loader = BAShapesLoader(data_dir=str(DATA_DIR), num_base_nodes=250, num_motifs=60, random_seed=seed)
        data = loader.load()

        model = GCN(in_channels=data.num_features, hidden_channels=64, out_channels=3)
        model.fit_model(data, epochs=80, lr=0.01)

        test_nodes = torch.where(data.test_mask)[0].tolist()[:num_test_nodes]
        evaluator = BenchmarkEvaluator(model=model, data=data)

        explainers = {
            "LT-DEF (Proposed)": LTDEFExplainer(model, num_hops=2, sparsity_target=0.70),
            "EiG-Search": EigSearchExplainer(model, num_hops=2, sparsity_target=0.70),
            "GNNExplainer": GNNExplainerWrapper(model, epochs=40),
            "SubgraphX": SubgraphXExplainer(model, num_hops=2, rollout_steps=10),
        }

        for name, exp in explainers.items():
            res = evaluator.evaluate_explainer(
                exp, test_nodes, ground_truth_motif_mask=data.ground_truth_mask
            )
            res["seed"] = seed
            res["explainer"] = name
            all_seed_results.append(res)
            print(f"  {name}: Motif-AUC = {res['motif_auc_mean']:.4f} | Sparsity = {res['sparsity_mean']:.3f}")

    df = pd.DataFrame(all_seed_results)
    out_file = REPORTS_DIR / "exp1_ba_shapes_motif_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved Experiment 1 results across all seeds to: {out_file}")

    # Summary table
    grouped = df.groupby("explainer")["motif_auc_mean"].agg(["mean", "std"])
    print("\nFinal Aggregate Motif-AUC (5 Seeds):")
    print(grouped.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    run_experiment_1(num_test_nodes=args.samples)
