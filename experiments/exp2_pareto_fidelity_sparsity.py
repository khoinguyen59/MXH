"""Experiment 2: Pareto Trade-off between Fidelity+ and Sparsity.

Evaluates how Fidelity+ scales across varying Sparsity constraints (0.50, 0.60, 0.70, 0.80, 0.90)
on real-world networks (Cora, Twitch) to construct Pareto frontier curves.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch

from src.config.settings import DATA_DIR, REPORTS_DIR
from src.datasets.cora import CoraLoader
from src.datasets.twitch import TwitchLoader
from src.evaluation.benchmark_evaluator import BenchmarkEvaluator
from src.explainers.eig_search import EigSearchExplainer
from src.explainers.lt_def import LTDEFExplainer
from src.models.graphsage import GraphSAGE


def run_experiment_2(dataset_name="cora", num_samples=15):
    print("=" * 65)
    print(f"Starting Experiment 2: Pareto Fidelity-Sparsity Analysis ({dataset_name})")
    print("=" * 65)

    if dataset_name == "cora":
        loader = CoraLoader(data_dir=str(DATA_DIR), random_seed=42)
    else:
        loader = TwitchLoader(data_dir=str(DATA_DIR), name="DE", random_seed=42)

    data = loader.load()
    model = GraphSAGE(data.num_features, 64, int(data.y.max().item()) + 1)
    model.fit_model(data, epochs=60, lr=0.01)

    test_nodes = torch.where(data.test_mask)[0].tolist()[:num_samples]
    sparsity_levels = [0.50, 0.60, 0.70, 0.80, 0.90]

    records = []
    for sp in sparsity_levels:
        print(f"\n--- Testing Sparsity Target: {sp:.2f} ---")
        evaluator = BenchmarkEvaluator(model=model, data=data, threshold=0.5)

        explainers = {
            "LT-DEF": LTDEFExplainer(model, num_hops=2, sparsity_target=sp),
            "EiG-Search": EigSearchExplainer(model, num_hops=2, sparsity_target=sp),
        }

        for name, exp in explainers.items():
            metrics = evaluator.evaluate_explainer(exp, test_nodes)
            records.append({
                "dataset": dataset_name,
                "target_sparsity": sp,
                "actual_sparsity": metrics["sparsity_mean"],
                "fidelity_plus": metrics["fidelity_plus_mean"],
                "fidelity_minus": metrics["fidelity_minus_mean"],
                "runtime_sec": metrics["runtime_mean_sec"],
                "explainer": name,
            })
            print(f"  {name} @ {sp}: Fid+ = {metrics['fidelity_plus_mean']:.4f} | Fid- = {metrics['fidelity_minus_mean']:.4f}")

    df = pd.DataFrame(records)
    out_file = REPORTS_DIR / f"exp2_pareto_{dataset_name}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved Experiment 2 Pareto table to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "twitch"])
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    run_experiment_2(dataset_name=args.dataset, num_samples=args.samples)
