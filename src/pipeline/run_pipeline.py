"""End-to-End Orchestrated Pipeline Runner.

Executes complete pipeline: Data Ingestion -> Anti-Leakage Split -> Model Training ->
Explanation Extraction -> Quantitative Evaluation -> NLE Grounding -> Report Generation.
Adheres strictly to Invariant-02 of the Master Governance Specification.
"""

import argparse
from pathlib import Path
import sys
import torch

from src.config.settings import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DEVICE,
    RANDOM_SEEDS,
    REPORTS_DIR,
)
from src.datasets.ba_shapes import BAShapesLoader
from src.datasets.cora import CoraLoader
from src.datasets.twitch import TwitchLoader
from src.evaluation.benchmark_evaluator import BenchmarkEvaluator
from src.explainers.eig_search import EigSearchExplainer
from src.explainers.gnn_explainer import GNNExplainerWrapper
from src.explainers.lt_def import LTDEFExplainer
from src.explainers.subgraphx import SubgraphXExplainer
from src.models.gat import GAT
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.nle_grounding.rejection_filter import RejectionFilter
from src.nle_grounding.verbalizer import NLEVerbalizer
from src.visualization.graph_plotter import GraphPlotter


def parse_args():
    parser = argparse.ArgumentParser(description="SNA C.10 End-to-End Pipeline Runner")
    parser.add_argument("--dataset", type=str, default="ba_shapes", choices=["ba_shapes", "cora", "twitch"])
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "graphsage", "gat"])
    parser.add_argument("--explainer", type=str, default="lt_def", choices=["all", "lt_def", "eig_search", "gnn_explainer", "subgraphx"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=10, help="Number of test nodes to explain")
    parser.add_argument("--dry-run", action="store_true", help="Perform sanity check with 3 epochs and 2 samples")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("SNA Topic C.10: End-to-End Explainable GNN Pipeline")
    print(f"Dataset: {args.dataset} | Model: {args.model} | Explainer: {args.explainer} | Seed: {args.seed}")
    print("=" * 70)

    torch.manual_seed(args.seed)

    # 1. Data Ingestion & Anti-Leakage Split
    print("[Step 1/6] Ingesting dataset with strict anti-leakage splitting...")
    if args.dataset == "ba_shapes":
        loader = BAShapesLoader(data_dir=str(DATA_DIR), random_seed=args.seed)
    elif args.dataset == "cora":
        loader = CoraLoader(data_dir=str(DATA_DIR), random_seed=args.seed)
    else:
        loader = TwitchLoader(data_dir=str(DATA_DIR), name="DE", random_seed=args.seed)

    data = loader.load()
    summary = loader.summary()
    print(f"Graph loaded: {summary['num_nodes']} nodes, {summary['num_edges']} edges, {summary['num_classes']} classes.")

    # 2. GNN Model Instantiation & Training
    print(f"[Step 2/6] Initializing and training {args.model.upper()} backbone...")
    in_dim = data.num_features
    out_dim = summary["num_classes"]

    if args.model == "graphsage":
        model = GraphSAGE(in_dim, 64, out_dim)
    elif args.model == "gat":
        model = GAT(in_dim, 16, out_dim, heads=4)
    else:
        model = GCN(in_dim, 64, out_dim)

    epochs = 3 if args.dry_run else args.epochs
    history = model.fit_model(data, epochs=epochs, lr=0.01, device=DEVICE)
    test_acc, test_loss = model.evaluate(data, data.test_mask)
    print(f"Training completed. Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Save Checkpoint
    ckpt_path = CHECKPOINT_DIR / f"{args.model}_{args.dataset}_seed{args.seed}.pt"
    model.save_checkpoint(ckpt_path)
    print(f"Saved frozen checkpoint to: {ckpt_path}")

    # 3. Explainer Configuration
    print(f"[Step 3/6] Setting up Explainability Engine ({args.explainer})...")
    num_samples = 2 if args.dry_run else args.samples
    test_indices = torch.where(data.test_mask)[0].tolist()[:num_samples]

    explainers_to_run = {}
    if args.explainer in ["all", "lt_def"]:
        explainers_to_run["LT-DEF (Proposed)"] = LTDEFExplainer(model=model, num_hops=2, sparsity_target=0.70)
    if args.explainer in ["all", "eig_search"]:
        explainers_to_run["EiG-Search"] = EigSearchExplainer(model=model, num_hops=2, sparsity_target=0.70)
    if args.explainer in ["all", "gnn_explainer"]:
        explainers_to_run["GNNExplainer"] = GNNExplainerWrapper(model=model, epochs=20 if args.dry_run else 50)
    if args.explainer in ["all", "subgraphx"]:
        explainers_to_run["SubgraphX"] = SubgraphXExplainer(model=model, num_hops=2, rollout_steps=5 if args.dry_run else 15)

    # 4. Quantitative Evaluation
    print(f"[Step 4/6] Benchmarking explainers over {len(test_indices)} test nodes...")
    evaluator = BenchmarkEvaluator(model=model, data=data, threshold=0.5)
    gt_motif_mask = getattr(data, "ground_truth_mask", None)

    summary_results = {}
    last_explanation = None

    for exp_name, exp_instance in explainers_to_run.items():
        print(f"  -> Running {exp_name}...")
        metrics = evaluator.evaluate_explainer(
            explainer=exp_instance,
            target_nodes=test_indices,
            ground_truth_motif_mask=gt_motif_mask,
        )
        summary_results[exp_name] = metrics
        # Keep an explanation for plotting and verbalization
        last_explanation = exp_instance.explain_node(test_indices[0], data.x, data.edge_index)

    df_results = BenchmarkEvaluator.format_results_table(summary_results)
    print("\n" + df_results.to_markdown(index=False) + "\n")

    # Save CSV Report
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_csv = tables_dir / f"benchmark_{args.dataset}_{args.model}.csv"
    df_results.to_csv(report_csv, index=False)
    print(f"Saved quantitative report table to: {report_csv}")

    # 5. Natural Language Grounding & Hallucination Audit
    print("[Step 5/6] Generating grounded Natural Language Explanation (NLE)...")
    if last_explanation is not None:
        verbalizer = NLEVerbalizer()
        nle_text = verbalizer.generate_explanation(last_explanation, data.x)
        print(f"\n[Generated NLE]:\n{nle_text}\n")

        audit_filter = RejectionFilter()
        is_acc, h_rate, issues = audit_filter.audit_explanation(nle_text, last_explanation)
        print(f"Audit Result: Accepted={is_acc} | Hallucination Rate: {h_rate:.2%}")
        if issues:
            print(f"Violations detected: {issues}")

    # 6. Visualization Export
    print("[Step 6/6] Rendering explanation subgraph figure...")
    if last_explanation is not None:
        plotter = GraphPlotter(output_dir=REPORTS_DIR / "figures")
        fig_path = REPORTS_DIR / "figures" / f"explanation_node_{test_indices[0]}.png"
        plotter.plot_explanation(last_explanation, data.edge_index, data.num_nodes, save_path=fig_path)
        print(f"Saved visualization figure to: {fig_path}")

    print("\nPipeline executed successfully without errors!")


if __name__ == "__main__":
    main()
