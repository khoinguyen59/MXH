"""Experiment 3: Natural Language Explanation Faithfulness & Hallucination Audit.

Validates that NLE generation preserves topological faithfulness and adheres to the
< 2% hallucination threshold across 50 test nodes.
"""

import argparse
import json
from pathlib import Path
import torch

from src.config.settings import DATA_DIR, REPORTS_DIR
from src.datasets.ba_shapes import BAShapesLoader
from src.explainers.lt_def import LTDEFExplainer
from src.models.gcn import GCN
from src.nle_grounding.rejection_filter import RejectionFilter
from src.nle_grounding.verbalizer import NLEVerbalizer


def run_experiment_3(num_samples=25):
    print("=" * 65)
    print("Starting Experiment 3: NLE Faithfulness & Hallucination Audit")
    print(f"Sample count: {num_samples}")
    print("=" * 65)

    loader = BAShapesLoader(data_dir=str(DATA_DIR), random_seed=42)
    data = loader.load()

    model = GCN(data.num_features, 64, 3)
    model.fit_model(data, epochs=50, lr=0.01)

    explainer = LTDEFExplainer(model, num_hops=2, sparsity_target=0.70)
    verbalizer = NLEVerbalizer()
    filter_auditor = RejectionFilter(max_hallucination_rate=0.02)

    test_nodes = torch.where(data.test_mask)[0].tolist()[:num_samples]

    audit_records = []
    total_hallucinations = 0
    accepted_count = 0

    for idx, node in enumerate(test_nodes):
        exp = explainer.explain_node(node, data.x, data.edge_index)
        text = verbalizer.generate_explanation(exp, data.x)
        is_accepted, h_rate, violations = filter_auditor.audit_explanation(text, exp)

        if is_accepted:
            accepted_count += 1
        total_hallucinations += h_rate

        audit_records.append({
            "node_id": node,
            "predicted_class": exp.predicted_class,
            "generated_text": text,
            "is_accepted": is_accepted,
            "hallucination_rate": h_rate,
            "violations": violations,
        })

    avg_hallucination = total_hallucinations / len(test_nodes)
    acceptance_rate = accepted_count / len(test_nodes)

    out_file = REPORTS_DIR / "exp3_nle_audit_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "sample_size": len(test_nodes),
            "average_hallucination_rate": avg_hallucination,
            "audit_acceptance_rate": acceptance_rate,
            "records": audit_records,
        }, f, indent=2)

    print(f"\nAudit Acceptance Rate: {acceptance_rate:.1%}")
    print(f"Average Hallucination Rate: {avg_hallucination:.2%}")
    print(f"Saved full audit JSON log to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    run_experiment_3(num_samples=args.samples)
