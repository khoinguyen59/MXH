# SNA Topic C.10 - Experimental Benchmark Suite

This directory contains standalone, reproducible experiment runners conforming to the course governance specification (`AGENTS.md`) and Special Lecture 01 (TS. Tran Hung Nghiep).

---

## Experiments Overview

| Runner / Script | Purpose & Benchmark | Output Artifact |
| :--- | :--- | :--- |
| [`run_phase1_experiments.py`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/experiments/run_phase1_experiments.py) | Master Exploring Experiments: Disagreement Problem (kappa@k, Jaccard, Tau), 3-way ANOVA multi-seed variance decomposition, and minimal edge budget k*. | `experiments/results/phase1/exploring_experiments_summary.json` |
| [`exp1_ba_shapes_motif.py`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/experiments/exp1_ba_shapes_motif.py) | Validates ground-truth motif discovery and Motif-AUC on BA-Shapes across 5 fixed seeds (`[42, 123, 456, 789, 999]`). | `product/reports/exp1_ba_shapes_motif_results.csv` |
| [`exp2_pareto_fidelity_sparsity.py`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/experiments/exp2_pareto_fidelity_sparsity.py) | Compares Pareto trade-off between Fidelity+ and Sparsity across LT-DEF, GNNExplainer, SubgraphX, and EiG-Search on Cora and Twitch. | `product/reports/exp2_pareto_results.csv` |
| [`exp3_nle_faithfulness.py`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/experiments/exp3_nle_faithfulness.py) | Assesses Natural Language Explanation faithfulness, hallucination rate (< 2% target), and human readability. | `product/reports/exp3_nle_audit_results.json` |
| [`pilot_study/`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/experiments/pilot_study) | Initial exploratory pilot benchmark runners, logs, and checkpoints on core & tier-2 datasets. | `experiments/pilot_study/logs/` |

---

## Directory Taxonomy

```text
experiments/
├── exp1_ba_shapes_motif.py          # Motif ground-truth validation runner
├── exp2_pareto_fidelity_sparsity.py # Pareto Fidelity-Sparsity frontier runner
├── exp3_nle_faithfulness.py         # NLE semantic faithfulness audit runner
├── run_phase1_experiments.py        # Master Phase 1 exploring experiments runner
├── pilot_study/                     # Pilot study benchmarks, checkpoints, and logs
│   ├── run_pilot_experiment.py      # Core datasets pilot benchmark runner
│   ├── run_tier2_experiments.py     # Tier-2 datasets pilot benchmark runner
│   ├── checkpoints/                 # Pilot frozen checkpoints
│   └── logs/                        # Pilot execution logs (JSON)
├── results/                         # Aggregated summary JSON/CSV outputs
│   └── phase1/                      # Phase 1 summary outputs
└── logs/                            # Full experiment run logs
```

---

## Execution Guidelines

Always run experiments with fixed random seeds (`[42, 123, 456, 789, 999]`):
```cmd
cmd /c ".venv\Scripts\python.exe -m experiments.run_phase1_experiments"
cmd /c ".venv\Scripts\python.exe -m experiments.exp1_ba_shapes_motif --seeds 42 123 456 789 999"
cmd /c ".venv\Scripts\python.exe -m experiments.exp2_pareto_fidelity_sparsity --dataset cora"
cmd /c ".venv\Scripts\python.exe -m experiments.exp3_nle_faithfulness"
```
