# Unit and Integration Test Suite (`tests/`)

This directory houses the automated test suite ensuring mathematical validity, anti-leakage compliance, and regression safety across all pipeline modules.

## Test Matrix

| Test Module | Verification Scope | Target Invariant |
| :--- | :--- | :--- |
| `test_anti_leakage.py` | Validates strictly disjoint train/val/test partitions (zero index intersection) and confirms feature scalers are fitted exclusively on training nodes. | Special Lecture 01 / Invariant-08 |
| `test_models.py` | Tests forward pass tensor dimensions, log-probability ranges, and latent embedding extraction for GCN, GraphSAGE, and GAT. | Invariant-05 |
| `test_explainers.py` | Verifies attribution masks for LT-DEF, EiG-Search, GNNExplainer, Saliency, Random, and Nearest-First explainers. | Invariant-02 / Invariant-05 |
| `test_metrics.py` | Verifies mathematical boundary conditions for Fidelity+, Fidelity-, Sparsity, Motif-AUC, Hallucination rate, Fidelity curves, Chance-Corrected Kappa, k*, Variance ANOVA, and Holm/TOST tests. | Invariant-07 / Invariant-09 |
| `test_datasets.py` | Verifies synthetic BA-Shapes graph generation, ground-truth motif attachment, and feature tensor dimensions. | Invariant-07 / Invariant-08 |
| `test_nle_grounding.py` | Verifies GraphSerializer prompt construction, NLEVerbalizer generation, and RejectionFilter topological hallucination detection. | Invariant-07 / Invariant-10 |

## Running Tests

Execute the full suite using pytest:
```cmd
cmd /c ".venv\Scripts\python.exe -m pytest tests/ -v"
```
