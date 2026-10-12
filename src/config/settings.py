"""Master configuration settings for SNA Topic C.10.

Defines reproducible random seeds, directory paths, hardware execution device,
model hyperparameters, explainer settings, and evaluation metrics.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import torch

# Base project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
REPORTS_DIR = BASE_DIR / "product" / "reports"
PROPOSAL_DIR = BASE_DIR / "product" / "proposal"
SLIDES_DIR = BASE_DIR / "product" / "slides"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
PAPERS_DIR = BASE_DIR / "docs" / "papers"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Mandatory multi-seed reproducibility standard (Invariant-09)
RANDOM_SEEDS = [42, 123, 456, 789, 999]
DEFAULT_SEED = 42

# Device selection (CUDA accelerated if available, CPU fallback)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anti-leakage data split proportions (Special Lecture 01 standard)
SPLIT_CONFIG = {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "stratify": True,
}

# GNN Backbone architectures and training hyperparameters
MODEL_CONFIG = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "max_epochs": 200,
    "patience": 20,
    "architectures": ["GCN", "GraphSAGE", "GAT"],
    "gat_heads": 4,
}

# Explainability Engine algorithms configuration (International XAI Standard)
EXPLAINER_CONFIG = {
    "core_explainers": ["GNNExplainer", "PGExplainer", "Saliency"],
    "reference_baselines": ["Random", "NearestFirst"],
    "advanced_explainers": ["SubgraphX", "EiGSearch", "LT-DEF"],
    "budget_k_grid": [1, 2, 3, 5, 10, 15, 20],
    "budget_pct_grid": [0.05, 0.10, 0.20, 0.30, 0.50],
    "gnn_explainer": {
        "epochs": 100,
        "lr": 0.01,
    },
    "pg_explainer": {
        "epochs": 30,
        "lr": 0.005,
    },
    "saliency": {
        "num_hops": 2,
    },
    "subgraphx": {
        "num_hops": 2,
        "rollout_steps": 10,
        "min_nodes": 4,
    },
    "eig_search": {
        "num_eigenvectors": 8,
        "alpha": 0.5,
    },
    "lt_def": {
        "top_k_edges": 10,
        "feature_percentile": 80,
        "eigen_pruning_threshold": 0.15,
    },
}

# Quantitative evaluation metrics and target criteria
EVALUATION_CONFIG = {
    "fidelity_types": ["fidelity_plus", "fidelity_minus"],
    "sparsity_target": 0.70,
    "disagreement_metrics": ["kappa_at_k", "jaccard_similarity", "rank_correlation"],
    "statistical_testing": {
        "alpha": 0.05,
        "tost_equivalence_bound": 0.05,
    },
    "runtime_benchmark_repeats": 5,
}

