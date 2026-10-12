"""Configuration package for SNA Topic C.10 Explainable GNN research."""

from src.config.settings import (
    BASE_DIR,
    CHECKPOINT_DIR,
    DATA_DIR,
    DEVICE,
    EVALUATION_CONFIG,
    EXPLAINER_CONFIG,
    MODEL_CONFIG,
    RANDOM_SEEDS,
    REPORTS_DIR,
    SPLIT_CONFIG,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "CHECKPOINT_DIR",
    "REPORTS_DIR",
    "RANDOM_SEEDS",
    "DEVICE",
    "MODEL_CONFIG",
    "EXPLAINER_CONFIG",
    "EVALUATION_CONFIG",
    "SPLIT_CONFIG",
]
