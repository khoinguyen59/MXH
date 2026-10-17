"""Adapter Service for BA-Shapes Handover Artifacts (Alias).

Re-exports BAShapesHandoverAdapter and BAShapesHandoverBundle from src.t02_explainers.adapters.
"""

from src.t02_explainers.adapters.bashapes_handover_adapter import (
    BAShapesHandoverAdapter,
    BAShapesHandoverBundle,
)

__all__ = ["BAShapesHandoverAdapter", "BAShapesHandoverBundle"]
