"""Separate Expected Shortfall optimizer — asset-only, no currency overlay (placeholder)."""

from __future__ import annotations

import numpy as np

from .base_optimizer import BaseOptimizer


class SeparateExpectedShortfallOptimizer(BaseOptimizer):
    """Separate ES optimizer: optimizes asset weights independently of currency."""

    def fit(self, train_bundle: dict[str, object]) -> None:
        pass

    def optimize(self) -> np.ndarray:
        pass

    def score(self, weights: np.ndarray, val_bundle: dict[str, object]) -> float:
        pass
