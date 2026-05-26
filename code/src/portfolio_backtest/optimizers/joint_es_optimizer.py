"""Joint Expected Shortfall optimizer with currency overlay (placeholder)."""

from __future__ import annotations

import numpy as np

from .base_optimizer import BaseOptimizer


class JointExpectedShortfallOptimizer(BaseOptimizer):
    """Joint ES optimizer: jointly optimizes asset weights and currency overlay."""

    def fit(self, train_bundle: dict[str, object]) -> None:
        pass

    def optimize(self) -> np.ndarray:
        pass

    def score(self, weights: np.ndarray, val_bundle: dict[str, object]) -> float:
        pass
