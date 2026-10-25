"""1/N benchmark: equal asset weights and no currency overlay."""

from __future__ import annotations

import numpy as np
from .base_optimizer import BaseOptimizer


class EqualWeightOptimizer(BaseOptimizer):
    def __init__(self, hyperparams: dict | None = None) -> None:
        super().__init__(hyperparams)
        self.num_assets: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """Determine the number of assets from the training bundle."""
        R_fh = train_bundle.get("R_fh")
        if R_fh is None:
            raise ValueError("train_bundle must contain 'R_fh' (fully hedged asset returns).")
        self.num_assets = R_fh.shape[1]

    def optimize(self) -> np.ndarray:
        """Return equal weights for all assets."""
        if self.num_assets == 0:
            raise ValueError("Optimizer must be fitted before calling optimize().")
        return np.full(self.num_assets, 1.0 / self.num_assets)

    def score(self, weights: np.ndarray, val_bundle: dict[str, object]) -> float:
        """Return 0.0 since the 1/N portfolio has no hyperparameters to optimize over."""
        return 0.0
