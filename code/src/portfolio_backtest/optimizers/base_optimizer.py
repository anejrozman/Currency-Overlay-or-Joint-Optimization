"""Interface of the optimizers run by the Backtester."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    def __init__(self, hyperparams: dict | None = None) -> None:
        self.hyperparams: dict = hyperparams if hyperparams is not None else {}

    @abstractmethod
    def fit(self, train_bundle: dict[str, object]) -> None:
        """Store what the optimizer needs from a training bundle."""

    @abstractmethod
    def optimize(self) -> np.ndarray | dict[str, np.ndarray]:
        """Asset weights, or {"x": asset weights, "psi": currency overlay}."""

    @abstractmethod
    def score(
        self,
        weights: np.ndarray | dict[str, np.ndarray],
        val_bundle: dict[str, object],
    ) -> float:
        """Validation score of the weights, lower is better; used by the Backtester's grid search."""
