"""
Abstract base class for all portfolio optimizers in the framework.

Every optimizer that interacts with the ``Backtester`` must subclass
``BaseOptimizer`` and implement the  methods below.

Author: Anej Rozman
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseOptimizer(ABC):

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Dictionary of model hyperparameters (regularization
                         penalties, confidence levels, etc.).  Defaults to an empty
                         dict for unregularized / parameter-free models.
        """
        self.hyperparams: dict = hyperparams if hyperparams is not None else {}

    @abstractmethod
    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store the components needed for optimization.

        Args:
            train_bundle: Dictionary containing training data. 
        """

    @abstractmethod
    def optimize(self) -> np.ndarray | dict[str, np.ndarray]:
        """
        Execute the optimization algorithm and return the compuetd optimal portfolio weights.

        Returns:
            Either a 1-D np.ndarray of asset weights (for asset-only
            strategies) or a dictionary with keys "x" (asset weights) and
            "psi" (currency overlay weights) for joint strategies.
        """

    @abstractmethod
    def score(
        self,
        weights: np.ndarray | dict[str, np.ndarray],
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate portfolio performance on validation dataset.
        Used by the ``Backtester`` during hyperparameter grid search.

        Args:
            weights: The output from method optimize.
            val_bundle: Validation data in the same format as train_bundle.

        Returns:
            A scalar risk score.
        """
