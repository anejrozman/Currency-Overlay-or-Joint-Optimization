"""
Asset-only Regularized Mean-Variance optimizer — ASSETS-FULL strategy (Ulrych & Lucescu 2026, Section 3 and Ulrych & Burkhardt 2023).

Uses fully hedged asset returns R_fh. The investor hedges all
currency risk via forward contracts and optimizes only the asset allocation.
The mean is estimated via the Bayes-Stein shrinkage estimator (Jorion 1986).

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


def bayes_stein(returns: np.ndarray) -> np.ndarray:
    """
    Bayes-Stein shrinkage estimator for the mean.

    Shrinks each assets sample mean toward the minimum variance portfolio
    mean with a shrinkage intensity.

    Args:
        (T, N) array of asset returns.

    Returns:
        1-D array of shape *(N,)* with shrunk mean estimates.
    """
    T, N = returns.shape
    mu = returns.mean(axis=0)
    try:
        Sigma = np.cov(returns, rowvar=False, ddof=1)
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return mu

    ones = np.ones(N)
    denom = float(ones @ Sigma_inv @ ones)
    mu_g = float(ones @ Sigma_inv @ mu) / denom

    diff = mu - mu_g * ones
    delta = float(T * (diff @ Sigma_inv @ diff))
    phi = (N + 2) / ((N + 2) + delta)

    return (1.0 - phi) * mu + phi * mu_g * ones


class RegularizedMeanVarianceAssetFullOptimizer(BaseOptimizer):
    """
    Asset only Regularized Mean-Variance optimizer.

    Minimizes a penalized mean-variance objective with L1 and L2
    regularization penalties, subject to a leverage limit and a full investment
    constraint. 
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``          (float) — risk aversion coefficient, default 3.0
                * ``lambda_l1``      (float) — L1 penalty weight, default 0.3
                * ``lambda_l2``      (float) — L2 penalty weight, default 0.3
                * ``leverage_limit`` (float) — L1 norm constraint, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
        """
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 3.0)
        self.lambda_l1: float = self.hyperparams.get("lambda_l1", 0.3)
        self.lambda_l2: float = self.hyperparams.get("lambda_l2", 0.3)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", cp.GUROBI)

        self.mu: np.ndarray | None = None
        self.Sigma: np.ndarray | None = None
        self.N: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store the fully hedged return matrix for optimization.

        Args:
            train_bundle: Must contain ``"R_fh"``.
        """
        if "R_fh" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_fh'.")
        R_fh = train_bundle["R_fh"]
        if R_fh.empty:
            raise ValueError("'R_fh' in train_bundle is empty.")
        Y = R_fh.values
        self.N = Y.shape[1]
        self.mu = bayes_stein(Y)
        self.Sigma = np.cov(Y, rowvar=False, ddof=1)

    def optimize(self) -> np.ndarray:
        """
        Solve the regularized mean-variance minimization problem.

        Returns:
            1-D numpy array of shape *(N,)* summing to 1.

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
        """
        if self.mu is None or self.N == 0:
            raise RuntimeError("fit() must be called before optimize().")

        x = cp.Variable(self.N)
        l1 = cp.norm(x, 1)

        objective = cp.Minimize(
            (self.gamma / 2) * cp.quad_form(x, cp.psd_wrap(self.Sigma))
            - self.mu @ x
            + self.lambda_l1 * l1
            + self.lambda_l2 * cp.sum_squares(x)
        )
        constraints = [cp.sum(x) == 1, l1 <= self.leverage_limit]
        problem = cp.Problem(objective, constraints)

        try:
            if self.solver == cp.GUROBI:
                problem.solve(
                    solver=cp.GUROBI,
                    verbose=False,
                    BarConvTol=1e-8,
                    Method=2,
                    ScaleFlag=0,
                    AggFill=0,
                    PreDual=1,
                    Presolve=1,
                    Threads=3,
                    FeasibilityTol=1e-9,
                    OptimalityTol=1e-9,
                )
            else:
                problem.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError as exc:
            raise RuntimeError(f"Solver error: {exc}") from exc

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Optimization failed: status={problem.status}")

        weights = x.value

        return (weights / weights.sum()).flatten()

    def score(
        self,
        weights: np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate validation portfolio variance of the candidate weights.

        Args:
            weights: 1-D array of portfolio weights from :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"``.

        Returns:
            Portfolio variance ``w' Sigma_val w`` on the validation window.
        """
        if "R_fh" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh'.")
        R_fh_val = val_bundle["R_fh"]
        if R_fh_val.empty or len(R_fh_val) < 2:
            return float("inf")
        Sigma_val = np.cov(R_fh_val.values, rowvar=False, ddof=1)
        return float(weights @ Sigma_val @ weights)
