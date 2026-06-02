"""
Regularized Joint Mean-Variance optimizer over asset and currency weights.

Minimizes a regularized MV objective of the combined portfolio return (R_fh @ x + R_c @ psi)
jointly over both asset weights x and currency overlay weights psi in a single optimization
problem. Bayes-Stein mean shrinkage is applied separately to assets and currencies.
Separate L1 and L2 penalties are applied to x and psi. Leverage cap on x only, psi is unconstrained.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer
from ..utils.estimation import bayes_stein

logger = logging.getLogger(__name__)


class RegularizedMeanVarianceJointOptimizer(BaseOptimizer):
    """
    Joint regularized mean-variance optimizer over asset and currency overlay weights.

    Solves a single CVXPY problem minimizing a regularized MV objective of (R_fh @ x + R_c @ psi)
    subject to full investment and leverage constraints on x. psi is unconstrained.
    Bayes-Stein shrinkage is applied separately to asset and currency means.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``           (float) — risk aversion coefficient, default 3.0
                * ``lambda_l1_asset`` (float) — L1 penalty for asset weights, default 0.3
                * ``lambda_l2_asset`` (float) — L2 penalty for asset weights, default 0.3 (floored at 1e-8)
                * ``lambda_l1_curr``  (float) — L1 penalty for currency overlay, default 0.3
                * ``lambda_l2_curr``  (float) — L2 penalty for currency overlay, default 0.3 (floored at 1e-8)
                * ``leverage_limit``  (float) — ||x||_1 cap, default 2.0
                * ``solver``                  — cvxpy solver, default ``cp.GUROBI``
        """
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 3.0)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 0.3)
        self.lambda_l2_asset: float = max(self.hyperparams.get("lambda_l2_asset", 0.3), 1e-8)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.3)
        self.lambda_l2_curr: float = max(self.hyperparams.get("lambda_l2_curr", 0.3), 1e-8)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", cp.GUROBI)

        self.Y_fh: np.ndarray | None = None
        self.Y_c: np.ndarray | None = None
        self.N: int = 0
        self.K: int = 0
        self.q: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store return matrices for joint optimization.

        Args:
            train_bundle: Must contain ``"R_fh"`` and ``"R_c"``.
        """
        if "R_fh" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_fh'.")
        if "R_c" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_c'.")
        R_fh = train_bundle["R_fh"]
        R_c = train_bundle["R_c"]
        self.Y_fh = R_fh.values
        self.Y_c = R_c.values
        self.q, self.N = self.Y_fh.shape
        self.K = self.Y_c.shape[1]

    def _solve(self, problem: cp.Problem) -> None:
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

    def _optimize_joint(
        self, Y_fh: np.ndarray, Y_c: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Jointly minimize regularized MV objective of (R_fh @ x + R_c @ psi) over x and psi."""
        q, N = Y_fh.shape
        K = Y_c.shape[1]
        mu = bayes_stein(Y_fh)
        mu_c = bayes_stein(Y_c)

        x = cp.Variable(N)
        psi = cp.Variable(K)
        r = Y_fh @ x + Y_c @ psi
        mean_r = cp.sum(r) / q
        var_r = cp.sum_squares(r - mean_r) / (q - 1)
        obj = (
            (self.gamma / 2) * var_r
            - mu @ x
            - mu_c @ psi
            + self.lambda_l1_asset * cp.norm(x, 1)
            + self.lambda_l2_asset * cp.sum_squares(x)
            + self.lambda_l1_curr * cp.norm(psi, 1)
            + self.lambda_l2_curr * cp.sum_squares(psi)
        )
        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(obj),
            [
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        self._solve(problem)

        weights = x.value
        if weights is None:
            raise RuntimeError("Solver returned None for asset weights.")
        overlay = psi.value
        if overlay is None:
            raise RuntimeError("Solver returned None for currency overlay weights.")

        return {
            "x": (weights / weights.sum()).flatten(),
            "psi": overlay.flatten(),
        }

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run the joint regularized optimization over training data.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        logger.debug(
            "Joint regularized optimization on %d obs, %d assets, %d currencies.",
            self.q, self.N, self.K,
        )
        return self._optimize_joint(self.Y_fh, self.Y_c)

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the mean-variance objective by rerunning the joint optimization on val data.

        Args:
            weights: Not used, joint optimization is rerun on val data.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Mean-variance objective.
        """
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty:
            return float("inf")

        Y_fh_val = R_fh_val.values
        Y_c_val = R_c_val.values

        result = self._optimize_joint(Y_fh_val, Y_c_val)
        r_val = Y_fh_val @ result["x"] + Y_c_val @ result["psi"]
        return float((self.gamma / 2) * np.var(r_val, ddof=1) - np.mean(r_val))
