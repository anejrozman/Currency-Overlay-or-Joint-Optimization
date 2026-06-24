"""
Asset only Regularized Mean-Variance optimizer — ASSETS-ZERO strategy (Ulrych & Lucescu 2026, Section 3 and Ulrych & Burkhardt 2023).

Uses unhedged asset returns R_u. The investor makes no currency hedging
and optimizes only the asset allocation over unhedged returns.
The mean is estimated via the Bayes-Stein shrinkage estimator (Jorion 1986).

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer
from ..utils.estimation import bayes_stein, ledoit_wolf_cc

logger = logging.getLogger(__name__)


class RegularizedMeanVarianceAssetZeroOptimizer(BaseOptimizer):
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

                * ``gamma``          (float) — risk aversion coefficient, default 50.0
                * ``lambda_l1``      (float) — L1 penalty weight, default 1e-4
                * ``lambda_l2``      (float) — L2 penalty weight, default 1e-4
                * ``leverage_limit`` (float) — L1 norm constraint, default 1.5
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
        """
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 50.0)
        self.lambda_l1: float = self.hyperparams.get("lambda_l1", 1e-4)
        self.lambda_l2: float = self.hyperparams.get("lambda_l2", 1e-4)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 1.5)
        self.solver = self.hyperparams.get("solver", cp.GUROBI)

        self.mu: np.ndarray | None = None
        self.Sigma: np.ndarray | None = None
        self.N: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store the unhedged return matrix for optimization.

        Args:
            train_bundle: Must contain ``"R_u"``.
        """
        if "R_u" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_u'.")
        R_u = train_bundle["R_u"]
        if R_u.empty:
            raise ValueError("'R_u' in train_bundle is empty.")
        Y = R_u.values
        self.N = Y.shape[1]
        self.mu = bayes_stein(Y)
        self.Sigma = ledoit_wolf_cc(Y)  # Ledoit-Wolf shrinkage toward constant-correlation target (appendix B)

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
        Evaluate the validation mean-variance utility of the candidate weights.

        Returns the optimization objective ``(gamma/2) Var(r) - E[r]`` on the
        validation window, where ``r`` is the realized portfolio return. Lower is
        better. This matches the optimizer objective, so model selection trades off
        return against risk rather than minimizing variance alone.

        Args:
            weights: 1-D array of portfolio weights from :meth:`optimize`.
            val_bundle: Must contain ``"R_u"``.

        Returns:
            Mean-variance utility on the validation window (lower is better).
        """
        if "R_u" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_u'.")
        R_u_val = val_bundle["R_u"]
        if R_u_val.empty or len(R_u_val) < 2:
            return float("inf")
        r_val = R_u_val.values @ weights
        return float((self.gamma / 2) * np.var(r_val, ddof=1) - np.mean(r_val))
