"""
Asset-only Regularized Expected Shortfall optimizer — ASSETS-ZERO strategy (Ulrych & Lucescu 2026, Section 3).

Uses unhedged asset returns (R_u). The investor holds no forward positions in currencies. 

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class RegularizedExpectedShortfallAssetZeroOptimizer(BaseOptimizer):
    """
    Asset-only Regularized Expected Shortfall optimizer.

    Minimizes Expected Shortfall at a given confidence level with L1 and L2
    regularization penalties, subject to a leverage limit and a full investment constraint.  
    Uses unhedged asset returns (R_u) for optimization, so
    the investor bears full currency risk and holds no forward positions.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — ES confidence level, default 0.80
                * ``lambda_l1``      (float) — L1 penalty weight, default 0.0
                * ``lambda_l2``      (float) — L2 penalty weight, default 0.0
                * ``leverage_limit`` (float) — L1 norm constraint, default 2.0
                * ``solver``                 — cvxpy solver constant, default ``cp.GUROBI`
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.lambda_l1: float = self.hyperparams.get("lambda_l1", 0.0)
        self.lambda_l2: float = self.hyperparams.get("lambda_l2", 0.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", cp.GUROBI)

        self.Y: np.ndarray | None = None
        self.N: int = 0
        self.q: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store the unhedged return matrix for optimization.

        Args:
            train_bundle: Must contain ``"R_u"`` mapped to a non-empty DataFrame.
        """
        if "R_u" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_u'.")
        R_u = train_bundle["R_u"]
        if R_u.empty:
            raise ValueError("'R_u' in train_bundle is empty.")
        self.Y = R_u.values
        self.q, self.N = self.Y.shape

    def optimize(self) -> np.ndarray:
        """
        Solve the regularized ES minimization problem.

        Returns:
            1-D numpy array of shape *(N,)* summing to 1.

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
            ValueError: If ``alpha`` is outside *(0, 1)*.
        """
        if self.Y is None or self.N == 0:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")

        x = cp.Variable(self.N)
        u = cp.Variable(self.q)
        eta = cp.Variable()

        l1_penalty = cp.norm(x, 1)
        es_term = eta + (1.0 / (self.q * tail_prob)) * cp.sum(u)
        objective = cp.Minimize(
            es_term
            + self.lambda_l1 * l1_penalty
            + self.lambda_l2 * cp.sum_squares(x)
        )
        constraints = [
            u >= 0,
            -self.Y @ x - eta <= u,
            cp.sum(x) == 1,
            l1_penalty <= self.leverage_limit,
        ]
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
        if weights is None:
            raise RuntimeError("Solver returned None weights.")

        # Normalize to correct numerical inaccuracy from the solver
        return (weights / weights.sum()).flatten()

    def score(self, weights: np.ndarray, val_bundle: dict[str, object]) -> float:
        """
        Evaluate validation Expected Shortfall of the candidate weights.

        Args:
            weights: 1-D array of portfolio weights from :meth:`optimize`.
            val_bundle: Must contain ``"R_u"``.

        Returns:
            Negative mean of the worst ``(1-alpha)`` fraction of portfolio
            returns.
        """
        if "R_u" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_u'.")
        R_u_val = val_bundle["R_u"]
        if R_u_val.empty:
            return float("inf")
        port_rets = R_u_val.values @ weights
        threshold = np.quantile(port_rets, 1.0 - self.alpha)
        tail = port_rets[port_rets <= threshold]
        return float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))
