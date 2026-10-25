"""Regularized ASSETS-FULL optimizer minimizing mean absolute deviation.

The asset weights x are optimized on fully hedged returns, with all currency risk hedged. x is fully invested
with a leverage cap on ||x||_1. L1 and L2 penalties on x (lambda_l1, lambda_l2). MAD is written in the lower
semi-absolute deviation form of Konno & Yamazaki (1991).
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

# Gurobi on the cluster, Clarabel where gurobipy is not installed
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False


class RegularizedMeanAbsoluteDeviationAssetFullOptimizer(BaseOptimizer):
    def __init__(self, hyperparams: dict | None = None) -> None:
        super().__init__(hyperparams)
        self.lambda_l1: float = self.hyperparams.get("lambda_l1", 2.5e-4)
        self.lambda_l2: float = self.hyperparams.get("lambda_l2", 0.1)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

        self.A: np.ndarray | None = None
        self.a_bar: np.ndarray | None = None
        self.N: int = 0
        self.T: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """Demean the fully hedged return matrix for optimization."""
        if "R_fh" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_fh'.")
        R_fh = train_bundle["R_fh"]
        if R_fh.empty:
            raise ValueError("'R_fh' in train_bundle is empty.")
        Y = np.asarray(R_fh.values, dtype=float)
        self.T, self.N = Y.shape
        self.a_bar = Y.mean(axis=0)
        self.A = Y - self.a_bar

    def optimize(self) -> np.ndarray:
        """Solve the regularized MAD minimization problem."""
        if self.A is None or self.N == 0:
            raise RuntimeError("fit() must be called before optimize().")

        x = cp.Variable(self.N)
        u = cp.Variable(self.T)

        l1_penalty = cp.norm(x, 1)
        # the factor 2 puts the lower semi-deviation on the MAD scale
        mad_term = (2.0 / self.T) * cp.sum(u)
        objective = cp.Minimize(
            mad_term
            + self.lambda_l1 * l1_penalty
            + self.lambda_l2 * cp.sum_squares(x)
        )
        constraints = [
            u >= 0,
            -self.A @ x <= u,
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

    def score(
        self,
        weights: np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """Validation MAD of the training-fitted weights."""
        if "R_fh" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh'.")
        R_fh_val = val_bundle["R_fh"]
        if R_fh_val.empty or len(R_fh_val) < 2:
            return float("inf")
        r_val = R_fh_val.values @ weights
        return float(np.mean(np.abs(r_val - r_val.mean())))
