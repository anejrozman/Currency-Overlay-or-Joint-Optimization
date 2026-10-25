"""Regularized ASSETS-ZERO optimizer minimizing second-order Expected Shortfall.

The asset weights x are optimized on fully hedged returns and realized unhedged (R_u), which the Backtester
handles. x is fully invested with a leverage cap on ||x||_1. L1 and L2 penalties on x (lambda_l1, lambda_l2).
The risk measure is HMCR_2 of Krokhmal (2007) at confidence level alpha.
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

# Gurobi on the cluster, Clarabel where gurobipy is not installed
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False


class RegularizedSecondOrderExpectedShortfallAssetZeroOptimizer(BaseOptimizer):
    def __init__(self, hyperparams: dict | None = None) -> None:
        super().__init__(hyperparams)
        # defaults are interior grid points, so a --no-grid run is regularized
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.lambda_l1: float = self.hyperparams.get("lambda_l1", 1e-2)
        self.lambda_l2: float = self.hyperparams.get("lambda_l2", 1.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

        self.Y: np.ndarray | None = None
        self.N: int = 0
        self.q: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """Extract and store the fully hedged return matrix for optimization."""
        if "R_fh" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_fh'.")
        R_fh = train_bundle["R_fh"]
        if R_fh.empty:
            raise ValueError("'R_fh' in train_bundle is empty.")
        self.Y = R_fh.values
        self.q, self.N = self.Y.shape

    def optimize(self) -> np.ndarray:
        """Solve the regularized second-order ES minimization problem."""
        if self.Y is None or self.N == 0:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")

        x = cp.Variable(self.N)
        u = cp.Variable(self.q)
        eta = cp.Variable()

        l1_penalty = cp.norm(x, 1)
        # ||u||_2 / sqrt(q) is the empirical ||(L - eta)_+||_2, unsquared so the measure stays coherent
        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(self.q))
        objective = cp.Minimize(
            soes_term
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
                    # the conic counterpart of BarConvTol, as the ||u||_2 epigraph makes this an SOCP
                    BarQCPConvTol=1e-8,
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
        """Validation HMCR_2 of the training-fitted weights."""
        if "R_fh" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh'.")
        R_fh_val = val_bundle["R_fh"]
        if R_fh_val.empty:
            return float("inf")

        tail_prob = 1.0 - self.alpha
        losses = -(R_fh_val.values @ weights)
        t_val = losses.shape[0]

        def _hmcr2_of_zeta(zeta: float) -> float:
            excess = np.maximum(losses - zeta, 0.0)
            return zeta + np.linalg.norm(excess) / (tail_prob * np.sqrt(t_val))

        result = minimize_scalar(
            _hmcr2_of_zeta,
            bounds=(float(losses.min()), float(losses.max())),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(result.fun)
