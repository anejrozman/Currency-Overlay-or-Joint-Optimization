"""
Asset-only Regularized Expected Shortfall optimizer — ASSETS-ZERO strategy (Lucescu & Ulrych 2026, Section 4.2).

ASSETS-ZERO is the *zero hedged* regularized asset portfolio: the asset
weights are optimized on fully hedged returns (R_fh) — the same asset-level
problem as ASSETS-FULL (Section 3.3 / Appendix A.3 of the paper, which
defines the asset optimization exclusively on R_fh) — and the portfolio is
then implemented WITHOUT currency hedging, so realized returns are unhedged
(R_u).  The realization on R_u is handled by the Backtester via its
``training_return_key`` / ``asset_return_key`` split; this class only ever
sees R_fh.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

# Production runs use Gurobi (cluster). Fall back to Clarabel locally, where
# gurobipy is not installed; weight differences are ~1e-6.
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False


class RegularizedExpectedShortfallAssetZeroOptimizer(BaseOptimizer):
    """
    Regularized ES asset optimizer for the ASSETS-ZERO strategy.

    Minimizes Expected Shortfall of the fully hedged asset returns (R_fh) at
    a given confidence level with L1 and L2 regularization penalties, subject
    to a leverage limit and a full investment constraint — identical to the
    ASSETS-FULL asset problem.  The zero-hedging policy is applied at
    realization: the Backtester evaluates the resulting weights on unhedged
    returns (R_u) and the investor holds no forward positions.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — ES confidence level, default 0.80
                * ``lambda_l1``      (float) — L1 penalty weight, default 0.3
                * ``lambda_l2``      (float) — L2 penalty weight, default 0.3
                * ``leverage_limit`` (float) — L1 norm constraint, default 2.0
                * ``solver``                 — cvxpy solver constant, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.lambda_l1: float = self.hyperparams.get("lambda_l1", 0.3)
        self.lambda_l2: float = self.hyperparams.get("lambda_l2", 0.3)
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
        """
        Extract and store the fully hedged return matrix for optimization.

        Args:
            train_bundle: Must contain ``"R_fh"`` mapped to a non-empty DataFrame.
        """
        if "R_fh" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_fh'.")
        R_fh = train_bundle["R_fh"]
        if R_fh.empty:
            raise ValueError("'R_fh' in train_bundle is empty.")
        self.Y = R_fh.values
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

        Scored on fully hedged returns, mirroring the training objective
        (hyperparameter selection is part of the asset-level problem and is
        therefore identical to ASSETS-FULL).

        Args:
            weights: 1-D array of portfolio weights from :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"``.

        Returns:
            Negative mean of the worst ``(1-alpha)`` fraction of portfolio
            returns.
        """
        if "R_fh" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh'.")
        R_fh_val = val_bundle["R_fh"]
        if R_fh_val.empty:
            return float("inf")
        port_rets = R_fh_val.values @ weights
        threshold = np.quantile(port_rets, 1.0 - self.alpha)
        tail = port_rets[port_rets <= threshold]
        return float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))
