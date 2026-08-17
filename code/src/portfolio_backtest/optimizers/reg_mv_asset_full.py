"""
Asset-only Regularized Mean-Variance optimizer — ASSETS-FULL strategy (Ulrych & Lucescu 2026, Section 3 and Ulrych & Burkhardt 2023).

Uses fully hedged asset returns R_fh: the investor hedges all
currency risk via forward contracts and optimizes only the asset allocation.

The objective carries no mean term.  The sample mean is estimated far too
imprecisely at these window lengths to be worth its variance cost: with it in
the objective the program spends its whole risk budget chasing the highest
estimated mean, collapsing onto a handful of names at the leverage bound.
Dropping it leaves a regularized minimum-variance problem, mirroring the
ES optimizers, which likewise penalize a risk measure alone.  This also removes
the need for a shrinkage estimator of the mean (Bayes-Stein, Jorion 1986); only
the covariance is estimated, via Ledoit-Wolf shrinkage toward the
constant-correlation target (Appendix B).

With the mean gone the risk-aversion coefficient gamma would be pure scaling on
the risk term — exactly equivalent to dividing both penalties by gamma/2 — so it
is dropped as a redundant knob and the penalty grids carry the tradeoff alone.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer
from ..utils.estimation import ledoit_wolf_cc

logger = logging.getLogger(__name__)

# Production runs use Gurobi (cluster). Fall back to Clarabel locally, where
# gurobipy is not installed; weight differences are ~1e-6.
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False


class RegularizedMeanVarianceAssetFullOptimizer(BaseOptimizer):
    """
    Asset-only Regularized Mean-Variance optimizer (ASSETS-FULL strategy).

    Minimizes portfolio variance with L1 and L2 regularization penalties,
    subject to a leverage limit and a full-investment constraint.  Uses only
    fully hedged asset returns (R_fh) for optimization.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``lambda_l1``      (float) — L1 penalty weight, default 5e-6
                * ``lambda_l2``      (float) — L2 penalty weight, default 1e-3
                * ``leverage_limit`` (float) — L1 norm constraint, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.lambda_l1: float = self.hyperparams.get("lambda_l1", 5e-6)
        self.lambda_l2: float = self.hyperparams.get("lambda_l2", 1e-3)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

        self.Sigma: np.ndarray | None = None
        self.N: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Estimate the covariance of the fully hedged returns for optimization.

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
        self.Sigma = ledoit_wolf_cc(Y)  # Ledoit-Wolf shrinkage toward constant-correlation target (appendix B)

    def optimize(self) -> np.ndarray:
        """
        Solve the regularized minimum-variance problem.

        Returns:
            1-D numpy array of shape *(N,)* summing to 1.

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
        """
        if self.Sigma is None or self.N == 0:
            raise RuntimeError("fit() must be called before optimize().")

        x = cp.Variable(self.N)
        l1 = cp.norm(x, 1)

        objective = cp.Minimize(
            cp.quad_form(x, cp.psd_wrap(self.Sigma))
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
        if weights is None:
            raise RuntimeError("Solver returned None weights.")

        # Normalize to correct numerical inaccuracy from the solver
        return (weights / weights.sum()).flatten()

    def score(
        self,
        weights: np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the validation variance of the candidate weights.

        Mirrors the training objective — the risk measure alone, without the
        penalties — exactly as the ES optimizers score on validation ES.

        Args:
            weights: 1-D array of portfolio weights from :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"``.

        Returns:
            Variance of the realized portfolio return on the validation
            window (lower is better).
        """
        if "R_fh" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh'.")
        R_fh_val = val_bundle["R_fh"]
        if R_fh_val.empty or len(R_fh_val) < 2:
            return float("inf")
        r_val = R_fh_val.values @ weights
        return float(np.var(r_val, ddof=1))
