"""
Asset-only Regularized Mean Absolute Deviation optimizer — ASSETS-ZERO strategy
(Konno & Yamazaki 1991, in the framework of Lucescu & Ulrych 2026, Section 4.2).

ASSETS-ZERO is the *zero hedged* regularized asset portfolio: the asset weights
are optimized on fully hedged returns (R_fh) — the same asset-level problem as
ASSETS-FULL (Section 3.3 / Appendix A.3 of the paper, which defines the asset
optimization exclusively on R_fh) — and the portfolio is then implemented
WITHOUT currency hedging, so realized returns are unhedged (R_u).  The
realization on R_u is handled by the Backtester via its ``training_return_key``
/ ``asset_return_key`` split; this class only ever sees R_fh.

The risk measure is the mean absolute deviation of the portfolio return about
its own sample mean,

    MAD(w) = (1/T) * sum_t |a_t' w - abar' w| = (1/T) * sum_t |A_t w|,

where A is the columnwise demeaned return matrix.  It is written here in
Konno & Yamazaki's lower semi-absolute deviation form: for any sample with
zero mean the mass below the mean equals the mass above it, so

    (1/T) * sum_t |A_t w| = (2/T) * sum_t max(0, -A_t w),

and the right-hand side is linear once the positive part is lifted into a
T-vector of auxiliary variables u >= -A w, u >= 0.  The factor of two is
carried explicitly so the objective is reported on the MAD scale rather than
the semi-deviation scale, which matters because the penalties are calibrated
against the size of the risk term.

As in the mean-variance optimizer the objective carries no mean term.  The
sample mean is estimated far too imprecisely at these window lengths to be
worth its variance cost; dropping it leaves a pure regularized minimum-risk
problem and removes the need for a shrinkage estimator of the mean.  Unlike
mean-variance, nothing beyond the demeaning has to be estimated at all — MAD is
computed from the empirical distribution directly, with no covariance matrix and
so no shrinkage target (contrast Ledoit-Wolf in reg_mv_asset_zero).

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


class RegularizedMeanAbsoluteDeviationAssetZeroOptimizer(BaseOptimizer):
    """
    Regularized MAD asset optimizer for the ASSETS-ZERO strategy.

    Minimizes the mean absolute deviation of the fully hedged asset returns
    (R_fh) with L1 and L2 regularization penalties, subject to a leverage limit
    and a full investment constraint — identical to the ASSETS-FULL asset
    problem.  The zero-hedging policy is applied at realization: the Backtester
    evaluates the resulting weights on unhedged returns (R_u) and the investor
    holds no forward positions.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``lambda_l1``      (float) — L1 penalty weight, default 2.5e-4
                * ``lambda_l2``      (float) — L2 penalty weight, default 0.1
                * ``leverage_limit`` (float) — L1 norm constraint, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
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
        """
        Demean the fully hedged return matrix for optimization.

        Args:
            train_bundle: Must contain ``"R_fh"`` mapped to a non-empty DataFrame.
        """
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
        """
        Solve the regularized MAD minimization problem.

        Returns:
            1-D numpy array of shape *(N,)* summing to 1.

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
        """
        if self.A is None or self.N == 0:
            raise RuntimeError("fit() must be called before optimize().")

        x = cp.Variable(self.N)
        u = cp.Variable(self.T)

        l1_penalty = cp.norm(x, 1)
        # Factor 2 puts the semi-deviation on the MAD scale (see module docstring).
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
        """
        Evaluate the validation mean absolute deviation of the candidate weights.

        Scored on fully hedged returns, mirroring the training objective — the
        risk measure alone, without the penalties — exactly as the ES and MV
        asset-zero optimizers do.  Hyperparameter selection is part of the
        asset-level problem and is therefore identical to ASSETS-FULL.

        The deviation is taken about the validation window's own mean, matching
        ``np.var(r_val, ddof=1)`` in the mean-variance optimizer.  Demeaning the
        validation sample by itself also keeps the semi-deviation identity intact
        out of sample, so this score is on the same scale as the training
        objective.

        Args:
            weights: 1-D array of portfolio weights from :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"``.

        Returns:
            Mean absolute deviation of the realized portfolio return on the
            validation window (lower is better).
        """
        if "R_fh" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh'.")
        R_fh_val = val_bundle["R_fh"]
        if R_fh_val.empty or len(R_fh_val) < 2:
            return float("inf")
        r_val = R_fh_val.values @ weights
        return float(np.mean(np.abs(r_val - r_val.mean())))
