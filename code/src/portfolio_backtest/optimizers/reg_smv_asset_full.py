"""
Asset-only Regularized Semi-Variance optimizer — ASSETS-FULL strategy
(Markowitz, Todd, Xu & Yamane 1993, in the framework of Ulrych & Lucescu 2026, Section 3).

Uses fully hedged asset returns R_fh: the investor hedges all currency risk via
forward contracts and optimizes only the asset allocation.

The risk measure is the semi-variance of the portfolio return about its own
sample mean — Markowitz et al.'s S_E, eq. (3.1a), with the target E set to the
portfolio's own expected return rather than an exogenous benchmark:

    SMV(w) = (1/T) * sum_t [ max(0, -A_t w) ]^2,

where A is the columnwise demeaned return matrix, so A_t w is the portfolio's
deviation from its own mean on day t and only the downside deviations are
counted.  Semi-variance is variance's one-sided counterpart: for a symmetric
return distribution the two coincide up to a factor of two, and they part
company exactly to the extent the sample is skewed, which is the point of
carrying it alongside the mean-variance program.

Their eqs. (3.4)-(3.7) lift the positive part into auxiliary variables to make
the objective quadratic.  The paper introduces 2T of them — Y for the signed
deviation and Z = Y^- for its negative part, tied together by BX - Y + Z = 0 —
because the Critical Line Algorithm needs the constraint set in equality form.
Only the downside half is needed here: minimizing sum z_t^2 subject to
z >= -A w and z >= 0 drives each z_t to max(0, -A_t w) on its own, so the
equality and the free variables Y drop out and the program is a QP in
(w, z) with T auxiliary variables rather than 2T.  It is solved directly by
an interior-point solver rather than by the Critical Line Algorithm: CLA traces
the whole E-S frontier, whereas one point of it is wanted per rebalancing date,
and the L1 penalty and leverage cap below are not in CLA's form to begin with.

The 1/sqrt(T) that eq. (3.8) folds into B is carried as an explicit 1/T on the
objective instead, which leaves A on the scale of the raw returns and keeps the
risk term numerically better conditioned.

As in the mean-variance and MAD optimizers the objective carries no mean term.
The sample mean is estimated far too imprecisely at these window lengths to be
worth its variance cost; dropping it leaves a pure regularized minimum-risk
problem and removes the need for a shrinkage estimator of the mean.  As with
MAD, and unlike mean-variance, nothing beyond the demeaning has to be estimated:
the semi-variance is read off the empirical distribution directly, with no
covariance matrix and so no shrinkage target (contrast Ledoit-Wolf in
reg_mv_asset_full).  Note that this is what makes the mean-free objective
coherent here — the demeaning still uses the sample mean, but only to locate the
threshold that separates downside from upside, not as a quantity the optimizer
trades risk against.

The risk term is reported as the semi-variance itself, not doubled onto the
variance scale, so :meth:`score` returns a number directly comparable with the
semi-variance quoted elsewhere.  It runs about half the size of the mean-variance
risk term on the same windows, which is why the penalties carry their own grids
(LAMBDA_L1_SMV_ASSET_GRID / LAMBDA_L2_SMV_ASSET_GRID in backtest_config.py)
rather than reusing the mean-variance ones.

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


class RegularizedSemiVarianceAssetFullOptimizer(BaseOptimizer):
    """
    Asset-only Regularized Semi-Variance optimizer (ASSETS-FULL strategy).

    Minimizes the semi-variance of the portfolio return about its own sample
    mean with L1 and L2 regularization penalties, subject to a leverage limit
    and a full-investment constraint.  Uses only fully hedged asset returns
    (R_fh) for optimization.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``lambda_l1``      (float) — L1 penalty weight, default 1e-5
                * ``lambda_l2``      (float) — L2 penalty weight, default 1.5e-3
                * ``leverage_limit`` (float) — L1 norm constraint, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        # Defaults are the interior points of the calibrated grids, so a --no-grid
        # run lands in the regularized regime rather than at an untested extreme.
        self.lambda_l1: float = self.hyperparams.get("lambda_l1", 1e-5)
        self.lambda_l2: float = self.hyperparams.get("lambda_l2", 1.5e-3)
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
        Solve the regularized semi-variance minimization problem.

        Returns:
            1-D numpy array of shape *(N,)* summing to 1.

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
        """
        if self.A is None or self.N == 0:
            raise RuntimeError("fit() must be called before optimize().")

        x = cp.Variable(self.N)
        z = cp.Variable(self.T)

        l1_penalty = cp.norm(x, 1)
        # z is driven to the downside deviation max(0, -A_t x) by the two
        # constraints below (see module docstring), so this is the semi-variance.
        smv_term = cp.sum_squares(z) / self.T
        objective = cp.Minimize(
            smv_term
            + self.lambda_l1 * l1_penalty
            + self.lambda_l2 * cp.sum_squares(x)
        )
        constraints = [
            z >= 0,
            -self.A @ x <= z,
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
        Evaluate the validation semi-variance of the candidate weights.

        Mirrors the training objective — the risk measure alone, without the
        penalties — exactly as the ES, MV and MAD optimizers score on validation
        ES, variance and MAD.

        The deviation is taken about the validation window's own mean, as in the
        MAD optimizer, so the downside/upside split is located by the same rule
        in and out of sample and the score is on the same scale as the training
        objective.

        Args:
            weights: 1-D array of portfolio weights from :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"``.

        Returns:
            Semi-variance of the realized portfolio return on the validation
            window (lower is better).
        """
        if "R_fh" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh'.")
        R_fh_val = val_bundle["R_fh"]
        if R_fh_val.empty or len(R_fh_val) < 2:
            return float("inf")
        r_val = R_fh_val.values @ weights
        downside = np.minimum(r_val - r_val.mean(), 0.0)
        return float(np.mean(downside ** 2))
