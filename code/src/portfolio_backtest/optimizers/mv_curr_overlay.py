"""
Unregularized Mean-Variance optimizer with currency overlay.

Step 1: Minimizes the variance of fully hedged asset returns (R_fh), subject to
        full investment and a leverage cap on asset weights.
Step 2: With asset weights fixed, minimizes the variance of the combined portfolio
        return (R_fh @ x_opt + R_c @ psi) by optimizing the currency overlay weights psi.
        psi is unconstrained; forward contracts are zero-investment instruments, so no
        slack weight is needed to fund the overlay.

The objective carries no mean term.  The variance coefficient is normalized to one,
which is the gamma -> infinity limit of the paper's objective after dividing through
by gamma / 2: the mean term vanishes and the cross-covariance term picks up its
factor of two directly from the expansion of Var(x' R_fh + psi' R_c).  This mirrors
the ES optimizers, which likewise minimize a risk measure alone, and removes the need
for a shrinkage estimator of the mean (Bayes-Stein, Jorion 1986).

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer
from ..utils.estimation import ledoit_wolf_cc, sample_cross_cov

logger = logging.getLogger(__name__)

# Production runs use Gurobi (cluster). Fall back to Clarabel locally, where
# gurobipy is not installed; weight differences are ~1e-6.
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False


class MeanVarianceCurrencyOverlayOptimizer(BaseOptimizer):
    """
    Unregularized mean-variance optimizer with separate currency overlay.

    Step 1 minimizes the variance of R_fh subject to full investment and leverage
    constraints. Step 2 fixes the asset weights and minimizes the variance of the
    combined portfolio return.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

        self.Y_fh: np.ndarray | None = None
        self.Y_c: np.ndarray | None = None
        self.N: int = 0
        self.K: int = 0
        self.q: int = 0
        self.x_opt: np.ndarray | None = None

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store return matrices for both optimization steps.

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

    def _optimize_assets(self, Y_fh: np.ndarray) -> np.ndarray:
        """Step 1 in optimization: minimum variance on R_fh with Ledoit-Wolf
        covariance, leverage constraint, and full investment."""
        N = Y_fh.shape[1]
        Sigma_a = ledoit_wolf_cc(Y_fh)

        x = cp.Variable(N)
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(x, cp.psd_wrap(Sigma_a))),
            [
                cp.sum(x) == 1,
                cp.norm(x, 1) <= self.leverage_limit,
            ],
        )
        self._solve(problem)

        weights = x.value
        return (weights / weights.sum()).flatten()

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray
    ) -> np.ndarray:
        """Step 2 in optimization: minimum-variance currency overlay over
        unconstrained psi.

        Minimizes psi' Sigma_c psi + 2 x*' Sigma_ac psi, the expansion of
        Var(x*' R_fh + psi' R_c) with the constant asset-only term dropped
        (appendix B, eq:mv_curr_expanded). Sigma_c is Ledoit-Wolf shrunk; the
        cross-covariance Sigma_ac is the un-shrunk sample estimate.
        """
        K = Y_c.shape[1]
        Sigma_c = ledoit_wolf_cc(Y_c)
        Sigma_ac = sample_cross_cov(Y_fh, Y_c)  # N x K, not shrunk
        cross = 2.0 * (x_fixed @ Sigma_ac)  # length-K constant vector

        psi = cp.Variable(K)
        obj = cp.quad_form(psi, cp.psd_wrap(Sigma_c)) + cross @ psi
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
        constraints = []
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)
        problem = cp.Problem(cp.Minimize(obj), constraints)
        self._solve(problem)

        return psi.value.flatten()

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run both optimization steps sequentially.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or any solver step fails.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        logger.debug(
            "Step 1: optimizing asset weights on %d obs, %d assets.", self.q, self.N
        )
        self.x_opt = self._optimize_assets(self.Y_fh)

        logger.debug(
            "Step 2: optimizing currency overlay on %d obs, %d currencies.", self.q, self.K
        )
        psi_opt = self._optimize_overlay(self.Y_fh, self.Y_c, self.x_opt)

        return {"x": self.x_opt, "psi": psi_opt}

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the variance of the combined portfolio on validation data.

        Uses ``self.x_opt`` (fixed from :meth:`optimize`), re-runs step 2 on validation
        returns to get psi_val, then evaluates ``var(r_val, ddof=1)``. Mirrors the
        unregularized ES overlay, which likewise re-solves step 2 on the validation
        window.

        Args:
            weights: Not used, asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Variance of the combined portfolio return (lower is better).
        """
        if self.x_opt is None:
            raise RuntimeError("optimize() must be called before score().")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty:
            return float("inf")

        Y_fh_val = R_fh_val.values
        Y_c_val = R_c_val.values

        psi_val = self._optimize_overlay(Y_fh_val, Y_c_val, self.x_opt)
        r_val = Y_fh_val @ self.x_opt + Y_c_val @ psi_val
        return float(np.var(r_val, ddof=1))
