"""
Unregularized Mean-Variance optimizer with currency overlay.

Step 1: Minimizes the mean-variance objective over fully hedged asset returns (R_fh),
        subject to full investment and a leverage cap on asset weights.
Step 2: With asset weights fixed, minimizes the mean-variance objective of the
        combined portfolio return (R_fh @ x_opt + R_c @ psi) by optimizing the
        currency overlay weights psi.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class MeanVarianceCurrencyOverlayOptimizer(BaseOptimizer):
    """
    Unregularized mean-variance optimizer with separate currency overlay.

    Step 1 minimizes the mean-variance objective on R_fh subject to full investment and leverage constraints.
    Step 2 fixes the asset weights and minimizes the mean-variance objective of the combined portfolio return.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``          (float) — risk aversion coefficient, default 3.0
                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
        """
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 3.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", cp.GUROBI)

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
        """Step 1 in optimization: unregularized mean-variance on R_fh with leverage constraint and full investment."""
        q, N = Y_fh.shape

        x = cp.Variable(N)
        r = Y_fh @ x
        mean_r = cp.sum(r) / q
        var_r = cp.sum_squares(r - mean_r) / (q - 1)
        mv_obj = (self.gamma / 2) * var_r - mean_r
        problem = cp.Problem(
            cp.Minimize(mv_obj),
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
        """Step 2 in optimization: minimize mean-variance of (R_fh @ x_fixed + R_c @ psi) over unconstrained psi."""
        q, K = Y_c.shape

        c = Y_fh @ x_fixed  
        psi = cp.Variable(K)
        r = c + Y_c @ psi
        mean_r = cp.sum(r) / q
        var_r = cp.sum_squares(r - mean_r) / (q - 1)
        mv_obj = (self.gamma / 2) * var_r - mean_r
        problem = cp.Problem(cp.Minimize(mv_obj))
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
        Evaluate the mean-variance objective on validation data.

        Uses ``self.x_opt`` (fixed from :meth:`optimize`), re-runs step 2 on validation
        returns to get psi_val, then evaluates ``(gamma/2) * var(r_val, ddof=1) - mean(r_val)``.

        Args:
            weights: Not used, asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Mean-variance objective of the portfolio.
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
        return float((self.gamma / 2) * np.var(r_val, ddof=1) - np.mean(r_val))
