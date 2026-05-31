"""
Two-step Expected Shortfall optimizer with currency overlay.

Step 1: Minimizes ES over fully hedged asset returns (R_fh), with a leverage cap on asset weights.
Step 2: With asset weights fixed, minimizes ES of the combined portfolio return
        (R_fh @ x_opt + R_c @ psi) by optimizing the currency overlay weights psi.
        psi is unconstrained, psi_USD acts as a free weight that makes the overlay a zero investment hedge.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class ExpectedShortfallCurrencyOverlayOptimizer(BaseOptimizer):
    """
    Two-step unregularized ES optimizer with sequential currency overlay.

    Step 1 minimizes ES on R_fh subject to full investment and leverage constraints.
    Step 2 fixes the asset weights and minimizes ES of the combined portfolio return..
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — ES confidence level, default 0.80
                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
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
            train_bundle: Must contain ``"R_fh"`` and ``"R_c``.
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
        """Step 1 in optimization: unregularized ES on R_fh with leverage constrain and full investment."""
        q, N = Y_fh.shape
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        u = cp.Variable(q)
        eta = cp.Variable()

        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)
        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(es_term),
            [
                u >= 0,
                -Y_fh @ x - eta <= u,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        self._solve(problem)

        weights = x.value
        return (weights / weights.sum()).flatten()

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray
    ) -> np.ndarray:
        """Step 2 in optimization: minimize ES of (R_fh @ x_fixed + R_c @ psi) over unconstrained psi."""
        q, K = Y_c.shape
        tail_prob = 1.0 - self.alpha

        c = Y_fh @ x_fixed  # constant asset return contribution, shape is (q,)

        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()

        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)
        problem = cp.Problem(
            cp.Minimize(es_term),
            [
                u >= 0,
                -c - Y_c @ psi - eta <= u,
            ],
        )
        self._solve(problem)

        overlay = psi.value
        return overlay.flatten()

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run both optimization steps sequentially.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or any solver step fails.
            ValueError: If ``alpha`` is outside *(0, 1)*.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")

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
        Evaluate validation Expected Shortfall of the candidate weights.

        Uses ``self.x_opt`` (asset weights from :meth:`optimize`) as fixed, re-runs
        step 2 on the validation returns to get psi_val, then evaluates combined ES.

        Args:
            weights: Not used directly, asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Negative mean of the worst ``(1-alpha)`` fraction of combined portfolio returns.
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
        port_rets = Y_fh_val @ self.x_opt + Y_c_val @ psi_val

        threshold = np.quantile(port_rets, 1.0 - self.alpha)
        tail = port_rets[port_rets <= threshold]
        return float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))
