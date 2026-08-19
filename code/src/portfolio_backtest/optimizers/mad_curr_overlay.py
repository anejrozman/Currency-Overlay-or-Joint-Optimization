"""
Unregularized Mean Absolute Deviation optimizer with currency overlay.

Step 1: Minimizes the MAD of fully hedged asset returns (R_fh), subject to full
        investment and a leverage cap on asset weights.
Step 2: With asset weights fixed, minimizes the MAD of the combined portfolio
        return (R_fh @ x_opt + R_c @ psi) by optimizing the currency overlay
        weights psi.  psi is unconstrained; forward contracts are zero-investment
        instruments, so no slack weight is needed to fund the overlay.

Both steps use Konno & Yamazaki's lower semi-absolute deviation form: for a
sample with zero mean the mass below the mean equals the mass above it, so
mean|z| = 2 * mean(max(0, -z)), and the right-hand side is linear once the
positive part is lifted into auxiliary variables.  The factor of two is carried
explicitly so both objectives are reported on the MAD scale.

Demeaning in step 2 covers both terms.  Writing c = R_fh @ x_opt for the fixed
asset contribution, mean(c + R_c psi) = mean(c) + mean(R_c) psi, so

    MAD(x_opt, psi) = mean| (c - mean c) + (R_c - mean R_c) psi |,

i.e. the fixed contribution and the currency matrix are each demeaned before the
program is formed.  The objective carries no mean term, mirroring the MV and ES
overlays, which likewise minimize a risk measure alone.

The base currency's excess return is identically zero, so its overlay weight
never enters the objective and would be left unidentified by the LP.  It is
pinned to zero, as in the MV overlays, rather than left free.

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


class MeanAbsoluteDeviationCurrencyOverlayOptimizer(BaseOptimizer):
    """
    Two-step unregularized MAD optimizer with sequential currency overlay.

    Step 1 minimizes the MAD of R_fh subject to full investment and leverage
    constraints.  Step 2 fixes the asset weights and minimizes the MAD of the
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
        self.Y_fh = np.asarray(R_fh.values, dtype=float)
        self.Y_c = np.asarray(R_c.values, dtype=float)
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
        """Step 1: unregularized MAD on R_fh with a leverage constraint and full investment."""
        T, N = Y_fh.shape
        A = Y_fh - Y_fh.mean(axis=0)

        x = cp.Variable(N)
        u = cp.Variable(T)

        problem = cp.Problem(
            cp.Minimize((2.0 / T) * cp.sum(u)),
            [
                u >= 0,
                -A @ x <= u,
                cp.sum(x) == 1,
                cp.norm(x, 1) <= self.leverage_limit,
            ],
        )
        self._solve(problem)

        weights = x.value
        if weights is None:
            raise RuntimeError("Solver returned None weights in step 1.")
        return (weights / weights.sum()).flatten()

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray
    ) -> np.ndarray:
        """Step 2: minimize the MAD of (R_fh @ x_fixed + R_c @ psi) over unconstrained psi."""
        T, K = Y_c.shape
        c = Y_fh @ x_fixed                  # fixed asset contribution, shape (T,)
        c_dm = c - c.mean()                 # demeaned; see module docstring
        A_c = Y_c - Y_c.mean(axis=0)

        psi = cp.Variable(K)
        u = cp.Variable(T)

        constraints = [
            u >= 0,
            -(c_dm + A_c @ psi) <= u,
        ]
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(cp.Minimize((2.0 / T) * cp.sum(u)), constraints)
        self._solve(problem)

        overlay = psi.value
        if overlay is None:
            raise RuntimeError("Solver returned None weights in step 2.")
        return overlay.flatten()

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
        Evaluate the MAD of the combined portfolio on validation data.

        Uses ``self.x_opt`` (fixed from :meth:`optimize`), re-runs step 2 on the
        validation returns to get psi_val, then evaluates the MAD of the combined
        return.  Mirrors the unregularized ES and MV overlays, which likewise
        re-solve step 2 on the validation window.

        The deviation is taken about the validation window's own mean, matching
        the asset-level MAD optimizers.

        Args:
            weights: Not used; asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Mean absolute deviation of the combined portfolio return (lower is better).
        """
        if self.x_opt is None:
            raise RuntimeError("optimize() must be called before score().")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty or len(R_fh_val) < 2:
            return float("inf")

        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)

        psi_val = self._optimize_overlay(Y_fh_val, Y_c_val, self.x_opt)
        r_val = Y_fh_val @ self.x_opt + Y_c_val @ psi_val
        return float(np.mean(np.abs(r_val - r_val.mean())))
