"""
Two-step unregularized Second-Order Expected Shortfall optimizer with currency
overlay (Krokhmal 2007, in the framework of Ulrych & Lucescu 2026, Section 3).

Step 1: Minimizes HMCR_2 over fully hedged asset returns (R_fh), with a leverage
        cap on asset weights and full investment.
Step 2: With asset weights fixed, minimizes HMCR_2 of the combined portfolio
        return (R_fh @ x_opt + R_c @ psi) by optimizing the currency overlay
        weights psi.  psi is unconstrained: forward contracts are zero-investment
        instruments, so no slack weight is needed to fund the overlay.

The risk measure, the unsquared L2 tail norm, and the sqrt(q) normalization are
identical to reg_soes_asset_full.py; see that module's docstring for why the
norm is not squared and why the normalization differs from ES.  Each step here
minimizes the same functional

    eta + ||u||_2 / ((1 - alpha) * sqrt(q)),   u = max(0, -(loss) - eta),

over its own decision variable (x in step 1, psi in step 2) with the other
fixed, exactly mirroring how es_curr_overlay.py sequences its two ES solves.
Both steps are SOCPs rather than LPs, for the same reason reg_soes_asset_full.py
is an SOCP rather than an LP: the epigraph of ||u||_2 is a second-order cone
constraint. ``optimal_inaccurate`` is correspondingly more likely here than for
the ES overlay's two LPs.

Base currency: R_c's base-currency column is identically zero (its excess
return over itself is zero by construction), so psi_base never appears in
either the SOC constraint or the objective of step 2, in either step's problem.
That is exactly the situation in es_curr_overlay.py's step 2 (a zero column in
an LP), not the one in mv/mad/smv_curr_overlay.py, where psi_base sits in a
quadratic term and an explicit pin is needed to keep the QP's Hessian
nonsingular in that direction. Step 2 here has no quadratic term in psi at all,
so the same non-issue applies as for ES: psi_base is left free rather than
pinned to zero.

With ``demean`` set, both return matrices are centered columnwise before either
step is solved, and the validation window is centered by its own column means
before step 2 is re-solved there. HMCR_2 is then measured on deviations from the
sample mean rather than on raw returns, which puts it on the same footing as the
deviation measures (mean-variance, MAD, semi-variance), all of which demean by
construction. The flag and its treatment are identical to es_curr_overlay.py's;
the switch is ``DEMEAN`` in backtest_config.py, off by default, mirrored into the
parameter grid there as ``DEMEAN_GRID`` and shared with the ES optimizers.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar

from portfolio_backtest.backtest_config import DEMEAN

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


class SecondOrderExpectedShortfallCurrencyOverlayOptimizer(BaseOptimizer):
    """
    Two-step unregularized second-order ES optimizer with sequential currency
    overlay.

    Step 1 minimizes HMCR_2 on R_fh subject to full investment and leverage
    constraints. Step 2 fixes the asset weights and minimizes HMCR_2 of the
    combined portfolio return.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — confidence level, default 0.80
                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
                * ``demean``         (bool)  — center R_fh and R_c columnwise before
                                               solving, default
                                               ``backtest_config.DEMEAN``
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.demean: bool = bool(self.hyperparams.get("demean", DEMEAN))
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
        if self.demean:
            self.Y_fh = self.Y_fh - self.Y_fh.mean(axis=0)
            self.Y_c = self.Y_c - self.Y_c.mean(axis=0)
        self.q, self.N = self.Y_fh.shape
        self.K = self.Y_c.shape[1]

    def _solve(self, problem: cp.Problem) -> None:
        try:
            if self.solver == cp.GUROBI:
                problem.solve(
                    solver=cp.GUROBI,
                    verbose=False,
                    BarConvTol=1e-8,
                    # BarConvTol does not govern the conic path; this is its
                    # QCP counterpart, and the SOC epigraph of ||u||_2 makes
                    # both steps SOCPs rather than the LPs the ES overlay solves.
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

    def _optimize_assets(self, Y_fh: np.ndarray) -> np.ndarray:
        """Step 1: unregularized HMCR_2 on R_fh with a leverage constraint and
        full investment."""
        q, N = Y_fh.shape
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        u = cp.Variable(q)
        eta = cp.Variable()

        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(q))
        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(soes_term),
            [
                u >= 0,
                -Y_fh @ x - eta <= u,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
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
        """Step 2: minimize HMCR_2 of (R_fh @ x_fixed + R_c @ psi) over
        unconstrained psi. psi_base is left free; see module docstring."""
        q, K = Y_c.shape
        tail_prob = 1.0 - self.alpha

        c = Y_fh @ x_fixed  # constant asset return contribution, shape (q,)

        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()

        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(q))
        problem = cp.Problem(
            cp.Minimize(soes_term),
            [
                u >= 0,
                -c - Y_c @ psi - eta <= u,
            ],
        )
        self._solve(problem)

        overlay = psi.value
        if overlay is None:
            raise RuntimeError("Solver returned None weights in step 2.")
        return overlay.flatten()

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run both optimization steps sequentially.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

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
        Evaluate validation second-order ES of the candidate weights.

        Uses ``self.x_opt`` (asset weights from :meth:`optimize`) as fixed,
        re-runs step 2 on the validation returns to get psi_val — mirroring the
        unregularized ES and SMV overlays, which likewise re-solve step 2 on the
        validation window — then evaluates the combined HMCR_2 independently of
        the eta the step-2 solve produced.

        That independent re-derivation is necessary, not merely mirrored: at
        p = 1 (the ES overlay) the empirical VaR of the realized returns is
        exactly the minimizing threshold regardless of how psi was obtained, so
        reading the score off a quantile is always exact. At p = 2 the
        minimizing eta is only exactly recovered by the step-2 solve itself
        (see reg_soes_asset_full.py for why VaR is not a valid substitute), and
        this SOCP is more prone to ``optimal_inaccurate`` than the ES overlay's
        LPs. Recomputing zeta with a bounded 1-D search over the realized
        combined returns — the same numerical approach used in
        reg_soes_asset_full.py / reg_soes_asset_zero.py's score() — verifies the
        score against the realized data directly rather than trusting the
        solver's eta.value under a status that may be inexact.

        Under ``demean`` the validation matrices are centered by their own column
        means, not the training ones: the score has to be the deviation HMCR_2 of
        this window, so the location it deviates from is this window's mean. The
        zeta search then runs on the centered losses, and its bracket
        (losses.min(), losses.max()) still contains the minimizer — HMCR_2 is
        translation-equivariant, so centering shifts both zeta and the score by
        the same constant.

        Args:
            weights: Not used; asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Second-order ES of the combined portfolio return on the validation
            window (lower is better).
        """
        if self.x_opt is None:
            raise RuntimeError("optimize() must be called before score().")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty:
            return float("inf")

        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)
        if self.demean:
            Y_fh_val = Y_fh_val - Y_fh_val.mean(axis=0)
            Y_c_val = Y_c_val - Y_c_val.mean(axis=0)

        psi_val = self._optimize_overlay(Y_fh_val, Y_c_val, self.x_opt)
        port_rets = Y_fh_val @ self.x_opt + Y_c_val @ psi_val

        tail_prob = 1.0 - self.alpha
        losses = -port_rets
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
