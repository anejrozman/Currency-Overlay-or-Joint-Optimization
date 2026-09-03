"""
Unregularized Joint Second-Order Expected Shortfall optimizer over asset and
currency weights (Krokhmal 2007, in the framework of Ulrych & Lucescu 2026,
Section 3).

Minimizes HMCR_2 of the combined portfolio return (R_fh @ x + R_c @ psi)
jointly over both asset weights x and currency overlay weights psi in a
single optimization scheme. No regularization. Leverage cap and budget on
the asset block only; psi is unconstrained.

The risk measure — the unsquared L2 tail norm and its sqrt(q) normalization —
and the reasoning for why this is an SOCP rather than an LP are identical to
reg_soes_asset_full.py; see that module's docstring for the full derivation.
Relative to soes_curr_overlay.py, this collapses the two sequential steps
(fixed x, then psi) into one joint program over (x, psi): both blocks enter
the same SOC constraint and are optimized together, rather than the overlay's
sequence of first optimizing x alone and then psi with x held fixed. This
mirrors how es_joint.py relates to es_curr_overlay.py.

Base currency: as in soes_curr_overlay.py and es_joint.py (and unlike
mv/mad/smv_joint.py), psi_base is left free rather than pinned to zero. R_c's
base-currency column is identically zero, so psi_base never enters the SOC
constraint or the objective regardless of its value; the QP-conditioning issue
that forces the mean-variance-type joint optimizers to pin it does not arise
here, since neither block enters through a quadratic term.

Unlike es_joint.py (and the other unregularized ES/joint optimizers, which
hardcode ``cp.GUROBI`` with no local fallback), this follows the convention
already used by the other four SOES optimizers built alongside it
(reg_soes_asset_full.py, reg_soes_asset_zero.py, soes_curr_overlay.py,
reg_soes_curr_overlay.py): fall back to Clarabel when gurobipy is unavailable,
so this class is usable locally without always overriding ``solver`` by hand.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar
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


class SecondOrderExpectedShortfallJointOptimizer(BaseOptimizer):
    """
    Joint unregularized second-order ES optimizer over asset and currency
    overlay weights.

    Solves a single CVXPY problem minimizing HMCR_2 of (R_fh @ x + R_c @ psi)
    subject to full investment and leverage constraints on x. psi is
    unconstrained.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — confidence level, default 0.80
                * ``leverage_limit`` (float) — ||x||_1 cap, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
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

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store return matrices for joint optimization.

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
                    # BarConvTol does not govern the conic path; this is its
                    # QCP counterpart, and the SOC epigraph of ||u||_2 makes
                    # this an SOCP rather than the LP the ES joint program solves.
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

    def _optimize_joint(
        self, Y_fh: np.ndarray, Y_c: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Jointly minimize HMCR_2 of (R_fh @ x + R_c @ psi) over x and psi.
        psi_base is left free; see module docstring."""
        q, N = Y_fh.shape
        K = Y_c.shape[1]
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()

        combined_return = Y_fh @ x + Y_c @ psi
        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(q))
        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(soes_term),
            [
                u >= 0,
                -combined_return - eta <= u,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        self._solve(problem)

        weights = x.value
        if weights is None:
            raise RuntimeError("Solver returned None for asset weights.")
        overlay = psi.value
        if overlay is None:
            raise RuntimeError("Solver returned None for currency overlay weights.")

        return {
            "x": (weights / weights.sum()).flatten(),
            "psi": overlay.flatten(),
        }

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run the joint optimization over training data.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
            ValueError: If ``alpha`` is outside *(0, 1)*.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")

        logger.debug(
            "Joint optimization on %d obs, %d assets, %d currencies.",
            self.q, self.N, self.K,
        )
        return self._optimize_joint(self.Y_fh, self.Y_c)

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate validation second-order ES by rerunning the joint optimization
        on validation data.

        Mirrors es_joint.py's score(), which likewise re-solves the joint
        program on the validation window rather than holding the training fit
        fixed. Unlike es_joint.py, the resulting HMCR_2 cannot be read off a
        quantile of the realized returns: the p = 1 VaR shortcut that makes
        that exact for ES does not carry over to p = 2 (see
        reg_soes_asset_full.py). The threshold zeta is instead solved for
        numerically with the same bounded 1-D search used throughout the SOES
        family.

        Args:
            weights: Not used, joint optimization is rerun on val data.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Second-order ES of the combined portfolio return on the validation
            sample (lower is better).
        """
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty:
            return float("inf")

        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)

        result = self._optimize_joint(Y_fh_val, Y_c_val)
        port_rets = Y_fh_val @ result["x"] + Y_c_val @ result["psi"]

        tail_prob = 1.0 - self.alpha
        losses = -port_rets
        t_val = losses.shape[0]

        def _hmcr2_of_zeta(zeta: float) -> float:
            excess = np.maximum(losses - zeta, 0.0)
            return zeta + np.linalg.norm(excess) / (tail_prob * np.sqrt(t_val))

        result_min = minimize_scalar(
            _hmcr2_of_zeta,
            bounds=(float(losses.min()), float(losses.max())),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(result_min.fun)
