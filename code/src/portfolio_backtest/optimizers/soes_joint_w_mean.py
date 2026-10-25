"""Joint asset and currency optimizer minimizing second-order Expected Shortfall with a mean term.

The asset weights x and the overlay psi are solved together on R_fh @ x + R_c @ psi. x is fully invested with
a leverage cap on ||x||_1. The objective subtracts (1 / gamma) times the expected return; asset means are
pooled by economy and James-Stein shrunk, currency means are the carry implied by the forward premium. psi has
an L1 cap (overlay_limit) and a zero base-currency weight. The risk measure is HMCR_2 of Krokhmal (2007) at
confidence level alpha.
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar

from ..utils.estimation import (
    conditional_currency_mean,
    economy_ids_from_indicator,
    pooled_asset_mean,
)
from ..utils.solver_fallback import solve_with_fallback

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

# Gurobi on the cluster, Clarabel where gurobipy is not installed
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False

# tolerances on |sum(x) - 1|: warn above the first, abort above the second
_BUDGET_WARN_TOL = 1e-6
_BUDGET_FAIL_TOL = 1e-4

# How far a fallback solution may exceed the leverage or overlay cap before it is
# rejected.  Matches the budget abort threshold.
_CAP_FAIL_TOL = 1e-4


class SecondOrderExpectedShortfallJointWithMeanOptimizer(BaseOptimizer):
    # One record per solve that needed a fallback solver, across every instance the
    # Backtester creates in this run; run_strategy.py writes it to the run directory.
    solver_fallbacks: list[dict] = []

    def __init__(self, hyperparams: dict | None = None) -> None:
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.gamma: float = self.hyperparams.get("gamma", 0.5)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.overlay_limit: float = self.hyperparams.get("overlay_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

        self.Y_fh: np.ndarray | None = None
        self.Y_c: np.ndarray | None = None
        self.economy_ids: np.ndarray | None = None
        self.mu_a: np.ndarray | None = None
        self.mu_c: np.ndarray | None = None
        self.N: int = 0
        self.K: int = 0
        self.q: int = 0
        self.window: tuple[str, str] | None = None

    def fit(self, train_bundle: dict[str, object]) -> None:
        """Extract the return matrices and estimate both mean vectors."""
        if "R_fh" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_fh'.")
        if "R_c" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_c'.")
        R_fh = train_bundle["R_fh"]
        R_c = train_bundle["R_c"]
        if R_fh.shape[0] != R_c.shape[0]:
            raise ValueError(
                f"R_fh and R_c must span the same window: got {R_fh.shape[0]} and "
                f"{R_c.shape[0]} observations."
            )
        self.Y_fh = np.ascontiguousarray(R_fh.values, dtype=float)
        self.Y_c = np.ascontiguousarray(R_c.values, dtype=float)
        self.q, self.N = self.Y_fh.shape
        self.K = self.Y_c.shape[1]
        self.window = (str(R_fh.index[0])[:10], str(R_fh.index[-1])[:10])

        self.economy_ids = economy_ids_from_indicator(train_bundle.get("C"), self.N)
        self.mu_a, w_a = pooled_asset_mean(self.Y_fh, self.economy_ids)
        self.mu_c = conditional_currency_mean(train_bundle.get("F"), self.Y_c)
        logger.debug(
            "Asset mean shrinkage w=%.4f (%d economies); carry mu_c gross=%.2e.",
            w_a, len(np.unique(self.economy_ids)), np.abs(self.mu_c).sum(),
        )

    def _solve(self, problem: cp.Problem, context: dict) -> str:
        """Solve ``problem`` and return the attempt that was accepted."""
        if self.solver == cp.GUROBI:
            return solve_with_fallback(
                problem,
                dict(
                    BarConvTol=1e-8,
                    # the conic counterpart of BarConvTol, as the ||u||_2 epigraph makes this an SOCP
                    BarQCPConvTol=1e-8,
                    Method=2,
                    ScaleFlag=0,
                    AggFill=0,
                    PreDual=1,
                    Presolve=1,
                    Threads=3,
                    FeasibilityTol=1e-9,
                    OptimalityTol=1e-9,
                ),
                context=context,
                events=type(self).solver_fallbacks,
                log=logger,
            )

        try:
            problem.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError as exc:
            raise RuntimeError(f"Solver error: {exc}") from exc

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Optimization failed: status={problem.status}")
        return str(self.solver)

    def _optimize_joint(
        self,
        Y_fh: np.ndarray,
        Y_c: np.ndarray,
        mu_a: np.ndarray,
        mu_c: np.ndarray,
        context: dict,
    ) -> dict[str, np.ndarray]:
        """Jointly minimize mean-HMCR_2 of (R_fh @ x + R_c @ psi) over x and psi."""
        q, N = Y_fh.shape
        K = Y_c.shape[1]
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()

        combined_return = Y_fh @ x + Y_c @ psi
        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(q))
        objective = soes_term - (1.0 / self.gamma) * (mu_a @ x + mu_c @ psi)
        l1_norm = cp.norm(x, 1)
        constraints = [
            u >= 0,
            -combined_return - eta <= u,
            cp.sum(x) == 1,                          # budget on the asset block only
            l1_norm <= self.leverage_limit,          # leverage on the asset block only
            cp.norm(psi, 1) <= self.overlay_limit,  # overlay cap
        ]

        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        attempt = self._solve(cp.Problem(cp.Minimize(objective), constraints), context)

        weights = x.value
        if weights is None:
            raise RuntimeError("Solver returned None for asset weights.")
        overlay = psi.value
        if overlay is None:
            raise RuntimeError("Solver returned None for currency overlay weights.")

        weights = np.asarray(weights, dtype=float).flatten()
        overlay = np.asarray(overlay, dtype=float).flatten()

        # sum(x) == 1 is a constraint; renormalizing would hide solver drift past the leverage cap
        budget_error = abs(weights.sum() - 1.0)
        if budget_error > _BUDGET_FAIL_TOL:
            raise RuntimeError(
                f"Budget constraint violated: sum(x) = {weights.sum():.9f}."
            )
        if budget_error > _BUDGET_WARN_TOL:
            logger.warning("sum(x) = %.9f, off budget by %.2e.", weights.sum(), budget_error)

        if attempt in ("gurobi_robust", "clarabel"):
            for name, gross, cap in (
                ("||x||_1", np.abs(weights).sum(), self.leverage_limit),
                ("||psi||_1", np.abs(overlay).sum(), self.overlay_limit),
            ):
                if gross > cap + _CAP_FAIL_TOL:
                    raise RuntimeError(
                        f"Fallback solution from {attempt} breaks the cap: "
                        f"{name} = {gross:.9f} > {cap}."
                    )

        return {"x": weights, "psi": overlay}

    def optimize(self) -> dict[str, np.ndarray]:
        """Run the joint optimization over training data."""
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")
        if not np.isfinite(self.overlay_limit) or self.overlay_limit <= 0.0:
            raise ValueError(
                "overlay_limit must be finite and positive: HMCR_2 is degree-one, so "
                "the joint program is unbounded below without it for gamma under ~0.026."
            )

        logger.debug(
            "Joint mean-HMCR_2 optimization on %d obs, %d assets, %d currencies.",
            self.q, self.N, self.K,
        )
        context = {
            "solve": "joint",
            "window_start": self.window[0],
            "window_end": self.window[1],
            "n_obs": self.q,
            "n_assets": self.N,
            "gamma": self.gamma,
        }
        return self._optimize_joint(self.Y_fh, self.Y_c, self.mu_a, self.mu_c, context)

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """Validation HMCR_2 minus mean return, after re-solving the joint program on the validation
        window.
        """
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty or len(R_fh_val) < 2:
            return float("inf")

        Y_fh_val = np.ascontiguousarray(R_fh_val.values, dtype=float)
        Y_c_val = np.ascontiguousarray(R_c_val.values, dtype=float)

        # the re-solve optimizes x and psi, so both means come from this window
        economy_ids = economy_ids_from_indicator(val_bundle.get("C"), Y_fh_val.shape[1])
        mu_a_val, _ = pooled_asset_mean(Y_fh_val, economy_ids)
        mu_c_val = conditional_currency_mean(val_bundle.get("F"), Y_c_val)

        context = {
            "solve": "joint, validation re-solve",
            "window_start": str(R_fh_val.index[0])[:10],
            "window_end": str(R_fh_val.index[-1])[:10],
            "n_obs": Y_fh_val.shape[0],
            "n_assets": Y_fh_val.shape[1],
            "gamma": self.gamma,
        }
        result = self._optimize_joint(Y_fh_val, Y_c_val, mu_a_val, mu_c_val, context)
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
        return float(result_min.fun) - float(port_rets.mean())
