"""Joint asset and currency optimizer minimizing variance with a mean term.

The asset weights x and the overlay psi are solved together on R_fh @ x + R_c @ psi. x is fully invested with
a leverage cap on ||x||_1. The objective subtracts (1 / gamma) times the expected return; asset means are
pooled by economy and James-Stein shrunk, currency means are the carry implied by the forward premium. psi has
an L1 cap (overlay_limit) and a zero base-currency weight. Covariances are Ledoit-Wolf shrunk.
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np

from ..utils.estimation import (
    conditional_currency_mean,
    economy_ids_from_indicator,
    ledoit_wolf_cc,
    pooled_asset_mean,
)

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


class MeanVarianceJointWithMeanOptimizer(BaseOptimizer):
    def __init__(self, hyperparams: dict | None = None) -> None:
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 30)
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

        self.economy_ids = economy_ids_from_indicator(train_bundle.get("C"), self.N)
        self.mu_a, w_a = pooled_asset_mean(self.Y_fh, self.economy_ids)
        self.mu_c = conditional_currency_mean(train_bundle.get("F"), self.Y_c)
        logger.debug(
            "Asset mean shrinkage w=%.4f (%d economies); carry mu_c gross=%.2e.",
            w_a, len(np.unique(self.economy_ids)), np.abs(self.mu_c).sum(),
        )

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

    def _optimize_joint(
        self,
        Y_fh: np.ndarray,
        Y_c: np.ndarray,
        mu_a: np.ndarray,
        mu_c: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Jointly minimize mean-variance of (R_fh @ x + R_c @ psi) over x and psi."""
        N = Y_fh.shape[1]
        K = Y_c.shape[1]
        Y = np.hstack([Y_fh, Y_c])  # q x (N + K) combined return matrix
        Sigma_joint = ledoit_wolf_cc(Y)

        x = cp.Variable(N)
        psi = cp.Variable(K)
        theta = cp.hstack([x, psi])
        objective = (
            cp.quad_form(theta, cp.psd_wrap(Sigma_joint))
            - (1.0 / self.gamma) * (mu_a @ x + mu_c @ psi)
        )

        constraints = [
            cp.sum(x) == 1,                          # budget on the asset block only
            cp.norm(x, 1) <= self.leverage_limit,    # leverage on the asset block only
            cp.norm(psi, 1) <= self.overlay_limit,  # overlay cap
        ]
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        self._solve(cp.Problem(cp.Minimize(objective), constraints))

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

        return {"x": weights, "psi": overlay}

    def optimize(self) -> dict[str, np.ndarray]:
        """Run the joint optimization over training data."""
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")

        logger.debug(
            "Joint mean-variance optimization on %d obs, %d assets, %d currencies.",
            self.q, self.N, self.K,
        )
        return self._optimize_joint(self.Y_fh, self.Y_c, self.mu_a, self.mu_c)

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """Validation standard deviation minus mean return, after re-solving the joint program on the
        validation window.
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

        result = self._optimize_joint(Y_fh_val, Y_c_val, mu_a_val, mu_c_val)
        port_rets = Y_fh_val @ result["x"] + Y_c_val @ result["psi"]

        return float(np.std(port_rets, ddof=1)) - float(port_rets.mean())
