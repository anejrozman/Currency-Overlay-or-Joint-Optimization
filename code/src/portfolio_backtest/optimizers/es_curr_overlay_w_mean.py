"""Two-step currency overlay optimizer minimizing Expected Shortfall with a mean term.

Step 1 solves for the asset weights x on the fully hedged returns R_fh; step 2 fixes x and solves for the
overlay psi on R_fh @ x + R_c @ psi. x is fully invested with a leverage cap on ||x||_1. The objective
subtracts (1 / gamma) times the expected return; asset means are pooled by economy and James-Stein shrunk,
currency means are the carry implied by the forward premium. psi has an L1 cap (overlay_limit) and a zero
base-currency weight. ES is taken at confidence level alpha.
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np

from ..utils.estimation import (
    conditional_currency_mean,
    economy_ids_from_indicator,
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


class ExpectedShortfallCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    def __init__(self, hyperparams: dict | None = None) -> None:
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.gamma: float = self.hyperparams.get("gamma", 2)
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
        self.x_opt: np.ndarray | None = None

    def fit(self, train_bundle: dict[str, object]) -> None:
        """Extract the return matrices and estimate both mean vectors."""
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

    def _optimize_assets(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
        """Step 1: mean-ES on R_fh with a leverage cap and full investment."""
        q, N = Y_fh.shape
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        u = cp.Variable(q)
        eta = cp.Variable()

        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)
        objective = es_term - (1.0 / self.gamma) * (mu_a @ x)
        problem = cp.Problem(
            cp.Minimize(objective),
            [
                u >= 0,
                -Y_fh @ x - eta <= u,
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
        self,
        Y_fh: np.ndarray,
        Y_c: np.ndarray,
        x_fixed: np.ndarray,
        mu_a: np.ndarray,
        mu_c: np.ndarray,
    ) -> np.ndarray:
        """Step 2: mean-ES of (R_fh @ x_fixed + R_c @ psi) over psi, capped in L1."""
        q, K = Y_c.shape
        tail_prob = 1.0 - self.alpha

        c = Y_fh @ x_fixed  # constant asset return contribution, shape (q,)
        mean_assets = float(mu_a @ x_fixed)  # constant; carried for the objective value

        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()

        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)
        objective = es_term - (1.0 / self.gamma) * (mean_assets + mu_c @ psi)
        constraints = [
            u >= 0,
            -c - Y_c @ psi - eta <= u,
            cp.norm(psi, 1) <= self.overlay_limit,
        ]

        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        self._solve(cp.Problem(cp.Minimize(objective), constraints))

        overlay = psi.value
        if overlay is None:
            raise RuntimeError("Solver returned None weights in step 2.")
        return overlay.flatten()

    def optimize(self) -> dict[str, np.ndarray]:
        """Run both optimization steps sequentially."""
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")

        logger.debug(
            "Step 1: optimizing asset weights on %d obs, %d assets.", self.q, self.N
        )
        self.x_opt = self._optimize_assets(self.Y_fh, self.mu_a)

        logger.debug(
            "Step 2: optimizing currency overlay on %d obs, %d currencies.", self.q, self.K
        )
        psi_opt = self._optimize_overlay(
            self.Y_fh, self.Y_c, self.x_opt, self.mu_a, self.mu_c
        )

        return {"x": self.x_opt, "psi": psi_opt}

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """Validation ES minus mean return, after re-solving step 2 on the validation window with x fixed."""
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

        # step 2 is re-solved with the gamma-scaled objective, so it needs this window's means
        economy_ids = economy_ids_from_indicator(val_bundle.get("C"), Y_fh_val.shape[1])
        mu_a_val, _ = pooled_asset_mean(Y_fh_val, economy_ids)
        mu_c_val = conditional_currency_mean(val_bundle.get("F"), Y_c_val)

        psi_val = self._optimize_overlay(
            Y_fh_val, Y_c_val, self.x_opt, mu_a_val, mu_c_val
        )
        port_rets = Y_fh_val @ self.x_opt + Y_c_val @ psi_val

        threshold = np.quantile(port_rets, 1.0 - self.alpha)
        tail = port_rets[port_rets <= threshold]
        es = float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))

        return es - float(port_rets.mean())
