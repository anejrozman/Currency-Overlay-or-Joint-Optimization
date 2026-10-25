"""Joint asset and currency optimizer minimizing second-order Expected Shortfall.

The asset weights x and the overlay psi are solved together on R_fh @ x + R_c @ psi. x is fully invested with
a leverage cap on ||x||_1. psi is otherwise unconstrained, the base-currency weight included. The risk measure
is HMCR_2 of Krokhmal (2007) at confidence level alpha. With demean, returns are centered columnwise before
solving and when scoring.
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar

from portfolio_backtest.backtest_config import DEMEAN

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

# Gurobi on the cluster, Clarabel where gurobipy is not installed
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False


class SecondOrderExpectedShortfallJointOptimizer(BaseOptimizer):
    def __init__(self, hyperparams: dict | None = None) -> None:
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

    def fit(self, train_bundle: dict[str, object]) -> None:
        """Extract and store return matrices for joint optimization."""
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
        """Jointly minimize HMCR_2 of (R_fh @ x + R_c @ psi) over x and psi. psi_base is left free; see
        module docstring.
        """
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
        """Run the joint optimization over training data."""
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
        """Validation HMCR_2, after re-solving the joint program on the validation window."""
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
