"""Regularized two-step currency overlay optimizer minimizing mean absolute deviation with a mean term.

Step 1 solves for the asset weights x on the fully hedged returns R_fh; step 2 fixes x and solves for the
overlay psi on R_fh @ x + R_c @ psi. x is fully invested with a leverage cap on ||x||_1. Separate L1 and L2
penalties on x and psi (lambda_l1_asset, lambda_l2_asset, lambda_l1_curr, lambda_l2_curr). The objective
subtracts (1 / gamma) times the expected return; asset means are pooled by economy and James-Stein shrunk,
currency means are the carry implied by the forward premium. psi has an L1 cap (overlay_limit) and a zero
base-currency weight. MAD is written in the lower semi-absolute deviation form of Konno & Yamazaki (1991).
"""

from __future__ import annotations

import hashlib
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


class RegularizedMeanAbsoluteDeviationCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    # class-level, shared by the per-grid-point instances; fit() resets it when the window changes
    _cache_key: str | None = None
    _asset_prob: dict | None = None
    _overlay_prob: dict | None = None
    _asset_solutions: dict[tuple[float, float], np.ndarray] = {}

    def __init__(self, hyperparams: dict | None = None) -> None:
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 0.5)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 2.5e-4)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.01)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.0)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 0.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.overlay_limit: float = self.hyperparams.get("overlay_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        for name in (
            "lambda_l1_asset",
            "lambda_l2_asset",
            "lambda_l1_curr",
            "lambda_l2_curr",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")
        if not np.isfinite(self.overlay_limit) or self.overlay_limit <= 0.0:
            raise ValueError(
                "overlay_limit must be finite and positive: MAD is degree-one, so "
                "step 2 is unbounded below without it for gamma under ~0.1."
            )

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

    # Caching

    @staticmethod
    def _frame_signature(df) -> tuple:
        """Identify a return-matrix slice by its endpoints, shape and columns."""
        return (
            str(df.index[0]),
            str(df.index[-1]),
            tuple(df.shape),
            tuple(map(str, df.columns)),
        )

    def _refresh_cache(self, R_fh, R_c) -> None:
        """Drop the compiled problems and cached step-1 solutions if anything baked into them has changed."""
        parts = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.gamma),
            float(self.leverage_limit),
            float(self.overlay_limit),
        )
        key = hashlib.blake2b(repr(parts).encode(), digest_size=16).hexdigest()

        cls = type(self)
        if key != cls._cache_key:
            cls._cache_key = key
            cls._asset_prob = None
            cls._overlay_prob = None
            cls._asset_solutions = {}
            logger.debug("Problem cache invalidated for new estimation window.")

    @classmethod
    def clear_cache(cls) -> None:
        """Drop all compiled problems and cached step-1 solutions."""
        cls._cache_key = None
        cls._asset_prob = None
        cls._overlay_prob = None
        cls._asset_solutions = {}

    def fit(self, train_bundle: dict[str, object]) -> None:
        """Extract the return matrices, estimate both mean vectors, and refresh the problem cache if
        anything baked into it changed.
        """
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

        self._refresh_cache(R_fh, R_c)

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

    # Step 1: assets

    def _build_asset_problem(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> dict:
        """Compile the step-1 problem once, with the penalties left as parameters."""
        T, N = Y_fh.shape
        A = Y_fh - Y_fh.mean(axis=0)

        x = cp.Variable(N)
        u = cp.Variable(T)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(
                (2.0 / T) * cp.sum(u)
                - (1.0 / self.gamma) * (mu_a @ x)
                + lam1 * l1_norm
                + lam2 * cp.sum_squares(x)
            ),
            [
                u >= 0,
                -A @ x <= u,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-1 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "x": x, "lam1": lam1, "lam2": lam2}

    def _optimize_assets(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
        """Step 1: penalized mean-MAD on R_fh with a leverage constraint and full investment.  Bounded by
        its constraints regardless of gamma.
        """
        cls = type(self)
        cache_id = (float(self.lambda_l1_asset), float(self.lambda_l2_asset))
        if cache_id in cls._asset_solutions:
            return cls._asset_solutions[cache_id].copy()

        if cls._asset_prob is None:
            cls._asset_prob = self._build_asset_problem(Y_fh, mu_a)

        p = cls._asset_prob
        p["lam1"].value = float(self.lambda_l1_asset)
        p["lam2"].value = float(self.lambda_l2_asset)
        self._solve(p["prob"])

        weights = p["x"].value
        if weights is None:
            raise RuntimeError("Solver returned None weights in step 1.")
        weights = (weights / weights.sum()).flatten()
        cls._asset_solutions[cache_id] = weights
        return weights.copy()

    # Step 2: currency overlay

    def _build_overlay_problem(self, Y_c: np.ndarray, mu_c: np.ndarray) -> dict:
        """Compile the step-2 problem once, with the penalties and the demeaned fixed asset contribution
        left as parameters.
        """
        T, K = Y_c.shape
        A_c = Y_c - Y_c.mean(axis=0)

        psi = cp.Variable(K)
        u = cp.Variable(T)
        c_dm = cp.Parameter(T)   # demeaned fixed asset contribution
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        constraints = [
            u >= 0,
            -(c_dm + A_c @ psi) <= u,
            cp.norm(psi, 1) <= self.overlay_limit,
        ]
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(
            cp.Minimize(
                (2.0 / T) * cp.sum(u)
                - (1.0 / self.gamma) * (mu_c @ psi)
                + lam1 * cp.norm(psi, 1)
                + lam2 * cp.sum_squares(psi)
            ),
            constraints,
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-2 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "psi": psi, "c_dm": c_dm, "lam1": lam1, "lam2": lam2}

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray, mu_c: np.ndarray
    ) -> np.ndarray:
        """Step 2: penalized mean-MAD currency overlay, capped in L1."""
        cls = type(self)
        if cls._overlay_prob is None:
            cls._overlay_prob = self._build_overlay_problem(Y_c, mu_c)

        c = Y_fh @ x_fixed
        p = cls._overlay_prob
        p["c_dm"].value = c - c.mean()
        p["lam1"].value = float(self.lambda_l1_curr)
        p["lam2"].value = float(self.lambda_l2_curr)
        self._solve(p["prob"])

        overlay = p["psi"].value
        if overlay is None:
            raise RuntimeError("Solver returned None weights in step 2.")
        return overlay.flatten()

    def optimize(self) -> dict[str, np.ndarray]:
        """Run both optimization steps sequentially."""
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        logger.debug(
            "Step 1: optimizing asset weights on %d obs, %d assets.", self.q, self.N
        )
        self.x_opt = self._optimize_assets(self.Y_fh, self.mu_a)

        logger.debug(
            "Step 2: optimizing currency overlay on %d obs, %d currencies.", self.q, self.K
        )
        psi_opt = self._optimize_overlay(self.Y_fh, self.Y_c, self.x_opt, self.mu_c)

        return {"x": self.x_opt, "psi": psi_opt}

    def score(
        self,
        weights: dict[str, np.ndarray],
        val_bundle: dict[str, object],
    ) -> float:
        """Validation MAD minus mean return of the training-fitted weights."""
        if not isinstance(weights, dict) or "x" not in weights or "psi" not in weights:
            raise ValueError("weights must be a dict with keys 'x' and 'psi'.")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")

        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty or len(R_fh_val) < 2:
            return float("inf")

        x = np.asarray(weights["x"], dtype=float).flatten()
        psi = np.asarray(weights["psi"], dtype=float).flatten()
        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)
        if x.shape[0] != Y_fh_val.shape[1] or psi.shape[0] != Y_c_val.shape[1]:
            raise ValueError(
                f"Weight/validation dimension mismatch: x {x.shape[0]} vs "
                f"{Y_fh_val.shape[1]} assets, psi {psi.shape[0]} vs "
                f"{Y_c_val.shape[1]} currencies."
            )

        port_rets = Y_fh_val @ x + Y_c_val @ psi
        mad = float(np.mean(np.abs(port_rets - port_rets.mean())))
        return mad - float(port_rets.mean())
