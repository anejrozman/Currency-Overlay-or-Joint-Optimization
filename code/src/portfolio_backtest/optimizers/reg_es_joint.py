"""
Joint Regularized Expected Shortfall optimizer over asset and currency weights.

Minimizes regularized ES of the combined portfolio return (R_fh @ x + R_c @ psi)
jointly over asset weights x and currency overlay weights psi in a single program.
Penalties follow the paper's S1, S2 diagonal structure: lambda^a on the first N
entries (assets) and lambda^c on the remaining M entries (currencies), so the four
coefficients lambda_l1_asset, lambda_l2_asset, lambda_l1_curr, lambda_l2_curr are
tuned independently.

Convention: gamma = infinity, so the mean term drops and the ES coefficient is
normalized to one. The Table 3 penalty bounds are read as already normalized.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import cvxpy as cp
import numpy as np

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

# Budget-violation tolerances on sum(x). Warn above the first, abort above the
# second. Both are loose relative to FeasibilityTol=1e-9; tripping either means
# the solve is not trustworthy rather than merely imprecise.
_BUDGET_WARN_TOL = 1e-6
_BUDGET_FAIL_TOL = 1e-4

_LEGACY_KEYS = ("lambda_l1", "lambda_l2")


class RegularizedExpectedShortfallJointOptimizer(BaseOptimizer):
    """
    Joint regularized ES optimizer over asset and currency overlay weights.

    Minimizes ES of (R_fh @ x + R_c @ psi) plus separate L1/L2 penalties on the
    asset and currency blocks, subject to full investment and a leverage cap on
    x. psi carries no budget or leverage constraint.

    The compiled problem is cached at class level and keyed on the estimation
    window, alpha and the leverage limit. The four penalties are DPP-compliant
    cp.Parameter objects, so a window is canonicalized once and reused across
    every grid point the backtester evaluates on it. Nothing factorizes in the
    joint program, so all grid points still require a full-size solve.
    """

    # window signature -> compiled problem bundle. Held at class level because
    # the backtester constructs a fresh optimizer instance per grid point.
    _PROBLEM_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
    # Per date the backtester touches two windows: train (every grid point) and
    # train+val (final refit). Two entries covers both; more risks holding
    # several dense 2520-by-(N+M) canonicalizations at top-100.
    _CACHE_MAXSIZE = 2

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``            (float) — ES confidence level, default 0.80
                * ``lambda_l1_asset``  (float) — L1 penalty on x, default 0.0
                * ``lambda_l2_asset``  (float) — L2 penalty on x, default 0.0
                * ``lambda_l1_curr``   (float) — L1 penalty on psi, default 0.0
                * ``lambda_l2_curr``   (float) — L2 penalty on psi, default 0.0
                * ``leverage_limit``   (float) — ||x||_1 cap, default 2.0
                * ``solver``                   — cvxpy solver, default ``cp.GUROBI``
        """
        super().__init__(hyperparams)

        legacy = [k for k in _LEGACY_KEYS if k in self.hyperparams]
        if legacy:
            raise ValueError(
                f"Shared penalty keys {legacy} are no longer accepted. The asset and "
                "currency grids differ by four to five orders of magnitude, so a "
                "shared coefficient cannot be in range for both blocks. Use "
                "lambda_l1_asset, lambda_l2_asset, lambda_l1_curr, lambda_l2_curr."
            )

        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 0.0)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.0)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.0)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 0.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", cp.GUROBI)

        for name in (
            "lambda_l1_asset",
            "lambda_l2_asset",
            "lambda_l1_curr",
            "lambda_l2_curr",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        self.Y_fh: np.ndarray | None = None
        self.Y_c: np.ndarray | None = None
        self.N: int = 0
        self.K: int = 0
        self.q: int = 0
        self._cache_key: tuple | None = None

    # Window identity and problem cache

    @staticmethod
    def _frame_signature(frame) -> tuple:
        """Cheap identity for a contiguous window slice."""
        index = frame.index
        return (
            frame.shape,
            tuple(frame.columns),
            index[0] if len(index) else None,
            index[-1] if len(index) else None,
        )

    @classmethod
    def _cache_put(cls, key: tuple, bundle: dict) -> None:
        cls._PROBLEM_CACHE[key] = bundle
        cls._PROBLEM_CACHE.move_to_end(key)
        while len(cls._PROBLEM_CACHE) > cls._CACHE_MAXSIZE:
            cls._PROBLEM_CACHE.popitem(last=False)

    @classmethod
    def clear_cache(cls) -> None:
        """Drop all compiled problems. Call between universes if memory is tight."""
        cls._PROBLEM_CACHE.clear()

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
        if R_fh.shape[0] != R_c.shape[0]:
            raise ValueError(
                f"R_fh and R_c must span the same window: got {R_fh.shape[0]} and "
                f"{R_c.shape[0]} observations."
            )

        self.Y_fh = np.ascontiguousarray(R_fh.values, dtype=float)
        self.Y_c = np.ascontiguousarray(R_c.values, dtype=float)
        self.q, self.N = self.Y_fh.shape
        self.K = self.Y_c.shape[1]

        self._cache_key = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.alpha),
            float(self.leverage_limit),
        )

    # Problem construction

    def _build_problem(self) -> dict:
        """Compile the joint program with the four penalties as DPP parameters."""
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(self.N, name="x")
        psi = cp.Variable(self.K, name="psi")
        u = cp.Variable(self.q, name="u")
        eta = cp.Variable(name="eta")

        p_l1_a = cp.Parameter(nonneg=True, name="lambda_l1_asset")
        p_l2_a = cp.Parameter(nonneg=True, name="lambda_l2_asset")
        p_l1_c = cp.Parameter(nonneg=True, name="lambda_l1_curr")
        p_l2_c = cp.Parameter(nonneg=True, name="lambda_l2_curr")

        combined_return = self.Y_fh @ x + self.Y_c @ psi
        es_term = eta + (1.0 / (self.q * tail_prob)) * cp.sum(u)

        # S1, S2 diagonal: lambda^a on the first N entries, lambda^c on the last M.
        penalty = (
            p_l1_a * cp.norm(x, 1)
            + p_l1_c * cp.norm(psi, 1)
            + p_l2_a * cp.sum_squares(x)
            + p_l2_c * cp.sum_squares(psi)
        )

        l1_norm_x = cp.norm(x, 1)
        constraints = [
            u >= 0,
            -combined_return - eta <= u,
            cp.sum(x) == 1,                    # budget on the asset block only
            l1_norm_x <= self.leverage_limit,  # leverage on the asset block only
        ]

        problem = cp.Problem(cp.Minimize(es_term + penalty), constraints)

        if not problem.is_dcp(dpp=True):
            raise RuntimeError(
                "Joint regularized ES problem is not DPP-compliant. Losing DPP means "
                "every grid point re-canonicalizes a dense q-by-(N+M) block."
            )

        return {
            "problem": problem,
            "x": x,
            "psi": psi,
            "params": {
                "lambda_l1_asset": p_l1_a,
                "lambda_l2_asset": p_l2_a,
                "lambda_l1_curr": p_l1_c,
                "lambda_l2_curr": p_l2_c,
            },
        }

    def _get_problem(self) -> dict:
        bundle = self._PROBLEM_CACHE.get(self._cache_key)
        if bundle is not None:
            self._PROBLEM_CACHE.move_to_end(self._cache_key)
            return bundle

        t0 = time.perf_counter()
        bundle = self._build_problem()
        logger.debug(
            "Compiled joint program (q=%d, N=%d, M=%d) in %.2fs.",
            self.q, self.N, self.K, time.perf_counter() - t0,
        )
        self._cache_put(self._cache_key, bundle)
        return bundle

    def _solve(self, problem: cp.Problem) -> None:
        """
        Solve with crossover left on. psi_USD is unidentified (the USD column of
        R_c is identically zero and psi carries no constraints), so the optimal
        face contains a line. Crossover picks a vertex; with Crossover=0 Gurobi
        hits its barrier iteration limit and returns status=user_limit.
        """
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

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run the joint regularized optimization on the fitted training window.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"``
            (currency overlay weights, shape M).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
            ValueError: If ``alpha`` is outside *(0, 1)*.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")

        bundle = self._get_problem()
        params = bundle["params"]
        params["lambda_l1_asset"].value = float(self.lambda_l1_asset)
        params["lambda_l2_asset"].value = float(self.lambda_l2_asset)
        params["lambda_l1_curr"].value = float(self.lambda_l1_curr)
        params["lambda_l2_curr"].value = float(self.lambda_l2_curr)

        t0 = time.perf_counter()
        self._solve(bundle["problem"])
        logger.debug(
            "Joint solve (q=%d, N=%d, M=%d, l1a=%.3g, l2a=%.3g, l1c=%.3g, l2c=%.3g) "
            "in %.2fs.",
            self.q, self.N, self.K,
            self.lambda_l1_asset, self.lambda_l2_asset,
            self.lambda_l1_curr, self.lambda_l2_curr,
            time.perf_counter() - t0,
        )

        weights = bundle["x"].value
        if weights is None:
            raise RuntimeError("Solver returned None for asset weights.")
        overlay = bundle["psi"].value
        if overlay is None:
            raise RuntimeError("Solver returned None for currency overlay weights.")

        weights = np.asarray(weights, dtype=float).flatten()
        overlay = np.asarray(overlay, dtype=float).flatten()

        # sum(x) == 1 is imposed, so no renormalization. Renormalizing would mask
        # solver drift and silently break the leverage cap; check instead.
        budget_error = abs(weights.sum() - 1.0)
        if budget_error > _BUDGET_FAIL_TOL:
            raise RuntimeError(
                f"Budget constraint violated: sum(x) = {weights.sum():.9f}."
            )
        if budget_error > _BUDGET_WARN_TOL:
            logger.warning("sum(x) = %.9f, off budget by %.2e.", weights.sum(), budget_error)

        return {"x": weights, "psi": overlay}

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the training-fitted weights on the validation window.

        No re-solve. Refitting on validation and then measuring the in-sample ES
        of that fit drives selection to the weakest penalty in the grid. The
        paper (p.25) specifies weights computed on training data and ES evaluated
        on the validation sample.

        The ES estimator matches the backtester's compute_performance_metrics:
        the mean of returns at or below the (1-alpha) quantile.

        Args:
            weights: The dict returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Negative mean of the worst ``(1-alpha)`` fraction of combined returns.
        """
        if not isinstance(weights, dict) or "x" not in weights or "psi" not in weights:
            raise ValueError("score() expects the dict with keys 'x' and 'psi' from optimize().")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")

        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty:
            return float("inf")

        Y_fh_val = R_fh_val.values
        Y_c_val = R_c_val.values

        x = np.asarray(weights["x"], dtype=float).flatten()
        psi = np.asarray(weights["psi"], dtype=float).flatten()
        if x.shape[0] != Y_fh_val.shape[1] or psi.shape[0] != Y_c_val.shape[1]:
            raise ValueError(
                f"Weight/validation dimension mismatch: x {x.shape[0]} vs "
                f"{Y_fh_val.shape[1]} assets, psi {psi.shape[0]} vs "
                f"{Y_c_val.shape[1]} currencies."
            )

        port_rets = Y_fh_val @ x + Y_c_val @ psi

        threshold = np.quantile(port_rets, 1.0 - self.alpha)
        tail = port_rets[port_rets <= threshold]
        return float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))