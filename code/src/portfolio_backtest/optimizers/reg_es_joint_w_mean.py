"""
Joint regularized mean-Expected Shortfall optimizer over asset and currency weights.

The mean-bearing counterpart of reg_es_joint.py, and the regularized counterpart
of es_joint_w_mean.py.  Minimizes

    ES_alpha(-(R_fh @ x + R_c @ psi)) - (1/gamma) * (mu_a' x + mu_c' psi)
      + lambda_l1_asset ||x||_1 + lambda_l2_asset ||x||^2
      + lambda_l1_curr  ||psi||_1 + lambda_l2_curr  ||psi||^2

jointly over x and psi in a single program, subject to full investment and a
leverage cap on x and an L1 cap on psi.  Penalties follow the paper's S1, S2
diagonal structure: lambda^a on the first N entries (assets) and lambda^c on the
remaining M entries (currencies), so the four coefficients are tuned
independently.

gamma is fixed at 2 rather than searched.  The penalty grid already has 256
combinations per rebalancing date and nothing factorizes in the joint program,
so every one of them is a full-size solve; multiplying that by a seven-point
gamma grid is not affordable at the universe sizes this is run on.  It remains
readable from ``hyperparams`` so it can be swept deliberately later without
touching this file, but it is deliberately absent from
``REG_ES_MEAN_JOINT_CONFIG``'s grid.  gamma = 2 was not calibrated for this
program -- the unregularized joint's selection favoured 1.0 with no penalties
competing against the mean term -- so it is a conservative choice rather than an
optimized one.

Means.  Identical to the unregularized joint's:

  * ``mu_a`` is pooled onto the 14 economies and James-Stein shrunk toward zero
    (:func:`~portfolio_backtest.utils.estimation.pooled_asset_mean`).
  * ``mu_c`` is conditioned on the forward premium, mu_c = -f_t
    (:func:`~portfolio_backtest.utils.estimation.conditional_currency_mean`).

Both are constants for a given estimation window, so they are compiled into the
program rather than passed as parameters; the window is already part of the
problem cache key, so nothing can go stale.  ``gamma`` and ``overlay_limit`` are
added to that key for the same reason ``alpha`` and the leverage limit are in
it: they are compiled in.

Overlay cap and base currency.  ES is positively homogeneous, so along psi = t*d
the objective behaves like t * [ES(R_c d) - mu_c' d / gamma] once the penalties
are zero -- which the grid does include, at lambda_l1_curr = lambda_l2_curr = 0.
Any direction d with ES(R_c d) < mu_c' d / gamma would then make the program
unbounded below.  At gamma = 2 that is comfortably out of reach, but the L1 cap
on psi closes it off by construction rather than by luck, and matches the
unregularized mean optimizers.  psi_base is pinned to zero: its R_c column is
identically zero and its forward premium likewise, so it would consume overlay
budget for nothing.  That pin also removes the unidentified direction that
reg_es_joint.py's ``_solve`` docstring works around -- there psi_USD is free and
the optimal face contains a line, which is why crossover has to stay on there.

No demeaning.  The ``demean`` switch reg_es_joint.py carries is deliberately
absent: centering would zero the very sample means this program is built around.
It is correspondingly gone from the cache key.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import cvxpy as cp
import numpy as np

from ..utils.estimation import (
    conditional_currency_mean,
    economy_ids_from_indicator,
    pooled_asset_mean,
)

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

# Production runs use Gurobi (cluster). Fall back to Clarabel locally, where
# gurobipy is not installed; weight differences are ~1e-6. This mirrors every
# other regularized optimizer in the package (reg_mv/mad/smv/soes) — only the
# original ES pair hard-defaults to Gurobi, which makes them unrunnable locally.
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False

# Budget-violation tolerances on sum(x). Warn above the first, abort above the
# second. Both are loose relative to FeasibilityTol=1e-9; tripping either means
# the solve is not trustworthy rather than merely imprecise.
_BUDGET_WARN_TOL = 1e-6
_BUDGET_FAIL_TOL = 1e-4

_LEGACY_KEYS = ("lambda_l1", "lambda_l2")


class RegularizedExpectedShortfallJointWithMeanOptimizer(BaseOptimizer):
    """
    Joint regularized mean-ES optimizer over asset and currency overlay weights.

    Minimizes ES of (R_fh @ x + R_c @ psi) minus the gamma-scaled mean of the same
    combination, plus separate L1/L2 penalties on the asset and currency blocks,
    subject to full investment and a leverage cap on x and an L1 cap on psi.

    The compiled problem is cached at class level and keyed on the estimation
    window and every constant compiled into it.  The four penalties are
    DPP-compliant ``cp.Parameter`` objects, so a window is canonicalized once and
    reused across every grid point the backtester evaluates on it.  Nothing
    factorizes in the joint program, so all grid points still require a
    full-size solve.
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
                * ``gamma``            (float) — risk aversion; the mean term enters
                                                 with coefficient 1 / gamma, default 2.
                                                 Fixed rather than searched; see the
                                                 module docstring.
                * ``lambda_l1_asset``  (float) — L1 penalty on x, default 0.0
                * ``lambda_l2_asset``  (float) — L2 penalty on x, default 0.0
                * ``lambda_l1_curr``   (float) — L1 penalty on psi, default 0.0
                * ``lambda_l2_curr``   (float) — L2 penalty on psi, default 0.0
                * ``leverage_limit``   (float) — ||x||_1 cap, default 2.0
                * ``overlay_limit``    (float) — ||psi||_1 cap, default 2.0
                * ``solver``                   — cvxpy solver, default ``cp.GUROBI``
                                                 (``cp.CLARABEL`` when gurobipy
                                                 is unavailable)
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
        self.gamma: float = self.hyperparams.get("gamma", 2)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 0.0)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.0)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.0)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 0.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.overlay_limit: float = self.hyperparams.get("overlay_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

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

        self.Y_fh: np.ndarray | None = None
        self.Y_c: np.ndarray | None = None
        self.economy_ids: np.ndarray | None = None
        self.mu_a: np.ndarray | None = None
        self.mu_c: np.ndarray | None = None
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
        Extract the return matrices and estimate both mean vectors.

        Args:
            train_bundle: Must contain ``"R_fh"``, ``"R_c"``, ``"C"`` and ``"F"``.
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

        self.economy_ids = economy_ids_from_indicator(train_bundle.get("C"), self.N)
        self.mu_a, w_a = pooled_asset_mean(self.Y_fh, self.economy_ids)
        self.mu_c = conditional_currency_mean(train_bundle.get("F"), self.Y_c)
        logger.debug(
            "Asset mean shrinkage w=%.4f (%d economies); carry mu_c gross=%.2e.",
            w_a, len(np.unique(self.economy_ids)), np.abs(self.mu_c).sum(),
        )

        # gamma and overlay_limit belong in the key for the same reason alpha and
        # the leverage limit do: they are compiled into the program as constants.
        # The mean vectors are compiled in too, but they are functions of the
        # window alone, which the frame signatures already identify.
        self._cache_key = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.alpha),
            float(self.gamma),
            float(self.leverage_limit),
            float(self.overlay_limit),
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
        mean_term = (1.0 / self.gamma) * (self.mu_a @ x + self.mu_c @ psi)

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
            cp.sum(x) == 1,                       # budget on the asset block only
            l1_norm_x <= self.leverage_limit,     # leverage on the asset block only
            cp.norm(psi, 1) <= self.overlay_limit,
        ]
        base = np.where(self.Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(cp.Minimize(es_term - mean_term + penalty), constraints)

        if not problem.is_dcp(dpp=True):
            raise RuntimeError(
                "Joint regularized mean-ES problem is not DPP-compliant. Losing DPP "
                "means every grid point re-canonicalizes a dense q-by-(N+M) block."
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
            "Compiled joint mean program (q=%d, N=%d, M=%d) in %.2fs.",
            self.q, self.N, self.K, time.perf_counter() - t0,
        )
        self._cache_put(self._cache_key, bundle)
        return bundle

    def _solve(self, problem: cp.Problem) -> None:
        """
        Solve with Gurobi's default crossover.

        Unlike reg_es_joint.py, psi_base is pinned and psi is L1-capped here, so
        the unidentified direction that forces crossover on there is gone.  The
        settings are nonetheless left identical to that optimizer's so the two
        are numerically comparable; there is no need to trade that off for speed
        when the penalties, not the solve, dominate the grid cost.
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
        Run the joint regularized mean-ES optimization on the fitted window.

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
            "Joint mean solve (q=%d, N=%d, M=%d, l1a=%.3g, l2a=%.3g, l1c=%.3g, "
            "l2c=%.3g) in %.2fs.",
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

        No re-solve, exactly as in reg_es_joint.py.  Refitting on validation and
        then measuring the in-sample ES of that fit drives selection to the
        weakest penalty in the grid; the paper (p.25) specifies weights computed
        on training data and evaluated on the validation sample.  The score is

            ES_alpha(combined loss) - 1.0 * mean(realized combined returns),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one.  The ES estimator matches the backtester's
        ``compute_performance_metrics``: the mean of returns at or below the
        (1-alpha) quantile.

        The reference coefficient is 1.0 rather than 1 / gamma so the number is
        directly comparable with the unregularized mean optimizers.  Either is
        unbiased here, since gamma is fixed across the whole grid: no candidate
        can earn a larger mean credit than another simply by carrying a smaller
        gamma, which is the failure the unregularized optimizers had to be
        protected from.  The realized mean is used rather than
        mu_a' x + mu_c' psi because a score should measure what the weights
        actually earned, not re-apply a forecast to them.

        Args:
            weights: The dict returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Validation ES minus realized mean return, lower is better.
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
        es = float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))

        return es - float(port_rets.mean())
