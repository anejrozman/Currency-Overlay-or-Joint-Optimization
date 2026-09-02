"""
Regularized Joint Semi-Variance optimizer over asset and currency weights
(Markowitz, Todd, Xu & Yamane 1993, in the framework of Lucescu & Ulrych 2026, Appendix A.3).

Minimizes a regularized semi-variance objective of the combined portfolio return
(R_fh @ x + R_c @ psi) jointly over both asset weights x and currency overlay weights
psi in a single optimization problem. Separate L1 and L2 penalties are applied to x
and psi, following the paper's S1, S2 diagonal structure. Leverage cap and budget on
the asset block only; psi is unconstrained apart from the base-currency pin.

The risk measure is Markowitz et al.'s S_E, eq. (3.1a), with the target E set to the
portfolio's own expected return rather than an exogenous benchmark:

    SMV(w) = (1/T) * sum_t [ max(0, -A_t w) ]^2,

where A is the columnwise demeaned return matrix, so only deviations below the
portfolio's own mean are counted. The program uses their eqs. (3.4)-(3.7)
reformulation, lifting the positive part into auxiliary variables to make the
objective quadratic. The paper introduces 2T of them — Y for the signed deviation and
Z = Y^- for its negative part, tied by BX - Y + Z = 0 — because the Critical Line
Algorithm needs the constraints in equality form. Only the downside half is needed
here: minimizing sum z_t^2 subject to z >= -A w and z >= 0 drives each z_t to
max(0, -A_t w) on its own, so the equality and the free variables Y drop out and the
program is a QP with T auxiliary variables rather than 2T. The 1/sqrt(T) that eq.
(3.8) folds into B is carried as an explicit 1/T on the objective instead, which
leaves A on the scale of the raw returns and keeps the risk term numerically better
conditioned. The risk term is reported as the semi-variance itself, not doubled onto
the variance scale, which matters because the penalty grids are calibrated against
the size of the risk term.

Demeaning covers both blocks. Since mean(x' R_fh + psi' R_c) = x' mean(R_fh) +
psi' mean(R_c), the deviation about the portfolio's own mean is

    SMV(x, psi) = (1/T) * sum_t [ max(0, -(A_fh x + A_c psi)_t) ]^2,

where A_fh and A_c are the columnwise demeaned return matrices. Note that no part of
the risk term separates into an asset-only and a currency-only term the way the
mean-variance quadratic form does: the positive part is taken of the combined return,
so the two blocks interact inside it. Only the penalties are block-separable.

That demeaning also keeps the program well posed. The residual A_fh x + A_c psi sums
to exactly zero over the window for every (x, psi), so it can be non-negative
throughout only if it vanishes identically — which needs a null vector of the stacked
demeaned matrix, impossible for T ~ 2500 against N + K ~ 145 columns of full rank.
The risk term is therefore strictly positive at the optimum even before the penalties
enter.

The objective carries no mean term, mirroring the MV, MAD and ES joint optimizers,
which likewise penalize a risk measure alone, and removing the need for a shrinkage
estimator of the mean (Bayes-Stein, Jorion 1986). Unlike mean-variance, nothing
beyond the demeaning is estimated: the semi-variance is read off the empirical
distribution directly, with no joint covariance matrix and so no shrinkage target
(contrast Ledoit-Wolf on the stacked returns in reg_mv_joint).

The base currency's excess return is identically zero, so its overlay weight never
enters the risk term and would be left unidentified. It is pinned to zero, as in
reg_mv_joint and reg_mad_joint, rather than left free: an unidentified variable would
take an arbitrary solver-dependent value, be charged the L1 penalty, and land in the
saved weight records.

Performance notes:

* The Backtester constructs a fresh optimizer instance for every hyperparameter
  combination, so the compiled problem and the window's demeaned matrices are cached
  at class level and keyed on the estimation window.
* Penalties enter as ``cp.Parameter``, so each problem canonicalizes once per window
  rather than once per grid point (DPP).
* Nothing factorizes in the joint program, so every grid point still needs a
  full-size solve; only the demeaning and the canonicalization are shared.

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

# Production runs use Gurobi (cluster). Fall back to Clarabel locally, where
# gurobipy is not installed; weight differences are ~1e-6.
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


class RegularizedSemiVarianceJointOptimizer(BaseOptimizer):
    """
    Joint regularized semi-variance optimizer over asset and currency overlay weights.

    Solves a single CVXPY problem minimizing the regularized semi-variance of
    (R_fh @ x + R_c @ psi) about its own sample mean, subject to full investment and
    leverage constraints on x. psi is unconstrained apart from the base-currency pin.

    The compiled problem and the window's demeaned return matrices are cached at class
    level and keyed on the estimation window and the leverage limit. The four
    penalties are DPP-compliant cp.Parameter objects, so a window is canonicalized
    once and reused across every grid point the backtester evaluates on it.
    """

    # window signature -> compiled problem bundle. Held at class level because
    # the backtester constructs a fresh optimizer instance per grid point.
    _PROBLEM_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
    # Per date the backtester touches two windows: train (every grid point) and
    # train+val (final refit). Two entries covers both.
    _CACHE_MAXSIZE = 2

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``lambda_l1_asset`` (float) — L1 penalty for asset weights, default 1e-5
                * ``lambda_l2_asset`` (float) — L2 penalty for asset weights, default 1.5e-3
                * ``lambda_l1_curr``  (float) — L1 penalty for currency overlay, default 2.5e-6
                * ``lambda_l2_curr``  (float) — L2 penalty for currency overlay, default 1e-5
                * ``leverage_limit``  (float) — ||x||_1 cap, default 2.0
                * ``solver``                  — cvxpy solver, default ``cp.GUROBI``
                                                (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)

        legacy = [k for k in _LEGACY_KEYS if k in self.hyperparams]
        if legacy:
            raise ValueError(
                f"Shared penalty keys {legacy} are no longer accepted. The asset and "
                "currency grids differ by orders of magnitude, so a shared coefficient "
                "cannot be in range for both blocks. Use lambda_l1_asset, "
                "lambda_l2_asset, lambda_l1_curr, lambda_l2_curr."
            )

        # Defaults are the interior points of the calibrated grids, so a --no-grid
        # run lands in the regularized regime rather than at an untested extreme.
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 1e-5)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 1.5e-3)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 2.5e-6)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 1e-5)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        for name in ("lambda_l1_asset", "lambda_l2_asset", "lambda_l1_curr", "lambda_l2_curr"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

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
            float(self.leverage_limit),
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

    # Problem construction

    def _build_problem(self) -> dict:
        """
        Compile the joint program with the four penalties as DPP parameters.

        The demeaned return matrices are baked in as constants, so the compiled problem
        is valid only for the window it was built from; the cache is keyed on that
        window, and :meth:`score` never re-solves on validation data.
        """
        A_fh = self.Y_fh - self.Y_fh.mean(axis=0)
        A_c = self.Y_c - self.Y_c.mean(axis=0)

        x = cp.Variable(self.N, name="x")
        psi = cp.Variable(self.K, name="psi")
        z = cp.Variable(self.q, name="z")

        p_l1_a = cp.Parameter(nonneg=True, name="lambda_l1_asset")
        p_l2_a = cp.Parameter(nonneg=True, name="lambda_l2_asset")
        p_l1_c = cp.Parameter(nonneg=True, name="lambda_l1_curr")
        p_l2_c = cp.Parameter(nonneg=True, name="lambda_l2_curr")

        l1_norm_x = cp.norm(x, 1)
        # S1, S2 diagonal: lambda^a on the first N entries, lambda^c on the last K.
        obj = (
            cp.sum_squares(z) / self.q
            + p_l1_a * l1_norm_x
            + p_l2_a * cp.sum_squares(x)
            + p_l1_c * cp.norm(psi, 1)
            + p_l2_c * cp.sum_squares(psi)
        )
        constraints = [
            z >= 0,
            -(A_fh @ x + A_c @ psi) <= z,
            cp.sum(x) == 1,                    # budget on the asset block only
            l1_norm_x <= self.leverage_limit,  # leverage on the asset block only
        ]
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
        base = np.where(self.Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(cp.Minimize(obj), constraints)
        if not problem.is_dcp(dpp=True):
            raise RuntimeError(
                "Joint regularized SMV problem is not DPP-compliant. Losing DPP means "
                "every grid point re-canonicalizes a dense q-by-(N+K) block."
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
            "Compiled joint SMV program (q=%d, N=%d, K=%d) in %.2fs.",
            self.q, self.N, self.K, time.perf_counter() - t0,
        )
        self._cache_put(self._cache_key, bundle)
        return bundle

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run the joint regularized optimization over training data.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        logger.debug(
            "Joint regularized optimization on %d obs, %d assets, %d currencies.",
            self.q, self.N, self.K,
        )

        bundle = self._get_problem()
        params = bundle["params"]
        params["lambda_l1_asset"].value = float(self.lambda_l1_asset)
        params["lambda_l2_asset"].value = float(self.lambda_l2_asset)
        params["lambda_l1_curr"].value = float(self.lambda_l1_curr)
        params["lambda_l2_curr"].value = float(self.lambda_l2_curr)
        self._solve(bundle["problem"])

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
        weights: dict[str, np.ndarray],
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the training-fitted weights on the validation window.

        No re-solve. Refitting on validation and then measuring the in-sample objective
        of that fit drives selection to the weakest penalty in the grid. The paper
        (p.25) specifies weights computed on training data and the risk measure
        evaluated on the validation sample.

        The deviation is taken about the validation window's own mean, matching the
        asset-level and overlay SMV optimizers.

        Args:
            weights: The dict returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Semi-variance of the combined portfolio return on the validation window
            (lower is better).
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

        r_val = Y_fh_val @ x + Y_c_val @ psi
        downside = np.minimum(r_val - r_val.mean(), 0.0)
        return float(np.mean(downside ** 2))
