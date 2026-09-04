"""
Joint Regularized Second-Order Expected Shortfall optimizer over asset and
currency weights (Krokhmal 2007, in the framework of Ulrych & Lucescu 2026,
Section 3).

Minimizes regularized HMCR_2 of the combined portfolio return
(R_fh @ x + R_c @ psi) jointly over asset weights x and currency overlay
weights psi in a single program. Penalties follow the paper's S1, S2 diagonal
structure: lambda^a on the asset block, lambda^c on the currency block, so the
four coefficients lambda_l1_asset, lambda_l2_asset, lambda_l1_curr,
lambda_l2_curr are tuned independently — exactly the structure
reg_es_joint.py and reg_smv_joint.py use.

The risk measure — the unsquared L2 tail norm and its sqrt(q) normalization —
and the reasoning for why this is an SOCP rather than an LP are identical to
reg_soes_asset_full.py; see that module's docstring for the full derivation.
Relative to soes_joint.py, this adds L1/L2 penalties on each block, exactly as
reg_soes_curr_overlay.py adds them to soes_curr_overlay.py's two steps.

Base currency: psi_base is left free rather than pinned to zero, as in
soes_joint.py, soes_curr_overlay.py and reg_soes_curr_overlay.py (and unlike
reg_mv_joint.py / reg_smv_joint.py, which pin it). R_c's base-currency column
is identically zero, so psi_base never enters the SOC constraint or the risk
term regardless of its value; whenever lambda_l1_curr or lambda_l2_curr is
positive the penalty alone drives it to exactly zero (confirmed on
reg_soes_curr_overlay.py's smoke tests), and at the (0, 0) grid corner it is
genuinely unidentified but harmless, since its realized contribution to
portfolio returns is zero regardless of the value a solver settles on.

That said, the same free direction matters for the solver, not just the
value: reg_es_joint.py's _solve() deliberately leaves Gurobi's Crossover at
its default (on) rather than following reg_es_curr_overlay.py's
Crossover=0-by-default convention, because turning crossover off on the full
joint program (not the smaller step-2-only overlay problem) makes Gurobi hit
its barrier iteration limit and return status=user_limit when psi_base is
unidentified. The same precaution is taken here for the same reason.

With ``demean`` set, both return matrices are centered columnwise before the
program is solved, so HMCR_2 is measured on deviations from the sample mean
rather than on raw returns and is comparable to the deviation measures
(mean-variance, MAD, semi-variance), all of which demean by construction.
:meth:`score` holds the trained weights fixed and re-centers only the validation
portfolio return, by its own mean. The flag enters the problem cache key because
it is baked into the compiled program through Y_fh and Y_c, exactly like alpha
and the leverage limit. All of this is identical to reg_es_joint.py; the switch
is ``DEMEAN`` in backtest_config.py, off by default, mirrored into the parameter
grid there as ``DEMEAN_GRID`` and shared with the ES optimizers.

Performance notes (identical rationale to reg_es_joint.py / reg_smv_joint.py):

* The compiled problem is cached at class level, keyed on the estimation
  window, alpha and the leverage limit, since those are baked in as
  constants rather than as parameters.
* The four penalties are DPP-compliant cp.Parameter objects, so a window is
  canonicalized once and reused across every grid point.
* Nothing factorizes in the joint program (unlike the two-step overlay, where
  step 1 can be cached independently of the currency penalties): every grid
  point still needs a full-size (x, psi, u, eta) solve.

Unlike reg_es_joint.py (which hardcodes ``cp.GUROBI`` with no local fallback),
this follows the convention already used by the other five SOES optimizers:
fall back to Clarabel when gurobipy is unavailable.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

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

# Budget-violation tolerances on sum(x). Warn above the first, abort above the
# second. Both are loose relative to FeasibilityTol=1e-9; tripping either means
# the solve is not trustworthy rather than merely imprecise.
_BUDGET_WARN_TOL = 1e-6
_BUDGET_FAIL_TOL = 1e-4

_LEGACY_KEYS = ("lambda_l1", "lambda_l2")


class RegularizedSecondOrderExpectedShortfallJointOptimizer(BaseOptimizer):
    """
    Joint regularized second-order ES optimizer over asset and currency
    overlay weights.

    Minimizes HMCR_2 of (R_fh @ x + R_c @ psi) plus separate L1/L2 penalties on
    the asset and currency blocks, subject to full investment and a leverage
    cap on x. psi carries no budget or leverage constraint.
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

                * ``alpha``            (float) — confidence level, default 0.80
                * ``lambda_l1_asset``  (float) — L1 penalty on x, default 0.0
                * ``lambda_l2_asset``  (float) — L2 penalty on x, default 0.0
                * ``lambda_l1_curr``   (float) — L1 penalty on psi, default 0.0
                * ``lambda_l2_curr``   (float) — L2 penalty on psi, default 0.0
                * ``leverage_limit``   (float) — ||x||_1 cap, default 2.0
                * ``demean``           (bool)  — center R_fh and R_c columnwise
                                                 before solving, default
                                                 ``backtest_config.DEMEAN``
                * ``solver``                   — cvxpy solver, default ``cp.GUROBI``
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

        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 0.0)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.0)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.0)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 0.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.demean: bool = bool(self.hyperparams.get("demean", DEMEAN))
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        for name in (
            "lambda_l1_asset",
            "lambda_l2_asset",
            "lambda_l1_curr",
            "lambda_l2_curr",
        ):
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
        if self.demean:
            self.Y_fh = self.Y_fh - self.Y_fh.mean(axis=0)
            self.Y_c = self.Y_c - self.Y_c.mean(axis=0)
        self.q, self.N = self.Y_fh.shape
        self.K = self.Y_c.shape[1]

        # demean belongs in the key for the same reason alpha and the leverage limit
        # do: it is baked into the compiled program as a constant, here through the
        # centered Y_fh and Y_c. Without it a demeaned and a raw run in the same
        # process would silently share a compiled problem.
        self._cache_key = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.alpha),
            float(self.leverage_limit),
            bool(self.demean),
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
        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(self.q))

        # S1, S2 diagonal: lambda^a on the asset block, lambda^c on the currency block.
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

        problem = cp.Problem(cp.Minimize(soes_term + penalty), constraints)

        if not problem.is_dcp(dpp=True):
            raise RuntimeError(
                "Joint regularized HMCR_2 problem is not DPP-compliant. Losing DPP "
                "means every grid point re-canonicalizes a dense q-by-(N+K) SOC block."
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
            "Compiled joint HMCR_2 program (q=%d, N=%d, K=%d) in %.2fs.",
            self.q, self.N, self.K, time.perf_counter() - t0,
        )
        self._cache_put(self._cache_key, bundle)
        return bundle

    def _solve(self, problem: cp.Problem) -> None:
        """
        Solve with crossover left on. psi_base is unidentified whenever both
        currency penalties are zero (the base-currency column of R_c is
        identically zero and psi carries no other constraint), so the optimal
        face contains a line. Crossover picks a vertex; with Crossover=0
        Gurobi hits its barrier iteration limit and returns status=user_limit
        — the same reasoning as reg_es_joint.py's _solve().
        """
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

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run the joint regularized optimization on the fitted training window.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"``
            (currency overlay weights, shape K).

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
            "Joint solve (q=%d, N=%d, K=%d, l1a=%.3g, l2a=%.3g, l1c=%.3g, l2c=%.3g) "
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

        No re-solve, mirroring reg_es_joint.py / reg_smv_joint.py: refitting on
        validation and then measuring the in-sample risk of that fit would drive
        penalty selection to the weakest point in the grid.

        Unlike reg_es_joint.py's score(), the result cannot be read off a
        quantile of the realized returns: the p = 1 VaR shortcut that makes
        that exact for ES does not carry over to p = 2 (see
        reg_soes_asset_full.py). The threshold zeta is instead solved for
        numerically with the same bounded 1-D search used throughout the SOES
        family.

        Under ``demean`` the validation portfolio return is centered by its own
        mean, not the training one: the score has to be the deviation HMCR_2 of
        this window, so the location it deviates from is this window's mean. The
        zeta search then runs on the centered losses, and its bracket
        (losses.min(), losses.max()) still contains the minimizer — HMCR_2 is
        translation-equivariant, so centering shifts both zeta and the score by
        the same constant.

        Args:
            weights: The dict returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Second-order ES of the combined portfolio return on the validation
            window (lower is better).
        """
        if not isinstance(weights, dict) or "x" not in weights or "psi" not in weights:
            raise ValueError("score() expects the dict with keys 'x' and 'psi' from optimize().")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")

        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty:
            return float("inf")

        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)

        x = np.asarray(weights["x"], dtype=float).flatten()
        psi = np.asarray(weights["psi"], dtype=float).flatten()
        if x.shape[0] != Y_fh_val.shape[1] or psi.shape[0] != Y_c_val.shape[1]:
            raise ValueError(
                f"Weight/validation dimension mismatch: x {x.shape[0]} vs "
                f"{Y_fh_val.shape[1]} assets, psi {psi.shape[0]} vs "
                f"{Y_c_val.shape[1]} currencies."
            )

        port_rets = Y_fh_val @ x + Y_c_val @ psi
        if self.demean:
            port_rets = port_rets - port_rets.mean()

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
