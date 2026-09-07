"""
Regularized joint mean-variance optimizer over asset and currency weights.

The mean-bearing counterpart of reg_mv_joint.py, and the regularized counterpart
of mv_joint_w_mean.py.  Minimizes

    Var(R_fh @ x + R_c @ psi) - (1 / gamma) * (mu_a' x + mu_c' psi)
    + lambda_l1_asset ||x||_1 + lambda_l2_asset ||x||^2
    + lambda_l1_curr  ||psi||_1 + lambda_l2_curr  ||psi||^2

jointly over x and psi in a single program, subject to full investment and a
leverage cap on x and an L1 cap on psi.  gamma is a risk-aversion coefficient in
the conventional direction: large gamma starves the mean term and the program
collapses onto reg_mv_joint, small gamma gives the mean its full weight.

The intended difference from reg_mv_curr_overlay_w_mean.py is the sequencing:
there x is fixed by an asset-only program and psi is fitted against it
afterwards, here both are chosen together against a single variance.  Two
differences that are not sequencing, both inherited from the mean-free pair and
both documented at length in mv_joint_w_mean.py: the cross-covariance is shrunk
inside one stacked Ledoit-Wolf estimate here rather than taken as a raw sample
block, and mu_a' x is a live term rather than a droppable constant.

gamma is fixed rather than searched, at 30.  This config already carries
4 x 4 x 4 x 4 = 256 penalty combinations per rebalancing date, and crossing that
with MV_GAMMA_GRID's five points is not affordable at these universe sizes -- the
same reasoning REG_ES_MEAN_JOINT_CONFIG records.  30 is the centre point of MV_GAMMA_GRID, whose full
curve for this program is tabulated in backtest_config.  On the fixed-gamma sweep
of the unregularized joint it earns a Sharpe of 0.373 against 0.162 mean-free, at
a drawdown of 25.9% against 27.9% -- so it still improves both axes, one step more
aggressive than the drawdown minimum at gamma = 50 (0.319 at 25.3%) and one step
short of where the caps saturate at gamma = 20.
It is still read from hyperparams, so it can be swept deliberately later without
touching this code.  The sweep that chose it had no penalties competing against
the mean term, so the penalty that is optimal here may sit lower than it does in
the mean-free program -- worth re-reading the selected-lambda series once these
have run.

Fixing gamma also removes the bias the unregularized pair had to reason about in
its score, and sidesteps a second problem specific to the unregularized joint.
There, :meth:`score` re-solves on the validation window and so re-estimates mu_a
on one year of data, where James-Stein shrinks it to exactly zero on 41% of
rebalancing dates -- selecting gamma against a program whose asset-mean channel
is largely switched off.  Here nothing is re-solved on validation at all, so the
scored weights are exactly the trained ones and mu_a is never re-estimated.

Overlay cap.  reg_mv_joint.py leaves psi unconstrained and relies on the
penalties alone.  The cap is carried here, matching mv_joint_w_mean.py and
reg_es_joint_w_mean.py, but it is slack at the shipped
gamma and becomes active below it.  Uncapped gross psi at the unpenalized corner,
measured on three real estimation windows (2000-06, 2005-06, 2010-06):

    gamma     100     50     30     20     10      5      2
    overlay  0.6-1.1  0.9-1.5  1.5-2.5  2.2-3.8  4.4-7.6  9.0-15.0  22-38
    joint    0.7-1.1  1.1-1.8  1.8-2.8  2.5-4.1  4.6-7.9  8.9-15.2  22-38

gamma = 30 is where the cap stops being decorative, but only just.  Sweeping the
full 256-point penalty grid on the three windows above, the overlay never reaches
it outright (its widest solution grazes 1.9999) and the joint sits on it at 16 of
256 points on the 2000-06 window, 9 of 256 on 2010-06 and none on 2005-06 -- 3.3%
of grid points overall, all of them in the lightly-penalized corner where nothing
else restrains psi.  By gamma = 10 the cap is the only thing between the overlay
and the double-digit gross notional recorded for the discarded mean-bearing MV
programs (74x - 98x).  Where it does bind, psi sits on a corner of the L1 ball
with its direction set by the carry ranking and its size by the cap rather than by
the variance-mean tradeoff, so those grid points are worth spotting in the
selected-lambda series rather than read as an interior optimum.

Means.  Estimated exactly as in mv_joint_w_mean.py: mu_a pooled onto the 14
economies and James-Stein shrunk toward zero, mu_c conditioned on the forward
premium as the carry -f_t.  What each channel forecasts is recorded in
mv_curr_overlay_w_mean.py.

Base currency.  Pinned to zero: its R_c column is identically zero so the stacked
covariance is singular in that direction, its mu_c entry is zero, and the overlay
cap would otherwise let it consume budget for nothing.

Budget.  sum(x) == 1 is imposed and the solution is checked rather than
renormalized, following reg_mv_joint.py: renormalizing would mask solver drift
and silently break the leverage cap.

No demeaning.  Variance is already location-free, so centering the return
matrices would leave the stacked covariance unchanged and only zero the mu terms
this program is built around.

Performance notes, inherited from reg_mv_joint.py:

* The Backtester constructs a fresh optimizer instance for every hyperparameter
  combination, so the compiled problem is cached at class level in a small LRU
  keyed on the estimation window.
* The four penalties enter as ``cp.Parameter``, so the dense (N+K)-by-(N+K)
  quadratic form canonicalizes once per window rather than once per grid point
  (DPP).  The mean vectors and gamma are compiled in as constants instead, which
  is why gamma joins the cache key; the mean vectors are functions of the window
  alone, so the frame signatures already cover them.

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
    ledoit_wolf_cc,
    pooled_asset_mean,
)

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


class RegularizedMeanVarianceJointWithMeanOptimizer(BaseOptimizer):
    """
    Joint regularized mean-variance optimizer over asset and currency overlay weights.

    Solves one CVXPY problem minimizing the penalized variance of
    (R_fh @ x + R_c @ psi) minus the gamma-scaled mean of the same combination,
    subject to full investment and a leverage cap on x and an L1 cap on psi.
    """

    _PROBLEM_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
    _CACHE_MAXSIZE = 2

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``           (float) — risk aversion; the mean term enters
                                                with coefficient 1 / gamma, default 30
                * ``lambda_l1_asset`` (float) — L1 penalty for asset weights, default 5e-6
                * ``lambda_l2_asset`` (float) — L2 penalty for asset weights, default 1e-3
                * ``lambda_l1_curr``  (float) — L1 penalty for the overlay, default 5e-6
                * ``lambda_l2_curr``  (float) — L2 penalty for the overlay, default 1e-4
                * ``leverage_limit``  (float) — ||x||_1 cap, default 2.0
                * ``overlay_limit``   (float) — ||psi||_1 cap, default 2.0
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

        self.gamma: float = self.hyperparams.get("gamma", 30)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 5e-6)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 1e-3)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 5e-6)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 1e-4)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.overlay_limit: float = self.hyperparams.get("overlay_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        for name in ("lambda_l1_asset", "lambda_l2_asset", "lambda_l1_curr", "lambda_l2_curr"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")

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

        # gamma and overlay_limit join the key because, unlike the penalties, they
        # are compiled in as constants rather than passed as DPP parameters.
        self._cache_key = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.gamma),
            float(self.leverage_limit),
            float(self.overlay_limit),
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

        The joint covariance (Ledoit-Wolf) is estimated on the stacked return
        vector [R_fh, R_c], so the asset-currency cross-covariance is shrunk
        together with the asset and currency blocks (appendix B,
        eq:mv_joint_obj).  It, the two mean vectors and gamma are baked in as
        constants, so the compiled problem is valid only for the window it was
        built from; the cache is keyed on that window and on gamma.
        """
        Y = np.hstack([self.Y_fh, self.Y_c])  # q x (N + K) combined return matrix
        Sigma_joint = ledoit_wolf_cc(Y)

        x = cp.Variable(self.N, name="x")
        psi = cp.Variable(self.K, name="psi")
        theta = cp.hstack([x, psi])

        p_l1_a = cp.Parameter(nonneg=True, name="lambda_l1_asset")
        p_l2_a = cp.Parameter(nonneg=True, name="lambda_l2_asset")
        p_l1_c = cp.Parameter(nonneg=True, name="lambda_l1_curr")
        p_l2_c = cp.Parameter(nonneg=True, name="lambda_l2_curr")

        l1_norm_x = cp.norm(x, 1)
        # S1, S2 diagonal: lambda^a on the first N entries, lambda^c on the last K.
        obj = (
            cp.quad_form(theta, cp.psd_wrap(Sigma_joint))
            - (1.0 / self.gamma) * (self.mu_a @ x + self.mu_c @ psi)
            + p_l1_a * l1_norm_x
            + p_l2_a * cp.sum_squares(x)
            + p_l1_c * cp.norm(psi, 1)
            + p_l2_c * cp.sum_squares(psi)
        )
        constraints = [
            cp.sum(x) == 1,                             # budget on the asset block only
            l1_norm_x <= self.leverage_limit,           # leverage on the asset block only
            cp.norm(psi, 1) <= self.overlay_limit,      # overlay cap; see module docstring
        ]
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
        base = np.where(self.Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(cp.Minimize(obj), constraints)
        if not problem.is_dcp(dpp=True):
            raise RuntimeError(
                "Joint regularized mean-MV problem is not DPP-compliant; every grid "
                "point would re-canonicalize the dense (N+K)-by-(N+K) quadratic form."
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
            "Compiled joint mean-MV program (q=%d, N=%d, K=%d) in %.2fs.",
            self.q, self.N, self.K, time.perf_counter() - t0,
        )
        self._cache_put(self._cache_key, bundle)
        return bundle

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run the joint regularized optimization over training data.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        logger.debug(
            "Joint regularized mean-MV optimization on %d obs, %d assets, %d currencies.",
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

        No re-solve.  Refitting on validation and then measuring the in-sample
        objective of that fit drives selection to the weakest penalty in the grid.
        The paper (p.25) specifies weights computed on training data and the risk
        measure evaluated on the validation sample.  The score is

            std(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one.

        The risk is the standard deviation, not the variance reg_mv_joint scores
        on.  Both have the same minimizer when no mean is subtracted, so the
        choice is free there; with a mean term it is not.  A daily variance on
        this data is about 2.6e-5 against a daily portfolio mean near 2.7e-4, so
        ``var - mean`` would be a ten-to-one mean maximization dressed up as a
        tradeoff.  The standard deviation is about 2.9e-3, and it is the
        degree-one convention ``in_sample_analysis.evaluate_risk`` already uses
        for ``"mv"`` -- the functional in_sample_vs_out_of_sample_w_mean.py scores
        these weights with, and the one the other three mean-bearing MV programs
        use, so the numbers are directly comparable across all four.

        The reference coefficient of 1.0 is unbiased here in a way it is not in
        the unregularized pair.  gamma is fixed across the whole grid, so no
        candidate can earn a larger mean credit than another simply by carrying a
        smaller gamma; the search is selecting penalties alone.  Not re-solving
        also avoids the validation-window mu_a degradation documented in
        mv_joint_w_mean.score, since mu_a is never re-estimated here.

        Args:
            weights: The dict returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Validation standard deviation minus realized mean return, lower is better.
        """
        if not isinstance(weights, dict) or "x" not in weights or "psi" not in weights:
            raise ValueError("score() expects the dict with keys 'x' and 'psi' from optimize().")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty or len(R_fh_val) < 2:
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
        return float(np.std(port_rets, ddof=1)) - float(port_rets.mean())
