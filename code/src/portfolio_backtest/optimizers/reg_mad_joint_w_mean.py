"""
Regularized joint mean-MAD optimizer over asset and currency weights.

The mean-bearing counterpart of reg_mad_joint.py, and the regularized counterpart
of mad_joint_w_mean.py.  Minimizes

    mean| A_fh x + A_c psi | - (1 / gamma) (mu_a' x + mu_c' psi)
    + lambda_l1_asset ||x||_1 + lambda_l2_asset ||x||^2
    + lambda_l1_curr  ||psi||_1 + lambda_l2_curr  ||psi||^2

jointly over x and psi in a single program, subject to full investment and a
leverage cap on x and an L1 cap on psi.  A_fh and A_c are the columnwise demeaned
return matrices, so the risk term is the mean absolute deviation of the combined
portfolio return about its own sample mean.  gamma is a risk-aversion coefficient
in the conventional direction: large gamma starves the mean term and the program
collapses onto reg_mad_joint, small gamma gives the mean its full weight.

The program keeps Konno & Yamazaki's lower semi-absolute deviation form from
reg_mad_joint.py: for a sample with zero mean, mean|z| = 2 * mean(max(0, -z)),
linear once the positive part is lifted into a T-vector of auxiliary variables.
The identity survives the mean term untouched, because the residual
A_fh x + A_c psi sums to exactly zero over the window for every x and psi.  The
factor of two is carried explicitly so the objective stays on the MAD scale, which
matters because the penalty grids are calibrated against the size of the risk term.

The intended difference from reg_mad_curr_overlay_w_mean.py is the sequencing:
there x is fixed by an asset-only program and psi is fitted against it afterwards,
here both are chosen together against a single deviation.  No part of the objective
separates into an asset-only and a currency-only risk term -- the absolute value is
taken of the combined return, so the two blocks interact inside it.  Unlike the
overlay's step 2, mu_a' x is a live term rather than a droppable constant.

Location.  The demeaning makes the MAD a *deviation* measure, invariant to a
location shift, so the mean reaches the objective only through the explicit, pooled
and shrunk mu terms and gamma -> infinity recovers reg_mad_joint.py.  Measuring
absolute deviation about zero instead would not be location-free and would reward a
high mean at fixed weight one on top of the explicit term -- the double-counting
reg_es_joint_w_mean.py carries because raw ES is not a deviation measure.  One
inconsistency the deviation form leaves, stated rather than fixed: the demeaning
subtracts the raw per-column sample mean while the mean term rewards the pooled and
shrunk mu_a.

gamma is fixed rather than searched, at 0.5.  This config already carries
4 x 4 x 4 x 4 = 256 penalty combinations per rebalancing date, and crossing that
with MAD_GAMMA_GRID's six points is not affordable -- the same reasoning
REG_MV_MEAN_JOINT_CONFIG and REG_SMV_MEAN_JOINT_CONFIG record.  0.5 is calibrated by
the same criterion that picked 30 for the MV and SMV pairs: on the fixed-gamma sweep
of the unregularized MAD joint it is the most aggressive grid point that still
improves drawdown over the mean-free program (MDD 25.6% against 27.3%) while more
than doubling the Sharpe, from 0.180 to 0.375.  The full curve is tabulated at
MAD_GAMMA_GRID in backtest_config.py.  It is still read from hyperparams, so it can
be swept deliberately later without touching this code.  gamma is not comparable
across risk measures: MAD is degree one, so its band sits two to three decades below
the MV and SMV ones, and 0.5 here and 30 there are comparable positions on their own
curves rather than comparable numbers.

Fixing gamma also removes the bias the unregularized pair had to reason about in its
score, and sidesteps a second problem specific to the unregularized joint.  There
:meth:`score` re-solves on the validation window and so re-estimates mu_a on one year
of data, where James-Stein shrinks it to exactly zero on 41% of rebalancing dates.
Here nothing is re-solved on validation at all, so the scored weights are exactly the
trained ones and mu_a is never re-estimated.

Overlay cap.  Required, not optional, and doing more work here than in any other
program in the suite.  The MV and SMV joints have degree-two risk terms, so the
quadratic eventually dominates the linear mean term in every direction.  MAD is
positively homogeneous of degree one and x is confined to a compact set by
sum(x) = 1 and the L1 leverage cap, so psi is the only runaway direction: along
psi = t * d the objective behaves like t * [MAD(A_c d) - mu_c' d / gamma], giving

    gamma_crit = max_d [ mu_c' d / MAD(A_c d) ],

measured at 0.062, 0.078 and 0.110 across estimation windows -- the same quantity as
the overlay's, as that argument predicts.  The shipped gamma of 0.5 sits about four
and a half times above the worst of those.

That asymptotic margin understates the case for the cap here, though.  The bound is
the growth rate as t -> infinity; at finite t the asset block can partially offset
A_c d, so the objective keeps falling well past where the overlay's would stop.
Measured uncapped on the window with the largest gamma_crit, the unregularized joint
reaches a gross psi of 4.7 at gamma = 1 and 11.6 at 0.2, and the solver reports the
problem unbounded at gamma = 0.5 -- the shipped value -- where the overlay's uncapped
psi is still below 0.2.  So for this program the cap is what makes the shipped
configuration solvable at all, not merely what keeps psi tidy.  The penalties do not
substitute for it: lambda_l1_curr = lambda_l2_curr = 0 is a corner of the grid.

The cap is set at 2.0, matching every other mean-bearing program, and costs nothing
at the conservative end: mean-free MAD runs a small overlay, so the cap is slack at
gamma -> infinity and this is a strict superset of reg_mad_joint.py.

Means.  Estimated exactly as in mad_joint_w_mean.py: mu_a pooled onto the 14
economies and James-Stein shrunk toward zero, mu_c conditioned on the forward premium
as the carry -f_t.  What each channel actually forecasts is recorded in
mv_curr_overlay_w_mean.py.

Base currency.  Pinned to zero: its R_c column is identically zero so it never enters
the risk term, its mu_c entry is zero, and the overlay cap would otherwise let it
consume budget while contributing nothing.

Budget.  sum(x) == 1 is imposed and the solution is checked rather than renormalized,
following reg_mad_joint.py: renormalizing would mask solver drift and silently break
the leverage cap.

Performance notes, inherited from reg_mad_joint.py:

* The Backtester constructs a fresh optimizer instance for every hyperparameter
  combination, so the compiled problem is cached at class level in a small LRU keyed
  on the estimation window.
* The four penalties enter as ``cp.Parameter``, so the dense q-by-(N+K) block
  canonicalizes once per window rather than once per grid point (DPP).  The mean
  vectors and gamma are compiled in as constants instead, which is why gamma joins
  the cache key.
* Unlike the overlay there is no asset-only sub-problem to cache, so all 256 grid
  points are full solves.

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


class RegularizedMeanAbsoluteDeviationJointWithMeanOptimizer(BaseOptimizer):
    """
    Joint regularized mean-MAD optimizer over asset and currency overlay weights.

    Solves one CVXPY problem minimizing the penalized MAD of
    (R_fh @ x + R_c @ psi) about its own sample mean, minus the gamma-scaled mean of
    the same combination, subject to full investment and a leverage cap on x and an
    L1 cap on psi -- which this program requires to be solvable at the shipped gamma.
    """

    _PROBLEM_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
    _CACHE_MAXSIZE = 2

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``           (float) — risk aversion; the mean term enters
                                                with coefficient 1 / gamma, default 0.5
                * ``lambda_l1_asset`` (float) — L1 penalty for asset weights, default 2.5e-4
                * ``lambda_l2_asset`` (float) — L2 penalty for asset weights, default 0.01
                * ``lambda_l1_curr``  (float) — L1 penalty for the overlay, default 1.5e-5
                * ``lambda_l2_curr``  (float) — L2 penalty for the overlay, default 2.5e-3
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

        # Defaults are the interior points of the calibrated grids, so a --no-grid
        # run lands in the regularized regime rather than at an untested extreme.
        self.gamma: float = self.hyperparams.get("gamma", 0.5)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 2.5e-4)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.01)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 1.5e-5)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 2.5e-3)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.overlay_limit: float = self.hyperparams.get("overlay_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        for name in ("lambda_l1_asset", "lambda_l2_asset", "lambda_l1_curr", "lambda_l2_curr"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")
        if not np.isfinite(self.overlay_limit) or self.overlay_limit <= 0.0:
            raise ValueError(
                "overlay_limit must be finite and positive: MAD is degree-one, so the "
                "joint program is not reliably solvable without it at the shipped gamma."
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

        The demeaned return matrices, the two mean vectors and gamma are baked in as
        constants, so the compiled problem is valid only for the window it was built
        from; the cache is keyed on that window and on gamma, and :meth:`score` never
        re-solves on validation data.
        """
        A_fh = self.Y_fh - self.Y_fh.mean(axis=0)
        A_c = self.Y_c - self.Y_c.mean(axis=0)

        x = cp.Variable(self.N, name="x")
        psi = cp.Variable(self.K, name="psi")
        u = cp.Variable(self.q, name="u")

        p_l1_a = cp.Parameter(nonneg=True, name="lambda_l1_asset")
        p_l2_a = cp.Parameter(nonneg=True, name="lambda_l2_asset")
        p_l1_c = cp.Parameter(nonneg=True, name="lambda_l1_curr")
        p_l2_c = cp.Parameter(nonneg=True, name="lambda_l2_curr")

        l1_norm_x = cp.norm(x, 1)
        # S1, S2 diagonal: lambda^a on the first N entries, lambda^c on the last K.
        obj = (
            (2.0 / self.q) * cp.sum(u)
            - (1.0 / self.gamma) * (self.mu_a @ x + self.mu_c @ psi)
            + p_l1_a * l1_norm_x
            + p_l2_a * cp.sum_squares(x)
            + p_l1_c * cp.norm(psi, 1)
            + p_l2_c * cp.sum_squares(psi)
        )
        constraints = [
            u >= 0,
            -(A_fh @ x + A_c @ psi) <= u,
            cp.sum(x) == 1,                            # budget on the asset block only
            l1_norm_x <= self.leverage_limit,          # leverage on the asset block only
            cp.norm(psi, 1) <= self.overlay_limit,     # overlay cap; see module docstring
        ]
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
        base = np.where(self.Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(cp.Minimize(obj), constraints)
        if not problem.is_dcp(dpp=True):
            raise RuntimeError(
                "Joint regularized mean-MAD problem is not DPP-compliant. Losing DPP "
                "means every grid point re-canonicalizes a dense q-by-(N+K) block."
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
            "Compiled joint mean-MAD program (q=%d, N=%d, K=%d) in %.2fs.",
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
            "Joint regularized mean-MAD optimization on %d obs, %d assets, %d currencies.",
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

            MAD(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk aversion
        of one.

        MAD needs no transformation to reach a degree-one score, unlike the MV and SMV
        pairs, which score on the standard deviation and the semi-deviation rather
        than on the variance and semi-variance their optimizers minimize.  MAD is
        already degree one: about 3.1e-3 daily against a daily portfolio mean near
        2.7e-4, an eleven-to-one ratio matching the transformed MV and SMV scores, and
        exactly what ``in_sample_analysis.evaluate_risk`` returns for ``"mad"``.

        The reference coefficient of 1.0 is unbiased here in a way it is not in the
        unregularized pair.  gamma is fixed across the whole grid, so no candidate can
        earn a larger mean credit than another simply by carrying a smaller gamma; the
        search is selecting penalties alone.  Not re-solving also avoids the
        validation-window mu_a degradation documented in mad_joint_w_mean.score, since
        mu_a is never re-estimated here.

        Args:
            weights: The dict returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Validation MAD minus realized mean return, lower is better.
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
        mad = float(np.mean(np.abs(port_rets - port_rets.mean())))
        return mad - float(port_rets.mean())
