"""
Regularized joint mean-Second-Order Expected Shortfall optimizer over asset and
currency weights (Krokhmal 2007, in the framework of Lucescu & Ulrych 2026, Appendix A.3).

The mean-bearing counterpart of reg_soes_joint.py, and the regularized counterpart
of soes_joint_w_mean.py.  Minimizes

    HMCR_2(-(R_fh @ x + R_c @ psi)) - (1 / gamma) (mu_a' x + mu_c' psi)
    + lambda_l1_asset ||x||_1 + lambda_l2_asset ||x||^2
    + lambda_l1_curr  ||psi||_1 + lambda_l2_curr  ||psi||^2

jointly over x and psi in a single program, subject to full investment and a
leverage cap on x and an L1 cap on psi.

The risk measure -- the unsquared L2 tail norm and its sqrt(q) normalization -- is
unchanged from reg_soes_joint.py; see reg_soes_asset_full.py for the derivation.
This is an SOCP, so ``optimal_inaccurate`` is more likely here than for the ES
joint's LP.

The intended difference from reg_soes_curr_overlay_w_mean.py is the sequencing:
there x is fixed by an asset-only program and psi is fitted against it afterwards,
here both blocks enter the same second-order cone constraint.  Unlike the
overlay's step 2, mu_a' x is a live term rather than a droppable constant.

No demeaning.  The ``demean`` switch reg_soes_joint.py carries is deliberately
absent, matching soes_joint_w_mean.py and reg_es_joint_w_mean.py: centering would
zero the very sample means this program is built around -- ``pooled_asset_mean`` on
a demeaned R_fh returns identically zero, James-Stein shrinking to w = 1.000.  So
mu_a and mu_c are estimated from the raw matrices, and the risk term sees those
same raw matrices.

HMCR_2 is translation-equivariant, so an undemeaned HMCR_2 decomposes as
HMCR_2(-r) = HMCR_2_deviation(-r) - mean(r): the risk term already rewards a high
mean at fixed weight one on top of the explicit 1/gamma term.  gamma -> infinity
therefore gives reg_soes_joint.py with ``demean=False``, not that module's default.
This mirrors the ES pair exactly, which is the point -- ES and SOES are the two
non-deviation measures in the suite and are treated identically.

gamma is fixed rather than searched, at 0.5.  This config already carries
4 x 4 x 4 x 4 = 256 penalty combinations per rebalancing date, and crossing that
with SOES_GAMMA_GRID's seven points is not affordable at SOES's solve cost.  0.5 is
a conservative choice rather than the swept optimum: on the fixed-gamma sweep of
the unregularized SOES joint it earns a Sharpe of 0.118 against 0.056 for the
mean-free program run under the same 2.0 cap, while the peak sits further out at
gamma = 0.05 (Sharpe 0.234).  Note this program's drawdown worsens monotonically
across most of the band -- 36.3% at the capped mean-free end to 39.3% at gamma =
0.1 -- so unlike the MV, SMV and MAD joints there is no gamma that improves both
axes, and 0.5 is chosen for caution rather than because it dominates.  The full
curve is tabulated at SOES_GAMMA_GRID in backtest_config.py.  gamma is still read
from hyperparams, so it can be swept deliberately later.  gamma is not comparable
across risk measures: HMCR_2 is degree-one, so its band sits two to three decades
below the MV and SMV ones.

Fixing gamma also removes the bias the unregularized pair had to reason about in
its score, and sidesteps a second problem specific to the unregularized joint:
there :meth:`score` re-solves on the validation window and so re-estimates mu_a on
one year of data, where James-Stein shrinks it to exactly zero on 41% of
rebalancing dates.  Here nothing is re-solved on validation, so the scored weights
are exactly the trained ones and mu_a is never re-estimated.  That also saves a
full SOCP solve per grid point, which matters at this program's cost.

Overlay cap.  Required, and doing more work here than in any other program in the
suite.  HMCR_2 is degree-one and x is confined to a compact set by sum(x) = 1 and
the L1 leverage cap, so psi is the only runaway direction, and along psi = t * d
the objective behaves like t * [HMCR_2(-(R_c d)) - mu_c' d / gamma], giving

    gamma_crit = max_d [ mu_c' d / HMCR_2(-(R_c d)) ],

measured at 0.0155, 0.0224 and 0.0264 -- identical to the overlay's, as that
argument predicts.  The shipped gamma of 0.5 clears the worst by nineteen times.

That asymptotic margin badly understates the case here.  At finite scale the asset
block partially offsets the carry direction, and the mean-free undemeaned joint
already runs a gross psi near 3, so the objective keeps falling far past where the
overlay's would stop.  Measured uncapped, the unregularized joint is reported
unbounded by the solver at EVERY gamma tested down from 1 -- 38x above gamma_crit,
and twice the shipped value -- where the uncapped overlay stayed bounded to 0.02.
For this program the cap is what makes the shipped configuration solvable at all.
The penalties do not substitute for it: lambda_l1_curr = lambda_l2_curr = 0 is a
corner of the grid.

The cap is set at 2.0, matching every other mean-bearing program.  That is not free:
the mean-free undemeaned SOES joint runs a gross psi of 1.61, 3.08 and 3.90 across
three estimation windows, so the cap binds on two of the three with the mean term
switched off entirely -- the largest overshoot of any program in the suite.  This
is therefore not a strict superset of reg_soes_joint.py.

Base currency.  Pinned to zero, which reg_soes_joint.py does not do.  There
psi_base is genuinely absent from an unconstrained SOCP; the overlay cap changes
that, since |psi_base| enters cp.norm(psi, 1) and would consume budget while
contributing nothing to the risk term or to the mean.

Budget.  sum(x) == 1 is imposed and the solution is checked rather than
renormalized, following reg_soes_joint.py.  That matters most in an SOCP, where
``optimal_inaccurate`` is the likeliest status in the family and a silent
rescaling could push ||x||_1 past the leverage cap.

Means.  Estimated exactly as in soes_joint_w_mean.py: mu_a pooled onto the 14
economies and James-Stein shrunk toward zero, mu_c conditioned on the forward
premium as the carry -f_t.

Performance notes, inherited from reg_soes_joint.py:

* The compiled problem is cached at class level in a small LRU keyed on the
  estimation window; the four penalties are DPP ``cp.Parameter`` objects, so the
  window canonicalizes once rather than once per grid point.  The mean vectors and
  gamma are compiled in as constants, which is why gamma joins the cache key.
* Unlike the overlay there is no asset-only sub-problem to cache, so all 256 grid
  points are full SOCP solves.  This is the most expensive program in the suite.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar

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
# second. Both are loose relative to FeasibilityTol=1e-9.
_BUDGET_WARN_TOL = 1e-6
_BUDGET_FAIL_TOL = 1e-4

_LEGACY_KEYS = ("lambda_l1", "lambda_l2")


class RegularizedSecondOrderExpectedShortfallJointWithMeanOptimizer(BaseOptimizer):
    """
    Joint regularized mean-HMCR_2 optimizer over asset and currency overlay weights.

    Solves one CVXPY problem minimizing the penalized HMCR_2 of
    (R_fh @ x + R_c @ psi) minus the gamma-scaled mean of the same combination,
    subject to full investment and a leverage cap on x and an L1 cap on psi --
    which this program requires to be solvable at the shipped gamma.
    """

    _PROBLEM_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
    _CACHE_MAXSIZE = 2

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``            (float) — confidence level, default 0.80
                * ``gamma``            (float) — risk aversion; the mean term enters
                                                 with coefficient 1 / gamma, default 0.5
                * ``lambda_l1_asset``  (float) — L1 penalty for asset weights, default 0.0
                * ``lambda_l2_asset``  (float) — L2 penalty for asset weights, default 0.0
                * ``lambda_l1_curr``   (float) — L1 penalty for the overlay, default 0.0
                * ``lambda_l2_curr``   (float) — L2 penalty for the overlay, default 0.0
                * ``leverage_limit``   (float) — ||x||_1 cap, default 2.0
                * ``overlay_limit``    (float) — ||psi||_1 cap, default 2.0
                * ``solver``                   — cvxpy solver, default ``cp.GUROBI``
                                                 (``cp.CLARABEL`` when gurobipy is unavailable)

        Note there is deliberately no ``demean`` key; see the module docstring.
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
        self.gamma: float = self.hyperparams.get("gamma", 0.5)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 0.0)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.0)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.0)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 0.0)
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
                "overlay_limit must be finite and positive: HMCR_2 is degree-one, and "
                "this joint program is not reliably solvable without it at the shipped "
                "gamma."
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

        The matrices are kept raw -- not centered; see the module docstring.

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

        # alpha, gamma, the leverage limit and the overlay cap join the key because,
        # unlike the penalties, they are compiled in as constants rather than passed
        # as DPP parameters.
        self._cache_key = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.alpha),
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

    # Problem construction

    def _build_problem(self) -> dict:
        """
        Compile the joint program with the four penalties as DPP parameters.

        The return matrices, the two mean vectors, alpha, gamma and the overlay cap
        are baked in as constants, so the compiled problem is valid only for the
        window it was built from; the cache is keyed on that window and on gamma,
        and :meth:`score` never re-solves on validation data.
        """
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
        l1_norm_x = cp.norm(x, 1)
        obj = (
            soes_term
            - (1.0 / self.gamma) * (self.mu_a @ x + self.mu_c @ psi)
            + p_l1_a * l1_norm_x
            + p_l2_a * cp.sum_squares(x)
            + p_l1_c * cp.norm(psi, 1)
            + p_l2_c * cp.sum_squares(psi)
        )
        constraints = [
            u >= 0,
            -combined_return - eta <= u,
            cp.sum(x) == 1,                            # budget on the asset block only
            l1_norm_x <= self.leverage_limit,          # leverage on the asset block only
            cp.norm(psi, 1) <= self.overlay_limit,     # overlay cap; see module docstring
        ]
        base = np.where(self.Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(cp.Minimize(obj), constraints)
        if not problem.is_dcp(dpp=True):
            raise RuntimeError(
                "Joint regularized mean-SOES problem is not DPP-compliant. Losing DPP "
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
            "Compiled joint mean-SOES program (q=%d, N=%d, K=%d) in %.2fs.",
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

        No re-solve, mirroring reg_soes_joint.py and the rest of the regularized
        family: refitting on validation and then measuring the in-sample objective
        of that fit drives selection to the weakest penalty in the grid.  The score
        is

            HMCR_2(combined loss) - mean(realized combined returns),

        the realized risk-return tradeoff of this window at a reference coefficient
        of one.

        HMCR_2 needs no transformation to reach a degree-one score, and it is
        re-derived here with a bounded 1-D search: at p = 2 the minimizing zeta has
        no VaR shortcut (see reg_soes_asset_full.py), and these SOCPs are the most
        likely in the family to return ``optimal_inaccurate``.

        Because the risk term is not demeaned, the reference coefficient is
        effectively one half rather than one -- HMCR_2 already contains a -mean(r)
        of its own -- so this score is HMCR_2_deviation - 2 * mean(r).  Stated
        rather than corrected, so it stays comparable with the ES pair and with the
        unregularized SOES pair.

        The reference coefficient of 1.0 is unbiased here in a way it is not in the
        unregularized pair: gamma is fixed across the whole grid, so the search
        selects penalties alone.  Not re-solving also avoids the validation-window
        mu_a degradation documented in soes_joint_w_mean.score.

        Args:
            weights: The dict returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Validation HMCR_2 minus realized mean return, lower is better.
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
        return float(result.fun) - float(port_rets.mean())
