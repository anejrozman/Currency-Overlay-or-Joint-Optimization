"""
Regularized mean-Second-Order Expected Shortfall optimizer with currency overlay
(Krokhmal 2007, in the framework of Lucescu & Ulrych 2026, Appendix A.3).

The mean-bearing counterpart of reg_soes_curr_overlay.py, and the regularized
counterpart of soes_curr_overlay_w_mean.py.  Both steps minimize the same
penalized mean-risk functional

    HMCR_2(portfolio loss) - (1 / gamma) mu' w
    + lambda_l1 ||w||_1 + lambda_l2 ||w||^2.

Step 1: minimizes HMCR_2(-(R_fh @ x)) - (1/gamma) mu_a' x plus the asset
        penalties, with full investment and a leverage cap on x.
Step 2: with asset weights fixed, minimizes

            HMCR_2(-(R_fh @ x* + R_c @ psi)) - (1/gamma) mu_c' psi
            + lambda_l1_curr ||psi||_1 + lambda_l2_curr ||psi||^2,

        subject to an L1 cap on the overlay.  The mu_a' x* term is a constant
        there and cannot move the argmin, so it is dropped -- which is also what
        lets :meth:`score` avoid estimating mu_a on the validation window.

The risk measure -- the unsquared L2 tail norm and its sqrt(q) normalization -- is
unchanged from reg_soes_curr_overlay.py; see reg_soes_asset_full.py for the
derivation.  Both steps are SOCPs, so ``optimal_inaccurate`` is more likely here
than for the ES overlays' LPs.

No demeaning.  The ``demean`` switch reg_soes_curr_overlay.py carries is
deliberately absent, matching soes_curr_overlay_w_mean.py and
reg_es_curr_overlay_w_mean.py: centering would zero the very sample means this
program is built around.  Running ``pooled_asset_mean`` on a demeaned R_fh returns
identically zero -- James-Stein shrinks to w = 1.000 -- against a max |mu_a| of
3.9e-4 to 1.4e-3 on the raw returns.  So mu_a and mu_c are estimated from the raw
matrices, and the risk term sees those same raw matrices.

That has a consequence worth stating.  HMCR_2 is translation-equivariant, so

    HMCR_2(-r) = HMCR_2_deviation(-r) - mean(r),

which means the risk term already rewards a high mean at fixed weight one on top
of the explicit 1/gamma term.  gamma -> infinity therefore gives
reg_soes_curr_overlay.py with ``demean=False``, not that module's default, and
the shrinkage buys less protection than it would in a deviation formulation.  This
mirrors the ES pair exactly: ES and SOES are the two non-deviation measures in the
suite and are treated identically, so an ES-versus-SOES comparison is clean even
though a SOES-versus-MV one carries this asymmetry.

gamma is fixed rather than searched, at 0.5.  This config already carries
4 x 4 x 4 x 4 = 256 penalty combinations per rebalancing date, and crossing that
with SOES_GAMMA_GRID's seven points is not affordable at SOES's solve cost -- the
same reasoning REG_MV_MEAN_OVERLAY_CONFIG and REG_MAD_MEAN_OVERLAY_CONFIG record.
0.5 is a conservative choice rather than the swept optimum: on the fixed-gamma
sweep of the unregularized SOES overlay it earns a Sharpe of 0.165 against 0.075
for the mean-free program, at a drawdown of 33.8% against 34.4% -- so it improves
both axes -- but the peak sits further out at gamma = 0.1 (Sharpe 0.291, drawdown
30.7%), where for this measure the Sharpe optimum and the drawdown minimum
coincide.  The full curve is tabulated at SOES_GAMMA_GRID in backtest_config.py.
gamma is still read from hyperparams, so it can be swept deliberately later.

Note gamma is not comparable across risk measures.  HMCR_2 is degree-one, like ES
and MAD and unlike MV and SMV, so its band sits two to three decades below theirs.

Fixing gamma also removes the one bias the unregularized pair had to reason about:
there :meth:`score` had to be gamma-free because a candidate carrying a smaller
gamma could earn an unboundedly larger mean credit; here every candidate carries
the same gamma, so the search selects penalties alone.

Overlay cap.  Required rather than optional.  HMCR_2 is positively homogeneous of
degree one, so along psi = t * d the step-2 objective behaves like
t * [HMCR_2(-(R_c d)) - mu_c' d / gamma] and runs to minus infinity for any gamma
below

    gamma_crit = max_d [ mu_c' d / HMCR_2(-(R_c d)) ],

measured at 0.0155, 0.0224 and 0.0264 on the raw return matrices this program
uses.  The shipped gamma of 0.5 clears the worst of those by about nineteen times,
so this program is comfortably well posed on its own terms and the cap is a
safeguard rather than the only thing holding it together -- unlike its joint
counterpart, where the cap is load-bearing at the shipped gamma.  The penalties do
not substitute for it: lambda_l1_curr = lambda_l2_curr = 0 is a corner of the grid.

The cap is set at 2.0, matching every other mean-bearing program.  That is not free
here: the mean-free undemeaned SOES overlay runs a gross psi of 0.62, 2.01 and 2.99
across three estimation windows, so the cap binds on two of them with the mean term
switched off entirely, and this program is not a strict superset of
reg_soes_curr_overlay.py.  ES and SMV are in the same position; MV and MAD are the
exceptions.

Base currency.  Pinned to zero, which reg_soes_curr_overlay.py does not do.  There
psi_base is genuinely absent from an unconstrained SOCP -- its R_c column is
identically zero, so it enters neither the cone constraint nor the objective.  The
overlay cap changes that: |psi_base| enters cp.norm(psi, 1), so a nonzero base
weight would consume overlay budget while contributing nothing to the risk term or
to the mean (its mu_c entry is zero too).

Means.  Estimated exactly as in soes_curr_overlay_w_mean.py: mu_a pooled onto the
14 economies and James-Stein shrunk toward zero, mu_c conditioned on the forward
premium as the carry -f_t.  mv_curr_overlay_w_mean.py records what each channel
actually forecasts over the following quarter.

Performance notes, inherited from reg_soes_curr_overlay.py:

* The Backtester constructs a fresh optimizer instance for every hyperparameter
  combination, so the compiled problems and the step-1 solutions are cached at
  class level and keyed on the estimation window.
* Penalties enter as ``cp.Parameter``, so each problem canonicalizes once per
  window rather than once per grid point (DPP).  The mean vectors and gamma are
  compiled in as constants instead, which is why gamma joins the cache key.
* Step 1 does not depend on the currency penalties, so it is solved once per
  (lambda_l1_asset, lambda_l2_asset) pair -- 16 solves per window rather than 256,
  which matters at SOES's SOCP cost.

Author: Anej Rozman
"""

from __future__ import annotations

import hashlib
import logging
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


class RegularizedSecondOrderExpectedShortfallCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step regularized mean-HMCR_2 optimizer with sequential currency overlay.

    Step 1 minimizes the penalized HMCR_2 of R_fh minus the pooled asset mean,
    subject to full investment and a leverage cap.  Step 2 fixes the asset weights
    and minimizes the penalized HMCR_2 of the combined portfolio return minus the
    carry mean of the overlay, subject to an L1 cap on psi.
    """

    _cache_key: str | None = None
    _asset_prob: dict | None = None
    _overlay_prob: dict | None = None
    _asset_solutions: dict[tuple[float, float], np.ndarray] = {}

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
                * ``leverage_limit``   (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``    (float) — ||psi||_1 cap for step 2, default 2.0
                * ``solver``                   — cvxpy solver, default ``cp.GUROBI``
                                                 (``cp.CLARABEL`` when gurobipy is unavailable)

        Note there is deliberately no ``demean`` key; see the module docstring.
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.gamma: float = self.hyperparams.get("gamma", 0.5)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 0.0)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.0)
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
                "overlay_limit must be finite and positive: HMCR_2 is degree-one, so "
                "step 2 is unbounded below without it for gamma under ~0.026."
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
        """
        Drop the compiled problems and cached step-1 solutions if anything baked
        into them has changed.

        ``alpha``, ``gamma``, ``leverage_limit`` and ``overlay_limit`` enter the key
        because they are compiled in as constants rather than passed as parameters.
        The two mean vectors are compiled in as well, but they are functions of the
        window alone, so the frame signatures already cover them.
        """
        parts = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.alpha),
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
        """
        Extract the return matrices, estimate both mean vectors, and refresh the
        problem cache if anything baked into it changed.

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

    # Step 1: assets

    def _build_asset_problem(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> dict:
        """Compile step 1 once, with the two asset penalties left as parameters.

        Y_fh, mu_a, alpha and gamma are baked in as constants, so the compiled
        problem is valid only for the window it was built from."""
        q, N = Y_fh.shape
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        u = cp.Variable(q)
        eta = cp.Variable()
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(q))
        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(
                soes_term
                - (1.0 / self.gamma) * (mu_a @ x)
                + lam1 * l1_norm
                + lam2 * cp.sum_squares(x)
            ),
            [
                u >= 0,
                -Y_fh @ x - eta <= u,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-1 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "x": x, "lam1": lam1, "lam2": lam2}

    def _optimize_assets(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
        """
        Step 1: penalized mean-HMCR_2 on R_fh with a leverage cap and full
        investment.  Bounded by its constraints regardless of gamma.

        The solution depends only on the asset penalties within a window, so it is
        cached on (lambda_l1_asset, lambda_l2_asset).
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
        """
        Compile step 2 once, with the penalties and the fixed asset return
        contribution left as parameters.

        Y_c, mu_c, alpha, gamma and the overlay cap are baked in as constants, so
        the compiled problem is valid only for the window it was built from.
        :meth:`score` never re-solves on validation data, so no other window ever
        reaches it.
        """
        q, K = Y_c.shape
        tail_prob = 1.0 - self.alpha

        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()
        c = cp.Parameter(q)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(q))
        constraints = [
            u >= 0,
            -c - Y_c @ psi - eta <= u,
            cp.norm(psi, 1) <= self.overlay_limit,
        ]
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(
            cp.Minimize(
                soes_term
                - (1.0 / self.gamma) * (mu_c @ psi)
                + lam1 * cp.norm(psi, 1)
                + lam2 * cp.sum_squares(psi)
            ),
            constraints,
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-2 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "psi": psi, "c": c, "lam1": lam1, "lam2": lam2}

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray, mu_c: np.ndarray
    ) -> np.ndarray:
        """Step 2: penalized mean-HMCR_2 currency overlay, capped in L1.

        The constant mu_a' x* is dropped: it cannot move the argmin."""
        cls = type(self)
        if cls._overlay_prob is None:
            cls._overlay_prob = self._build_overlay_problem(Y_c, mu_c)

        p = cls._overlay_prob
        p["c"].value = Y_fh @ x_fixed  # fixed asset return contribution, shape (q,)
        p["lam1"].value = float(self.lambda_l1_curr)
        p["lam2"].value = float(self.lambda_l2_curr)
        self._solve(p["prob"])

        overlay = p["psi"].value
        if overlay is None:
            raise RuntimeError("Solver returned None weights in step 2.")
        return overlay.flatten()

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run both optimization steps sequentially.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or any solver step fails.
            ValueError: If ``alpha`` is outside *(0, 1)*.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")

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
        """
        Evaluate train-fitted candidate weights on the validation window.

        Both the asset weights and the currency overlay are taken as given from
        :meth:`optimize` on the training window and held fixed; nothing is
        re-solved on the validation sample, matching reg_soes_curr_overlay.py and
        the rest of the regularized family.  Refitting on validation and then
        measuring the in-sample objective of that fit drives selection to the
        weakest penalty in the grid.  The score is

            HMCR_2(combined loss) - mean(realized combined returns),

        the realized risk-return tradeoff of this window at a reference coefficient
        of one.

        HMCR_2 needs no transformation to reach a degree-one score -- it is
        positively homogeneous of degree one -- and it is exactly what
        ``in_sample_analysis.evaluate_risk`` returns for ``"soes"``.  It is
        re-derived here with a bounded 1-D search rather than read off any solver
        eta: at p = 2 the minimizing zeta has no VaR shortcut (see
        reg_soes_asset_full.py), and these SOCPs are the most likely in the family
        to return ``optimal_inaccurate``.

        Because the risk term is not demeaned, the reference coefficient is
        effectively one half rather than one: HMCR_2 already contains a -mean(r) of
        its own, so this score is HMCR_2_deviation - 2 * mean(r).  Stated rather
        than corrected, so it stays comparable with the ES pair and with
        soes_curr_overlay_w_mean.

        The reference coefficient of 1.0 is unbiased here in a way it is not in the
        unregularized pair: gamma is fixed across the whole grid, so no candidate
        can earn a larger mean credit than another simply by carrying a smaller
        gamma, and the search selects penalties alone.

        Args:
            weights: Dict with ``"x"`` and ``"psi"`` as returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Validation HMCR_2 minus realized mean return, lower is better.
        """
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
