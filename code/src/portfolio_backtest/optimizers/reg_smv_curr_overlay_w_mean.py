"""
Regularized mean-Semi-Variance optimizer with currency overlay
(Markowitz, Todd, Xu & Yamane 1993, in the framework of Lucescu & Ulrych 2026, Appendix A.3).

The mean-bearing counterpart of reg_smv_curr_overlay.py, and the regularized
counterpart of smv_curr_overlay_w_mean.py.  Both steps minimize the same penalized
mean-risk functional

    SMV(portfolio return) - (1 / gamma) mu' w
    + lambda_l1 ||w||_1 + lambda_l2 ||w||^2,

with gamma a risk-aversion coefficient in the conventional direction: large gamma
starves the mean term and the program collapses onto reg_smv_curr_overlay, small
gamma gives the mean its full weight.

Step 1: minimizes (1/T) sum_t [max(0, -(A x)_t)]^2 - (1/gamma) mu_a' x plus the
        asset penalties, with full investment and a leverage cap on x.
Step 2: with asset weights fixed, minimizes

            (1/T) sum_t [max(0, -(c_dm + A_c psi)_t)]^2 - (1/gamma) mu_c' psi
            + lambda_l1_curr ||psi||_1 + lambda_l2_curr ||psi||^2,

        subject to an L1 cap on the overlay.  The mu_a' x* term is a constant
        there and cannot move the argmin, so it is dropped -- which is also what
        lets :meth:`score` avoid estimating mu_a on the validation window.

The risk measure and its reformulation are unchanged from reg_smv_curr_overlay.py:
Markowitz et al.'s S_E, eq. (3.1a), with the target E set to the portfolio's own
expected return, lifted into T auxiliary variables via their eqs. (3.4)-(3.7),
keeping only the downside half so each step is a QP with T auxiliary variables
rather than 2T.  The risk term is reported as the semi-variance itself, not
doubled onto the variance scale, which matters because the penalty grids are
calibrated against the size of the risk term.

Location.  The demeaning is what makes the mean term well defined here.
A = Y - Y.mean(axis=0) turns the semi-variance into a *deviation* measure,
invariant to a location shift, so the mean reaches the objective only through the
explicit, pooled and shrunk mu terms -- as a variance does in
reg_mv_curr_overlay_w_mean.py, and why gamma -> infinity recovers
reg_smv_curr_overlay.py up to the overlay cap.  Dropping the demeaning would turn
the risk term into a lower partial moment about zero, not location-free, and it
would then reward a high mean at fixed weight one on top of the explicit term --
the double-counting reg_es_curr_overlay_w_mean.py carries because raw ES is not a
deviation measure either.

One inconsistency the deviation form leaves, stated rather than fixed: the
demeaning subtracts the raw per-column sample mean while the mean term rewards
the pooled and shrunk mu_a, so risk is measured about a noisy target and return
forecast with a deliberately quiet one.

gamma is fixed rather than searched, at 30.  This config already carries
4 x 4 x 4 x 4 = 256 penalty combinations per rebalancing date, and crossing that
with SMV_GAMMA_GRID's six points is not affordable at these universe sizes, still
less at SMV's solve cost -- the same reasoning REG_MV_MEAN_OVERLAY_CONFIG records.
30 is calibrated rather than conservative: on the fixed-gamma sweep of the
unregularized SMV overlay it is the most aggressive grid point that still improves
drawdown over the mean-free program (MDD 30.5% against 32.2%) while raising the
Sharpe from 0.083 to 0.381.  The full curve is tabulated at SMV_GAMMA_GRID in
backtest_config.py.  It is still read from hyperparams, so it can be swept
deliberately later without touching this code.  Note that 30 sits closer to the
SMV Sharpe peak (15-20) than the same number does on the MV curve (peak 10), so
this pair is running a relatively more aggressive tilt than the MV pair at the
same nominal gamma; gamma is not comparable across risk measures.

Fixing gamma also removes the one bias the unregularized pair had to reason
about.  There :meth:`score` had to be gamma-free because a candidate carrying a
smaller gamma could earn an unboundedly larger mean credit; here every candidate
carries the same gamma, so the reference coefficient of 1.0 below is unbiased by
construction and the search is selecting penalties alone.

Overlay cap.  Carried at 2.0, matching smv_curr_overlay_w_mean.py and the
mean-bearing MV pair, so every mean-bearing program in the thesis shares one
overlay budget.  Unlike the MV pair this is not free: the mean-free SMV overlay
already runs gross psi of 0.40 to 2.54 across fourteen estimation windows (median
1.40), so the cap binds on about one window in five *with the mean term switched
off entirely*, and this program is therefore not a strict superset of
reg_smv_curr_overlay.py.  Measured at the backtest level the effect is small:
turning the cap on with the mean term off is worth +0.005 Sharpe on the
unregularized overlay, against +0.053 for the mean term alone.  Where it binds,
psi sits on a corner of the L1 ball with its size set by the budget rather than by
the risk-return tradeoff.

Means.  Estimated exactly as in smv_curr_overlay_w_mean.py: mu_a pooled onto the
14 economies and James-Stein shrunk toward zero, mu_c conditioned on the forward
premium as the carry -f_t.  What each channel actually forecasts over the
following quarter is recorded in mv_curr_overlay_w_mean.py -- mu_c genuinely, mu_a
not, cross-sectionally.

Base currency.  Pinned to zero: its R_c column is identically zero so it never
enters the risk term, its mu_c entry is zero, and the overlay cap would otherwise
let it consume budget while contributing nothing.

Performance notes, inherited from reg_smv_curr_overlay.py:

* The Backtester constructs a fresh optimizer instance for every hyperparameter
  combination, so the compiled problems and the step-1 solutions are cached at
  class level and keyed on the estimation window.
* Penalties enter as ``cp.Parameter``, so each problem canonicalizes once per
  window rather than once per grid point (DPP).  The mean vectors and gamma are
  compiled in as constants instead, which is why gamma joins the cache key; the
  mean vectors are functions of the window alone, so the frame signatures already
  cover them.
* Step 1 does not depend on the currency penalties, so it is solved once per
  (lambda_l1_asset, lambda_l2_asset) pair -- 16 solves per window rather than 256.
  That matters more here than for MV: each SMV solve carries T ~ 2600 auxiliary
  variables and costs roughly fifty times a mean-variance one.

Author: Anej Rozman
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

# Production runs use Gurobi (cluster). Fall back to Clarabel locally, where
# gurobipy is not installed; weight differences are ~1e-6.
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False


class RegularizedSemiVarianceCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step regularized mean-semi-variance optimizer with sequential currency overlay.

    Step 1 minimizes the penalized semi-variance of R_fh minus the pooled asset
    mean, subject to full investment and a leverage cap.  Step 2 fixes the asset
    weights and minimizes the penalized semi-variance of the combined portfolio
    return minus the carry mean of the overlay, subject to an L1 cap on psi.
    """

    # Shared across the per-grid-point instances the Backtester creates. Reset by
    # fit() whenever anything compiled into the problems changes.
    _cache_key: str | None = None
    _asset_prob: dict | None = None
    _overlay_prob: dict | None = None
    _asset_solutions: dict[tuple[float, float], np.ndarray] = {}

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``           (float) — risk aversion; the mean term enters
                                                with coefficient 1 / gamma, default 30
                * ``lambda_l1_asset`` (float) — L1 penalty for asset weights, default 1e-5
                * ``lambda_l2_asset`` (float) — L2 penalty for asset weights, default 1.5e-3
                * ``lambda_l1_curr``  (float) — L1 penalty for the overlay, default 0.0
                * ``lambda_l2_curr``  (float) — L2 penalty for the overlay, default 0.0
                * ``leverage_limit``  (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``   (float) — ||psi||_1 cap for step 2, default 2.0
                * ``solver``                  — cvxpy solver, default ``cp.GUROBI``
                                                (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 30)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 1e-5)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 1.5e-3)
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

        ``gamma``, ``leverage_limit`` and ``overlay_limit`` enter the key because
        they are compiled in as constants rather than passed as parameters.  The
        two mean vectors are compiled in as well, but they are functions of the
        window alone (``mu_a`` of R_fh and the economy map, ``mu_c`` of the forward
        premium aligned to R_c), so the frame signatures already cover them.
        """
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
        """
        Extract the return matrices, estimate both mean vectors, and refresh the
        problem cache if anything baked into it changed.

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
        """Compile the step-1 problem once, with the penalties left as parameters.

        The demeaned return matrix, mu_a and gamma are baked in as constants, so
        the compiled problem is valid only for the window it was built from;
        ``_refresh_cache`` drops it whenever the window or gamma changes."""
        T, N = Y_fh.shape
        A = Y_fh - Y_fh.mean(axis=0)

        x = cp.Variable(N)
        z = cp.Variable(T)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(
                cp.sum_squares(z) / T
                - (1.0 / self.gamma) * (mu_a @ x)
                + lam1 * l1_norm
                + lam2 * cp.sum_squares(x)
            ),
            [
                z >= 0,
                -A @ x <= z,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-1 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "x": x, "lam1": lam1, "lam2": lam2}

    def _optimize_assets(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
        """
        Step 1: penalized mean-semi-variance on R_fh with a leverage constraint and
        full investment.

        The solution depends only on the asset penalties within a window, so it is
        cached on (lambda_l1_asset, lambda_l2_asset); gamma is already part of the
        window cache key.
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
        Compile the step-2 problem once, with the penalties and the demeaned fixed
        asset contribution left as parameters.

        The demeaned currency matrix, mu_c and gamma are baked in as constants, so
        the compiled problem is valid only for the window it was built from.
        ``_refresh_cache`` drops it whenever the window changes, and :meth:`score`
        never re-solves on validation data, so no other window ever reaches it.
        """
        T, K = Y_c.shape
        A_c = Y_c - Y_c.mean(axis=0)

        psi = cp.Variable(K)
        z = cp.Variable(T)
        c_dm = cp.Parameter(T)   # demeaned fixed asset contribution
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        constraints = [
            z >= 0,
            -(c_dm + A_c @ psi) <= z,
            cp.norm(psi, 1) <= self.overlay_limit,
        ]
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(
            cp.Minimize(
                cp.sum_squares(z) / T
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
        """Step 2: penalized mean-semi-variance of (R_fh @ x_fixed + R_c @ psi)
        over psi, capped in L1.

        The constant mu_a' x* is dropped: it cannot move the argmin."""
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
        """
        Run both optimization steps sequentially.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or any solver step fails.
        """
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
        """
        Evaluate train-fitted candidate weights on the validation window.

        Both the asset weights and the currency overlay are taken as given from
        :meth:`optimize` on the training window and held fixed; nothing is
        re-solved on the validation sample, matching reg_smv_curr_overlay.py and
        the rest of the regularized family.  Refitting on validation and then
        measuring the in-sample objective of that fit drives selection to the
        weakest penalty in the grid.  The score is

            semideviation(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one.

        The risk is the semi-deviation, not the semi-variance
        reg_smv_curr_overlay.py scores on.  Both have the same minimizer when no
        mean is subtracted, so the choice is free there; with a mean term it is
        not.  A daily semi-variance on this data is on the order of 1e-5 against a
        daily portfolio mean near 2.7e-4, so ``semivariance - mean`` would be a
        mean maximization dressed up as a tradeoff.  The semi-deviation is on the
        order of 3e-3, and it is the degree-one convention
        ``in_sample_analysis.evaluate_risk`` already uses for ``"smv"`` -- the
        functional in_sample_vs_out_of_sample_w_mean.py scores these weights with,
        and the one the other three mean-bearing SMV and MV programs use, so the
        numbers are directly comparable across all of them.

        The reference coefficient of 1.0 is unbiased here in a way it is not in
        the unregularized pair.  gamma is fixed across the whole grid, so no
        candidate can earn a larger mean credit than another simply by carrying a
        smaller gamma; the search is selecting penalties alone.  The realized mean
        is used rather than mu_a' x + mu_c' psi because a score should measure what
        the weights actually earned on the window, not re-apply a shrunk or
        conditional forecast to them.

        Args:
            weights: Dict with ``"x"`` and ``"psi"`` as returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Validation semi-deviation minus realized mean return, lower is better.
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
        downside = np.minimum(port_rets - port_rets.mean(), 0.0)
        semidev = float(np.sqrt(np.mean(downside ** 2)))
        return semidev - float(port_rets.mean())
