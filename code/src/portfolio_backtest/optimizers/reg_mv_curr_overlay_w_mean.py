"""
Regularized mean-variance optimizer with currency overlay.

The mean-bearing counterpart of reg_mv_curr_overlay.py, and the regularized
counterpart of mv_curr_overlay_w_mean.py.  Both steps minimize the same penalized
mean-risk functional

    Var(portfolio return) - (1 / gamma) * mu' w + lambda_l1 ||w||_1 + lambda_l2 ||w||^2,

with gamma a risk-aversion coefficient in the conventional direction: large gamma
starves the mean term and the program collapses onto reg_mv_curr_overlay, small
gamma gives the mean its full weight.

Step 1: minimizes x' Sigma_a x - (1 / gamma) mu_a' x plus the asset penalties,
        with full investment and a leverage cap on x.  Sigma_a is Ledoit-Wolf
        shrunk toward constant correlation.
Step 2: with asset weights fixed, minimizes

            psi' Sigma_c psi + 2 x*' Sigma_ac psi - (1 / gamma) mu_c' psi
            + lambda_l1_curr ||psi||_1 + lambda_l2_curr ||psi||^2,

        subject to an L1 cap on the overlay.  As in the unregularized pair, the
        two terms that are constant in psi are dropped rather than carried: the
        asset-only variance x*' Sigma_a x*, and the asset mean mu_a' x*.

gamma is fixed rather than searched.  This config already carries
4 x 4 x 4 x 4 = 256 penalty combinations per rebalancing date, and crossing that
with MV_GAMMA_GRID's five points is not affordable at these universe sizes -- the
same reasoning REG_ES_MEAN_OVERLAY_CONFIG records.  The default of 30 is the
centre point of MV_GAMMA_GRID, where the full curve is tabulated.  On the
fixed-gamma sweep of the unregularized overlay it earns a Sharpe of 0.372 against
0.140 mean-free, at a drawdown of 26.6% against 28.2% -- so it still improves both
axes, one step more aggressive than the drawdown minimum at gamma = 50 (0.317 at
25.4%) and one step short of where the caps saturate at gamma = 20.
It is still read from hyperparams, so it can be swept deliberately later without
touching this code.  Note the sweep that chose it had no penalties competing
against the mean term, so the penalty that is optimal here may sit lower than it
does in the mean-free program -- worth re-reading the selected-lambda series once
these have run.

Fixing gamma also removes the one bias the unregularized pair had to reason
about.  There, :meth:`score` had to be gamma-free because a candidate carrying a
smaller gamma could earn an unboundedly larger mean credit; here every candidate
carries the same gamma, so no candidate can buy score that way and the reference
coefficient of 1.0 below is unbiased by construction.

Overlay cap.  reg_mv_curr_overlay.py leaves psi unconstrained and relies on the
penalties alone.  The cap is carried here, matching mv_curr_overlay_w_mean.py and
reg_es_curr_overlay_w_mean.py, but it is slack at the shipped
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

Means.  Estimated exactly as in mv_curr_overlay_w_mean.py: mu_a pooled onto the
14 economies and James-Stein shrunk toward zero, mu_c conditioned on the forward
premium as the carry -f_t.  That module records what each channel forecasts --
mu_c genuinely (slope +0.75 on the realized next quarter, tercile spread
+2.66 bp/day), mu_a not, cross-sectionally (slope -0.26, spread -0.82 bp/day).

Base currency.  Pinned to zero: its R_c column is identically zero so Sigma_c is
singular in that direction, its mu_c entry is zero, and the overlay cap would
otherwise let it consume budget for nothing.

No demeaning.  Variance is already location-free, so centering the return
matrices would leave Sigma_a, Sigma_c and Sigma_ac unchanged and only zero the mu
terms this program is built around.

Performance notes, inherited from reg_mv_curr_overlay.py:

* The Backtester constructs a fresh optimizer instance for every hyperparameter
  combination, so the compiled problems and the step-1 solutions are cached at
  class level and keyed on the estimation window.
* Penalties enter as ``cp.Parameter``, so each problem canonicalizes once per
  window rather than once per grid point (DPP).  The mean vectors and gamma are
  compiled in as constants instead, which is why gamma joins the cache key; the
  mean vectors are functions of the window alone, so the frame signatures already
  cover them.
* Step 1 does not depend on the currency penalties, so it is solved once per
  (lambda_l1_asset, lambda_l2_asset) pair rather than once per grid point.

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
    ledoit_wolf_cc,
    pooled_asset_mean,
    sample_cross_cov,
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


class RegularizedMeanVarianceCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step regularized mean-variance optimizer with sequential currency overlay.

    Step 1 minimizes the penalized variance of R_fh minus the pooled asset mean,
    subject to full investment and a leverage cap.  Step 2 fixes the asset weights
    and minimizes the penalized variance of the combined portfolio return minus
    the carry mean of the overlay, subject to an L1 cap on psi.
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
                * ``lambda_l1_asset`` (float) — L1 penalty for asset weights, default 5e-6
                * ``lambda_l2_asset`` (float) — L2 penalty for asset weights, default 1e-3
                * ``lambda_l1_curr``  (float) — L1 penalty for the overlay, default 5e-6
                * ``lambda_l2_curr``  (float) — L2 penalty for the overlay, default 1e-4
                * ``leverage_limit``  (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``   (float) — ||psi||_1 cap for step 2, default 2.0
                * ``solver``                  — cvxpy solver, default ``cp.GUROBI``
                                                (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 30)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 5e-6)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 1e-3)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 5e-6)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 1e-4)
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

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract the return matrices, estimate both mean vectors, and refresh the
        problem cache if the window changed.

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

    # Step 1: asset weights

    def _build_asset_problem(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> dict:
        """Compile step 1 once, with the two asset penalties left as parameters.

        Sigma_a, mu_a and gamma are baked in as constants, so the compiled problem
        is valid only for the window it was built from; ``_refresh_cache`` drops it
        whenever the window or gamma changes."""
        N = Y_fh.shape[1]
        Sigma_a = ledoit_wolf_cc(Y_fh)

        x = cp.Variable(N)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        l1_norm = cp.norm(x, 1)
        obj = (
            cp.quad_form(x, cp.psd_wrap(Sigma_a))
            - (1.0 / self.gamma) * (mu_a @ x)
            + lam1 * l1_norm
            + lam2 * cp.sum_squares(x)
        )
        problem = cp.Problem(
            cp.Minimize(obj),
            [
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-1 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "x": x, "lam1": lam1, "lam2": lam2}

    def _optimize_assets(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
        """
        Step 1: penalized mean-variance on R_fh with a Ledoit-Wolf covariance, a
        leverage constraint and full investment.

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

    def _build_overlay_problem(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, mu_c: np.ndarray
    ) -> dict:
        """
        Compile step 2 once, with the penalties and the fixed asset
        cross-covariance contribution left as parameters.

        The moment estimates, mu_c and gamma are baked in as constants, so the
        compiled problem is valid only for the window it was built from.
        ``_refresh_cache`` drops it whenever the window changes, and :meth:`score`
        never re-solves on validation data, so no other window ever reaches it.
        """
        K = Y_c.shape[1]
        Sigma_c = ledoit_wolf_cc(Y_c)
        Sigma_ac = sample_cross_cov(Y_fh, Y_c)  # N x K, not shrunk

        psi = cp.Variable(K)
        cross = cp.Parameter(K)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        obj = (
            cp.quad_form(psi, cp.psd_wrap(Sigma_c))
            + cross @ psi
            - (1.0 / self.gamma) * (mu_c @ psi)
            + lam1 * cp.norm(psi, 1)
            + lam2 * cp.sum_squares(psi)
        )
        # The cap is load-bearing here, unlike in the mean-free program: the grid
        # contains lambda_l1_curr = lambda_l2_curr = 0. See the module docstring.
        constraints = [cp.norm(psi, 1) <= self.overlay_limit]
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(cp.Minimize(obj), constraints)
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-2 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "psi": psi, "cross": cross,
                "lam1": lam1, "lam2": lam2, "Sigma_ac": Sigma_ac}

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray, mu_c: np.ndarray
    ) -> np.ndarray:
        """Step 2: penalized mean-variance currency overlay, capped in L1.

        Minimizes psi' Sigma_c psi + 2 x*' Sigma_ac psi - mu_c' psi / gamma plus
        the overlay penalties, the expansion of Var(x*' R_fh + psi' R_c) less the
        mean term, with both terms that are constant in psi dropped -- the
        asset-only variance and the asset mean mu_a' x*."""
        cls = type(self)
        if cls._overlay_prob is None:
            cls._overlay_prob = self._build_overlay_problem(Y_fh, Y_c, mu_c)

        p = cls._overlay_prob
        p["cross"].value = 2.0 * (x_fixed @ p["Sigma_ac"])  # length-K vector
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
        re-solved on the validation sample, exactly as in
        reg_mv_curr_overlay.py.  Refitting on validation and then measuring the
        in-sample objective of that fit drives selection to the weakest penalty in
        the grid.  The score is

            std(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one.

        The risk is the standard deviation, not the variance reg_mv_curr_overlay
        scores on.  Both have the same minimizer when no mean is subtracted, so
        the choice is free there; with a mean term it is not.  A daily variance on
        this data is about 2.6e-5 against a daily portfolio mean near 2.7e-4, so
        ``var - mean`` would be a ten-to-one mean maximization dressed up as a
        tradeoff.  The standard deviation is about 2.9e-3, and it is the
        degree-one convention ``in_sample_analysis.evaluate_risk`` already uses for
        ``"mv"`` -- the functional in_sample_vs_out_of_sample_w_mean.py scores
        these weights with, and the one the unregularized mean-bearing MV pair
        uses, so the numbers are directly comparable across all four.

        The reference coefficient of 1.0 is unbiased here in a way it is not in
        the unregularized pair.  gamma is fixed across the whole grid, so unlike
        there, no candidate can earn a larger mean credit than another simply by
        carrying a smaller gamma; the search is selecting penalties alone.  The
        realized mean is used rather than mu_a' x + mu_c' psi because a score
        should measure what the weights actually earned on the window, not
        re-apply a shrunk or conditional forecast to them.

        Args:
            weights: Dict with ``"x"`` and ``"psi"`` as returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Validation standard deviation minus realized mean return, lower is better.
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
        return float(np.std(port_rets, ddof=1)) - float(port_rets.mean())
