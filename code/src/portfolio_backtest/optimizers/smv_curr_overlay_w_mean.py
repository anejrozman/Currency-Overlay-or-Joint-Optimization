"""
Two-step mean-Semi-Variance optimizer with currency overlay
(Markowitz, Todd, Xu & Yamane 1993, in the framework of Ulrych & Lucescu 2026, Section 3).

The mean-bearing counterpart of smv_curr_overlay.py.  Both steps minimize the
same mean-risk functional

    SMV(portfolio return) - (1 / gamma) * mu' (decision variable),

so gamma is a risk-aversion coefficient in the conventional direction: large
gamma starves the mean term and the program collapses onto the pure-SMV overlay,
small gamma gives the mean its full weight.  The normalization is by 1 / gamma
rather than gamma / 2 multiplying the risk term so that the objective stays on
the semi-variance scale for every gamma, matching mv_curr_overlay_w_mean.py; the
two are identical up to a positive rescaling and give the same weights, since
this program is unregularized and its constraints do not scale with the objective.

Step 1: minimizes (1/T) sum_t [max(0, -(A x)_t)]^2 - (1/gamma) mu_a' x over the
        demeaned fully hedged asset returns, with full investment and a leverage
        cap on x.
Step 2: with asset weights fixed, minimizes

            (1/T) sum_t [max(0, -(c_dm + A_c psi)_t)]^2 - (1/gamma) mu_c' psi,

        over the currency overlay psi, subject to an L1 cap on the overlay.  The
        mu_a' x* term is a constant there and cannot move the argmin, so it is
        dropped rather than carried -- which is also what lets :meth:`score` skip
        re-estimating mu_a on the validation window.

The risk measure and its reformulation are unchanged from smv_curr_overlay.py:
Markowitz et al.'s S_E, eq. (3.1a), with the target E set to the portfolio's own
expected return, lifted into T auxiliary variables via their eqs. (3.4)-(3.7).
Only the downside half is needed -- minimizing sum z_t^2 subject to z >= -A w and
z >= 0 drives each z_t to max(0, -A_t w) on its own -- so each step is a QP with
T auxiliary variables rather than 2T, and the 1/sqrt(T) of eq. (3.8) is carried
as an explicit 1/T on the objective.

Location.  The demeaning is what makes this work, and it is worth being explicit
about why.  A = Y - Y.mean(axis=0) turns the semi-variance into a *deviation*
measure: it is invariant to a location shift of the return distribution, so the
mean reaches the objective only through the explicit, pooled and shrunk mu terms,
exactly as a variance does in mv_curr_overlay_w_mean.py.  Dropping the demeaning
would turn the risk term into a lower partial moment about zero, which is not
location-free and would therefore reward a high mean at fixed weight one on top
of the explicit term -- the double-counting that es_curr_overlay_w_mean.py carries
because raw ES is not a deviation measure either.  The deviation form is used
here for that reason, and it is why gamma -> infinity recovers smv_curr_overlay.py.

One inconsistency the deviation form does leave, stated rather than fixed: the
demeaning subtracts the raw per-column sample mean, while the mean term rewards
the pooled and shrunk mu_a.  Risk is measured about a noisy target and return is
forecast with a deliberately quiet one.  Subtracting mu_a instead would be
internally consistent but would break the correspondence with the mean-free
program and depart from the risk measure as Markowitz et al. define it.

Overlay cap.  Carried at 2.0 to match mv_curr_overlay_w_mean.py, so that every
mean-bearing program in the thesis shares one overlay budget and the MV/SMV and
joint/overlay comparisons are not confounded by different feasible sets.  Note
what this costs, because it does not cost the MV pair anything.  The mean-free
SMV overlay already runs a larger book than the MV one -- across fourteen
estimation windows its gross psi is 0.40 to 2.54 (median 1.40), against 0.55 to
1.52 for mv_curr_overlay -- so a cap of 2.0 binds on 2 of those 14 windows (14%)
*with the mean term switched off entirely*.  This program is therefore not a strict
superset of smv_curr_overlay.py: at gamma -> infinity it reproduces it only on
windows where the mean-free solution already fits inside the cap.  Where the cap
binds, psi sits on a corner of the L1 ball with its size set by the budget rather
than by the risk-return tradeoff.

Means.  Estimated exactly as in mv_curr_overlay_w_mean.py, and for the same
reasons: mu_a pooled onto the 14 economies and James-Stein shrunk toward zero
(:func:`~portfolio_backtest.utils.estimation.pooled_asset_mean`), mu_c not
estimated from returns at all but conditioned on the forward premium as the carry
-f_t (:func:`~portfolio_backtest.utils.estimation.conditional_currency_mean`).
That module records what each channel actually forecasts over the following
quarter -- mu_c genuinely (slope +0.75, tercile spread +2.66 bp/day), mu_a not,
cross-sectionally (slope -0.26, spread -0.82 bp/day).

Economy mapping.  Taken from the ``"C"`` bundle entry, the asset-to-currency
indicator matrix the Backtester already assembles and restricts to the window's
valid assets.  Its columns are R_c's columns in order, so economy k and currency
k are the same object.

Base currency.  Pinned to zero.  Its R_c column is identically zero so it never
enters the risk term, its mu_c entry is zero, and the overlay cap would otherwise
let it consume budget while contributing nothing.

Cost.  Unlike the MV programs, each step here is a QP in T + N variables rather
than N, so a single solve is roughly fifty times slower (about 1.0s against 0.02s
on a ten-year window of ~120 assets).  That is worth knowing before crossing this
optimizer with a large grid.

Author: Anej Rozman
"""

from __future__ import annotations

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


class SemiVarianceCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step mean-semi-variance optimizer with sequential currency overlay.

    Step 1 minimizes the semi-variance of R_fh minus the pooled asset mean,
    subject to full investment and a leverage cap.  Step 2 fixes the asset
    weights and minimizes the semi-variance of the combined portfolio return
    minus the carry mean of the overlay, subject to an L1 cap on psi.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``          (float) — risk aversion; the mean term enters
                                               with coefficient 1 / gamma, default 15
                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``  (float) — ||psi||_1 cap for step 2, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 15)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.overlay_limit: float = self.hyperparams.get("overlay_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

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
        self.Y_fh = np.asarray(R_fh.values, dtype=float)
        self.Y_c = np.asarray(R_c.values, dtype=float)
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

    def _optimize_assets(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
        """Step 1: mean-semi-variance on R_fh with a leverage cap and full
        investment.  The demeaning makes the risk term location-free, so the mean
        enters only through mu_a; see the module docstring."""
        T, N = Y_fh.shape
        A = Y_fh - Y_fh.mean(axis=0)

        x = cp.Variable(N)
        z = cp.Variable(T)

        objective = cp.sum_squares(z) / T - (1.0 / self.gamma) * (mu_a @ x)
        problem = cp.Problem(
            cp.Minimize(objective),
            [
                z >= 0,
                -A @ x <= z,
                cp.sum(x) == 1,
                cp.norm(x, 1) <= self.leverage_limit,
            ],
        )
        self._solve(problem)

        weights = x.value
        if weights is None:
            raise RuntimeError("Solver returned None weights in step 1.")
        return (weights / weights.sum()).flatten()

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray, mu_c: np.ndarray
    ) -> np.ndarray:
        """Step 2: mean-semi-variance of (R_fh @ x_fixed + R_c @ psi) over psi,
        capped in L1.

        The constant mu_a' x* is dropped: it cannot move the argmin.  The
        base-currency weight is pinned to zero; see the module docstring."""
        T, K = Y_c.shape
        c = Y_fh @ x_fixed                  # fixed asset contribution, shape (T,)
        c_dm = c - c.mean()                 # demeaned; see module docstring
        A_c = Y_c - Y_c.mean(axis=0)

        psi = cp.Variable(K)
        z = cp.Variable(T)

        objective = cp.sum_squares(z) / T - (1.0 / self.gamma) * (mu_c @ psi)
        constraints = [
            z >= 0,
            -(c_dm + A_c @ psi) <= z,
            cp.norm(psi, 1) <= self.overlay_limit,
        ]
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        self._solve(cp.Problem(cp.Minimize(objective), constraints))

        overlay = psi.value
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
            ValueError: If ``gamma`` is not strictly positive.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")

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
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the candidate weights on a common, gamma-free yardstick.

        Uses ``self.x_opt`` (asset weights from :meth:`optimize`) as fixed,
        re-runs step 2 on the validation returns to get psi_val -- mirroring
        smv_curr_overlay.py and the other unregularized overlays -- then returns

            semideviation(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one, identical for every candidate.

        The risk is the semi-deviation, not the semi-variance smv_curr_overlay.py
        scores on.  Both have the same minimizer when no mean is subtracted, so
        the choice is free there; with a mean term it is not.  A daily
        semi-variance on this data is on the order of 1e-5 against a daily
        portfolio mean near 2.7e-4, so ``semivariance - mean`` would be a mean
        maximization dressed up as a tradeoff.  The semi-deviation is on the order
        of 3e-3, and it is the degree-one convention
        ``in_sample_analysis.evaluate_risk`` already uses for ``"smv"``
        ("semi-deviation, not semi-variance ... so the ratios stay comparable") --
        the functional in_sample_vs_out_of_sample_w_mean.py scores these weights
        with, and the one the mean-bearing MV pair uses, so the numbers are
        directly comparable across measures.

        The realized mean is used directly rather than reconstructed from mu_a and
        mu_c: a score measures what the weights actually earned on the window, and
        re-applying a shrunk or conditional forecast there would let a candidate
        score well on a mean it never realized.  mu_a is not re-estimated on the
        validation window at all, since step 2 drops the constant mu_a' x*.

        On the MV pair this score was measured to lean conservative -- it selects
        the largest gamma in the band on most dates rather than locating an
        interior optimum, and alternative scores move which endpoint is preferred
        rather than making the search discriminate (see
        mv_curr_overlay_w_mean.score).  Read the selected-gamma series here with
        the same caution.

        Args:
            weights: Not used directly; asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"``, ``"R_c"`` and ``"F"``.

        Returns:
            Validation semi-deviation minus realized mean return, lower is better.
        """
        if self.x_opt is None:
            raise RuntimeError("optimize() must be called before score().")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty or len(R_fh_val) < 2:
            return float("inf")

        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)

        # Step 2 is re-solved here, so it still needs this window's own carry
        # vector -- the objective it minimizes is the gamma-scaled one.  Only the
        # score returned below is gamma-free.  Note the Backtester ends the
        # validation window at the rebalancing date, the same date the full
        # estimation window ends on, so this reads the same last forward-premium
        # row the production refit conditions on.
        mu_c_val = conditional_currency_mean(val_bundle.get("F"), Y_c_val)

        psi_val = self._optimize_overlay(Y_fh_val, Y_c_val, self.x_opt, mu_c_val)
        port_rets = Y_fh_val @ self.x_opt + Y_c_val @ psi_val

        downside = np.minimum(port_rets - port_rets.mean(), 0.0)
        semidev = float(np.sqrt(np.mean(downside ** 2)))
        return semidev - float(port_rets.mean())
