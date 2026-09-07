"""
Two-step mean-variance optimizer with currency overlay.

The mean-bearing counterpart of mv_curr_overlay.py.  Both steps minimize the same
mean-risk functional

    Var(portfolio return) - (1 / gamma) * mu' (decision variable),

so gamma is a risk-aversion coefficient in the conventional direction: large
gamma starves the mean term and the program collapses onto the pure-variance
overlay, small gamma gives the mean its full weight.  The normalization is by
1 / gamma rather than gamma / 2 multiplying the variance, so the objective stays
on the variance scale for every gamma and the structure matches
es_curr_overlay_w_mean.py; the two are identical up to a positive rescaling and
give the same weights, since this program is unregularized and its constraints do
not scale with the objective.  Note that the conventions differ by a factor of
two: a gamma here is 2 / gamma in the (gamma / 2) * Var - mu' w form the discarded
mean-bearing MV programs used (backtest_config.py).

Step 1: minimizes x' Sigma_a x - (1 / gamma) * mu_a' x over fully hedged asset
        returns, with full investment and a leverage cap on x.  Sigma_a is
        Ledoit-Wolf shrunk toward constant correlation, as in mv_curr_overlay.py.
Step 2: with asset weights fixed, minimizes

            psi' Sigma_c psi + 2 x*' Sigma_ac psi - (1 / gamma) * mu_c' psi,

        the expansion of Var(x*' R_fh + psi' R_c) less the mean term, over the
        currency overlay psi, subject to an L1 cap on the overlay.  Sigma_c is
        Ledoit-Wolf shrunk; the cross-covariance Sigma_ac is the un-shrunk sample
        estimate, both as in mv_curr_overlay.py.  Two terms that are constant in
        psi are dropped rather than carried: the asset-only variance
        x*' Sigma_a x*, which mv_curr_overlay.py likewise drops, and the asset
        mean mu_a' x*.  Neither can move the argmin, and dropping the second is
        what lets :meth:`score` skip re-estimating mu_a on the validation window
        altogether.

Location.  This is where the MV program is cleaner than its ES sibling, and the
first reason the ES gamma grid cannot be reused.  Variance is a deviation
measure, invariant to a location shift of the return distribution, so the mean
reaches the objective only through the explicit, pooled and shrunk mu terms and
gamma -> infinity recovers mv_curr_overlay.py exactly.  Raw ES is not a deviation
measure -- ES(-r) = ES_deviation(-r) - E[r] -- so es_curr_overlay_w_mean.py, which
deliberately does not demean, carries a second implicit mean channel at fixed
weight one, estimated by the raw portfolio sample mean.  Its gamma is doing a
different job than gamma here, on top of the three-orders-of-magnitude scale gap
between an ES_80 and a variance.  The grid is therefore calibrated separately;
see MV_GAMMA_GRID in backtest_config.py.

Means.  Estimated exactly as in es_curr_overlay_w_mean.py, and for the same
reasons: the two estimation problems are not comparable in difficulty.

  * Assets.  N is 100-130 in the top-10 universe against q ~ 2500 daily
    observations, and a per-asset sample mean at that signal-to-noise ratio is
    almost pure noise -- which is why every mean-free optimizer in this package
    dropped the mean term outright (see the gamma = 3 blow-ups recorded in
    backtest_config.py).  The mean here is pooled onto the 14 economies before it
    is used: each economy's series is the equal-weighted average of its
    constituents, that panel is James-Stein shrunk toward zero, and every asset
    inherits its economy's shrunk mean
    (:func:`~portfolio_backtest.utils.estimation.pooled_asset_mean`).  Expected
    returns then vary only between economies, not within them.
  * Currencies.  Not estimated from returns at all: the mean is conditional on
    the forward premium, mu_c = -f_t, which is the carry
    (:func:`~portfolio_backtest.utils.estimation.conditional_currency_mean`).  An
    unconditional sample mean shrinks to exactly zero in most windows, since 13
    currency means over a 2500-day window carry almost no signal, which would
    leave step 2 with no mean term whatsoever.  The base currency is held at zero.

What the two channels are worth.  The justification above is about estimation
difficulty -- pooling makes mu_a stable -- which is not the same as making it
informative, so both channels were measured against what they actually predict
over the following quarter, on the top-10 universe:

  * mu_c is a genuine conditional forecast.  Regressed on the realized next-quarter
    excess currency return it has a slope of +0.75 against the +1 the estimator
    assumes, a correlation of +0.11, and sorting currencies on it separates realized
    returns by +2.66 bp/day between the top and bottom terciles.  Its level is
    unbiased (+0.02%/yr).
  * mu_a is not, at least not in the direct sense.  Regressed on the realized
    next-quarter economy return it has a slope of -0.26 and a correlation of -0.04,
    and the tercile sort runs the wrong way, -0.82 bp/day.  A ten-year trailing mean
    is a stable estimate of something that does not forecast the next quarter
    cross-sectionally.

Its level runs 2.4%/yr above realized, which is the point-in-time universe showing
through -- today's top-10 names are the ones that rose -- but that bias is harmless
here: with sum(x) = 1 a constant added to every entry of mu_a leaves the argmin
untouched, and step 2 drops mu_a' x* outright.  Only the cross-sectional dispersion
reaches the solution.

Despite the negative slope the backtest prefers mu_a on: at gamma = 10 it earns a
Sharpe of 0.481 against 0.390 with mu_a zeroed, and permuting the economy means
across economies -- preserving dispersion and scale exactly while destroying the
assignment -- gives 0.028, 0.296 and 0.338 over three seeds, worse than zeroing it.
So the specific assignment carries something the tercile sort does not capture,
plausibly the interaction with Sigma_a rather than raw predictability.  Read that as
suggestive rather than settled: three permutations spanning 0.03 to 0.34 put the
true-versus-permuted gap at roughly 1.6 permutation standard deviations, and a
persistent country tilt that this particular 2000-2023 sample rewards would look the
same.  More seeds would settle it if the write-up needs to claim mu_a contributes.

Economy mapping.  Taken from the ``"C"`` bundle entry, the asset-to-currency
indicator matrix the Backtester already assembles and restricts to the window's
valid assets.  Its columns are R_c's columns in order, so economy k and currency
k are the same object and no separate mapping has to be plumbed through.

Base currency.  Pinned to zero for two independent reasons here.  Its R_c column
is identically zero, so Sigma_c is singular in that direction and psi_base is
unidentified by the risk term -- which is why mv_curr_overlay.py pins it even
without a cap.  And the overlay cap puts |psi_base| into cp.norm(psi, 1), where a
nonzero base weight would consume overlay budget while contributing nothing to the
variance, the cross-covariance or the mean (its mu_c entry is zero as well).

Overlay cap.  mv_curr_overlay.py leaves psi unconstrained, and can afford to: with
the base currency pinned, the remaining Sigma_c block is positive definite, so its
QP is strictly convex and bounded.  Adding the linear mean term keeps the program
bounded but shifts the solution by Sigma_c^-1 mu_c / (2 * gamma), and that tilt is
large.  On a ten-year window its gross L1 is about 141, so an uncapped overlay
would reach a gross psi of 2 at gamma = 35, 7x notional at gamma = 10 and 70x at
gamma = 1 -- the pathology recorded for the discarded mean-bearing MV overlay,
which reached 98x.  The cap is therefore carried, as in es_curr_overlay_w_mean.py,
and it binds across most of the useful gamma range rather than being a formality.

No demeaning.  The ``demean`` switch the ES and SOES overlays carry has no
analogue here.  Variance is already location-free, so centering the return
matrices leaves Sigma_a, Sigma_c and Sigma_ac unchanged and would only zero the mu
terms this program is built around.

Author: Anej Rozman
"""

from __future__ import annotations

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


class MeanVarianceCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step mean-variance optimizer with sequential currency overlay.

    Step 1 minimizes the variance of R_fh minus the pooled asset mean, subject to
    full investment and a leverage cap.  Step 2 fixes the asset weights and
    minimizes the variance of the combined portfolio return minus the carry mean
    of the overlay, subject to an L1 cap on psi.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``          (float) — risk aversion; the mean term enters
                                               with coefficient 1 / gamma, default 100
                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``  (float) — ||psi||_1 cap for step 2, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 100)
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
        """Step 1: mean-variance on R_fh with a Ledoit-Wolf covariance, a leverage
        cap and full investment."""
        N = Y_fh.shape[1]
        Sigma_a = ledoit_wolf_cc(Y_fh)

        x = cp.Variable(N)
        objective = cp.quad_form(x, cp.psd_wrap(Sigma_a)) - (1.0 / self.gamma) * (mu_a @ x)
        problem = cp.Problem(
            cp.Minimize(objective),
            [
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
        self,
        Y_fh: np.ndarray,
        Y_c: np.ndarray,
        x_fixed: np.ndarray,
        mu_c: np.ndarray,
    ) -> np.ndarray:
        """Step 2: mean-variance of (R_fh @ x_fixed + R_c @ psi) over psi, capped
        in L1.

        Minimizes psi' Sigma_c psi + 2 x*' Sigma_ac psi - mu_c' psi / gamma, the
        expansion of Var(x*' R_fh + psi' R_c) less the mean term, with both terms
        that are constant in psi dropped -- the asset-only variance and the asset
        mean mu_a' x*.  The base-currency weight is pinned to zero; see the module
        docstring."""
        K = Y_c.shape[1]
        Sigma_c = ledoit_wolf_cc(Y_c)
        Sigma_ac = sample_cross_cov(Y_fh, Y_c)  # N x K, not shrunk
        cross = 2.0 * (x_fixed @ Sigma_ac)      # length-K constant vector

        psi = cp.Variable(K)
        objective = (
            cp.quad_form(psi, cp.psd_wrap(Sigma_c))
            + cross @ psi
            - (1.0 / self.gamma) * (mu_c @ psi)
        )
        constraints = [cp.norm(psi, 1) <= self.overlay_limit]

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
        mv_curr_overlay.py and the other unregularized overlays -- then returns

            std(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one, identical for every candidate.

        The risk is the standard deviation, not the variance mv_curr_overlay.py
        scores on.  Both have the same minimizer when no mean is subtracted, so
        the choice is free there; with a mean term it is not.  A daily variance on
        this data is about 2.6e-5 against a daily portfolio mean near 2.7e-4, so
        ``var - mean`` would be a ten-to-one mean maximization dressed up as a
        tradeoff.  The standard deviation is about 2.9e-3, an eleven-to-one
        risk-to-mean ratio in the same regime as the ES score's, and it is the
        degree-one convention ``in_sample_analysis.evaluate_risk`` already uses for
        ``"mv"`` ("standard deviation, not variance ... so the ratios stay
        comparable") -- which matters, because that is the functional
        in_sample_vs_out_of_sample_w_mean.py scores these weights with.

        The realized mean is used directly rather than reconstructed from mu_a and
        mu_c.  A score measures what the weights actually earned on the window;
        re-applying a shrunk or conditional forecast there would let a candidate
        score well on a mean it never realized.  mu_a is not re-estimated on the
        validation window at all, since step 2 drops the constant mu_a' x* and
        cannot use it.

        Scoring each candidate at its own gamma -- as an earlier version of the
        mean-bearing ES overlay did -- is what this avoids.  The mean credit
        mu' w / gamma grows without bound as gamma falls while the risk term does
        not, so that search ran to whichever gamma was smallest in the grid,
        overshooting the gamma that was best out of sample.

        This score has the opposite bias, and it is strong enough that the
        validation search barely functions as one.  A reference risk aversion of
        one in ``std - mean`` units means demanding a daily Sharpe of 1.0 before
        taking on any risk, which is severe, so the search runs to the most
        conservative point available: on MV_GAMMA_GRID's [10, 100] band it picks
        gamma = 100 on 87% of the 91 quarterly top-10 dates and never once reaches
        the lower endpoint, making the run a fixed gamma = 100 with noise.  Two
        alternative scores were measured on the same grid, and the picture is the
        same in the other direction -- scoring on the validation Sharpe selects
        gamma = 10 on 47% of dates, and scoring on the realized mean alone on 80%:

            score              SR      MDD     selection
            std - mean       0.2702   26.4%   87% at gamma = 100 (upper endpoint)
            val. Sharpe      0.4443   34.4%   47% at gamma = 10  (lower endpoint)
            realized mean    0.4523   35.2%   80% at gamma = 10  (lower endpoint)
            fixed gamma=100  0.2576   26.4%   --
            fixed gamma=10   0.4808   33.4%   --

        Each variant tracks a fixed gamma at one end of the band, and the two
        aggressive ones land slightly below simply fixing gamma at 10.  So the
        search is expressing an endpoint preference rather than locating an
        interior optimum, and switching scores moves which endpoint it prefers
        rather than making it discriminate.  The conservative one is kept: it is
        the variant whose selections are defensible out of sample without
        appealing to the realized mean of the scoring window, and its drawdown
        (26.4%) is the lowest of the three.  Read the selected-gamma series as a
        statement about the band's upper end, not as a fitted optimum.

        Args:
            weights: Not used directly; asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"``, ``"R_c"`` and ``"F"``.

        Returns:
            Validation standard deviation minus realized mean return, lower is better.
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

        return float(np.std(port_rets, ddof=1)) - float(port_rets.mean())
