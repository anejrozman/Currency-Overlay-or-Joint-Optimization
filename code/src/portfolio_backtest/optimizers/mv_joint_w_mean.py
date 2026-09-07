"""
Unregularized joint mean-variance optimizer over asset and currency weights.

The mean-bearing counterpart of mv_joint.py, and the joint counterpart of
mv_curr_overlay_w_mean.py.  Minimizes

    Var(R_fh @ x + R_c @ psi) - (1 / gamma) * (mu_a' x + mu_c' psi)

jointly over x and psi in a single program, subject to full investment and a
leverage cap on x and an L1 cap on psi.  gamma is a risk-aversion coefficient in
the conventional direction: large gamma starves the mean term and the program
collapses onto mv_joint, small gamma gives the mean its full weight.  As in the
overlay, the 1 / gamma normalization keeps the objective on the variance scale
for every gamma rather than rescaling the risk term; the two are identical up to
a positive rescaling and give the same weights, since this program is
unregularized and its constraints do not scale with the objective.  The
convention differs by a factor of two from the (gamma / 2) * Var - mu' w form the
discarded mean-bearing MV programs used (backtest_config.py).

The intended difference from mv_curr_overlay_w_mean.py is the sequencing.  There,
x is fixed by an asset-only program and psi is fitted against it afterwards; here
both are chosen together, so the mean tilt on the asset book and the carry tilt
on the overlay are traded off against a single variance rather than in sequence.

Two differences that are not sequencing, and that matter when the two programs
are compared as a gap (in_sample_vs_out_of_sample_w_mean.py):

  * The cross-covariance estimator differs, inherited from the mean-free pair.
    mv_curr_overlay.py, and so the overlay here, estimates Sigma_c with
    Ledoit-Wolf and takes the asset-currency block Sigma_ac as the raw sample
    cross-covariance.  mv_joint.py, and so this program, shrinks the whole
    stacked covariance of [R_fh, R_c] in one Ledoit-Wolf estimate, so the
    cross-block is shrunk along with the diagonal blocks.  The joint program is
    therefore working with a differently regularized cross-block, not only a
    different decision order.
  * mu_a' x is a live term here.  In the overlay's step 2 the asset weights are
    already fixed, so mu_a' x* is an additive constant that cannot move the
    argmin and is dropped -- which is what lets that optimizer's score() skip
    estimating mu_a on the validation window.  Here x is a decision variable, so
    the term is part of the objective and :meth:`score` has to estimate mu_a on
    the validation window as well.

Means.  Identical to the overlay's, and for the same reasons:

  * Assets.  mu_a is pooled onto the 14 economies and James-Stein shrunk toward
    zero (:func:`~portfolio_backtest.utils.estimation.pooled_asset_mean`), which
    cuts the estimated dimension from N ~ 100-130 to at most 14 and restricts the
    cross-sectional spread of expected returns to the between-economy dimension.
    The pooling is what does that work rather than the shrinkage on top of it:
    the James-Stein weight has a median of 0.19 across estimation windows, so
    most of the raw economy mean survives.
  * Currencies.  mu_c is not estimated from returns at all but conditioned on the
    forward premium, mu_c = -f_t
    (:func:`~portfolio_backtest.utils.estimation.conditional_currency_mean`).

The measurements of what each channel forecasts, recorded in
mv_curr_overlay_w_mean.py, apply unchanged here: mu_c is a genuine conditional
predictor of the next quarter (slope +0.75, tercile spread +2.66 bp/day) while
mu_a is not, cross-sectionally (slope -0.26, tercile spread -0.82 bp/day).

Overlay cap.  Unlike es_joint_w_mean.py, this program does not need the cap for
boundedness.  ES is positively homogeneous, so a direction d with
ES(R_c d) < mu_c' d / gamma sends that program to minus infinity; a variance is
quadratic, so with the base currency pinned the stacked covariance is positive
definite on the free directions and the objective grows without bound in every
direction of psi.  The cap is carried anyway, for two reasons.  It keeps the
constraint set identical to mv_curr_overlay_w_mean.py, without which a joint
versus overlay gap would confound the sequencing with a difference in what each
program is allowed to hold.  And the unconstrained solution is large in practice:
the mean shifts the overlay by Sigma^-1 mu / (2 * gamma), whose gross L1 runs to
double digits over the useful gamma range, the pathology recorded for the
discarded mean-bearing MV overlay at 98x notional.

Base currency.  Pinned to zero.  Its R_c column is identically zero, so the
stacked covariance is singular in that direction and psi_base is unidentified by
the risk term, and the overlay cap would otherwise let it consume budget while
contributing nothing to the variance or the mean (its mu_c entry is zero too).

Budget.  sum(x) == 1 is imposed and the solution is checked rather than
renormalized, following mv_joint.py: renormalizing would mask solver drift and
silently break the leverage cap.

No demeaning.  The ``demean`` switch the ES and SOES programs carry has no
analogue here.  Variance is already location-free, so centering the return
matrices leaves the stacked covariance unchanged and would only zero the mu terms
this program is built around.

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
# the solve is not trustworthy rather than merely imprecise. Same values as
# mv_joint.py.
_BUDGET_WARN_TOL = 1e-6
_BUDGET_FAIL_TOL = 1e-4


class MeanVarianceJointWithMeanOptimizer(BaseOptimizer):
    """
    Joint unregularized mean-variance optimizer over asset and currency overlay weights.

    Solves one CVXPY problem minimizing the variance of (R_fh @ x + R_c @ psi)
    minus the gamma-scaled mean of the same combination, subject to full
    investment and a leverage cap on x and an L1 cap on psi.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``          (float) — risk aversion; the mean term enters
                                               with coefficient 1 / gamma, default 100
                * ``leverage_limit`` (float) — ||x||_1 cap, default 2.0
                * ``overlay_limit``  (float) — ||psi||_1 cap, default 2.0
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

    def _optimize_joint(
        self,
        Y_fh: np.ndarray,
        Y_c: np.ndarray,
        mu_a: np.ndarray,
        mu_c: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Jointly minimize mean-variance of (R_fh @ x + R_c @ psi) over x and psi.

        The joint covariance (Ledoit-Wolf) is estimated on the stacked return
        vector [R_fh, R_c], so the asset-currency cross-covariance is shrunk
        together with the asset and currency blocks (appendix B,
        eq:mv_joint_obj), and the mean enters as the stacked mu against the same
        theta."""
        N = Y_fh.shape[1]
        K = Y_c.shape[1]
        Y = np.hstack([Y_fh, Y_c])  # q x (N + K) combined return matrix
        Sigma_joint = ledoit_wolf_cc(Y)

        x = cp.Variable(N)
        psi = cp.Variable(K)
        theta = cp.hstack([x, psi])
        objective = (
            cp.quad_form(theta, cp.psd_wrap(Sigma_joint))
            - (1.0 / self.gamma) * (mu_a @ x + mu_c @ psi)
        )

        constraints = [
            cp.sum(x) == 1,                          # budget on the asset block only
            cp.norm(x, 1) <= self.leverage_limit,    # leverage on the asset block only
            cp.norm(psi, 1) <= self.overlay_limit,   # overlay cap; see module docstring
        ]
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        self._solve(cp.Problem(cp.Minimize(objective), constraints))

        weights = x.value
        if weights is None:
            raise RuntimeError("Solver returned None for asset weights.")
        overlay = psi.value
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

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run the joint optimization over training data.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
            ValueError: If ``gamma`` is not strictly positive.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")

        logger.debug(
            "Joint mean-variance optimization on %d obs, %d assets, %d currencies.",
            self.q, self.N, self.K,
        )
        return self._optimize_joint(self.Y_fh, self.Y_c, self.mu_a, self.mu_c)

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the candidate on a common, gamma-free yardstick.

        Re-runs the joint optimization on the validation window at this
        candidate's gamma -- mirroring mv_joint.py, which likewise re-solves
        rather than carrying the trained weights over -- and then returns

            std(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one, identical for every candidate.

        Unlike mv_curr_overlay_w_mean.score, this has to estimate mu_a on the
        validation window: the re-solve here optimizes over x, so mu_a' x is a
        live term rather than the additive constant it is in the overlay's step 2.

        The risk is the standard deviation, not the variance mv_joint.py scores
        on.  Both have the same minimizer when no mean is subtracted, so the
        choice is free there; with a mean term it is not.  A daily variance on
        this data is about 2.6e-5 against a daily portfolio mean near 2.7e-4, so
        ``var - mean`` would be a ten-to-one mean maximization dressed up as a
        tradeoff.  The standard deviation is about 2.9e-3, and it is the
        degree-one convention ``in_sample_analysis.evaluate_risk`` already uses
        for ``"mv"`` -- which matters, because that is the functional
        in_sample_vs_out_of_sample_w_mean.py scores these weights with.

        The realized mean is used directly rather than reconstructed from mu_a
        and mu_c.  A score measures what the weights actually earned on the
        window; re-applying a shrunk or conditional forecast there would let a
        candidate score well on a mean it never realized.

        One asymmetry to be aware of when reading a selected-gamma series, and it
        is specific to the joint program.  mu_a is re-estimated here on the
        one-year validation window, where James-Stein has far less to work with
        than on the ten-year estimation window: the shrinkage weight rises from a
        median of 0.19 to 0.74, and reaches 1.0 -- mu_a identically zero -- on 41%
        of rebalancing dates.  mu_c does not degrade the same way, since it is read
        off the last forward-premium row, which is the same row in both windows.
        So gamma is chosen against a program whose asset-mean channel is largely
        or entirely switched off while its carry channel runs at full strength,
        then applied to a production refit where both are live.  The overlay
        avoids this by construction rather than by design: mu_a' x* is an additive
        constant in its step 2, so it is dropped and never re-estimated.

        Carrying the training window's mu_a into this re-solve instead was tried,
        which makes the scoring program structurally match production and removes
        the mild in-sample use of the mean.  It does not help: selection gets
        flatter (87 of 91 dates at gamma = 100 rather than 82) and the realized
        Sharpe falls from 0.309 to 0.274.  Holding the trained weights fixed
        instead of re-solving gives 0.272 with a near-identical selection.  All
        three variants differ on roughly five of the 91 dates, which is worth
        knowing before any of these Sharpe figures is quoted to three decimals:
        the spread across defensible implementations of this score is about the
        same size as the gap between the joint program and the overlay.

        On the overlay this score was measured to lean conservative -- it selects
        the largest gamma in the band on most dates rather than locating an
        interior optimum, and alternative scores move which endpoint is preferred
        rather than making the search discriminate (mv_curr_overlay_w_mean.score
        records the comparison).  The joint program trades both tilts against a
        single variance, so the same check is worth running here before its
        selected-gamma series is read as a fitted optimum.

        Args:
            weights: Not used; the joint optimization is rerun on validation data.
            val_bundle: Must contain ``"R_fh"``, ``"R_c"``, ``"C"`` and ``"F"``.

        Returns:
            Validation standard deviation minus realized mean return, lower is better.
        """
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty or len(R_fh_val) < 2:
            return float("inf")

        Y_fh_val = np.ascontiguousarray(R_fh_val.values, dtype=float)
        Y_c_val = np.ascontiguousarray(R_c_val.values, dtype=float)

        # The re-solve optimizes over x as well as psi, so both mean vectors are
        # live and both have to come from this window.  Note the Backtester ends
        # the validation window at the rebalancing date, the same date the full
        # estimation window ends on, so mu_c here reads the same last
        # forward-premium row the production refit conditions on.
        economy_ids = economy_ids_from_indicator(val_bundle.get("C"), Y_fh_val.shape[1])
        mu_a_val, _ = pooled_asset_mean(Y_fh_val, economy_ids)
        mu_c_val = conditional_currency_mean(val_bundle.get("F"), Y_c_val)

        result = self._optimize_joint(Y_fh_val, Y_c_val, mu_a_val, mu_c_val)
        port_rets = Y_fh_val @ result["x"] + Y_c_val @ result["psi"]

        return float(np.std(port_rets, ddof=1)) - float(port_rets.mean())
