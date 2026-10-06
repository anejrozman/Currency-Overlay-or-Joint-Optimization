"""
Two-step mean-Second-Order Expected Shortfall optimizer with currency overlay
(Krokhmal 2007, in the framework of Ulrych & Lucescu 2026, Section 3).

The mean-bearing counterpart of soes_curr_overlay.py, and the second-order
counterpart of es_curr_overlay_w_mean.py.  Both steps minimize the same mean-risk
functional

    HMCR_2(portfolio loss) - (1 / gamma) * mu' (decision variable),

so gamma is a risk-aversion coefficient in the conventional direction: large gamma
starves the mean term and the program collapses onto the pure-SOES overlay, small
gamma gives the mean its full weight.  The normalization is by 1 / gamma rather
than gamma multiplying the risk term so that the objective stays on the HMCR_2
scale for every gamma, which matters for the validation score; the two are
identical up to a positive rescaling and give the same weights, since this program
is unregularized and its constraints do not scale with the objective.

Step 1: minimizes HMCR_2(-(R_fh @ x)) - (1/gamma) mu_a' x over fully hedged asset
        returns, with full investment and a leverage cap on x.
Step 2: with asset weights fixed, minimizes

            HMCR_2(-(R_fh @ x* + R_c @ psi)) - (1/gamma) mu_c' psi,

        over the currency overlay psi, subject to an L1 cap on the overlay.  The
        mu_a' x* term is a constant there and cannot move the argmin, so it is
        dropped rather than carried -- which is also what lets :meth:`score` skip
        re-estimating mu_a on the validation window.

The risk measure, the unsquared L2 tail norm and the sqrt(q) normalization are
unchanged from soes_curr_overlay.py; see reg_soes_asset_full.py for why the norm
is not squared and why the normalization differs from ES.  Both steps are SOCPs
rather than LPs, because the epigraph of ||u||_2 is a second-order cone
constraint, so ``optimal_inaccurate`` is more likely here than for the ES
overlay's two LPs.

No demeaning, and what that costs.  The ``demean`` switch soes_curr_overlay.py
carries is deliberately absent, exactly as in es_curr_overlay_w_mean.py: centering
the return matrices would zero the very sample means this program is built around.
Measured on three estimation windows, running ``pooled_asset_mean`` on a demeaned
R_fh returns identically zero -- James-Stein shrinks to w = 1.000 -- against a
max |mu_a| of 3.9e-4 to 1.4e-3 on the raw returns.  So mu_a and mu_c are both
estimated from the raw matrices here, and the risk term sees those same raw
matrices.

That choice has a consequence worth stating plainly, because it is the one place
this program differs in kind from the MV, SMV and MAD mean-bearing pairs.  HMCR_2
is translation-equivariant, so an undemeaned HMCR_2 decomposes as

    HMCR_2(-r) = HMCR_2_deviation(-r) - mean(r),

which means the risk term already rewards a high mean at fixed weight one, on top
of the explicit 1 / gamma term.  The effective objective is therefore
HMCR_2_deviation - (1 + 1/gamma) * (a mean), where the implicit "1" multiplies the
raw portfolio sample mean while the explicit 1/gamma multiplies the pooled and
shrunk mu.  Two things follow.  gamma -> infinity does not give a mean-free
program; it gives soes_curr_overlay.py with ``demean=False``, which is not that
module's default.  And the shrinkage buys less protection than it would in a
deviation formulation, because an unshrunk mean channel sits alongside it at fixed
weight.  This mirrors es_curr_overlay_w_mean.py exactly, which is the point: ES and
SOES are the two non-deviation measures in the suite and are treated identically,
so an ES-versus-SOES comparison is clean even though an SOES-versus-MV one carries
this asymmetry.

Overlay cap.  Required rather than optional, and for the same reason it is in the
ES and MAD pairs.  HMCR_2 is positively homogeneous of degree one, so along
psi = t * d the step-2 objective behaves like

    t * [ HMCR_2(-(R_c d)) - mu_c' d / gamma ],

and runs to minus infinity for any gamma below

    gamma_crit = max_d [ mu_c' d / HMCR_2(-(R_c d)) ],

measured at 0.0155, 0.0224 and 0.0264 on the raw return matrices this program uses
(0.0152 to 0.0265 demeaned -- the mean is small enough against HMCR_2 that
centering barely moves the threshold).  Nothing in soes_curr_overlay's constraint
set rules that out; it is safe only because it carries no mean term.  The L1 cap
closes it off, and SOES_GAMMA_GRID stays above every measured gamma_crit.

The cap is set at 2.0, matching every other mean-bearing program.  Unlike the MV
and MAD pairs that is not free here: the mean-free SOES overlay already runs a
gross psi of 0.62, 2.01 and 2.99 on those three windows, so the cap binds on two of
them with the mean term switched off entirely, and this program is not a strict
superset of soes_curr_overlay.py.  ES and SMV are in the same position; MV and MAD
are the exceptions.

Base currency.  Pinned to zero, which soes_curr_overlay.py does not do.  There
psi_base is genuinely absent from an unconstrained SOCP -- its R_c column is
identically zero, so it enters neither the cone constraint nor the objective -- and
leaving it free is harmless.  The overlay cap changes that: |psi_base| enters
cp.norm(psi, 1), so a nonzero base weight would consume overlay budget while
contributing nothing to the risk term or to the mean (its mu_c entry is zero as
well).  This is the same argument es_curr_overlay_w_mean.py makes against
es_curr_overlay.py's free psi_base.

Means.  Estimated exactly as in es_curr_overlay_w_mean.py and the rest of the
mean-bearing family: mu_a pooled onto the 14 economies and James-Stein shrunk
toward zero (:func:`~portfolio_backtest.utils.estimation.pooled_asset_mean`), mu_c
not estimated from returns at all but conditioned on the forward premium as the
carry -f_t (:func:`~portfolio_backtest.utils.estimation.conditional_currency_mean`).
mv_curr_overlay_w_mean.py records what each channel actually forecasts over the
following quarter -- mu_c genuinely (slope +0.75, tercile spread +2.66 bp/day),
mu_a not, cross-sectionally (slope -0.26, spread -0.82 bp/day).

Economy mapping.  Taken from the ``"C"`` bundle entry, the asset-to-currency
indicator matrix the Backtester already assembles and restricts to the window's
valid assets.  Its columns are R_c's columns in order, so economy k and currency k
are the same object.

Solver fallback.  The production Gurobi settings stalled short of their tolerances
partway through every cluster run of this program over the full gamma grid, which
cvxpy reports as ``user_limit``.  Each solve in both steps therefore goes through
``utils.solver_fallback.solve_with_fallback``: the production solve first,
unchanged, then Gurobi with robust settings, then Clarabel.  A fallback solves to
looser tolerances than the production settings, so its solution is also checked
against the budget and the leverage or overlay cap before it is used.  Every
fallback is logged and kept in :attr:`solver_fallbacks`, which run_strategy.py
writes to ``solver_fallbacks.csv``.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar

from ..utils.estimation import (
    conditional_currency_mean,
    economy_ids_from_indicator,
    pooled_asset_mean,
)
from ..utils.solver_fallback import solve_with_fallback

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

# How far a fallback solution may miss the budget or exceed the leverage or overlay
# cap before it is rejected.
_CAP_FAIL_TOL = 1e-4


class SecondOrderExpectedShortfallCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step mean-HMCR_2 optimizer with sequential currency overlay.

    Step 1 minimizes HMCR_2 on R_fh minus the pooled asset mean, subject to full
    investment and a leverage cap.  Step 2 fixes the asset weights and minimizes
    HMCR_2 of the combined portfolio return minus the carry mean of the overlay,
    subject to an L1 cap on psi -- which this program, being degree-one, requires
    for boundedness.
    """

    # One record per solve that needed a fallback solver, across every instance the
    # Backtester creates in this run; run_strategy.py writes it to the run directory.
    solver_fallbacks: list[dict] = []

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — confidence level, default 0.80
                * ``gamma``          (float) — risk aversion; the mean term enters
                                               with coefficient 1 / gamma, default 0.5
                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``  (float) — ||psi||_1 cap for step 2, default 2.0.
                                               Required for boundedness below
                                               gamma_crit ~ 0.026; see the module
                                               docstring.
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)

        Note there is deliberately no ``demean`` key; see the module docstring.
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.gamma: float = self.hyperparams.get("gamma", 0.5)
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
        self.window: tuple[str, str] | None = None
        self.x_opt: np.ndarray | None = None

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract the return matrices and estimate both mean vectors.

        The matrices are kept raw -- not centered -- for both the risk term and the
        mean estimates; see the module docstring.

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
        self.window = (str(R_fh.index[0])[:10], str(R_fh.index[-1])[:10])

        self.economy_ids = economy_ids_from_indicator(train_bundle.get("C"), self.N)
        self.mu_a, w_a = pooled_asset_mean(self.Y_fh, self.economy_ids)
        self.mu_c = conditional_currency_mean(train_bundle.get("F"), self.Y_c)
        logger.debug(
            "Asset mean shrinkage w=%.4f (%d economies); carry mu_c gross=%.2e.",
            w_a, len(np.unique(self.economy_ids)), np.abs(self.mu_c).sum(),
        )

    def _solve(self, problem: cp.Problem, context: dict) -> str:
        """Solve ``problem`` and return the attempt that was accepted.

        On Gurobi this is ``"gurobi"``, ``"gurobi_robust"`` or ``"clarabel"``; see
        the module docstring.  Any other configured solver is used on its own."""
        if self.solver == cp.GUROBI:
            return solve_with_fallback(
                problem,
                dict(
                    BarConvTol=1e-8,
                    # BarConvTol does not govern the conic path; this is its QCP
                    # counterpart, and the SOC epigraph of ||u||_2 makes both steps
                    # SOCPs rather than the LPs the ES overlay solves.
                    BarQCPConvTol=1e-8,
                    Method=2,
                    ScaleFlag=0,
                    AggFill=0,
                    PreDual=1,
                    Presolve=1,
                    Threads=3,
                    FeasibilityTol=1e-9,
                    OptimalityTol=1e-9,
                ),
                context=context,
                events=type(self).solver_fallbacks,
                log=logger,
            )

        try:
            problem.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError as exc:
            raise RuntimeError(f"Solver error: {exc}") from exc

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Optimization failed: status={problem.status}")
        return str(self.solver)

    def _optimize_assets(self, Y_fh: np.ndarray, mu_a: np.ndarray, context: dict) -> np.ndarray:
        """Step 1: mean-HMCR_2 on R_fh with a leverage cap and full investment.

        Bounded by its constraints: sum(x) = 1 with an L1 leverage cap is a compact
        feasible set, so unlike step 2 this needs no cap of its own.  ``context``
        identifies the solve if it needs a fallback solver."""
        q, N = Y_fh.shape
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        u = cp.Variable(q)
        eta = cp.Variable()

        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(q))
        objective = soes_term - (1.0 / self.gamma) * (mu_a @ x)
        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(objective),
            [
                u >= 0,
                -Y_fh @ x - eta <= u,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        attempt = self._solve(problem, context)

        weights = x.value
        if weights is None:
            raise RuntimeError("Solver returned None weights in step 1.")
        if attempt in ("gurobi_robust", "clarabel"):
            budget, gross = weights.sum(), np.abs(weights).sum()
            if abs(budget - 1.0) > _CAP_FAIL_TOL or gross > self.leverage_limit + _CAP_FAIL_TOL:
                raise RuntimeError(
                    f"Fallback solution from {attempt} breaks step 1's constraints: "
                    f"sum(x) = {budget:.9f}, ||x||_1 = {gross:.9f} > {self.leverage_limit}."
                )
        return (weights / weights.sum()).flatten()

    def _optimize_overlay(
        self,
        Y_fh: np.ndarray,
        Y_c: np.ndarray,
        x_fixed: np.ndarray,
        mu_c: np.ndarray,
        context: dict,
    ) -> np.ndarray:
        """Step 2: mean-HMCR_2 of (R_fh @ x_fixed + R_c @ psi) over psi, capped in L1.

        The constant mu_a' x* is dropped: it cannot move the argmin.  The
        base-currency weight is pinned to zero so it cannot consume overlay budget
        for nothing; see the module docstring.  ``context`` identifies the solve if
        it needs a fallback solver."""
        q, K = Y_c.shape
        tail_prob = 1.0 - self.alpha

        c = Y_fh @ x_fixed  # constant asset return contribution, shape (q,)

        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()

        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(q))
        objective = soes_term - (1.0 / self.gamma) * (mu_c @ psi)
        constraints = [
            u >= 0,
            -c - Y_c @ psi - eta <= u,
            cp.norm(psi, 1) <= self.overlay_limit,
        ]

        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        attempt = self._solve(cp.Problem(cp.Minimize(objective), constraints), context)

        overlay = psi.value
        if overlay is None:
            raise RuntimeError("Solver returned None weights in step 2.")
        if attempt in ("gurobi_robust", "clarabel"):
            gross = np.abs(overlay).sum()
            if gross > self.overlay_limit + _CAP_FAIL_TOL:
                raise RuntimeError(
                    f"Fallback solution from {attempt} breaks the overlay cap: "
                    f"||psi||_1 = {gross:.9f} > {self.overlay_limit}."
                )
        return overlay.flatten()

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run both optimization steps sequentially.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or any solver step fails.
            ValueError: If ``alpha`` is outside *(0, 1)*, ``gamma`` is not positive,
                        or ``overlay_limit`` is not finite and positive -- an
                        infinite cap would make step 2 unbounded below
                        gamma_crit ~ 0.026.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")
        if not np.isfinite(self.overlay_limit) or self.overlay_limit <= 0.0:
            raise ValueError(
                "overlay_limit must be finite and positive: HMCR_2 is degree-one, so "
                "step 2 is unbounded below without it for gamma under ~0.026."
            )

        context = {
            "window_start": self.window[0],
            "window_end": self.window[1],
            "n_obs": self.q,
            "n_assets": self.N,
            "gamma": self.gamma,
        }

        logger.debug(
            "Step 1: optimizing asset weights on %d obs, %d assets.", self.q, self.N
        )
        self.x_opt = self._optimize_assets(
            self.Y_fh, self.mu_a, {"solve": "step 1 (assets)", **context}
        )

        logger.debug(
            "Step 2: optimizing currency overlay on %d obs, %d currencies.", self.q, self.K
        )
        psi_opt = self._optimize_overlay(
            self.Y_fh, self.Y_c, self.x_opt, self.mu_c, {"solve": "step 2 (overlay)", **context}
        )

        return {"x": self.x_opt, "psi": psi_opt}

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the candidate weights on a common, gamma-free yardstick.

        Uses ``self.x_opt`` (asset weights from :meth:`optimize`) as fixed, re-runs
        step 2 on the validation returns to get psi_val -- mirroring
        soes_curr_overlay.py and es_curr_overlay_w_mean.py -- then returns

            HMCR_2(combined loss) - 1.0 * mean(realized combined returns),

        the realized risk-return tradeoff of this window at a reference coefficient
        of one, identical for every candidate.

        HMCR_2 needs no transformation to reach a degree-one score: it is positively
        homogeneous of degree one, about 1.0e-2 daily against a daily portfolio mean
        near 3e-4, and it is exactly what ``in_sample_analysis.evaluate_risk``
        returns for ``"soes"`` -- the functional
        in_sample_vs_out_of_sample_w_mean.py scores these weights with.

        The HMCR_2 is re-derived here rather than read off the step-2 solve's eta,
        and that is necessary rather than decorative.  At p = 1 the empirical VaR of
        the realized returns is exactly the minimizing threshold however psi was
        obtained, so the ES overlays can read their score off a quantile.  At p = 2
        the minimizing zeta is only exactly recovered by the solve itself (see
        reg_soes_asset_full.py), and this SOCP is more prone to
        ``optimal_inaccurate`` than the ES overlay's LPs, so zeta is recomputed with
        the same bounded 1-D search the other SOES optimizers use.

        Because the risk term is not demeaned, the reference coefficient is
        effectively one half rather than one: HMCR_2 already contains a -mean(r) of
        its own by translation-equivariance, so this score is
        HMCR_2_deviation - 2 * mean(r).  That is a property of the undemeaned
        formulation, shared with es_curr_overlay_w_mean.score, and it is stated here
        rather than corrected so the two remain directly comparable.

        The realized mean is used directly rather than reconstructed from mu_a and
        mu_c: a score measures what the weights actually earned on the window, and
        re-applying a shrunk or conditional forecast there would let a candidate
        score well on a mean it never realized.  mu_a is not re-estimated on the
        validation window at all, since step 2 drops the constant mu_a' x*.

        On the MV pair this score was measured to lean conservative -- it selects the
        largest gamma in the band on most dates rather than locating an interior
        optimum, and alternative scores move which endpoint is preferred rather than
        making the search discriminate (see mv_curr_overlay_w_mean.score).  Read the
        selected-gamma series here with the same caution.

        Args:
            weights: Not used directly; asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"``, ``"R_c"`` and ``"F"``.

        Returns:
            Validation HMCR_2 minus realized mean return, lower is better.
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
        # estimation window ends on, so this reads the same last forward-premium row
        # the production refit conditions on.
        mu_c_val = conditional_currency_mean(val_bundle.get("F"), Y_c_val)

        context = {
            "solve": "step 2 (overlay), validation re-solve",
            "window_start": str(R_fh_val.index[0])[:10],
            "window_end": str(R_fh_val.index[-1])[:10],
            "n_obs": Y_fh_val.shape[0],
            "n_assets": Y_fh_val.shape[1],
            "gamma": self.gamma,
        }
        psi_val = self._optimize_overlay(Y_fh_val, Y_c_val, self.x_opt, mu_c_val, context)
        port_rets = Y_fh_val @ self.x_opt + Y_c_val @ psi_val

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
