"""
Unregularized joint mean-Second-Order Expected Shortfall optimizer over asset and
currency weights (Krokhmal 2007, in the framework of Ulrych & Lucescu 2026, Section 3).

The mean-bearing counterpart of soes_joint.py, and the joint counterpart of
soes_curr_overlay_w_mean.py.  Minimizes

    HMCR_2(-(R_fh @ x + R_c @ psi)) - (1 / gamma) * (mu_a' x + mu_c' psi)

jointly over x and psi in a single program, subject to full investment and a
leverage cap on x and an L1 cap on psi.  gamma is a risk-aversion coefficient in
the conventional direction: large gamma starves the mean term and the program
collapses onto soes_joint with ``demean=False``, small gamma gives the mean its
full weight.  The 1 / gamma normalization keeps the objective on the HMCR_2 scale
for every gamma; the two forms are identical up to a positive rescaling and give
the same weights, since this program is unregularized and its constraints do not
scale with the objective.

The risk measure -- the unsquared L2 tail norm and its sqrt(q) normalization -- is
unchanged from soes_joint.py; see reg_soes_asset_full.py for the derivation and
for why this is an SOCP rather than an LP.  ``optimal_inaccurate`` is
correspondingly more likely here than for the ES joint's LP.

The intended difference from soes_curr_overlay_w_mean.py is the sequencing: there
x is fixed by an asset-only program and psi is fitted against it afterwards, here
both blocks enter the same second-order cone constraint and are optimized
together.  One consequence matters for the code: unlike the overlay's step 2,
mu_a' x is a live term rather than a droppable constant, so :meth:`score` has to
estimate mu_a on the validation window as well.

No demeaning, and what that costs.  The ``demean`` switch soes_joint.py carries is
deliberately absent, exactly as in soes_curr_overlay_w_mean.py and
es_joint_w_mean.py: centering the return matrices would zero the very sample means
this program is built around.  Running ``pooled_asset_mean`` on a demeaned R_fh
returns identically zero -- James-Stein shrinks to w = 1.000 -- against a max
|mu_a| of 3.9e-4 to 1.4e-3 on the raw returns.  So mu_a and mu_c are estimated
from the raw matrices, and the risk term sees those same raw matrices.

That has a consequence worth stating plainly.  HMCR_2 is translation-equivariant,
so an undemeaned HMCR_2 decomposes as

    HMCR_2(-r) = HMCR_2_deviation(-r) - mean(r),

which means the risk term already rewards a high mean at fixed weight one, on top
of the explicit 1 / gamma term.  The effective objective is therefore
HMCR_2_deviation - (1 + 1/gamma) * (a mean), where the implicit "1" multiplies the
raw portfolio sample mean and the explicit 1/gamma multiplies the pooled and
shrunk mu.  gamma -> infinity therefore gives soes_joint.py with ``demean=False``,
which is not that module's default, and the shrinkage buys less protection than it
would in a deviation formulation.  This mirrors es_joint_w_mean.py exactly, which
is the point: ES and SOES are the two non-deviation measures in the suite and are
treated identically, so an ES-versus-SOES comparison is clean even though a
SOES-versus-MV one carries this asymmetry.

Overlay cap.  Required rather than optional.  HMCR_2 is positively homogeneous of
degree one, and x is confined to a compact set by sum(x) = 1 and the L1 leverage
cap, so psi is the only direction in which the program can run away: along
psi = t * d with x bounded, R_fh x stays bounded and mu_a' x with it, so the
objective behaves like

    t * [ HMCR_2(-(R_c d)) - mu_c' d / gamma ],

the same growth rate the overlay's step 2 has.  The critical value is therefore
the same quantity,

    gamma_crit = max_d [ mu_c' d / HMCR_2(-(R_c d)) ],

measured at 0.0155, 0.0224 and 0.0264 on the raw return matrices this program
uses.  Below it an uncapped joint program diverges in psi; nothing in soes_joint's
constraint set rules that out, and soes_joint is safe only because it carries no
mean term.  The L1 cap closes that off, and SOES_JOINT_GAMMA_GRID stays above
every measured gamma_crit.  The cap is set at 2.0, matching every other
mean-bearing program.

Base currency.  Pinned to zero, which soes_joint.py does not do.  There psi_base
is genuinely absent from an unconstrained SOCP -- its R_c column is identically
zero, so it enters neither the cone constraint nor the objective -- and leaving it
free is harmless.  The overlay cap changes that: |psi_base| enters
cp.norm(psi, 1), so a nonzero base weight would consume overlay budget while
contributing nothing to the risk term or to the mean (its mu_c entry is zero too).

Budget.  sum(x) == 1 is imposed and the solution is checked rather than
renormalized.  This departs from soes_joint.py, which divides through by
weights.sum(), and the departure is deliberate: renormalizing masks solver drift
and can silently push ||x||_1 past the leverage cap.  That matters more here than
anywhere else in the suite, because this is an SOCP and ``optimal_inaccurate`` is
the most likely status in the family -- exactly the case where a silent rescaling
would do damage.  mv_joint.py, smv_joint.py and mad_joint.py already check rather
than renormalize; this follows them.

Means.  Estimated exactly as in soes_curr_overlay_w_mean.py: mu_a pooled onto the
14 economies and James-Stein shrunk toward zero, mu_c not estimated from returns
at all but conditioned on the forward premium as the carry -f_t.
mv_curr_overlay_w_mean.py records what each channel actually forecasts over the
following quarter -- mu_c genuinely, mu_a not, cross-sectionally.

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


class SecondOrderExpectedShortfallJointWithMeanOptimizer(BaseOptimizer):
    """
    Joint unregularized mean-HMCR_2 optimizer over asset and currency overlay weights.

    Solves one CVXPY problem minimizing HMCR_2 of (R_fh @ x + R_c @ psi) minus the
    gamma-scaled mean of the same combination, subject to full investment and a
    leverage cap on x and an L1 cap on psi -- which this program, being degree-one,
    requires for boundedness.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — confidence level, default 0.80
                * ``gamma``          (float) — risk aversion; the mean term enters
                                               with coefficient 1 / gamma, default 0.1
                * ``leverage_limit`` (float) — ||x||_1 cap, default 2.0
                * ``overlay_limit``  (float) — ||psi||_1 cap, default 2.0.  Required
                                               for boundedness below
                                               gamma_crit ~ 0.026; see the module
                                               docstring.
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)

        Note there is deliberately no ``demean`` key; see the module docstring.
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.gamma: float = self.hyperparams.get("gamma", 0.1)
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
                    # BarConvTol does not govern the conic path; this is its QCP
                    # counterpart, and the SOC epigraph of ||u||_2 makes this an
                    # SOCP rather than the LP the ES joint program solves.
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

    def _optimize_joint(
        self,
        Y_fh: np.ndarray,
        Y_c: np.ndarray,
        mu_a: np.ndarray,
        mu_c: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Jointly minimize mean-HMCR_2 of (R_fh @ x + R_c @ psi) over x and psi.

        The base-currency weight is pinned to zero and psi is capped in L1; both
        are required once the mean term is present.  See the module docstring."""
        q, N = Y_fh.shape
        K = Y_c.shape[1]
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()

        combined_return = Y_fh @ x + Y_c @ psi
        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(q))
        objective = soes_term - (1.0 / self.gamma) * (mu_a @ x + mu_c @ psi)
        l1_norm = cp.norm(x, 1)
        constraints = [
            u >= 0,
            -combined_return - eta <= u,
            cp.sum(x) == 1,                          # budget on the asset block only
            l1_norm <= self.leverage_limit,          # leverage on the asset block only
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

        # sum(x) == 1 is imposed, so no renormalization -- unlike soes_joint.py.
        # Renormalizing would mask solver drift and can silently break the leverage
        # cap, which matters most in an SOCP where optimal_inaccurate is likely.
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
            ValueError: If ``alpha`` is outside *(0, 1)*, ``gamma`` is not positive,
                        or ``overlay_limit`` is not finite and positive -- an
                        infinite cap would make the program unbounded below
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
                "the joint program is unbounded below without it for gamma under ~0.026."
            )

        logger.debug(
            "Joint mean-HMCR_2 optimization on %d obs, %d assets, %d currencies.",
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

        Re-runs the joint optimization on the validation window at this candidate's
        gamma -- mirroring soes_joint.py, which likewise re-solves rather than
        carrying the trained weights over -- and then returns

            HMCR_2(combined loss) - 1.0 * mean(realized combined returns),

        the realized risk-return tradeoff of this window at a reference coefficient
        of one, identical for every candidate.

        Unlike soes_curr_overlay_w_mean.score, this has to estimate mu_a on the
        validation window: the re-solve here optimizes over x, so mu_a' x is a live
        term rather than the additive constant it is in the overlay's step 2.

        HMCR_2 needs no transformation to reach a degree-one score: it is positively
        homogeneous of degree one, about 1.0e-2 daily against a daily portfolio mean
        near 3e-4, and it is exactly what ``in_sample_analysis.evaluate_risk``
        returns for ``"soes"``.

        The HMCR_2 is re-derived here with a bounded 1-D search rather than read off
        the solve's eta, and that is necessary rather than decorative.  At p = 1 the
        empirical VaR of the realized returns is exactly the minimizing threshold
        however the weights were obtained, so the ES programs can read their score
        off a quantile.  At p = 2 the minimizing zeta is only exactly recovered by
        the solve itself (see reg_soes_asset_full.py), and this SOCP is the most
        likely in the family to return ``optimal_inaccurate``.

        Because the risk term is not demeaned, the reference coefficient is
        effectively one half rather than one: HMCR_2 already contains a -mean(r) of
        its own by translation-equivariance, so this score is
        HMCR_2_deviation - 2 * mean(r).  That is a property of the undemeaned
        formulation, shared with es_joint_w_mean.score and
        soes_curr_overlay_w_mean.score, and it is stated rather than corrected so
        the four remain directly comparable.

        Two caveats inherited from the rest of the family, both measured on the MV
        pair.  This score leans conservative -- it selects the largest gamma in the
        band on most dates rather than locating an interior optimum, and alternative
        scores move which endpoint is preferred rather than making the search
        discriminate.  And mu_a is re-estimated here on the one-year validation
        window, where James-Stein shrinks it far harder than on the ten-year
        estimation window: on the MV joint its shrinkage weight rose from a median
        of 0.19 to 0.74 and reached 1.0 -- mu_a identically zero -- on 41% of
        rebalancing dates, while mu_c does not degrade the same way.  Carrying the
        training mu_a instead was tried there and made selection flatter and the
        realized Sharpe worse, so the re-solve is kept for consistency with
        soes_joint.py and the rest of the unregularized family.

        Args:
            weights: Not used; the joint optimization is rerun on validation data.
            val_bundle: Must contain ``"R_fh"``, ``"R_c"``, ``"C"`` and ``"F"``.

        Returns:
            Validation HMCR_2 minus realized mean return, lower is better.
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
        # live and both have to come from this window.  Note the Backtester ends the
        # validation window at the rebalancing date, the same date the full
        # estimation window ends on, so mu_c here reads the same last
        # forward-premium row the production refit conditions on.
        economy_ids = economy_ids_from_indicator(val_bundle.get("C"), Y_fh_val.shape[1])
        mu_a_val, _ = pooled_asset_mean(Y_fh_val, economy_ids)
        mu_c_val = conditional_currency_mean(val_bundle.get("F"), Y_c_val)

        result = self._optimize_joint(Y_fh_val, Y_c_val, mu_a_val, mu_c_val)
        port_rets = Y_fh_val @ result["x"] + Y_c_val @ result["psi"]

        tail_prob = 1.0 - self.alpha
        losses = -port_rets
        t_val = losses.shape[0]

        def _hmcr2_of_zeta(zeta: float) -> float:
            excess = np.maximum(losses - zeta, 0.0)
            return zeta + np.linalg.norm(excess) / (tail_prob * np.sqrt(t_val))

        result_min = minimize_scalar(
            _hmcr2_of_zeta,
            bounds=(float(losses.min()), float(losses.max())),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(result_min.fun) - float(port_rets.mean())
