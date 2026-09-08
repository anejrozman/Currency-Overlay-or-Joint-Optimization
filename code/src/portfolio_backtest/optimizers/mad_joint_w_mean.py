"""
Unregularized joint mean-MAD optimizer over asset and currency weights.

The mean-bearing counterpart of mad_joint.py, and the joint counterpart of
mad_curr_overlay_w_mean.py.  Minimizes

    mean| A_fh x + A_c psi | - (1 / gamma) * (mu_a' x + mu_c' psi)

jointly over x and psi in a single program, subject to full investment and a
leverage cap on x and an L1 cap on psi.  A_fh and A_c are the columnwise demeaned
return matrices, so the risk term is the mean absolute deviation of the combined
portfolio return about its own sample mean.  gamma is a risk-aversion coefficient
in the conventional direction: large gamma starves the mean term and the program
collapses onto mad_joint, small gamma gives the mean its full weight.

The program keeps Konno & Yamazaki's lower semi-absolute deviation form from
mad_joint.py: for a sample with zero mean the mass below the mean equals the mass
above it, so mean|z| = 2 * mean(max(0, -z)), and the right-hand side is linear
once the positive part is lifted into a T-vector of auxiliary variables.  That
identity survives the mean term untouched, because the mean term does not enter z:
the residual A_fh x + A_c psi sums to exactly zero over the window by
construction, for every x and every psi.  The factor of two is carried explicitly
so the objective stays on the MAD scale.

The intended difference from mad_curr_overlay_w_mean.py is the sequencing: there x
is fixed by an asset-only program and psi is fitted against it afterwards, here
both are chosen together against a single deviation.  No part of the objective
separates into an asset-only and a currency-only risk term -- the absolute value
is taken of the combined return, so the two blocks interact inside it.  One
consequence matters for the code: unlike the overlay's step 2, mu_a' x is a live
term rather than a droppable constant, so :meth:`score` has to estimate mu_a on
the validation window as well.

Location.  The demeaning is what makes the mean term well defined.  A_fh and A_c
turn the MAD into a *deviation* measure, invariant to a location shift, so the
mean reaches the objective only through the explicit, pooled and shrunk mu terms,
and gamma -> infinity recovers mad_joint.py exactly.  Measuring absolute deviation
about zero instead would not be location-free and would reward a high mean at
fixed weight one on top of the explicit term -- the double-counting
es_joint_w_mean.py carries because raw ES is not a deviation measure.

One inconsistency the deviation form leaves, stated rather than fixed: the
demeaning subtracts the raw per-column sample mean while the mean term rewards
the pooled and shrunk mu_a.

Overlay cap.  As in the overlay pair, and unlike the MV and SMV joints, the cap is
what makes this program well posed rather than merely tidy.  Their risk terms are
degree two, so the quadratic eventually dominates the linear mean term in every
direction.  MAD is positively homogeneous of degree one, and x is confined to a
compact set by sum(x) = 1 and the L1 leverage cap, so psi is the only direction in
which the program can run away: along psi = t * d with x bounded, A_fh x stays
bounded and mu_a' x with it, so the objective behaves like

    t * [ MAD(A_c d) - mu_c' d / gamma ],

which is the same growth rate the overlay's step 2 has.  The critical value is
therefore the same quantity,

    gamma_crit = max_d [ mu_c' d / MAD(A_c d) ],

measured at 0.062, 0.078 and 0.110 across estimation windows -- identical to the
overlay's, as the argument above predicts.  Below it an uncapped joint program
diverges in psi; nothing in mad_joint's constraint set rules that out, and
mad_joint is safe only because it carries no mean term.  The L1 cap closes that
off, and MAD_JOINT_GAMMA_GRID stays above every measured gamma_crit.

That asymptotic threshold understates how much the cap is doing here, though, and
this is the one place the joint behaves materially worse than the overlay.  The
bound above is the growth rate as t -> infinity; at finite t the asset block can
partially offset A_c d, so MAD(A_fh x + t A_c d) sits below t * MAD(A_c d) over a
long range and the objective keeps falling well past where the overlay's would
stop.  Measured uncapped on the window with the largest gamma_crit (0.110), gross
psi is 4.7 at gamma = 1, 11.6 at 0.2 and 22.7 at 0.12, and the solver reports the
problem unbounded at several gammas in between -- all of them comfortably ABOVE
gamma_crit, where the overlay's uncapped psi is still below 0.5.  So for the joint
the cap is load-bearing across the whole band rather than only near the threshold,
and an uncapped run would be numerically unusable long before it became genuinely
unbounded.  Capped, the program is well behaved: sum(x) holds to 1e-6 and psi
saturates the budget from about gamma = 0.3 down.

The cap is set at 2.0, matching every other mean-bearing program, and it costs
nothing at the conservative end: mean-free MAD runs a small overlay, so the cap is
slack at gamma -> infinity and this program is a strict superset of mad_joint.py.
At the aggressive end psi saturates against it, and there the overlay is a corner
of the L1 ball with its direction set by the carry ranking rather than by the
risk-return tradeoff.

Means.  Estimated exactly as in mad_curr_overlay_w_mean.py: mu_a pooled onto the
14 economies and James-Stein shrunk toward zero, mu_c conditioned on the forward
premium as the carry -f_t.  What each channel actually forecasts over the
following quarter is recorded in mv_curr_overlay_w_mean.py -- mu_c genuinely, mu_a
not, cross-sectionally.

Base currency.  Pinned to zero.  Its R_c column is identically zero so it never
enters the risk term, its mu_c entry is zero, and the overlay cap would otherwise
let it consume budget while contributing nothing.  Without the pin the LP would
leave it unidentified.

Budget.  sum(x) == 1 is imposed and the solution is checked rather than
renormalized, following mad_joint.py: renormalizing would mask solver drift and
silently break the leverage cap.

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

# Budget-violation tolerances on sum(x). Warn above the first, abort above the
# second. Both are loose relative to FeasibilityTol=1e-9; tripping either means
# the solve is not trustworthy rather than merely imprecise.
_BUDGET_WARN_TOL = 1e-6
_BUDGET_FAIL_TOL = 1e-4


class MeanAbsoluteDeviationJointWithMeanOptimizer(BaseOptimizer):
    """
    Joint unregularized mean-MAD optimizer over asset and currency overlay weights.

    Solves one CVXPY problem minimizing the MAD of (R_fh @ x + R_c @ psi) about its
    own sample mean, minus the gamma-scaled mean of the same combination, subject
    to full investment and a leverage cap on x and an L1 cap on psi -- which this
    program, unlike the MV and SMV joints, requires for boundedness.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``          (float) — risk aversion; the mean term enters
                                               with coefficient 1 / gamma, default 0.5
                * ``leverage_limit`` (float) — ||x||_1 cap, default 2.0
                * ``overlay_limit``  (float) — ||psi||_1 cap, default 2.0.  Required
                                               for boundedness at the aggressive end
                                               of the band; see the module docstring.
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
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
        """Jointly minimize mean-MAD of (R_fh @ x + R_c @ psi) over x and psi.

        Both return blocks are demeaned columnwise, so the risk term is the
        deviation of the combined portfolio return about its own sample mean and is
        location-free; the mean enters only through mu_a and mu_c.  The L1 cap on
        psi is load-bearing rather than cosmetic; see the module docstring."""
        T, N = Y_fh.shape
        K = Y_c.shape[1]
        A_fh = Y_fh - Y_fh.mean(axis=0)
        A_c = Y_c - Y_c.mean(axis=0)

        x = cp.Variable(N)
        psi = cp.Variable(K)
        u = cp.Variable(T)

        l1_norm = cp.norm(x, 1)
        objective = (
            (2.0 / T) * cp.sum(u)
            - (1.0 / self.gamma) * (mu_a @ x + mu_c @ psi)
        )
        constraints = [
            u >= 0,
            -(A_fh @ x + A_c @ psi) <= u,
            cp.sum(x) == 1,                          # budget on the asset block only
            l1_norm <= self.leverage_limit,          # leverage on the asset block only
            cp.norm(psi, 1) <= self.overlay_limit,   # overlay cap; see module docstring
        ]
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
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
            ValueError: If ``gamma`` is not strictly positive, or ``overlay_limit``
                        is not finite and positive -- an infinite cap would make the
                        program unbounded below gamma_crit ~ 0.06-0.11.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")
        if not np.isfinite(self.overlay_limit) or self.overlay_limit <= 0.0:
            raise ValueError(
                "overlay_limit must be finite and positive: MAD is degree-one, so "
                "the joint program is unbounded below without it for gamma under ~0.1."
            )

        logger.debug(
            "Joint mean-MAD optimization on %d obs, %d assets, %d currencies.",
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
        gamma -- mirroring mad_joint.py, which likewise re-solves rather than
        carrying the trained weights over -- and then returns

            MAD(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one, identical for every candidate.

        Unlike mad_curr_overlay_w_mean.score, this has to estimate mu_a on the
        validation window: the re-solve here optimizes over x, so mu_a' x is a live
        term rather than the additive constant it is in the overlay's step 2.

        MAD needs no transformation to reach a degree-one score, unlike the MV and
        SMV pairs, which score on the standard deviation and the semi-deviation
        rather than on the variance and semi-variance their optimizers minimize.
        MAD is already degree one: about 3.1e-3 daily against a daily portfolio
        mean near 2.7e-4, an eleven-to-one ratio matching the transformed MV and
        SMV scores, and exactly what ``in_sample_analysis.evaluate_risk`` returns
        for ``"mad"`` -- the functional in_sample_vs_out_of_sample_w_mean.py scores
        these weights with.

        The realized mean is used directly rather than reconstructed from the
        estimators: a score measures what the weights actually earned on the
        window, not what a shrunk or conditional forecast predicted for them.

        Two caveats, both inherited from the MV and SMV pairs and both measured
        there.  This score leans conservative -- on mv_curr_overlay_w_mean it
        selects the largest gamma in the band on most dates rather than locating an
        interior optimum, and alternative scores move which endpoint is preferred
        rather than making the search discriminate.  And mu_a is re-estimated here
        on the one-year validation window, where James-Stein shrinks it far harder
        than on the ten-year estimation window: on the MV joint its shrinkage weight
        rose from a median of 0.19 to 0.74 and reached 1.0, i.e. mu_a identically
        zero, on 41% of rebalancing dates.  mu_c does not degrade the same way,
        being read off the last forward-premium row.  So gamma is chosen against a
        program whose asset-mean channel is largely switched off while its carry
        channel runs at full strength.  Carrying the training mu_a instead was tried
        on the MV joint and made selection flatter and the realized Sharpe worse, so
        the re-solve is kept here for consistency with mad_joint.py and the rest of
        the unregularized family.

        Args:
            weights: Not used; the joint optimization is rerun on validation data.
            val_bundle: Must contain ``"R_fh"``, ``"R_c"``, ``"C"`` and ``"F"``.

        Returns:
            Validation MAD minus realized mean return, lower is better.
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

        mad = float(np.mean(np.abs(port_rets - port_rets.mean())))
        return mad - float(port_rets.mean())
