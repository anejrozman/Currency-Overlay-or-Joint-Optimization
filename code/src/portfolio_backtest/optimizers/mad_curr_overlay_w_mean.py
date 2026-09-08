"""
Two-step mean-MAD optimizer with currency overlay.

The mean-bearing counterpart of mad_curr_overlay.py.  Both steps minimize the same
mean-risk functional

    MAD(portfolio return) - (1 / gamma) * mu' (decision variable),

so gamma is a risk-aversion coefficient in the conventional direction: large gamma
starves the mean term and the program collapses onto the pure-MAD overlay, small
gamma gives the mean its full weight.  The normalization is by 1 / gamma rather
than gamma multiplying the risk term so that the objective stays on the MAD scale
for every gamma, matching mv_curr_overlay_w_mean.py and smv_curr_overlay_w_mean.py;
the two are identical up to a positive rescaling and give the same weights, since
this program is unregularized and its constraints do not scale with the objective.

Step 1: minimizes MAD(R_fh @ x) - (1/gamma) mu_a' x, with full investment and a
        leverage cap on x.
Step 2: with asset weights fixed, minimizes

            MAD(R_fh @ x* + R_c @ psi) - (1/gamma) mu_c' psi,

        over the currency overlay psi, subject to an L1 cap on the overlay.  The
        mu_a' x* term is a constant there and cannot move the argmin, so it is
        dropped -- which is also what lets :meth:`score` skip re-estimating mu_a on
        the validation window.

Both steps keep Konno & Yamazaki's lower semi-absolute deviation form from
mad_curr_overlay.py: for a sample with zero mean the mass below the mean equals
the mass above it, so mean|z| = 2 * mean(max(0, -z)), and the right-hand side is
linear once the positive part is lifted into auxiliary variables.  That identity
survives the mean term untouched, because the mean term does not enter z: the
residual A x in step 1 and c_dm + A_c psi in step 2 each sum to exactly zero over
the window by construction, for every x and every psi.  The factor of two is
carried explicitly so both objectives stay on the MAD scale.

Location.  The demeaning is what makes the mean term well defined.  A = Y - Y.mean
turns the MAD into a *deviation* measure, invariant to a location shift, so the
mean reaches the objective only through the explicit, pooled and shrunk mu terms,
and gamma -> infinity recovers mad_curr_overlay.py exactly.  Measuring absolute
deviation about zero instead would not be location-free and would reward a high
mean at fixed weight one on top of the explicit term -- the double-counting
es_curr_overlay_w_mean.py carries because raw ES is not a deviation measure.

One inconsistency the deviation form leaves, stated rather than fixed: the
demeaning subtracts the raw per-column sample mean while the mean term rewards the
pooled and shrunk mu_a, so risk is measured about a noisy target and return
forecast with a deliberately quiet one.  The same holds in the MV and SMV pairs.

Overlay cap.  Here the cap is what makes the program well posed at all, which is
not true of the MV and SMV pairs and is the one structural difference from them.
Their risk terms are degree two, so the quadratic always eventually dominates the
linear mean term and psi is pinned down by the data.  MAD is positively
homogeneous of degree one, so along psi = t * d the step-2 objective behaves like

    t * [ MAD(A_c d) - mu_c' d / gamma ],

and runs to minus infinity for any direction d with mu_c' d / gamma > MAD(A_c d),
that is for any

    gamma < gamma_crit = max_d [ mu_c' d / MAD(A_c d) ].

Nothing in mad_curr_overlay's constraint set rules that out; it is safe only
because it carries no mean term.  Measured on three estimation windows, gamma_crit
is 0.062, 0.078 and 0.110 -- small in absolute terms, but the useful gamma band for
this program sits at 3 down to 0.1 (MAD is degree one and about a hundred times a
daily variance in size, so its band is two to three decades below the MV and SMV
ones), which puts the divergence threshold just under the aggressive end of the
grid rather than far away from it.  Verified directly: with the cap lifted, gross
psi is 0.41 at gamma = 0.1 and 20.4 at gamma = 0.05.  The L1 cap closes that off.
A run that silently dropped it would not merely hold a large overlay at the
aggressive end, it would fail to have a solution.

Two consequences worth carrying into the results.  Approaching gamma_crit the LP
loses its interior optimum in psi and sits on the cap, with its direction tending
to the max-carry-per-MAD vector, so gamma stops tuning the overlay there and only
tunes the asset block -- gross psi reaches the 2.0 cap by gamma = 0.1.  And step 1
is never at risk, since sum(x) = 1 with an L1 leverage cap is already a compact
feasible set.

The cap is set at 2.0, matching every other mean-bearing program.  Unlike the SMV
and ES pairs, that costs nothing here: mean-free MAD runs a gross psi of 0.19 to
1.18 (median 0.66) across those eleven windows, never above 2.0, so the cap is
slack at gamma -> infinity and this program is a strict superset of
mad_curr_overlay.py.

Means.  Estimated exactly as in mv_curr_overlay_w_mean.py: mu_a pooled onto the 14
economies and James-Stein shrunk toward zero
(:func:`~portfolio_backtest.utils.estimation.pooled_asset_mean`), mu_c not
estimated from returns at all but conditioned on the forward premium as the carry
-f_t (:func:`~portfolio_backtest.utils.estimation.conditional_currency_mean`).
That module records what each channel actually forecasts over the following
quarter -- mu_c genuinely (slope +0.75, tercile spread +2.66 bp/day), mu_a not,
cross-sectionally (slope -0.26, spread -0.82 bp/day).

Economy mapping.  Taken from the ``"C"`` bundle entry, the asset-to-currency
indicator matrix the Backtester already assembles and restricts to the window's
valid assets.  Its columns are R_c's columns in order, so economy k and currency k
are the same object.

Base currency.  Pinned to zero.  Its R_c column is identically zero so it never
enters the risk term, its mu_c entry is zero, and the overlay cap would otherwise
let it consume budget while contributing nothing.  Without the pin the LP would
leave it unidentified.

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


class MeanAbsoluteDeviationCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step mean-MAD optimizer with sequential currency overlay.

    Step 1 minimizes the MAD of R_fh minus the pooled asset mean, subject to full
    investment and a leverage cap.  Step 2 fixes the asset weights and minimizes
    the MAD of the combined portfolio return minus the carry mean of the overlay,
    subject to an L1 cap on psi -- which this program, unlike the MV and SMV pairs,
    requires for boundedness.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``          (float) — risk aversion; the mean term enters
                                               with coefficient 1 / gamma, default 0.5
                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``  (float) — ||psi||_1 cap for step 2, default 2.0.
                                               Required for boundedness at the
                                               aggressive end of the band; see the
                                               module docstring.
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
        """Step 1: mean-MAD on R_fh with a leverage cap and full investment.

        Bounded by its constraints: sum(x) = 1 with an L1 leverage cap is a
        compact feasible set, so unlike step 2 this needs no cap of its own."""
        T, N = Y_fh.shape
        A = Y_fh - Y_fh.mean(axis=0)

        x = cp.Variable(N)
        u = cp.Variable(T)

        objective = (2.0 / T) * cp.sum(u) - (1.0 / self.gamma) * (mu_a @ x)
        problem = cp.Problem(
            cp.Minimize(objective),
            [
                u >= 0,
                -A @ x <= u,
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
        """Step 2: mean-MAD of (R_fh @ x_fixed + R_c @ psi) over psi, capped in L1.

        The constant mu_a' x* is dropped: it cannot move the argmin.  The L1 cap is
        load-bearing rather than cosmetic here -- without it this LP is unbounded
        below for any gamma under roughly 28; see the module docstring."""
        T, K = Y_c.shape
        c = Y_fh @ x_fixed                  # fixed asset contribution, shape (T,)
        c_dm = c - c.mean()                 # demeaned; see module docstring
        A_c = Y_c - Y_c.mean(axis=0)

        psi = cp.Variable(K)
        u = cp.Variable(T)

        objective = (2.0 / T) * cp.sum(u) - (1.0 / self.gamma) * (mu_c @ psi)
        constraints = [
            u >= 0,
            -(c_dm + A_c @ psi) <= u,
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
            ValueError: If ``gamma`` is not strictly positive, or ``overlay_limit``
                        is not finite and positive -- an infinite cap would make
                        step 2 unbounded below gamma_crit ~ 0.06-0.11.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")
        if not np.isfinite(self.overlay_limit) or self.overlay_limit <= 0.0:
            raise ValueError(
                "overlay_limit must be finite and positive: MAD is degree-one, so "
                "step 2 is unbounded below without it for gamma under ~0.1."
            )

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

        Uses ``self.x_opt`` (asset weights from :meth:`optimize`) as fixed, re-runs
        step 2 on the validation returns to get psi_val -- mirroring
        mad_curr_overlay.py and the other unregularized overlays -- then returns

            MAD(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one, identical for every candidate.

        MAD needs no transformation to get there, unlike the MV and SMV pairs.
        Those score on the standard deviation and the semi-deviation rather than
        on the variance and the semi-variance their optimizers minimize, because a
        degree-two risk term against a degree-one mean would make ``risk - mean``
        a mean maximization in disguise.  MAD is already degree one: about 3.1e-3
        daily against a daily portfolio mean near 2.7e-4, an eleven-to-one ratio
        matching the transformed MV and SMV scores.  It is also exactly what
        ``in_sample_analysis.evaluate_risk`` returns for ``"mad"``, the functional
        in_sample_vs_out_of_sample_w_mean.py scores these weights with.

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
        the same caution, and note the extra wrinkle that at the aggressive end of
        the band psi is pinned to the cap, so neighbouring grid points there differ
        mostly in their asset block.

        Args:
            weights: Not used directly; asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"``, ``"R_c"`` and ``"F"``.

        Returns:
            Validation MAD minus realized mean return, lower is better.
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

        mad = float(np.mean(np.abs(port_rets - port_rets.mean())))
        return mad - float(port_rets.mean())
