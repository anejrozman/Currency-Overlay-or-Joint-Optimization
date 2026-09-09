"""
Two-step mean-Expected Shortfall optimizer with currency overlay.

The mean-bearing counterpart of es_curr_overlay.py.  Both steps minimize the
same mean-risk functional

    ES_alpha(portfolio loss) - (1 / gamma) * mu' (decision variable),

so gamma is a risk-aversion coefficient in the conventional direction: large
gamma starves the mean term and the program collapses onto the pure-ES overlay,
gamma = 1 gives the mean its full weight.  The normalization is by 1 / gamma
rather than gamma multiplying the risk term so that the objective stays on the
ES scale for every gamma, which matters for the validation score (see
:meth:`score`); the two are identical up to a positive rescaling and give the
same weights, since this program is unregularized and its constraints do not
scale with the objective.

Step 1: minimizes ES(R_fh @ x) - (1/gamma) * mu_a' x over fully hedged asset
        returns, with full investment and a leverage cap on x.
Step 2: with asset weights fixed, minimizes
        ES(R_fh @ x* + R_c @ psi) - (1/gamma) * (mu_a' x* + mu_c' psi)
        over the currency overlay psi, subject to an L1 cap on the overlay.
        The mu_a' x* term is a constant there and does not move the argmin; it
        is carried so the reported objective is the full one.

Selection.  The objective is gamma-scaled but :meth:`score` is not: candidates
are compared on a common reference yardstick, ES minus the realized mean of the
validation window.  Scoring each candidate at its own gamma makes the search run
to whichever gamma is smallest in the grid, for reasons and with measurements
recorded in :meth:`score`.

Means.  The two mean vectors are estimated very differently, because the two
estimation problems are not comparable in difficulty.

  * Assets.  N is 100-130 in the top-10 universe against q ~ 2500 daily
    observations, and a per-asset sample mean at that signal-to-noise ratio is
    almost pure noise -- which is exactly why every other optimizer in this
    package dropped the mean term outright (see the gamma = 3 blow-ups recorded
    in backtest_config.py).  The mean here is therefore pooled onto the 14
    economies before it is used: each economy's series is the equal-weighted
    average of its constituents, that panel is James-Stein shrunk toward zero,
    and every asset inherits its economy's shrunk mean
    (:func:`~portfolio_backtest.utils.estimation.pooled_asset_mean`).  Expected
    returns then vary only between economies, not within them.
  * Currencies.  Not estimated from returns at all: the mean is conditional on
    the forward premium, mu_c = -f_t, which is the carry (see
    ``_conditional_currency_mean``).  An unconditional sample mean was tried
    first and shrank to exactly zero in five of six test windows, since 13
    currency means over a 2500-day window carry almost no signal -- leaving step
    2 with no mean term whatsoever.  The forward premium is observed rather than
    estimated and is a genuine predictor here (pooled slope -0.83 against -1
    under random-walk spot, 0 under UIP; cross-currency correlation between mean
    -f and mean R_c of +0.72).  The base currency is held at zero.

Economy mapping.  Taken from the ``"C"`` bundle entry, the asset-to-currency
indicator matrix the Backtester already assembles and restricts to the window's
valid assets.  Its columns are R_c's columns in order, so economy k and currency
k are the same object and no separate mapping has to be plumbed through.

Base currency.  Unlike es_curr_overlay.py, where psi_base is genuinely absent
from step 2 (a zero column in an unconstrained LP) and can be left free, the
overlay cap here puts |psi_base| into cp.norm(psi, 1).  A nonzero base weight
would then consume overlay budget while contributing nothing to either the
return (its R_c column is zero) or the mean term (its mu_c entry is zero), so it
is pinned to zero, exactly as mv/mad/smv_curr_overlay.py pin theirs.

No demeaning.  The ``demean`` switch the other ES overlays carry is deliberately
absent: centering the return matrices would zero the very sample means this
program is built around, and the explicit mu terms already give the mean a
controlled, shrunk, dimensionality-reduced channel into the objective rather
than the raw one that centering was introduced to suppress.

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


class ExpectedShortfallCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step mean-ES optimizer with sequential currency overlay.

    Step 1 minimizes ES minus the pooled asset mean on R_fh subject to full
    investment and a leverage cap.  Step 2 fixes the asset weights and minimizes
    ES minus the currency mean of the combined portfolio return, subject to an
    L1 cap on the overlay.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — ES confidence level, default 0.80
                * ``gamma``          (float) — risk aversion; the mean term enters
                                               with coefficient 1 / gamma, default 2
                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``  (float) — ||psi||_1 cap for step 2, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.gamma: float = self.hyperparams.get("gamma", 2)
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
        """Step 1: mean-ES on R_fh with a leverage cap and full investment."""
        q, N = Y_fh.shape
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        u = cp.Variable(q)
        eta = cp.Variable()

        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)
        objective = es_term - (1.0 / self.gamma) * (mu_a @ x)
        problem = cp.Problem(
            cp.Minimize(objective),
            [
                u >= 0,
                -Y_fh @ x - eta <= u,
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
        mu_a: np.ndarray,
        mu_c: np.ndarray,
    ) -> np.ndarray:
        """Step 2: mean-ES of (R_fh @ x_fixed + R_c @ psi) over psi, capped in L1.

        The base-currency weight is pinned to zero so it cannot consume overlay
        budget for no return; see the module docstring."""
        q, K = Y_c.shape
        tail_prob = 1.0 - self.alpha

        c = Y_fh @ x_fixed  # constant asset return contribution, shape (q,)
        mean_assets = float(mu_a @ x_fixed)  # constant; carried for the objective value

        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()

        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)
        objective = es_term - (1.0 / self.gamma) * (mean_assets + mu_c @ psi)
        constraints = [
            u >= 0,
            -c - Y_c @ psi - eta <= u,
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
            ValueError: If ``alpha`` is outside *(0, 1)* or ``gamma`` is not positive.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")

        logger.debug(
            "Step 1: optimizing asset weights on %d obs, %d assets.", self.q, self.N
        )
        self.x_opt = self._optimize_assets(self.Y_fh, self.mu_a)

        logger.debug(
            "Step 2: optimizing currency overlay on %d obs, %d currencies.", self.q, self.K
        )
        psi_opt = self._optimize_overlay(
            self.Y_fh, self.Y_c, self.x_opt, self.mu_a, self.mu_c
        )

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
        es_curr_overlay.py and the other unregularized overlays -- then returns

            ES_alpha(combined loss) - 1.0 * mean(realized combined returns),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one, identical for every candidate.

        The realized mean is used directly rather than reconstructed from the
        estimators: mean(Y_fh @ x + Y_c @ psi) equals mu_a' x + mu_c' psi exactly
        when mu is the raw sample mean, so this *is* the reference-gamma form,
        with the estimator noise removed.  A score is a measure of what the
        weights actually earned on the window; re-applying a shrunk or
        conditional forecast there would let a candidate score well on a mean it
        never realized.

        Scoring each candidate at its own gamma -- the previous behaviour -- is
        what this replaces, and the reason is measured rather than theoretical.
        The mean credit mu' w / gamma grows without bound as gamma falls while
        the risk term does not, so the search ran to whichever gamma was smallest
        in the grid: 61.5% of dates at the bottom of the old [100 ... 1] grid, and
        54.9% at the bottom of the recentered [10 ... 0.1] one, in both cases
        overshooting the gamma that was actually best out of sample.  On the
        recentered grid that cost roughly a third of the available Sharpe (0.115
        selected against 0.154 at a fixed gamma of 0.2) and raised drawdown from
        29.3% to 32.8%.  A gamma-free score removes that channel: gamma now has
        to earn its selection through the weights it produces.

        Args:
            weights: Not used directly; asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"``, ``"R_c"``, ``"C"`` and ``"F"``.

        Returns:
            Validation ES minus realized mean return, lower is better.
        """
        if self.x_opt is None:
            raise RuntimeError("optimize() must be called before score().")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty:
            return float("inf")

        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)

        # Step 2 is re-solved here, so it still needs this window's own mean
        # vectors -- the objective it minimizes is the gamma-scaled one.  Only the
        # score returned below is gamma-free.
        economy_ids = economy_ids_from_indicator(val_bundle.get("C"), Y_fh_val.shape[1])
        mu_a_val, _ = pooled_asset_mean(Y_fh_val, economy_ids)
        mu_c_val = conditional_currency_mean(val_bundle.get("F"), Y_c_val)

        psi_val = self._optimize_overlay(
            Y_fh_val, Y_c_val, self.x_opt, mu_a_val, mu_c_val
        )
        port_rets = Y_fh_val @ self.x_opt + Y_c_val @ psi_val

        threshold = np.quantile(port_rets, 1.0 - self.alpha)
        tail = port_rets[port_rets <= threshold]
        es = float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))

        return es - float(port_rets.mean())
