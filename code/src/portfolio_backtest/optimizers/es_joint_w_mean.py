"""
Unregularized joint mean-Expected Shortfall optimizer over asset and currency weights.

The mean-bearing counterpart of es_joint.py, and the joint counterpart of
es_curr_overlay_w_mean.py.  Minimizes

    ES_alpha(-(R_fh @ x + R_c @ psi)) - (1 / gamma) * (mu_a' x + mu_c' psi)

jointly over x and psi in a single program, subject to full investment and a
leverage cap on x and an L1 cap on psi.  gamma is a risk-aversion coefficient in
the conventional direction: large gamma starves the mean term and the program
collapses onto es_joint, small gamma gives the mean its full weight.  As in the
overlay, the 1 / gamma normalization keeps the objective on the ES scale for
every gamma rather than rescaling the risk term; the two are identical up to a
positive rescaling and give the same weights, since this program is unregularized
and its constraints do not scale with the objective.

The difference from es_curr_overlay_w_mean.py is only the sequencing.  There, x
is fixed by an asset-only program and psi is then fitted against it; here both
are chosen together, so the mean tilt on the asset book and the carry tilt on the
overlay are traded off against a single tail rather than in sequence.  Everything
else -- the estimators, the caps, the base-currency pin, the absence of a demean
switch, and the gamma-free validation score -- is shared with the overlay, and
the two shared estimators live in utils/estimation.py.

Means.  Identical to the overlay's, and for the same reasons:

  * Assets.  mu_a is pooled onto the 14 economies and James-Stein shrunk toward
    zero (:func:`~portfolio_backtest.utils.estimation.pooled_asset_mean`), which
    cuts the estimated dimension from N ~ 100-130 to at most 14 and restricts the
    cross-sectional spread of expected returns to the between-economy dimension.
  * Currencies.  mu_c is not estimated from returns at all but conditioned on the
    forward premium, mu_c = -f_t
    (:func:`~portfolio_backtest.utils.estimation.conditional_currency_mean`).

Overlay cap.  ES is positively homogeneous, so along psi = t * d the objective
behaves like t * [ES(R_c d) - mu_c' d / gamma].  Any direction d with
ES(R_c d) < mu_c' d / gamma would make the program unbounded below, and nothing
in es_joint's constraint set rules that out -- es_joint is safe only because it
carries no mean term at all.  The L1 cap on psi closes that off, and at the same
time matches the overlay optimizer.  The base-currency weight is pinned to zero
so it cannot consume the cap while contributing nothing: its R_c column is
identically zero and its forward premium likewise.

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


class ExpectedShortfallJointWithMeanOptimizer(BaseOptimizer):
    """
    Joint unregularized mean-ES optimizer over asset and currency overlay weights.

    Solves one CVXPY problem minimizing ES of (R_fh @ x + R_c @ psi) minus the
    gamma-scaled mean of the same combination, subject to full investment and a
    leverage cap on x and an L1 cap on psi.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — ES confidence level, default 0.80
                * ``gamma``          (float) — risk aversion; the mean term enters
                                               with coefficient 1 / gamma, default 2
                * ``leverage_limit`` (float) — ||x||_1 cap, default 2.0
                * ``overlay_limit``  (float) — ||psi||_1 cap, default 2.0
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

    def _optimize_joint(
        self,
        Y_fh: np.ndarray,
        Y_c: np.ndarray,
        mu_a: np.ndarray,
        mu_c: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Jointly minimize mean-ES of (R_fh @ x + R_c @ psi) over x and psi."""
        q, N = Y_fh.shape
        K = Y_c.shape[1]
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()

        combined_return = Y_fh @ x + Y_c @ psi
        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)
        objective = es_term - (1.0 / self.gamma) * (mu_a @ x + mu_c @ psi)
        constraints = [
            u >= 0,
            -combined_return - eta <= u,
            cp.sum(x) == 1,
            cp.norm(x, 1) <= self.leverage_limit,
            cp.norm(psi, 1) <= self.overlay_limit,
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

        return {
            "x": (weights / weights.sum()).flatten(),
            "psi": overlay.flatten(),
        }

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run the joint optimization over training data.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
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
            "Joint mean-ES optimization on %d obs, %d assets, %d currencies.",
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
        candidate's gamma -- mirroring es_joint, which likewise re-solves rather
        than carrying the trained weights over -- and then scores the result as

            ES_alpha(combined loss) - 1.0 * mean(realized combined returns),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one, identical for every candidate.

        The realized mean is used directly rather than reconstructed from the
        estimators: mean(Y_fh @ x + Y_c @ psi) equals mu_a' x + mu_c' psi exactly
        when mu is the raw sample mean, so this is the reference-gamma form with
        the estimator noise removed.  Scoring each candidate at its own gamma --
        the natural-looking alternative -- makes the mean credit mu' w / gamma
        grow without bound as gamma falls while the risk term does not, so the
        search runs to whichever gamma is smallest in the grid; that failure was
        measured on the overlay optimizer and is documented in its ``score``.

        Args:
            weights: Not used; the joint program is re-solved on validation data.
            val_bundle: Must contain ``"R_fh"``, ``"R_c"``, ``"C"`` and ``"F"``.

        Returns:
            Validation ES minus realized mean return, lower is better.
        """
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty:
            return float("inf")

        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)

        # The re-solve minimizes the gamma-scaled objective, so it needs this
        # window's own mean vectors. Only the score returned below is gamma-free.
        economy_ids = economy_ids_from_indicator(val_bundle.get("C"), Y_fh_val.shape[1])
        mu_a_val, _ = pooled_asset_mean(Y_fh_val, economy_ids)
        mu_c_val = conditional_currency_mean(val_bundle.get("F"), Y_c_val)

        result = self._optimize_joint(Y_fh_val, Y_c_val, mu_a_val, mu_c_val)
        port_rets = Y_fh_val @ result["x"] + Y_c_val @ result["psi"]

        threshold = np.quantile(port_rets, 1.0 - self.alpha)
        tail = port_rets[port_rets <= threshold]
        es = float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))

        return es - float(port_rets.mean())
