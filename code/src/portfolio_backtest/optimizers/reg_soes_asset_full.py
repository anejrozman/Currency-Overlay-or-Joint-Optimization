"""
Asset-only Regularized Second-Order Expected Shortfall optimizer — ASSETS-FULL
strategy (Krokhmal 2007, in the framework of Ulrych & Lucescu 2026, Section 3).

Uses fully hedged asset returns R_fh: the investor hedges all currency risk via
forward contracts and optimizes only the asset allocation.

The risk measure is Krokhmal's higher-moment coherent risk measure (HMCR) at
order p = 2 — the second-order counterpart of Expected Shortfall.  Writing L for
the portfolio loss (the negated return), HMCR is defined by the same
Rockafellar-Uryasev style scalar minimization that defines ES, with the tail
mean replaced by a tail L_p norm (Krokhmal 2007, eq. (13)):

    HMCR_{p,alpha}(L) = min_eta { eta + (1 - alpha)^{-1} || (L - eta)_+ ||_p },

where ||X||_p = (E[|X|^p])^{1/p}.  At p = 1 this collapses to ES exactly; at
p = 2 the tail is penalized by its root-mean-square rather than its mean, so
the measure is sensitive to the dispersion of the losses beyond the threshold
and not only to their average.  That is the point of carrying it alongside ES:
two return distributions with identical ES can have very different tail shapes,
and HMCR_2 separates them.  Krokhmal shows the family is coherent, consistent
with second-order stochastic dominance, and (unlike ES) strictly sensitive to
the higher moments of the tail.

On an equiprobable sample of q days the empirical L_2 norm is

    || (L - eta)_+ ||_2 = ( (1/q) * sum_t u_t^2 )^{1/2} = ||u||_2 / sqrt(q),

with u_t = max(0, -Y_t x - eta), which is the objective implemented below.  Two
things about that expression are worth stating explicitly, because both are easy
to get wrong:

* **The norm is not squared.**  Squaring would give a smooth quadratic in u and
  a slightly easier program, but it destroys positive homogeneity — the squared
  functional scales as lambda^2 under L -> lambda*L rather than lambda — and the
  measure stops being coherent.  The 2-norm enters linearly, exactly as in
  Krokhmal's definition.

* **The normalization differs from ES.**  The ES program divides the sum of the
  tail excesses by q; this one divides their 2-norm by sqrt(q).  Both leave the
  risk term on the scale of daily returns, which is what makes the two measures
  and their penalty grids comparable.  Dividing the 2-norm by q instead would
  shrink the risk term by a factor of ~sqrt(q) (~50 at these window lengths) and
  silently hand the whole objective to the penalties.

Everything else is unchanged from reg_es_asset_full: the same auxiliary
variables u and threshold eta, the same constraints u >= 0 and -Y x - eta <= u
that drive u_t to the positive part of the excess loss, the same budget
constraint, the same leverage cap, and the same L1/L2 penalties on x.  Only the
functional applied to u differs.

The program is a second-order cone program rather than a linear one: the
epigraph of ||u||_2 is an SOC constraint, so cvxpy hands Gurobi a QCP where the
ES formulation hands it an LP.  Two consequences for the solver block below.
First, ``optimal_inaccurate`` is materially more likely than for the LP — the
interior-point method has no simplex crossover to clean up after it on a conic
problem — and is accepted here as it is elsewhere; the weights are normalized on
return in any case.  Second, ``BarConvTol`` governs Gurobi's LP/QP barrier and
does *not* apply to the conic path, so ``BarQCPConvTol`` is set alongside it
(default 1e-6) to hold the SOCP to the same tolerance the rest of the block asks
for.  It is the one addition to the Gurobi settings used by the other
optimizers.

As with the ES program the objective carries no mean term.  The sample mean is
estimated far too imprecisely at these window lengths to be worth its variance
cost; dropping it leaves a pure regularized minimum-risk problem.  Nothing
beyond the empirical distribution has to be estimated: like ES and MAD, and
unlike mean-variance, the measure is read off the sample directly, with no
covariance matrix and so no shrinkage target.

The penalties carry their own grids (LAMBDA_L1_SOES_ASSET_GRID /
LAMBDA_L2_SOES_ASSET_GRID in backtest_config.py) rather than reusing the ES
ones.  HMCR_2 sits on the same daily-return scale as ES_80 but runs materially
larger — it dominates ES by construction, since the L_2 norm of the tail
excesses dominates their mean — so the ES grids are close but not right.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar
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


class RegularizedSecondOrderExpectedShortfallAssetFullOptimizer(BaseOptimizer):
    """
    Asset-only Regularized Second-Order Expected Shortfall optimizer
    (ASSETS-FULL strategy).

    Minimizes the second-order HMCR of the portfolio loss at a given confidence
    level with L1 and L2 regularization penalties, subject to a leverage limit
    and a full-investment constraint.  Uses only fully hedged asset returns
    (R_fh) for optimization.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``          (float) — confidence level, default 0.80
                * ``lambda_l1``      (float) — L1 penalty weight, default 5e-5
                * ``lambda_l2``      (float) — L2 penalty weight, default 0.3
                * ``leverage_limit`` (float) — L1 norm constraint, default 2.0
                * ``solver``                 — cvxpy solver constant, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        # Defaults are the interior points of the calibrated grids, so a --no-grid
        # run lands in the regularized regime rather than at an untested extreme.
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.lambda_l1: float = self.hyperparams.get("lambda_l1", 5e-5)
        self.lambda_l2: float = self.hyperparams.get("lambda_l2", 0.3)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

        self.Y: np.ndarray | None = None
        self.N: int = 0
        self.q: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store the fully hedged return matrix for optimization.

        Args:
            train_bundle: Must contain ``"R_fh"`` mapped to a non-empty DataFrame.
        """
        if "R_fh" not in train_bundle:
            raise ValueError("train_bundle must contain 'R_fh'.")
        R_fh = train_bundle["R_fh"]
        if R_fh.empty:
            raise ValueError("'R_fh' in train_bundle is empty.")
        self.Y = R_fh.values
        self.q, self.N = self.Y.shape

    def optimize(self) -> np.ndarray:
        """
        Solve the regularized second-order ES minimization problem.

        Returns:
            1-D numpy array of shape *(N,)* summing to 1.

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
            ValueError: If ``alpha`` is outside *(0, 1)*.
        """
        if self.Y is None or self.N == 0:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")

        x = cp.Variable(self.N)
        u = cp.Variable(self.q)
        eta = cp.Variable()

        l1_penalty = cp.norm(x, 1)
        # Empirical ||(L - eta)_+||_2 = ||u||_2 / sqrt(q). The norm enters
        # unsquared: squaring it would break positive homogeneity and the
        # measure would no longer be coherent (see module docstring).
        soes_term = eta + cp.norm(u, 2) / (tail_prob * np.sqrt(self.q))
        objective = cp.Minimize(
            soes_term
            + self.lambda_l1 * l1_penalty
            + self.lambda_l2 * cp.sum_squares(x)
        )
        constraints = [
            u >= 0,
            -self.Y @ x - eta <= u,
            cp.sum(x) == 1,
            l1_penalty <= self.leverage_limit,
        ]
        problem = cp.Problem(objective, constraints)

        try:
            if self.solver == cp.GUROBI:
                problem.solve(
                    solver=cp.GUROBI,
                    verbose=False,
                    BarConvTol=1e-8,
                    # BarConvTol does not govern the conic path; this is its
                    # QCP counterpart, and the SOC epigraph of ||u||_2 makes
                    # this an SOCP rather than the LP the ES program solves.
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

        weights = x.value
        if weights is None:
            raise RuntimeError("Solver returned None weights.")

        # Normalize to correct numerical inaccuracy from the solver
        return (weights / weights.sum()).flatten()

    def score(
        self,
        weights: np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the validation second-order ES of the candidate weights.

        Mirrors the training objective — the risk measure alone, without the
        penalties — exactly as the ES, MV, MAD and SMV optimizers score on
        validation ES, variance, MAD and semi-variance.

        The threshold zeta is solved for numerically rather than plugged in from
        the empirical VaR.  At p = 1 the empirical VaR of the losses at level
        ``alpha`` is exactly the minimizer of the Rockafellar-Uryasev functional,
        which is why the ES optimizer can substitute it directly and read off
        the exact tail mean.  That shortcut does not carry over to p = 2: the
        stationarity condition there is

            E[(L - eta)^+] / ||(L - eta)^+||_2 = 1 - alpha,

        whose solution sits well above VaR_alpha rather than coinciding with it
        (for normal losses at alpha = 0.80 the minimizing eta is near the 93rd
        percentile, so plugging in VaR alone would overstate the measure by
        roughly 9%).  The bias from using VaR would also not be common across
        grid candidates — it depends on the conditional shape of the tail, which
        the weights do move — so it would distort the grid search rather than
        merely offsetting every candidate by the same amount.  The functional
        is convex in zeta, so it is minimized directly with a bounded 1-D search
        over the range of the validation losses.

        Note that this is *not* the same functional as the ES score: it
        evaluates the measure being minimized in :meth:`optimize`, and a tail
        mean does not transfer.

        Args:
            weights: 1-D array of portfolio weights from :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"``.

        Returns:
            Second-order ES of the realized portfolio loss on the validation
            window (lower is better).
        """
        if "R_fh" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh'.")
        R_fh_val = val_bundle["R_fh"]
        if R_fh_val.empty:
            return float("inf")

        tail_prob = 1.0 - self.alpha
        losses = -(R_fh_val.values @ weights)
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
        return float(result.fun)
