"""
Unregularized Joint Semi-Variance optimizer over asset and currency weights
(Markowitz, Todd, Xu & Yamane 1993, in the framework of Ulrych & Lucescu 2026, Section 3).

Minimizes the semi-variance of the combined portfolio return (R_fh @ x + R_c @ psi)
jointly over both asset weights x and currency overlay weights psi in a single
optimization scheme.  Leverage cap and budget on the asset block only; psi is
unconstrained apart from the base-currency pin.

The risk measure is Markowitz et al.'s S_E, eq. (3.1a), with the target E set to the
portfolio's own expected return rather than an exogenous benchmark:

    SMV(w) = (1/T) * sum_t [ max(0, -A_t w) ]^2,

where A is the columnwise demeaned return matrix, so only deviations below the
portfolio's own mean are counted.  The program uses their eqs. (3.4)-(3.7)
reformulation, lifting the positive part into auxiliary variables to make the
objective quadratic.  The paper introduces 2T of them — Y for the signed deviation
and Z = Y^- for its negative part, tied by BX - Y + Z = 0 — because the Critical Line
Algorithm needs the constraints in equality form.  Only the downside half is needed
here: minimizing sum z_t^2 subject to z >= -A w and z >= 0 drives each z_t to
max(0, -A_t w) on its own, so the equality and the free variables Y drop out and the
program is a QP with T auxiliary variables rather than 2T.  It is solved directly by
an interior-point solver rather than by the Critical Line Algorithm: CLA traces the
whole E-S frontier, whereas one point of it is wanted per rebalancing date, and the
leverage cap is not in CLA's form to begin with.  The 1/sqrt(T) that eq. (3.8) folds
into B is carried as an explicit 1/T on the objective instead, which leaves A on the
scale of the raw returns and keeps the risk term numerically better conditioned.

Demeaning covers both blocks.  Since mean(x' R_fh + psi' R_c) = x' mean(R_fh) +
psi' mean(R_c), the deviation about the portfolio's own mean is

    SMV(x, psi) = (1/T) * sum_t [ max(0, -(A_fh x + A_c psi)_t) ]^2,

where A_fh and A_c are the columnwise demeaned return matrices.  Note that no part of
the objective separates into an asset-only and a currency-only term the way the
mean-variance quadratic form does: the positive part is taken of the combined return,
so the two blocks interact inside it.

That demeaning also keeps the program well posed.  The residual A_fh x + A_c psi sums
to exactly zero over the window for every (x, psi), so it can be non-negative
throughout only if it vanishes identically — which needs a null vector of the stacked
demeaned matrix, impossible for T ~ 2500 against N + K ~ 145 columns of full rank.
The objective is therefore strictly positive at the optimum, and it grows without
bound as ||psi|| does, so psi is pinned down by the data rather than by a constraint.

The objective carries no mean term, mirroring the MV, MAD and ES joint optimizers,
which likewise minimize a risk measure alone, and removing the need for a shrinkage
estimator of the mean (Bayes-Stein, Jorion 1986).  Unlike mean-variance, nothing
beyond the demeaning is estimated: the semi-variance is read off the empirical
distribution directly, with no joint covariance matrix and so no shrinkage target
(contrast Ledoit-Wolf on the stacked returns in mv_joint).

The base currency's excess return is identically zero, so its overlay weight never
enters the objective and would be left unidentified.  It is pinned to zero, as in
mv_joint and mad_joint, rather than left free.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
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


class SemiVarianceJointOptimizer(BaseOptimizer):
    """
    Joint unregularized semi-variance optimizer over asset and currency overlay weights.

    Solves a single CVXPY problem minimizing the semi-variance of
    (R_fh @ x + R_c @ psi) about its own sample mean, subject to full investment and
    leverage constraints on x.  psi is unconstrained apart from the base-currency pin.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``leverage_limit`` (float) — ||x||_1 cap, default 2.0
                * ``solver``                 — cvxpy solver, default ``cp.GUROBI``
                                               (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

        self.Y_fh: np.ndarray | None = None
        self.Y_c: np.ndarray | None = None
        self.N: int = 0
        self.K: int = 0
        self.q: int = 0

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store return matrices for joint optimization.

        Args:
            train_bundle: Must contain ``"R_fh"`` and ``"R_c"``.
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
        self, Y_fh: np.ndarray, Y_c: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Jointly minimize the semi-variance of (R_fh @ x + R_c @ psi) over x and psi.

        Both return blocks are demeaned columnwise, so the objective is the downside
        deviation of the combined portfolio return about its own sample mean (see
        module docstring)."""
        T, N = Y_fh.shape
        K = Y_c.shape[1]
        A_fh = Y_fh - Y_fh.mean(axis=0)
        A_c = Y_c - Y_c.mean(axis=0)

        x = cp.Variable(N)
        psi = cp.Variable(K)
        z = cp.Variable(T)

        l1_norm = cp.norm(x, 1)
        constraints = [
            z >= 0,
            -(A_fh @ x + A_c @ psi) <= z,
            cp.sum(x) == 1,                  # budget on the asset block only
            l1_norm <= self.leverage_limit,  # leverage on the asset block only
        ]
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(cp.Minimize(cp.sum_squares(z) / T), constraints)
        self._solve(problem)

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
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or the solver fails.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        logger.debug(
            "Joint optimization on %d obs, %d assets, %d currencies.",
            self.q, self.N, self.K,
        )
        return self._optimize_joint(self.Y_fh, self.Y_c)

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the semi-variance by rerunning the joint optimization on validation data.

        Mirrors the unregularized ES, MV and MAD joint optimizers, which likewise
        re-solve on the validation window. (The regularized variants instead hold the
        training fit fixed, because re-solving would drive penalty selection to the
        weakest point in the grid; with no penalties to select there is nothing for
        that to bias here.)

        The deviation is taken about the validation window's own mean, matching the
        asset-level and overlay SMV optimizers.

        Args:
            weights: Not used, joint optimization is rerun on val data.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Semi-variance of the combined portfolio return (lower is better).
        """
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")
        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty or len(R_fh_val) < 2:
            return float("inf")

        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)

        result = self._optimize_joint(Y_fh_val, Y_c_val)
        r_val = Y_fh_val @ result["x"] + Y_c_val @ result["psi"]
        downside = np.minimum(r_val - r_val.mean(), 0.0)
        return float(np.mean(downside ** 2))
