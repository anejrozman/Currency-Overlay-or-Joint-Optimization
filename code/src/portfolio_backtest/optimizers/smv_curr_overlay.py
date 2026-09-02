"""
Unregularized Semi-Variance optimizer with currency overlay
(Markowitz, Todd, Xu & Yamane 1993, in the framework of Ulrych & Lucescu 2026, Section 3).

Step 1: Minimizes the semi-variance of fully hedged asset returns (R_fh), subject
        to full investment and a leverage cap on asset weights.
Step 2: With asset weights fixed, minimizes the semi-variance of the combined
        portfolio return (R_fh @ x_opt + R_c @ psi) by optimizing the currency
        overlay weights psi.  psi is unconstrained; forward contracts are
        zero-investment instruments, so no slack weight is needed to fund the
        overlay.

The risk measure is Markowitz et al.'s S_E, eq. (3.1a), with the target E set to
the portfolio's own expected return rather than an exogenous benchmark:

    SMV(w) = (1/T) * sum_t [ max(0, -A_t w) ]^2,

where A is the columnwise demeaned return matrix, so only deviations below the
portfolio's own mean are counted.  Both steps use their eqs. (3.4)-(3.7)
reformulation, lifting the positive part into auxiliary variables to make the
objective quadratic.  The paper introduces 2T of them — Y for the signed
deviation and Z = Y^- for its negative part, tied by BX - Y + Z = 0 — because
the Critical Line Algorithm needs the constraints in equality form.  Only the
downside half is needed here: minimizing sum z_t^2 subject to z >= -A w and
z >= 0 drives each z_t to max(0, -A_t w) on its own, so the equality and the
free variables Y drop out and each step is a QP with T auxiliary variables
rather than 2T.  The 1/sqrt(T) that eq. (3.8) folds into B is carried as an
explicit 1/T on the objective instead, which leaves A on the scale of the raw
returns and keeps the risk term numerically better conditioned.

Demeaning in step 2 covers both terms.  Writing c = R_fh @ x_opt for the fixed
asset contribution, mean(c + R_c psi) = mean(c) + mean(R_c) psi for any psi, so

    SMV(x_opt, psi) = (1/T) * sum_t [ max(0, -( (c - mean c) + (R_c - mean R_c) psi )_t ) ]^2,

i.e. the fixed contribution and the currency matrix are each demeaned before the
program is formed.  Unlike the mean-variance overlay, no term of the objective
is constant in psi and none can be dropped: the positive part is taken of the
combined return, so the asset contribution stays inside it.

That demeaning also keeps step 2 well posed.  The residual c_dm + A_c psi sums
to exactly zero over the window for every psi, so it can be non-negative
throughout only if it vanishes identically — which needs c_dm to lie in the
column space of A_c, impossible for T ~ 2500 against a dozen currencies.  The
objective is therefore strictly positive at the optimum and psi is pinned down
by the downside rows, rather than sitting on a flat face of zero-risk solutions.

The objective carries no mean term, mirroring the MV, MAD and ES overlays, which
likewise minimize a risk measure alone.  As with MAD, and unlike mean-variance,
nothing has to be estimated beyond the demeaning: the semi-variance is read off
the empirical distribution directly, with no covariance or cross-covariance
matrix and so no shrinkage target (contrast Ledoit-Wolf and sample_cross_cov in
mv_curr_overlay).

The base currency's excess return is identically zero, so its overlay weight
never enters the objective and would be left unidentified.  It is pinned to
zero, as in the MV and MAD overlays, rather than left free.

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


class SemiVarianceCurrencyOverlayOptimizer(BaseOptimizer):
    """
    Two-step unregularized semi-variance optimizer with sequential currency overlay.

    Step 1 minimizes the semi-variance of R_fh subject to full investment and
    leverage constraints.  Step 2 fixes the asset weights and minimizes the
    semi-variance of the combined portfolio return.
    """

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``leverage_limit`` (float) — ||x||_1 cap for step 1, default 2.0
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
        self.x_opt: np.ndarray | None = None

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract and store return matrices for both optimization steps.

        Args:
            train_bundle: Must contain ``"R_fh"`` and ``"R_c"``.
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

    def _optimize_assets(self, Y_fh: np.ndarray) -> np.ndarray:
        """Step 1: unregularized semi-variance on R_fh with a leverage constraint
        and full investment."""
        T, N = Y_fh.shape
        A = Y_fh - Y_fh.mean(axis=0)

        x = cp.Variable(N)
        z = cp.Variable(T)

        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(z) / T),
            [
                z >= 0,
                -A @ x <= z,
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
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray
    ) -> np.ndarray:
        """Step 2: minimize the semi-variance of (R_fh @ x_fixed + R_c @ psi) over
        unconstrained psi."""
        T, K = Y_c.shape
        c = Y_fh @ x_fixed                  # fixed asset contribution, shape (T,)
        c_dm = c - c.mean()                 # demeaned; see module docstring
        A_c = Y_c - Y_c.mean(axis=0)

        psi = cp.Variable(K)
        z = cp.Variable(T)

        constraints = [
            z >= 0,
            -(c_dm + A_c @ psi) <= z,
        ]
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(cp.Minimize(cp.sum_squares(z) / T), constraints)
        self._solve(problem)

        overlay = psi.value
        if overlay is None:
            raise RuntimeError("Solver returned None weights in step 2.")
        return overlay.flatten()

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run both optimization steps sequentially.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or any solver step fails.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        logger.debug(
            "Step 1: optimizing asset weights on %d obs, %d assets.", self.q, self.N
        )
        self.x_opt = self._optimize_assets(self.Y_fh)

        logger.debug(
            "Step 2: optimizing currency overlay on %d obs, %d currencies.", self.q, self.K
        )
        psi_opt = self._optimize_overlay(self.Y_fh, self.Y_c, self.x_opt)

        return {"x": self.x_opt, "psi": psi_opt}

    def score(
        self,
        weights: dict[str, np.ndarray] | np.ndarray,
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the semi-variance of the combined portfolio on validation data.

        Uses ``self.x_opt`` (fixed from :meth:`optimize`), re-runs step 2 on the
        validation returns to get psi_val, then evaluates the semi-variance of the
        combined return.  Mirrors the unregularized ES, MV and MAD overlays, which
        likewise re-solve step 2 on the validation window.

        The deviation is taken about the validation window's own mean, matching
        the asset-level SMV optimizers.

        Args:
            weights: Not used; asset weights are taken from ``self.x_opt``.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Semi-variance of the combined portfolio return (lower is better).
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

        psi_val = self._optimize_overlay(Y_fh_val, Y_c_val, self.x_opt)
        r_val = Y_fh_val @ self.x_opt + Y_c_val @ psi_val
        downside = np.minimum(r_val - r_val.mean(), 0.0)
        return float(np.mean(downside ** 2))
