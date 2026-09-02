"""
Regularized Semi-Variance optimizer with currency overlay
(Markowitz, Todd, Xu & Yamane 1993, in the framework of Lucescu & Ulrych 2026, Appendix A.3).

Step 1: Minimizes the regularized semi-variance of fully hedged asset returns
        (R_fh), with L1/L2 penalties, full investment and a leverage cap.
Step 2: With asset weights fixed, minimizes the regularized semi-variance of the
        combined portfolio return (R_fh @ x_opt + R_c @ psi) over unconstrained psi.

Following Appendix A.3 of Lucescu & Ulrych (2026), psi carries neither a budget
constraint nor a leverage constraint. Forward contracts are zero-investment
instruments, so no slack weight is needed to fund the overlay.

The risk measure is Markowitz et al.'s S_E, eq. (3.1a), with the target E set to
the portfolio's own expected return rather than an exogenous benchmark:

    SMV(w) = (1/T) * sum_t [ max(0, -A_t w) ]^2,

where A is the columnwise demeaned return matrix, so only deviations below the
portfolio's own mean are counted. Both steps use their eqs. (3.4)-(3.7)
reformulation, lifting the positive part into auxiliary variables to make the
objective quadratic. The paper introduces 2T of them — Y for the signed deviation
and Z = Y^- for its negative part, tied by BX - Y + Z = 0 — because the Critical
Line Algorithm needs the constraints in equality form. Only the downside half is
needed here: minimizing sum z_t^2 subject to z >= -A w and z >= 0 drives each z_t
to max(0, -A_t w) on its own, so the equality and the free variables Y drop out
and each step is a QP with T auxiliary variables rather than 2T. The 1/sqrt(T)
that eq. (3.8) folds into B is carried as an explicit 1/T on the objective
instead, which leaves A on the scale of the raw returns and keeps the risk term
numerically better conditioned. The risk term is reported as the semi-variance
itself, not doubled onto the variance scale, which matters because the penalty
grids are calibrated against the size of the risk term.

Demeaning in step 2 covers both terms. Writing c = R_fh @ x_opt for the fixed
asset contribution, mean(c + R_c psi) = mean(c) + mean(R_c) psi for any psi, so

    SMV(x_opt, psi) = (1/T) * sum_t [ max(0, -( (c - mean c) + (R_c - mean R_c) psi )_t ) ]^2,

i.e. the fixed contribution and the currency matrix are each demeaned before the
program is formed. Unlike the mean-variance overlay, no term of the objective is
constant in psi and none can be dropped: the positive part is taken of the
combined return, so the asset contribution stays inside it.

That demeaning also keeps step 2 well posed. The residual c_dm + A_c psi sums to
exactly zero over the window for every psi, so it can be non-negative throughout
only if it vanishes identically — which needs c_dm to lie in the column space of
A_c, impossible for T ~ 2500 against a dozen currencies. The risk term is
therefore strictly positive at the optimum even before the penalties enter.

The objective carries no mean term, mirroring the MV, MAD and ES overlays, which
likewise penalize a risk measure alone. As with MAD, and unlike mean-variance,
nothing has to be estimated beyond the demeaning: the semi-variance is read off
the empirical distribution directly, with no covariance or cross-covariance matrix
and so no shrinkage target (contrast Ledoit-Wolf and sample_cross_cov in
reg_mv_curr_overlay).

The base currency's excess return is identically zero, so its overlay weight never
enters the risk term and would be left unidentified. It is pinned to zero, as in
the MV and MAD overlays, rather than left free: an unidentified variable would take
an arbitrary solver-dependent value, be charged the L1 penalty, and land in the
saved weight records.

Performance notes:

* The Backtester constructs a fresh optimizer instance for every hyperparameter
  combination, so the compiled problems and the step-1 solutions are cached at class
  level and keyed on the estimation window.
* Penalties enter as ``cp.Parameter``, so each problem canonicalizes once per window
  rather than once per grid point (DPP).
* Step 1 does not depend on the currency penalties, so it is solved once per
  (lambda_l1_asset, lambda_l2_asset) pair rather than once per grid point.

Author: Anej Rozman
"""

from __future__ import annotations

import hashlib
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


class RegularizedSemiVarianceCurrencyOverlayOptimizer(BaseOptimizer):
    """
    Two-step regularized semi-variance optimizer with sequential currency overlay.

    Step 1 minimizes the regularized semi-variance of R_fh subject to full
    investment and leverage constraints. Step 2 fixes the asset weights and
    minimizes the regularized semi-variance of the combined portfolio return.
    """

    # Shared across the per-grid-point instances the Backtester creates. Reset by
    # fit() whenever the estimation window or leverage_limit changes.
    _cache_key: str | None = None
    _asset_prob: dict | None = None
    _overlay_prob: dict | None = None
    _asset_solutions: dict[tuple[float, float], np.ndarray] = {}

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``lambda_l1_asset`` (float) — L1 penalty for asset weights, default 1e-5
                * ``lambda_l2_asset`` (float) — L2 penalty for asset weights, default 1.5e-3
                * ``lambda_l1_curr``  (float) — L1 penalty for the overlay, default 0.0
                * ``lambda_l2_curr``  (float) — L2 penalty for the overlay, default 0.0
                * ``leverage_limit``  (float) — ||x||_1 cap for step 1, default 2.0
                * ``solver``                  — cvxpy solver, default ``cp.GUROBI``
                                                (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 1e-5)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 1.5e-3)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.0)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 0.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        for name in (
            "lambda_l1_asset",
            "lambda_l2_asset",
            "lambda_l1_curr",
            "lambda_l2_curr",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")

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

    # Caching

    @staticmethod
    def _frame_signature(df) -> tuple:
        """Identify a return-matrix slice by its endpoints, shape and columns."""
        return (
            str(df.index[0]),
            str(df.index[-1]),
            tuple(df.shape),
            tuple(map(str, df.columns)),
        )

    def _refresh_cache(self, R_fh, R_c) -> None:
        """
        Drop the compiled problems and cached step-1 solutions if the estimation
        window has changed.

        ``leverage_limit`` enters the key because it is baked into the compiled problem
        as a constant rather than as a parameter. If it is ever added to the grid, the
        cache invalidates instead of silently reusing the wrong problem.
        """
        parts = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.leverage_limit),
        )
        key = hashlib.blake2b(repr(parts).encode(), digest_size=16).hexdigest()

        cls = type(self)
        if key != cls._cache_key:
            cls._cache_key = key
            cls._asset_prob = None
            cls._overlay_prob = None
            cls._asset_solutions = {}
            logger.debug("Problem cache invalidated for new estimation window.")

    @classmethod
    def clear_cache(cls) -> None:
        """Drop all compiled problems and cached step-1 solutions."""
        cls._cache_key = None
        cls._asset_prob = None
        cls._overlay_prob = None
        cls._asset_solutions = {}

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

        self._refresh_cache(R_fh, R_c)

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

    # Step 1: assets

    def _build_asset_problem(self, Y_fh: np.ndarray) -> dict:
        """Compile the step-1 problem once, with the penalties left as parameters."""
        T, N = Y_fh.shape
        A = Y_fh - Y_fh.mean(axis=0)

        x = cp.Variable(N)
        z = cp.Variable(T)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(
                cp.sum_squares(z) / T + lam1 * l1_norm + lam2 * cp.sum_squares(x)
            ),
            [
                z >= 0,
                -A @ x <= z,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-1 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "x": x, "lam1": lam1, "lam2": lam2}

    def _optimize_assets(self, Y_fh: np.ndarray) -> np.ndarray:
        """
        Step 1: regularized semi-variance on R_fh with a leverage constraint and
        full investment.

        The solution depends only on the asset penalties, so it is cached on
        (lambda_l1_asset, lambda_l2_asset) for the current estimation window.
        """
        cls = type(self)
        cache_id = (float(self.lambda_l1_asset), float(self.lambda_l2_asset))
        if cache_id in cls._asset_solutions:
            return cls._asset_solutions[cache_id].copy()

        if cls._asset_prob is None:
            cls._asset_prob = self._build_asset_problem(Y_fh)

        p = cls._asset_prob
        p["lam1"].value = float(self.lambda_l1_asset)
        p["lam2"].value = float(self.lambda_l2_asset)
        self._solve(p["prob"])

        weights = p["x"].value
        if weights is None:
            raise RuntimeError("Solver returned None weights in step 1.")
        weights = (weights / weights.sum()).flatten()
        cls._asset_solutions[cache_id] = weights
        return weights.copy()

    # Step 2: currency overlay

    def _build_overlay_problem(self, Y_c: np.ndarray) -> dict:
        """
        Compile the step-2 problem once, with the penalties and the demeaned fixed
        asset contribution left as parameters.

        The demeaned currency matrix is baked in as a constant, so the compiled problem
        is valid only for the window it was built from.  ``_refresh_cache`` drops it
        whenever the window changes, and :meth:`score` never re-solves on validation
        data, so no other window ever reaches this problem.
        """
        T, K = Y_c.shape
        A_c = Y_c - Y_c.mean(axis=0)

        psi = cp.Variable(K)
        z = cp.Variable(T)
        c_dm = cp.Parameter(T)   # demeaned fixed asset contribution
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        constraints = [
            z >= 0,
            -(c_dm + A_c @ psi) <= z,
        ]
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(
            cp.Minimize(
                cp.sum_squares(z) / T
                + lam1 * cp.norm(psi, 1)
                + lam2 * cp.sum_squares(psi)
            ),
            constraints,
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-2 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "psi": psi, "c_dm": c_dm, "lam1": lam1, "lam2": lam2}

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray
    ) -> np.ndarray:
        """Step 2: regularized semi-variance of (R_fh @ x_fixed + R_c @ psi) over
        unconstrained psi."""
        cls = type(self)
        if cls._overlay_prob is None:
            cls._overlay_prob = self._build_overlay_problem(Y_c)

        c = Y_fh @ x_fixed
        p = cls._overlay_prob
        p["c_dm"].value = c - c.mean()
        p["lam1"].value = float(self.lambda_l1_curr)
        p["lam2"].value = float(self.lambda_l2_curr)
        self._solve(p["prob"])

        overlay = p["psi"].value
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
        weights: dict[str, np.ndarray],
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate the validation semi-variance of train-fitted candidate weights.

        Both the asset weights and the currency overlay are taken as given from
        :meth:`optimize` on the training window and held fixed. Nothing is re-solved on
        the validation sample, matching the regularized ES, MV and MAD overlays: the
        paper (p.25) specifies weights computed on training data and the risk measure
        evaluated on the validation sample.

        The deviation is taken about the validation window's own mean, matching the
        asset-level SMV optimizers.

        Args:
            weights: Dict with ``"x"`` and ``"psi"`` as returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Semi-variance of the combined portfolio return on the validation sample
            (lower is better).
        """
        if not isinstance(weights, dict) or "x" not in weights or "psi" not in weights:
            raise ValueError("weights must be a dict with keys 'x' and 'psi'.")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")

        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty or len(R_fh_val) < 2:
            return float("inf")

        x = np.asarray(weights["x"], dtype=float)
        psi = np.asarray(weights["psi"], dtype=float)
        r_val = R_fh_val.values @ x + R_c_val.values @ psi
        downside = np.minimum(r_val - r_val.mean(), 0.0)
        return float(np.mean(downside ** 2))
