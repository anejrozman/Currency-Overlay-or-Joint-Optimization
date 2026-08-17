"""
Regularized Mean-Variance optimizer with currency overlay.

Step 1: Minimizes the regularized variance of fully hedged asset returns (R_fh),
        with L1/L2 penalties, full investment and a leverage cap.
Step 2: With asset weights fixed, minimizes the regularized variance of the combined
        portfolio return (R_fh @ x_opt + R_c @ psi) over unconstrained psi.

Following Appendix A.3 of Lucescu & Ulrych (2026), psi carries neither a budget
constraint nor a leverage constraint. Forward contracts are zero-investment
instruments, so no slack weight is needed to fund the overlay.

The objective carries no mean term.  The variance coefficient is normalized to one,
which is the gamma -> infinity limit of the paper's objective after dividing through
by gamma / 2: the mean term vanishes, the cross-covariance term picks up its factor
of two directly from the expansion of Var(x' R_fh + psi' R_c), and the penalties are
read as the already-normalized quantities 2 * lambda / gamma.  This mirrors the ES
optimizers, which likewise penalize a risk measure alone, and removes the need for a
shrinkage estimator of the mean (Bayes-Stein, Jorion 1986).

Keeping the mean makes the overlay wildly unstable: at gamma = 3 the unpenalized
gross overlay reaches 20-70x the portfolio notional against roughly 1x once the mean
is dropped (see results/runs/reg_mv_curr_overlay/run_1).

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
from ..utils.estimation import ledoit_wolf_cc, sample_cross_cov

logger = logging.getLogger(__name__)

# Production runs use Gurobi (cluster). Fall back to Clarabel locally, where
# gurobipy is not installed; weight differences are ~1e-6.
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False


class RegularizedMeanVarianceCurrencyOverlayOptimizer(BaseOptimizer):
    """
    Two-step regularized mean-variance optimizer with sequential currency overlay.

    Step 1 minimizes the regularized variance of R_fh subject to full investment and
    leverage constraints. Step 2 fixes the asset weights and minimizes the regularized
    variance of the combined portfolio return.
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

                * ``lambda_l1_asset`` (float) — L1 penalty for asset weights, default 5e-6
                * ``lambda_l2_asset`` (float) — L2 penalty for asset weights, default 1e-3
                * ``lambda_l1_curr``  (float) — L1 penalty for the overlay, default 5e-6
                * ``lambda_l2_curr``  (float) — L2 penalty for the overlay, default 1e-4
                * ``leverage_limit``  (float) — ||x||_1 cap for step 1, default 2.0
                * ``solver``                  — cvxpy solver, default ``cp.GUROBI``
                                                (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 5e-6)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 1e-3)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 5e-6)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 1e-4)
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

        self.Y_fh = R_fh.values
        self.Y_c = R_c.values
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
        N = Y_fh.shape[1]
        Sigma_a = ledoit_wolf_cc(Y_fh)

        x = cp.Variable(N)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        l1_norm = cp.norm(x, 1)
        obj = (
            cp.quad_form(x, cp.psd_wrap(Sigma_a))
            + lam1 * l1_norm
            + lam2 * cp.sum_squares(x)
        )
        problem = cp.Problem(
            cp.Minimize(obj),
            [
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-1 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "x": x, "lam1": lam1, "lam2": lam2}

    def _optimize_assets(self, Y_fh: np.ndarray) -> np.ndarray:
        """
        Step 1: regularized minimum variance on R_fh with Ledoit-Wolf covariance,
        a leverage constraint and full investment.

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
        weights = (weights / weights.sum()).flatten()
        cls._asset_solutions[cache_id] = weights
        return weights.copy()

    # Step 2: currency overlay

    def _build_overlay_problem(self, Y_fh: np.ndarray, Y_c: np.ndarray) -> dict:
        """
        Compile the step-2 problem once, with the penalties and the fixed asset
        cross-covariance contribution left as parameters.

        The moment estimates are baked in as constants, so the compiled problem is
        valid only for the window it was built from.  ``_refresh_cache`` drops it
        whenever the window changes, and :meth:`score` never re-solves on validation
        data, so no other window ever reaches this problem.
        """
        K = Y_c.shape[1]
        Sigma_c = ledoit_wolf_cc(Y_c)
        Sigma_ac = sample_cross_cov(Y_fh, Y_c)  # N x K, not shrunk

        psi = cp.Variable(K)
        cross = cp.Parameter(K)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        obj = (
            cp.quad_form(psi, cp.psd_wrap(Sigma_c))
            + cross @ psi
            + lam1 * cp.norm(psi, 1)
            + lam2 * cp.sum_squares(psi)
        )
        # Base currency (zero-variance excess return) is the residual numeraire,
        # not an overlay decision variable; pin its weight to zero.
        constraints = []
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)
        problem = cp.Problem(cp.Minimize(obj), constraints)
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-2 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "psi": psi, "cross": cross,
                "lam1": lam1, "lam2": lam2, "Sigma_ac": Sigma_ac}

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray
    ) -> np.ndarray:
        """Step 2: regularized minimum-variance currency overlay over unconstrained psi.

        Minimizes psi' Sigma_c psi + 2 x*' Sigma_ac psi + penalties, the expansion of
        Var(x*' R_fh + psi' R_c) with the constant asset-only term dropped
        (appendix B, eq:mv_curr_expanded). Sigma_c is Ledoit-Wolf shrunk; the
        cross-covariance Sigma_ac is the un-shrunk sample estimate.
        """
        cls = type(self)
        if cls._overlay_prob is None:
            cls._overlay_prob = self._build_overlay_problem(Y_fh, Y_c)

        p = cls._overlay_prob
        p["cross"].value = 2.0 * (x_fixed @ p["Sigma_ac"])  # length-K vector
        p["lam1"].value = float(self.lambda_l1_curr)
        p["lam2"].value = float(self.lambda_l2_curr)
        self._solve(p["prob"])

        return p["psi"].value.flatten()

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
        Evaluate the validation variance of train-fitted candidate weights.

        Both the asset weights and the currency overlay are taken as given from
        :meth:`optimize` on the training window and held fixed. Nothing is re-solved on
        the validation sample, mirroring the regularized ES overlay.

        Args:
            weights: Dict with ``"x"`` and ``"psi"`` as returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Variance of the combined portfolio return on the validation sample
            (lower is better).
        """
        if not isinstance(weights, dict) or "x" not in weights or "psi" not in weights:
            raise ValueError("weights must be a dict with keys 'x' and 'psi'.")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")

        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty:
            return float("inf")

        x = np.asarray(weights["x"], dtype=float)
        psi = np.asarray(weights["psi"], dtype=float)
        r_val = R_fh_val.values @ x + R_c_val.values @ psi
        return float(np.var(r_val, ddof=1))
