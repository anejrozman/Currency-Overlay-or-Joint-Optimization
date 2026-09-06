"""
Two-step regularized mean-Expected Shortfall optimizer with currency overlay.

The mean-bearing counterpart of reg_es_curr_overlay.py, and the regularized
counterpart of es_curr_overlay_w_mean.py.

Step 1: minimizes  ES(R_fh @ x) - (1/gamma) * mu_a' x  + lambda_l1_asset ||x||_1
        + lambda_l2_asset ||x||^2, subject to full investment and a leverage cap.
Step 2: with x fixed, minimizes
        ES(R_fh @ x* + R_c @ psi) - (1/gamma) * (mu_a' x* + mu_c' psi)
        + lambda_l1_curr ||psi||_1 + lambda_l2_curr ||psi||^2,
        subject to an L1 cap on psi.  The mu_a' x* term is an additive constant
        there and does not move the argmin; it is carried as a parameter so the
        reported objective is the full one.

gamma is fixed at 2 rather than searched.  The penalty grid already has 256
combinations per rebalancing date, and multiplying that by a seven-point gamma
grid is not affordable at the universe sizes this is run on.  It remains
readable from ``hyperparams`` so it can be swept deliberately later without
touching this file, but it is deliberately absent from
``REG_ES_MEAN_OVERLAY_CONFIG``'s grid.  Note that gamma = 2 was not calibrated
for this program: the unregularized overlay's sweep put its optimum near 0.2 on
the top-10 universe, but that program had no penalties competing with the mean
term, so the two are not comparable and 2 is a deliberately conservative choice.

Means.  Identical to the unregularized overlay's:

  * ``mu_a`` is pooled onto the 14 economies and James-Stein shrunk toward zero
    (:func:`~portfolio_backtest.utils.estimation.pooled_asset_mean`).
  * ``mu_c`` is conditioned on the forward premium, mu_c = -f_t
    (:func:`~portfolio_backtest.utils.estimation.conditional_currency_mean`).

Both are constants for a given estimation window, so they are baked into the
compiled problems rather than passed as parameters; the window is already the
problem cache key, so nothing can go stale.  ``gamma`` is added to that key for
the same reason ``alpha`` and the leverage limit are in it: it is compiled in.

Base currency.  psi_base is pinned to zero.  Its R_c column is identically zero
and its forward premium likewise, so it would consume overlay budget while
contributing nothing to either the return or the mean term.  The pin also
removes the unidentified direction that makes the optimal face of the
unpenalized program contain a line.

No demeaning.  The ``demean`` switch reg_es_curr_overlay.py carries is
deliberately absent: centering would zero the very sample means this program is
built around.  It is correspondingly gone from the cache key.

Performance notes, inherited from reg_es_curr_overlay.py:

* The Backtester constructs a fresh optimizer instance for every hyperparameter
  combination, so the compiled problems and the step-1 solutions are cached at
  class level and keyed on the estimation window.
* Penalties enter as ``cp.Parameter``, so each problem canonicalizes once per
  window rather than once per grid point (DPP).
* Step 1 does not depend on the currency penalties, so it is solved once per
  (lambda_l1_asset, lambda_l2_asset) pair rather than once per grid point.

Author: Anej Rozman
"""

from __future__ import annotations

import hashlib
import logging
import os

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
# gurobipy is not installed; weight differences are ~1e-6. This mirrors every
# other regularized optimizer in the package (reg_mv/mad/smv/soes) — only the
# original ES pair hard-defaults to Gurobi, which makes them unrunnable locally.
try:
    import gurobipy  # noqa: F401
    _DEFAULT_SOLVER = cp.GUROBI
except ImportError:
    _DEFAULT_SOLVER = cp.CLARABEL

_warned_fallback = False


class RegularizedExpectedShortfallCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step regularized mean-ES optimizer with sequential currency overlay.

    Step 1 minimizes penalized mean-ES on R_fh subject to full investment and a
    leverage cap.  Step 2 fixes the asset weights and minimizes penalized mean-ES
    of the combined portfolio return subject to an L1 cap on the overlay.
    """

    # Shared across the per-grid-point instances the Backtester creates. Reset by
    # fit() whenever the estimation window or a compiled-in constant changes.
    _cache_key: str | None = None
    _asset_prob: dict | None = None
    _overlay_prob: dict | None = None
    _asset_solutions: dict[tuple[float, float], np.ndarray] = {}

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``            (float) — ES confidence level, default 0.80
                * ``gamma``            (float) — risk aversion; the mean term enters
                                                 with coefficient 1 / gamma, default 2.
                                                 Fixed rather than searched; see the
                                                 module docstring.
                * ``lambda_l1_asset``  (float) — L1 penalty on asset weights, default 0.0
                * ``lambda_l2_asset``  (float) — L2 penalty on asset weights, default 0.0
                * ``lambda_l1_curr``   (float) — L1 penalty on the overlay, default 0.0
                * ``lambda_l2_curr``   (float) — L2 penalty on the overlay, default 0.0
                * ``leverage_limit``   (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``    (float) — ||psi||_1 cap for step 2, default 2.0
                * ``crossover``        (bool)  — run Gurobi crossover after the barrier
                                                 solve, default ``False``. Off is faster;
                                                 on returns vertex solutions with exact
                                                 zeros under an L1 penalty.
                * ``threads``          (int)   — Gurobi thread count, default from
                                                 ``SLURM_CPUS_PER_TASK`` or 4
                * ``solver``                   — cvxpy solver, default ``cp.GUROBI``
                                                 (``cp.CLARABEL`` when gurobipy
                                                 is unavailable)
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.gamma: float = self.hyperparams.get("gamma", 2)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 0.0)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.0)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.0)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 0.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.overlay_limit: float = self.hyperparams.get("overlay_limit", 2.0)
        self.crossover: bool = bool(self.hyperparams.get("crossover", False))
        self.threads: int = int(
            self.hyperparams.get("threads", os.environ.get("SLURM_CPUS_PER_TASK", 4))
        )
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

        global _warned_fallback
        if self.solver is not cp.GUROBI and _DEFAULT_SOLVER is not cp.GUROBI and not _warned_fallback:
            logger.warning("gurobipy not installed — falling back to Clarabel solver.")
            _warned_fallback = True

        for name in (
            "lambda_l1_asset",
            "lambda_l2_asset",
            "lambda_l1_curr",
            "lambda_l2_curr",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")

        self.Y_fh: np.ndarray | None = None
        self.Y_c: np.ndarray | None = None
        self.economy_ids: np.ndarray | None = None
        self.mu_a: np.ndarray | None = None
        self.mu_c: np.ndarray | None = None
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
        Drop the compiled problems and cached step-1 solutions if anything baked
        into them has changed.

        ``alpha``, ``gamma``, ``leverage_limit`` and ``overlay_limit`` enter the key
        because they are compiled in as constants rather than passed as parameters.
        The two mean vectors are compiled in as well, but they are functions of the
        window alone (``mu_a`` of R_fh and the economy map, ``mu_c`` of the forward
        premium aligned to R_c), so the frame signatures already cover them.
        """
        parts = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.alpha),
            float(self.gamma),
            float(self.leverage_limit),
            float(self.overlay_limit),
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

        self._refresh_cache(R_fh, R_c)

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
                    Method=2,
                    Crossover=-1 if self.crossover else 0,
                    BarConvTol=1e-8,
                    Threads=self.threads,
                )
            else:
                problem.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError as exc:
            raise RuntimeError(f"Solver error: {exc}") from exc

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Optimization failed: status={problem.status}")

    # Step 1: assets

    def _build_asset_problem(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> dict:
        """Compile the step-1 problem once, with the penalties left as parameters."""
        q, N = Y_fh.shape
        tail_prob = 1.0 - self.alpha

        x = cp.Variable(N)
        u = cp.Variable(q)
        eta = cp.Variable()
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        l1_norm = cp.norm(x, 1)
        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)
        mean_term = (1.0 / self.gamma) * (mu_a @ x)

        problem = cp.Problem(
            cp.Minimize(es_term - mean_term + lam1 * l1_norm + lam2 * cp.sum_squares(x)),
            [
                u >= 0,
                -Y_fh @ x - eta <= u,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-1 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "x": x, "lam1": lam1, "lam2": lam2}

    def _optimize_assets(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
        """
        Step 1: penalized mean-ES on R_fh with a leverage constraint and full
        investment.

        The solution depends only on the asset penalties, so it is cached on
        (lambda_l1_asset, lambda_l2_asset) for the current estimation window.
        """
        cls = type(self)
        cache_id = (float(self.lambda_l1_asset), float(self.lambda_l2_asset))
        if cache_id in cls._asset_solutions:
            return cls._asset_solutions[cache_id].copy()

        if cls._asset_prob is None:
            cls._asset_prob = self._build_asset_problem(Y_fh, mu_a)

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

    def _build_overlay_problem(self, Y_c: np.ndarray, mu_c: np.ndarray) -> dict:
        """
        Compile the step-2 problem once, with the penalties, the fixed asset return
        contribution and the constant asset mean left as parameters.
        """
        q, K = Y_c.shape
        tail_prob = 1.0 - self.alpha

        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()
        c = cp.Parameter(q)
        mean_assets = cp.Parameter()
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)
        mean_term = (1.0 / self.gamma) * (mean_assets + mu_c @ psi)

        constraints = [
            u >= 0,
            -c - Y_c @ psi - eta <= u,
            cp.norm(psi, 1) <= self.overlay_limit,
        ]
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(
            cp.Minimize(
                es_term - mean_term
                + lam1 * cp.norm(psi, 1)
                + lam2 * cp.sum_squares(psi)
            ),
            constraints,
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-2 problem is not DPP-compliant; parameters would recompile.")
        return {
            "prob": problem,
            "psi": psi,
            "c": c,
            "mean_assets": mean_assets,
            "lam1": lam1,
            "lam2": lam2,
        }

    def _optimize_overlay(
        self,
        Y_fh: np.ndarray,
        Y_c: np.ndarray,
        x_fixed: np.ndarray,
        mu_a: np.ndarray,
        mu_c: np.ndarray,
    ) -> np.ndarray:
        """Step 2: penalized mean-ES of (R_fh @ x_fixed + R_c @ psi) over capped psi."""
        cls = type(self)
        if cls._overlay_prob is None:
            cls._overlay_prob = self._build_overlay_problem(Y_c, mu_c)

        p = cls._overlay_prob
        p["c"].value = Y_fh @ x_fixed  # fixed asset return contribution, shape (q,)
        p["mean_assets"].value = float(mu_a @ x_fixed)
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
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or any solver step fails.
            ValueError: If ``alpha`` is outside *(0, 1)*.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

        tail_prob = 1.0 - self.alpha
        if not (0.0 < tail_prob < 1.0):
            raise ValueError("alpha must be strictly in (0, 1).")

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
        weights: dict[str, np.ndarray],
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate train-fitted candidate weights on the validation window.

        Both the asset weights and the currency overlay are taken as given from
        :meth:`optimize` on the training window and held fixed; nothing is
        re-solved on the validation sample, exactly as in
        reg_es_curr_overlay.py.  The score is

            ES_alpha(combined loss) - 1.0 * mean(realized combined returns),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one.

        The reference coefficient is 1.0 rather than 1 / gamma so the number is
        directly comparable with the unregularized mean optimizers, which use the
        same yardstick.  Either is unbiased here: gamma is fixed across the whole
        grid, so unlike the unregularized case there is no candidate that can
        earn a larger mean credit than another simply by carrying a smaller
        gamma.  The realized mean is used rather than mu_a' x + mu_c' psi because
        a score should measure what the weights actually earned on the window,
        not re-apply a shrunk or conditional forecast to them.

        Args:
            weights: Dict with ``"x"`` and ``"psi"`` as returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Validation ES minus realized mean return, lower is better.
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
        port_rets = R_fh_val.values @ x + R_c_val.values @ psi

        threshold = np.quantile(port_rets, 1.0 - self.alpha)
        tail = port_rets[port_rets <= threshold]
        es = float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))

        return es - float(port_rets.mean())
