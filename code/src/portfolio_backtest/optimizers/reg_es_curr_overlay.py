"""
Two-step regularized Expected Shortfall optimizer with currency overlay.

Step 1: Minimizes regularized ES over fully hedged asset returns (R_fh), subject to
        full investment and a leverage cap on asset weights.
Step 2: With asset weights fixed, minimizes regularized ES of the combined portfolio
        return (R_fh @ x_opt + R_c @ psi) over the currency overlay weights psi.

Following Appendix A.3 of Lucescu & Ulrych (2026), psi carries neither a budget
constraint nor a leverage constraint. Forward contracts are zero-investment
instruments, so no slack weight is needed to fund the overlay.

The ES coefficient is normalized to one, which is the gamma -> infinity limit of the
paper's objective after dividing through by gamma / 2. The mean term vanishes and the
penalties are read as the already-normalized quantities 2 * lambda / gamma, so the
Table 3 bounds apply directly to lambda_*_asset and lambda_*_curr as used here.

With ``demean`` set, both return matrices are centered columnwise before either
step is solved, so ES is measured on deviations from the sample mean rather than
on raw returns and is comparable to the deviation measures (mean-variance, MAD,
semi-variance), all of which demean by construction. :meth:`score` holds the
trained weights fixed and re-centers only the validation portfolio return, by its
own mean. The flag enters the cache key because it is baked into the compiled
problems through Y_fh and Y_c, exactly like alpha and the leverage limit. Off by
default; the switch is ``DEMEAN`` in backtest_config.py, mirrored into the
parameter grid there as ``DEMEAN_GRID``.

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
import os

import cvxpy as cp
import numpy as np

from portfolio_backtest.backtest_config import DEMEAN

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class RegularizedExpectedShortfallCurrencyOverlayOptimizer(BaseOptimizer):
    """
    Two-step regularized ES optimizer with sequential currency overlay.

    Step 1 minimizes regularized ES on R_fh subject to full investment and leverage
    constraints. Step 2 fixes the asset weights and minimizes regularized ES of the
    combined portfolio return.
    """

    # Shared across the per-grid-point instances the Backtester creates. Reset by
    # fit() whenever the estimation window, alpha or leverage_limit changes.
    _cache_key: str | None = None
    _asset_prob: dict | None = None
    _overlay_prob: dict | None = None
    _asset_solutions: dict[tuple[float, float], np.ndarray] = {}

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``alpha``            (float) — ES confidence level, default 0.80
                * ``lambda_l1_asset``  (float) — L1 penalty on asset weights, default 0.0
                * ``lambda_l2_asset``  (float) — L2 penalty on asset weights, default 0.0
                * ``lambda_l1_curr``   (float) — L1 penalty on the overlay, default 0.0
                * ``lambda_l2_curr``   (float) — L2 penalty on the overlay, default 0.0
                * ``leverage_limit``   (float) — ||x||_1 cap for step 1, default 2.0
                * ``demean``           (bool)  — center R_fh and R_c columnwise
                                                 before solving, default
                                                 ``backtest_config.DEMEAN``
                * ``crossover``        (bool)  — run Gurobi crossover after the barrier
                                                 solve, default ``False``. Off is faster;
                                                 on returns vertex solutions with exact
                                                 zeros under an L1 penalty.
                * ``threads``          (int)   — Gurobi thread count, default from
                                                 ``SLURM_CPUS_PER_TASK`` or 4
                * ``solver``                   — cvxpy solver, default ``cp.GUROBI``
        """
        super().__init__(hyperparams)
        self.alpha: float = self.hyperparams.get("alpha", 0.80)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 0.0)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.0)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.0)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 0.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.demean: bool = bool(self.hyperparams.get("demean", DEMEAN))
        self.crossover: bool = bool(self.hyperparams.get("crossover", False))
        self.threads: int = int(
            self.hyperparams.get("threads", os.environ.get("SLURM_CPUS_PER_TASK", 4))
        )
        self.solver = self.hyperparams.get("solver", cp.GUROBI)

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

        ``alpha``, ``leverage_limit`` and ``demean`` enter the key because they are
        baked into the compiled problems as constants rather than as parameters
        (``demean`` through the centered Y_fh and Y_c). If any of them is ever added
        to the grid, the cache invalidates instead of silently reusing the wrong
        problem — or, for ``demean``, letting a demeaned and a raw run in the same
        process share problems and cached step-1 solutions.
        """
        parts = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
            float(self.alpha),
            float(self.leverage_limit),
            bool(self.demean),
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
        if self.demean:
            self.Y_fh = self.Y_fh - self.Y_fh.mean(axis=0)
            self.Y_c = self.Y_c - self.Y_c.mean(axis=0)
        self.q, self.N = self.Y_fh.shape
        self.K = self.Y_c.shape[1]

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

    def _build_asset_problem(self, Y_fh: np.ndarray) -> dict:
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

        problem = cp.Problem(
            cp.Minimize(es_term + lam1 * l1_norm + lam2 * cp.sum_squares(x)),
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

    def _optimize_assets(self, Y_fh: np.ndarray) -> np.ndarray:
        """
        Step 1: regularized ES on R_fh with a leverage constraint and full investment.

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

    def _build_overlay_problem(self, Y_c: np.ndarray) -> dict:
        """
        Compile the step-2 problem once, with the penalties and the fixed asset return
        contribution left as parameters.
        """
        q, K = Y_c.shape
        tail_prob = 1.0 - self.alpha

        psi = cp.Variable(K)
        u = cp.Variable(q)
        eta = cp.Variable()
        c = cp.Parameter(q)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        es_term = eta + (1.0 / (q * tail_prob)) * cp.sum(u)

        problem = cp.Problem(
            cp.Minimize(es_term + lam1 * cp.norm(psi, 1) + lam2 * cp.sum_squares(psi)),
            [
                u >= 0,
                -c - Y_c @ psi - eta <= u,
            ],
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-2 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "psi": psi, "c": c, "lam1": lam1, "lam2": lam2}

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray
    ) -> np.ndarray:
        """Step 2: regularized ES of (R_fh @ x_fixed + R_c @ psi) over unconstrained psi."""
        cls = type(self)
        if cls._overlay_prob is None:
            cls._overlay_prob = self._build_overlay_problem(Y_c)

        p = cls._overlay_prob
        p["c"].value = Y_fh @ x_fixed  # fixed asset return contribution, shape (q,)
        p["lam1"].value = float(self.lambda_l1_curr)
        p["lam2"].value = float(self.lambda_l2_curr)
        self._solve(p["prob"])

        return p["psi"].value.flatten()

    def optimize(self) -> dict[str, np.ndarray]:
        """
        Run both optimization steps sequentially.

        Returns:
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency overlay,
            shape K).

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
        Evaluate validation Expected Shortfall of train-fitted candidate weights.

        Both the asset weights and the currency overlay are taken as given from
        :meth:`optimize` on the training window and held fixed. Nothing is re-solved on
        the validation sample.

        Under ``demean`` the validation portfolio return is centered by its own mean,
        not the training one: the score has to be the deviation ES of this window, so
        the location it deviates from is this window's mean.

        Args:
            weights: Dict with ``"x"`` and ``"psi"`` as returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Negative mean of the worst ``(1-alpha)`` fraction of combined portfolio
            returns on the validation sample.
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
        if self.demean:
            port_rets = port_rets - port_rets.mean()
        threshold = np.quantile(port_rets, 1.0 - self.alpha)
        tail = port_rets[port_rets <= threshold]
        return float(-np.mean(tail)) if len(tail) > 0 else float(-np.mean(port_rets))