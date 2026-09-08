"""
Regularized mean-MAD optimizer with currency overlay.

The mean-bearing counterpart of reg_mad_curr_overlay.py, and the regularized
counterpart of mad_curr_overlay_w_mean.py.  Both steps minimize the same penalized
mean-risk functional

    MAD(portfolio return) - (1 / gamma) mu' w
    + lambda_l1 ||w||_1 + lambda_l2 ||w||^2,

with gamma a risk-aversion coefficient in the conventional direction: large gamma
starves the mean term and the program collapses onto reg_mad_curr_overlay, small
gamma gives the mean its full weight.

Step 1: minimizes MAD(R_fh @ x) - (1/gamma) mu_a' x plus the asset penalties, with
        full investment and a leverage cap on x.
Step 2: with asset weights fixed, minimizes

            MAD(R_fh @ x* + R_c @ psi) - (1/gamma) mu_c' psi
            + lambda_l1_curr ||psi||_1 + lambda_l2_curr ||psi||^2,

        subject to an L1 cap on the overlay.  The mu_a' x* term is a constant there
        and cannot move the argmin, so it is dropped -- which is also what lets
        :meth:`score` avoid estimating mu_a on the validation window.

Both steps keep Konno & Yamazaki's lower semi-absolute deviation form from
reg_mad_curr_overlay.py: for a sample with zero mean, mean|z| = 2 * mean(max(0, -z)),
which is linear once the positive part is lifted into auxiliary variables.  The
identity survives the mean term untouched, because the mean term does not enter z:
the residual A x in step 1 and c_dm + A_c psi in step 2 each sum to exactly zero
over the window by construction.  The factor of two is carried explicitly so both
objectives stay on the MAD scale, which matters because the penalty grids are
calibrated against the size of the risk term.

Location.  The demeaning makes the MAD a *deviation* measure, invariant to a
location shift, so the mean reaches the objective only through the explicit,
pooled and shrunk mu terms and gamma -> infinity recovers reg_mad_curr_overlay.py.
Measuring absolute deviation about zero instead would not be location-free and
would reward a high mean at fixed weight one on top of the explicit term -- the
double-counting reg_es_curr_overlay_w_mean.py carries because raw ES is not a
deviation measure.  One inconsistency the deviation form leaves, stated rather
than fixed: the demeaning subtracts the raw per-column sample mean while the mean
term rewards the pooled and shrunk mu_a.

gamma is fixed rather than searched, at 0.5.  This config already carries
4 x 4 x 4 x 4 = 256 penalty combinations per rebalancing date, and crossing that
with MAD_GAMMA_GRID's six points is not affordable -- the same reasoning
REG_MV_MEAN_OVERLAY_CONFIG and REG_SMV_MEAN_OVERLAY_CONFIG record.  0.5 is
calibrated by the same criterion that picked 30 for the MV and SMV pairs: on the
fixed-gamma sweep of the unregularized MAD overlay it is the most aggressive grid
point that still improves drawdown over the mean-free program (MDD 26.5% against
27.7%) while more than doubling the Sharpe, from 0.160 to 0.338.  The full curve is
tabulated at MAD_GAMMA_GRID in backtest_config.py.  It is still read from
hyperparams, so it can be swept deliberately later without touching this code.

Note that gamma is not comparable across risk measures.  MAD is degree one, like
ES and unlike MV and SMV, so its gamma is a pure information ratio and its band
sits two to three decades below theirs; 0.5 here and 30 there are comparable
positions on their own curves, not comparable numbers.

Fixing gamma also removes the one bias the unregularized pair had to reason about.
There :meth:`score` had to be gamma-free because a candidate carrying a smaller
gamma could earn an unboundedly larger mean credit; here every candidate carries
the same gamma, so the reference coefficient of 1.0 below is unbiased by
construction and the search selects penalties alone.

Overlay cap.  Required, not optional, and for a reason the MV and SMV pairs do not
share.  Their risk terms are degree two, so the quadratic eventually dominates the
linear mean term in every direction.  MAD is positively homogeneous of degree one,
so along psi = t * d the step-2 objective behaves like t * [MAD(A_c d) - mu_c' d /
gamma] and runs to minus infinity for any gamma below

    gamma_crit = max_d [ mu_c' d / MAD(A_c d) ],

measured at 0.062, 0.078 and 0.110 across estimation windows.  The shipped gamma of
0.5 sits about four and a half times above the worst of those, so this program is
well posed on its own terms and the cap is a safeguard rather than the only thing
holding it together -- but a run at a materially smaller gamma without the cap
would diverge rather than merely hold a large overlay.  The cap is set at 2.0,
matching every other mean-bearing program, and costs nothing at the conservative
end: mean-free MAD runs a gross psi of 0.19 to 1.18 across those windows, so the
cap is slack at gamma -> infinity and this is a strict superset of
reg_mad_curr_overlay.py.

Means.  Estimated exactly as in mad_curr_overlay_w_mean.py: mu_a pooled onto the 14
economies and James-Stein shrunk toward zero, mu_c conditioned on the forward
premium as the carry -f_t.  What each channel actually forecasts over the following
quarter is recorded in mv_curr_overlay_w_mean.py -- mu_c genuinely, mu_a not,
cross-sectionally.

Base currency.  Pinned to zero: its R_c column is identically zero so it never
enters the risk term, its mu_c entry is zero, and the overlay cap would otherwise
let it consume budget while contributing nothing.

Performance notes, inherited from reg_mad_curr_overlay.py:

* The Backtester constructs a fresh optimizer instance for every hyperparameter
  combination, so the compiled problems and the step-1 solutions are cached at
  class level and keyed on the estimation window.
* Penalties enter as ``cp.Parameter``, so each problem canonicalizes once per
  window rather than once per grid point (DPP).  The mean vectors and gamma are
  compiled in as constants instead, which is why gamma joins the cache key; the
  mean vectors are functions of the window alone, so the frame signatures already
  cover them.
* Step 1 does not depend on the currency penalties, so it is solved once per
  (lambda_l1_asset, lambda_l2_asset) pair -- 16 solves per window rather than 256.

Author: Anej Rozman
"""

from __future__ import annotations

import hashlib
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


class RegularizedMeanAbsoluteDeviationCurrencyOverlayWithMeanOptimizer(BaseOptimizer):
    """
    Two-step regularized mean-MAD optimizer with sequential currency overlay.

    Step 1 minimizes the penalized MAD of R_fh minus the pooled asset mean, subject
    to full investment and a leverage cap.  Step 2 fixes the asset weights and
    minimizes the penalized MAD of the combined portfolio return minus the carry
    mean of the overlay, subject to an L1 cap on psi.
    """

    # Shared across the per-grid-point instances the Backtester creates. Reset by
    # fit() whenever anything compiled into the problems changes.
    _cache_key: str | None = None
    _asset_prob: dict | None = None
    _overlay_prob: dict | None = None
    _asset_solutions: dict[tuple[float, float], np.ndarray] = {}

    def __init__(self, hyperparams: dict | None = None) -> None:
        """
        Args:
            hyperparams: Optional dict with keys:

                * ``gamma``           (float) — risk aversion; the mean term enters
                                                with coefficient 1 / gamma, default 0.5
                * ``lambda_l1_asset`` (float) — L1 penalty for asset weights, default 2.5e-4
                * ``lambda_l2_asset`` (float) — L2 penalty for asset weights, default 0.01
                * ``lambda_l1_curr``  (float) — L1 penalty for the overlay, default 0.0
                * ``lambda_l2_curr``  (float) — L2 penalty for the overlay, default 0.0
                * ``leverage_limit``  (float) — ||x||_1 cap for step 1, default 2.0
                * ``overlay_limit``   (float) — ||psi||_1 cap for step 2, default 2.0
                * ``solver``                  — cvxpy solver, default ``cp.GUROBI``
                                                (``cp.CLARABEL`` when gurobipy is unavailable)
        """
        super().__init__(hyperparams)
        self.gamma: float = self.hyperparams.get("gamma", 0.5)
        self.lambda_l1_asset: float = self.hyperparams.get("lambda_l1_asset", 2.5e-4)
        self.lambda_l2_asset: float = self.hyperparams.get("lambda_l2_asset", 0.01)
        self.lambda_l1_curr: float = self.hyperparams.get("lambda_l1_curr", 0.0)
        self.lambda_l2_curr: float = self.hyperparams.get("lambda_l2_curr", 0.0)
        self.leverage_limit: float = self.hyperparams.get("leverage_limit", 2.0)
        self.overlay_limit: float = self.hyperparams.get("overlay_limit", 2.0)
        self.solver = self.hyperparams.get("solver", _DEFAULT_SOLVER)

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
        if not np.isfinite(self.overlay_limit) or self.overlay_limit <= 0.0:
            raise ValueError(
                "overlay_limit must be finite and positive: MAD is degree-one, so "
                "step 2 is unbounded below without it for gamma under ~0.1."
            )

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

        ``gamma``, ``leverage_limit`` and ``overlay_limit`` enter the key because
        they are compiled in as constants rather than passed as parameters.  The two
        mean vectors are compiled in as well, but they are functions of the window
        alone, so the frame signatures already cover them.
        """
        parts = (
            self._frame_signature(R_fh),
            self._frame_signature(R_c),
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

    @classmethod
    def clear_cache(cls) -> None:
        """Drop all compiled problems and cached step-1 solutions."""
        cls._cache_key = None
        cls._asset_prob = None
        cls._overlay_prob = None
        cls._asset_solutions = {}

    def fit(self, train_bundle: dict[str, object]) -> None:
        """
        Extract the return matrices, estimate both mean vectors, and refresh the
        problem cache if anything baked into it changed.

        Args:
            train_bundle: Must contain ``"R_fh"``, ``"R_c"``, ``"C"`` and ``"F"``.
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

        self._refresh_cache(R_fh, R_c)

        self.Y_fh = np.ascontiguousarray(R_fh.values, dtype=float)
        self.Y_c = np.ascontiguousarray(R_c.values, dtype=float)
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

    # Step 1: assets

    def _build_asset_problem(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> dict:
        """Compile the step-1 problem once, with the penalties left as parameters.

        The demeaned return matrix, mu_a and gamma are baked in as constants, so the
        compiled problem is valid only for the window it was built from."""
        T, N = Y_fh.shape
        A = Y_fh - Y_fh.mean(axis=0)

        x = cp.Variable(N)
        u = cp.Variable(T)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        l1_norm = cp.norm(x, 1)
        problem = cp.Problem(
            cp.Minimize(
                (2.0 / T) * cp.sum(u)
                - (1.0 / self.gamma) * (mu_a @ x)
                + lam1 * l1_norm
                + lam2 * cp.sum_squares(x)
            ),
            [
                u >= 0,
                -A @ x <= u,
                cp.sum(x) == 1,
                l1_norm <= self.leverage_limit,
            ],
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-1 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "x": x, "lam1": lam1, "lam2": lam2}

    def _optimize_assets(self, Y_fh: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
        """
        Step 1: penalized mean-MAD on R_fh with a leverage constraint and full
        investment.  Bounded by its constraints regardless of gamma.

        The solution depends only on the asset penalties within a window, so it is
        cached on (lambda_l1_asset, lambda_l2_asset); gamma is already part of the
        window cache key.
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
        Compile the step-2 problem once, with the penalties and the demeaned fixed
        asset contribution left as parameters.

        The demeaned currency matrix, mu_c and gamma are baked in as constants, so
        the compiled problem is valid only for the window it was built from.
        ``_refresh_cache`` drops it whenever the window changes, and :meth:`score`
        never re-solves on validation data, so no other window ever reaches it.
        """
        T, K = Y_c.shape
        A_c = Y_c - Y_c.mean(axis=0)

        psi = cp.Variable(K)
        u = cp.Variable(T)
        c_dm = cp.Parameter(T)   # demeaned fixed asset contribution
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)

        constraints = [
            u >= 0,
            -(c_dm + A_c @ psi) <= u,
            cp.norm(psi, 1) <= self.overlay_limit,
        ]
        base = np.where(Y_c.var(axis=0) <= 0.0)[0]
        if base.size:
            constraints.append(psi[base] == 0)

        problem = cp.Problem(
            cp.Minimize(
                (2.0 / T) * cp.sum(u)
                - (1.0 / self.gamma) * (mu_c @ psi)
                + lam1 * cp.norm(psi, 1)
                + lam2 * cp.sum_squares(psi)
            ),
            constraints,
        )
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("Step-2 problem is not DPP-compliant; parameters would recompile.")
        return {"prob": problem, "psi": psi, "c_dm": c_dm, "lam1": lam1, "lam2": lam2}

    def _optimize_overlay(
        self, Y_fh: np.ndarray, Y_c: np.ndarray, x_fixed: np.ndarray, mu_c: np.ndarray
    ) -> np.ndarray:
        """Step 2: penalized mean-MAD currency overlay, capped in L1.

        The constant mu_a' x* is dropped: it cannot move the argmin."""
        cls = type(self)
        if cls._overlay_prob is None:
            cls._overlay_prob = self._build_overlay_problem(Y_c, mu_c)

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
            Dict with ``"x"`` (asset weights, shape N) and ``"psi"`` (currency
            overlay, shape K).

        Raises:
            RuntimeError: If :meth:`fit` was not called, or any solver step fails.
        """
        if self.Y_fh is None or self.Y_c is None:
            raise RuntimeError("fit() must be called before optimize().")

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
        weights: dict[str, np.ndarray],
        val_bundle: dict[str, object],
    ) -> float:
        """
        Evaluate train-fitted candidate weights on the validation window.

        Both the asset weights and the currency overlay are taken as given from
        :meth:`optimize` on the training window and held fixed; nothing is re-solved
        on the validation sample, matching reg_mad_curr_overlay.py and the rest of
        the regularized family.  Refitting on validation and then measuring the
        in-sample objective of that fit drives selection to the weakest penalty in
        the grid.  The score is

            MAD(combined return) - mean(combined return),

        the realized risk-return tradeoff of this window at a reference risk
        aversion of one.

        MAD needs no transformation to reach a degree-one score, unlike the MV and
        SMV pairs, which score on the standard deviation and the semi-deviation
        rather than on the variance and semi-variance their optimizers minimize.
        MAD is already degree one: about 3.1e-3 daily against a daily portfolio mean
        near 2.7e-4, an eleven-to-one ratio matching the transformed MV and SMV
        scores, and exactly what ``in_sample_analysis.evaluate_risk`` returns for
        ``"mad"`` -- the functional in_sample_vs_out_of_sample_w_mean.py scores these
        weights with, so the numbers are comparable across the whole mean-bearing
        family.

        The reference coefficient of 1.0 is unbiased here in a way it is not in the
        unregularized pair.  gamma is fixed across the whole grid, so no candidate
        can earn a larger mean credit than another simply by carrying a smaller
        gamma; the search is selecting penalties alone.  The realized mean is used
        rather than mu_a' x + mu_c' psi because a score should measure what the
        weights actually earned on the window, not re-apply a shrunk or conditional
        forecast to them.

        Args:
            weights: Dict with ``"x"`` and ``"psi"`` as returned by :meth:`optimize`.
            val_bundle: Must contain ``"R_fh"`` and ``"R_c"``.

        Returns:
            Validation MAD minus realized mean return, lower is better.
        """
        if not isinstance(weights, dict) or "x" not in weights or "psi" not in weights:
            raise ValueError("weights must be a dict with keys 'x' and 'psi'.")
        if "R_fh" not in val_bundle or "R_c" not in val_bundle:
            raise ValueError("val_bundle must contain 'R_fh' and 'R_c'.")

        R_fh_val = val_bundle["R_fh"]
        R_c_val = val_bundle["R_c"]
        if R_fh_val.empty or R_c_val.empty or len(R_fh_val) < 2:
            return float("inf")

        x = np.asarray(weights["x"], dtype=float).flatten()
        psi = np.asarray(weights["psi"], dtype=float).flatten()
        Y_fh_val = np.asarray(R_fh_val.values, dtype=float)
        Y_c_val = np.asarray(R_c_val.values, dtype=float)
        if x.shape[0] != Y_fh_val.shape[1] or psi.shape[0] != Y_c_val.shape[1]:
            raise ValueError(
                f"Weight/validation dimension mismatch: x {x.shape[0]} vs "
                f"{Y_fh_val.shape[1]} assets, psi {psi.shape[0]} vs "
                f"{Y_c_val.shape[1]} currencies."
            )

        port_rets = Y_fh_val @ x + Y_c_val @ psi
        mad = float(np.mean(np.abs(port_rets - port_rets.mean())))
        return mad - float(port_rets.mean())
