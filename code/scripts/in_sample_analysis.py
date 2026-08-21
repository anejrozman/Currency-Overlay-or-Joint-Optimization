"""
In-sample against realized risk for the unregularized joint and overlay programs.

At every quarterly rebalancing date the unregularized joint and the unregularized
overlay program are fit on exactly the ten-year estimation window the production
backtest uses, and the resulting weight pair (x, psi) is then scored twice by one
shared risk function: once on that same estimation window, giving the in-sample
risk, and once on the following quarter's holding window, giving the realized
risk.  Two questions follow from those two numbers:

  * noise  — whether estimation error is large for each program on its own, and
    whether it grows with the number of assets.  This is what the figure shows:
    one panel per risk measure, each carrying the mean overfitting ratio of the
    two programs, that program's realized / in-sample risk divided by 1/N's on
    the same two windows, with circular block bootstrap bands and a reference at
    1.0.  The two lines are the same diagnostic run twice, not a comparison
    between the programs — each is read against 1.0 on its own.
  * ceiling — how much lower the joint program's risk is than the overlay's, and
    whether its in-sample advantage survives out of sample.  The per-measure
    summary CSV still carries it as median (overlay - joint) / overlay on both
    windows; the figure devoted to it now lives in in_sample_vs_out_of_sample.py,
    which measures the same gap with intervals and a regression on the currency
    share.

Design choices, each of which the measurement depends on:

  * One shared scorer.  Both evaluations go through ``evaluate_risk`` rather than
    through the optimizers' own ``score()`` methods.  Those methods do not
    evaluate a common object: the joint ``score`` re-solves the whole program on
    the scoring window and the overlay ``score`` re-solves its second step, so a
    gap measured through them would partly be an artifact of the two classes
    scoring different things.
  * Unregularized programs only.  Regularization restricts the feasible set and
    is itself the thing this diagnostic measures, so the ``reg_*`` twins are out
    of scope.  Their unregularized counterparts declare no penalties, so the
    production parameter grid collapses to a singleton and the backtester's
    validation search is a no-op — refitting directly on the full estimation
    window reproduces its refit step exactly.
  * No transaction costs.  Costs land on the realized side alone and would bias
    the realized / in-sample ratio upward by an amount unrelated to estimation
    noise.  Weights are likewise held fixed over the holding quarter (no drift),
    so both windows score the same weight vector.
  * The production leverage cap only, read from ``backtest_config.py``.  The l1
    cap on x is an implicit penalty that tightens as the number of weights grows,
    so a non-binding second cap was swept alongside it while the diagnostic was
    being built, to check that the cap was not manufacturing the universe-size
    trend by itself.  On top-10 it moved nothing (gross exposure sits under the
    cap, and the two legs agreed to four decimals), so the sweep now solves the
    production cap alone.
  * A 1/N control at every date, which is what the top panel divides by.  The
    raw realized / in-sample ratio is the product of two effects: a ten-year
    window spanning the dot-com crash, the GFC and COVID is riskier than a median
    single quarter, which drags the ratio down whatever the weights are, and
    estimation error, which pushes it up.  Scoring 1/N through the same windows
    and the same scorer isolates the first — it optimizes nothing, so nothing
    about it can be overfit — and it comes out near-identical across measures on
    top-10 (0.756 for ES, 0.758 for volatility), confirming it is a property of
    the windows rather than of the risk measure.  Dividing it out per date leaves
    the estimation-error part alone, which is the only version comparable across
    risk measures: the raw ES ratio lands above 1.0 and the raw MV ratio below it
    on the same universe purely because ES overfits harder (overfitting ratio
    1.318 against 1.234 on top-10), not because the two disagree about the data.
  * MV is reported as a standard deviation, not a variance.  The MV programs
    minimize variance and a monotone transform leaves their solution untouched,
    but variance is quadratic in scale, so variance ratios would not be
    comparable with the ES and MAD ratios plotted beside them.

One row is written per date, strategy, universe and leverage cap; nothing is
aggregated into the row-level CSV.  Aggregation happens only in the plotting
step, which reads those CSVs back and writes both the per-measure medians and
overfitting_means.csv, the means and intervals behind the figure.  The figure
spans all three risk measures, so it is drawn from every CSV present in the
output directory rather than from the measure this invocation swept.

Usage:
    python code/scripts/in_sample_analysis.py es
    python code/scripts/in_sample_analysis.py mv --universes 10 50
    python code/scripts/in_sample_analysis.py mad --plot-only

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio_backtest import config
from portfolio_backtest.backtest.backtester import Backtester
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.bootstrap import block_bootstrap_mean
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (
    compute_daily_forward_premium,
    compute_excess_currency_returns,
    compute_fully_hedged_returns,
)

logger = logging.getLogger("in_sample_analysis")

MEASURES = ("mv", "es", "mad")
ROLES = ("joint", "overlay")
UNIVERSE_SIZES = (10, 50, 100)

# 1/N control. Both optimized ratios sit on a common denominator effect that has
# nothing to do with either program: a ten-year estimation window spanning the
# dot-com crash, the GFC and COVID is riskier than a median single quarter, which
# drags every realized / in-sample ratio down regardless of how the weights were
# chosen. Scoring 1/N through the same windows and the same scorer measures that
# effect alone — it optimizes nothing, so nothing about it can be overfit — and
# the optimized ratio divided by it is the part attributable to estimation error.
# Without the control the ratio's level is not comparable across risk measures.
CONTROL_STRATEGY = "equal_weight"
CONTROL_STRATEGY_KEY = "equally_weighted"  # STRATEGY_REGISTRY entry
# 1/N holds ||x||_1 = 1, so no leverage cap can bind and the control is solved
# once per date rather than once per cap.
CONTROL_LEVERAGE_LABEL = "none"

OUT_DIR = config.RESULTS_DIR / "in_sample_analysis"

MEASURE_LABELS = {
    "es": "Expected Shortfall",
    "mv": "Volatility",
    "mad": "Mean Absolute Deviation",
}
ROLE_LABELS = {"joint": "Joint", "overlay": "Overlay", CONTROL_STRATEGY: "1/N control"}
ROLE_COLORS = {"joint": "C0", "overlay": "C3", CONTROL_STRATEGY: "dimgrey"}

ROW_COLUMNS = [
    "measure",
    "universe",
    "leverage_label",
    "leverage_limit",
    "date",
    "strategy",
    "alpha",
    "n_assets",
    "n_obs_in_sample",
    "n_obs_realized",
    "in_sample_risk",
    "realized_risk",
]


# Setup


def _pick_solver():
    if "GUROBI" in cp.installed_solvers():
        return cp.GUROBI, "GUROBI"
    return cp.CLARABEL, "CLARABEL"


def _resolve_strategies(measure: str) -> dict[str, str]:
    """Map each role onto the single *unregularized* strategy of that measure."""
    resolved: dict[str, str] = {}
    for role in ROLES:
        matches = [
            name
            for name in STRATEGY_REGISTRY
            if "reg" not in name.split("_") and {measure, role}.issubset(set(name.split("_")))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one unregularized {measure} {role} strategy in "
                f"STRATEGY_REGISTRY, found {matches}. The diagnostic compares the "
                f"overlay against the joint program; running on one alone measures "
                f"nothing."
            )
        resolved[role] = matches[0]
    return resolved


def _fixed_hyperparams(strategy: str) -> dict[str, float]:
    """
    Non-penalty settings of an unregularized strategy, read off its production grid.

    The unregularized programs declare no ``lambda*`` keys, so every entry must be
    a singleton; anything else would mean the production backtest selects it by
    validation and this script has no basis for picking one value.
    """
    param_grid = STRATEGY_REGISTRY[strategy]["config"].param_grid
    fixed: dict[str, float] = {}
    for key, values in param_grid.items():
        if key.startswith("lambda"):
            raise ValueError(
                f"{strategy} declares the penalty '{key}'. This diagnostic runs the "
                f"unregularized programs only — regularization is what it measures."
            )
        if len(values) != 1:
            raise ValueError(
                f"Hyperparameter '{key}' of {strategy} has {len(values)} candidates "
                f"{values}; every non-penalty setting must be fixed."
            )
        fixed[key] = values[0]
    return fixed


# Data


def _build_universe_mask(universe_file: Path | str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Daily universe mask (1.0 where a stock is a constituent, NaN otherwise),
    forward-filled from the annual reconstitution dates.

    Built with a single-key groupby rather than ``DataHandler``'s ``pivot_table``:
    that reshape silently duplicates index labels on the top-100 universe file,
    which makes the subsequent ``reindex`` fail.  Same construction as
    ``asset_currency_dependance.build_universe_mask``.
    """
    universe_df = pd.read_csv(universe_file)
    universe_df["clean_ric"] = (
        universe_df["ric_code"]
        .astype(str)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )
    universe_df["date"] = pd.to_datetime(universe_df["date"])

    constituents = universe_df.groupby("date")["clean_ric"].agg(set)
    all_rics = sorted(universe_df["clean_ric"].unique())
    mask = pd.DataFrame(np.nan, index=constituents.index, columns=all_rics)
    for recon_date, rics in constituents.items():
        mask.loc[recon_date, list(rics)] = 1.0

    return mask.reindex(index, method="ffill")


def _load_universe_data(universe: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fully hedged asset returns, excess currency returns and the daily universe
    mask, recomputed from the processed data exactly as ``run_strategy.py`` does
    for a hedged strategy.
    """
    universe_path = config.RAW_DIR / config.UNIVERSE_FILES[universe]

    R_local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    R_local = R_local.where(R_local.abs() < config.RETURN_FILTER_BACKTEST)
    e = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    f_raw = load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE)
    f_daily = compute_daily_forward_premium(f_raw, config.TENOR_DAYS_3M)

    asset_to_currency = build_asset_to_currency(universe_path)
    R_c = compute_excess_currency_returns(e, f_daily)
    R_fh = compute_fully_hedged_returns(R_local, e, f_daily, asset_to_currency)

    # DataHandler restricts the return frame to the universe's own assets.
    R_fh = R_fh[[col for col in R_fh.columns if col in asset_to_currency]]

    common = R_fh.index.intersection(R_c.index)
    R_fh = R_fh.loc[common]
    R_c = R_c.loc[common]
    universe_mask = _build_universe_mask(universe_path, common)
    return R_fh, R_c, universe_mask


def _rebalancing_windows(
    R_fh: pd.DataFrame,
    universe_mask: pd.DataFrame,
    backtest_cfg,
) -> list[dict]:
    """
    Every rebalancing date the production backtester would actually trade on,
    with its estimation and holding boundaries and its investable universe.

    Uses ``Backtester``'s own helpers and reproduces the guards in
    ``Backtester.run`` (full lookback available, splits computable, non-empty
    mask, complete return history) so the windows are identical to production.
    """
    probe = Backtester(
        optimizer=STRATEGY_REGISTRY["equally_weighted"]["optimizer"],
        window_years=backtest_cfg.window_years,
        val_years=backtest_cfg.val_years,
        rebalancing_frequency=backtest_cfg.rebalancing_frequency,
    )
    dates = R_fh.index
    rebalance_dates = probe._get_rebalance_dates(dates)
    data_start = dates[0]

    windows: list[dict] = []
    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]
        if pd.Timestamp(t) - pd.DateOffset(years=backtest_cfg.window_years) < data_start:
            continue
        try:
            splits = probe._get_estimation_and_validation_splits(dates, t, rebalance_dates[i + 1])
        except ValueError:
            continue

        mask_row = universe_mask.loc[t] if t in universe_mask.index else pd.Series(dtype=float)
        current_universe = [a for a in mask_row[mask_row.notna()].index if a in R_fh.columns]
        if not current_universe:
            continue

        est_slice = R_fh.loc[
            splits["full_estimation_start"] : splits["full_estimation_end"], current_universe
        ]
        try:
            assets = probe._filter_universe(est_slice)
        except ValueError:
            continue

        windows.append(
            {
                "date": t,
                "estimation_start": splits["full_estimation_start"],
                "estimation_end": splits["full_estimation_end"],
                "holding_start": splits["holding_start"],
                "holding_end": splits["holding_end"],
                "assets": assets,
            }
        )
    return windows


# The shared scorer


def evaluate_risk(
    measure: str,
    R_fh_window: pd.DataFrame,
    R_c_window: pd.DataFrame,
    x: np.ndarray,
    psi: np.ndarray,
    alpha: float | None = None,
) -> float:
    """
    Risk of the combined portfolio ``R_fh @ x + R_c @ psi`` on one window.

    The single evaluation path for both the in-sample and the realized side, and
    for the joint and the overlay strategy alike: same portfolio return series,
    same estimator, same window semantics.  Nothing here re-solves anything — the
    weights come in fixed.

    Args:
        measure: ``"es"``, ``"mv"`` or ``"mad"``.
        R_fh_window: Fully hedged asset returns over the window, columns aligned to *x*.
        R_c_window: Excess currency returns over the window, columns aligned to *psi*.
        x: Asset weights.
        psi: Currency overlay weights.
        alpha: ES confidence level; required for ``"es"``, ignored otherwise.

    Returns:
        ES at *alpha* (mean loss over the worst ``1-alpha`` tail), the standard
        deviation, or the mean absolute deviation about the window's own mean.
        Lower is better in every case.  NaN on an empty window.
    """
    if R_fh_window.empty or R_c_window.empty:
        return float("nan")
    if len(R_fh_window) != len(R_c_window):
        raise ValueError(
            f"Asset and currency windows differ in length ({len(R_fh_window)} vs "
            f"{len(R_c_window)}); the combined return would misalign them."
        )

    r = R_fh_window.to_numpy() @ x + R_c_window.to_numpy() @ psi
    if r.size == 0:
        return float("nan")

    if measure == "es":
        if alpha is None:
            raise ValueError("alpha is required to evaluate ES.")
        threshold = np.quantile(r, 1.0 - alpha)
        tail = r[r <= threshold]
        return float(-np.mean(tail)) if tail.size else float(-np.mean(r))
    if measure == "mv":
        # Standard deviation, not variance: same minimizer, but linear in scale
        # so the ratios stay comparable with the ES and MAD ones.
        return float(np.std(r, ddof=1)) if r.size > 1 else float("nan")
    if measure == "mad":
        return float(np.mean(np.abs(r - r.mean())))
    raise ValueError(f"Unknown measure '{measure}'.")


# Sweep


def _solve_weights(
    optimizer_cls,
    hyperparams: dict,
    R_fh_window: pd.DataFrame,
    R_c_window: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    optimizer = optimizer_cls(hyperparams=hyperparams)
    optimizer.fit({"R_fh": R_fh_window, "R_c": R_c_window})
    weights = optimizer.optimize()
    if isinstance(weights, dict):
        x = np.asarray(weights["x"], dtype=float).flatten()
        psi = np.asarray(weights["psi"], dtype=float).flatten()
    else:
        # Asset-only optimizers (the 1/N control) carry no overlay, exactly as
        # Backtester.run treats a bare weight array.
        x = np.asarray(weights, dtype=float).flatten()
        psi = np.zeros(R_c_window.shape[1], dtype=float)
    return x, psi


def run_universe(
    measure: str,
    universe: int,
    strategies: dict[str, str],
    leverages: dict[str, float],
    solver,
) -> pd.DataFrame:
    """Solve and score every (date, strategy, leverage) cell of one universe."""
    backtest_cfgs = {role: STRATEGY_REGISTRY[name]["config"] for role, name in strategies.items()}
    windows_cfg = {
        (cfg.window_years, cfg.val_years, cfg.rebalancing_frequency)
        for cfg in backtest_cfgs.values()
    }
    if len(windows_cfg) != 1:
        raise ValueError(
            f"The {measure} overlay and joint configs disagree on the estimation window "
            f"{windows_cfg}. Both must use the same window for the gap to mean anything."
        )

    logger.info("Loading top-%d universe data.", universe)
    R_fh, R_c, universe_mask = _load_universe_data(universe)
    windows = _rebalancing_windows(R_fh, universe_mask, backtest_cfgs["joint"])
    logger.info(
        "Top-%d: %d rebalancing dates, %s to %s.",
        universe,
        len(windows),
        windows[0]["date"].date() if windows else "-",
        windows[-1]["date"].date() if windows else "-",
    )

    hyperparams = {role: _fixed_hyperparams(name) for role, name in strategies.items()}
    optimizers = {role: STRATEGY_REGISTRY[name]["optimizer"] for role, name in strategies.items()}

    # The control reuses the production 1/N optimizer. It ignores hyperparams
    # entirely; alpha is carried only so the ES scorer runs at the same
    # confidence level as the programs it is a baseline for.
    optimizers[CONTROL_STRATEGY] = STRATEGY_REGISTRY[CONTROL_STRATEGY_KEY]["optimizer"]
    hyperparams[CONTROL_STRATEGY] = {
        key: value for key, value in hyperparams["joint"].items() if key == "alpha"
    }

    rows: list[dict] = []
    for i, window in enumerate(windows, start=1):
        assets = window["assets"]
        R_fh_est = R_fh.loc[window["estimation_start"] : window["estimation_end"], assets]
        R_c_est = R_c.loc[window["estimation_start"] : window["estimation_end"]]
        # Same treatment of holding-period gaps as Backtester.run: a name that
        # stops trading contributes zero, it does not drop the whole window.
        R_fh_hold = R_fh.loc[window["holding_start"] : window["holding_end"], assets].fillna(0.0)
        R_c_hold = R_c.loc[window["holding_start"] : window["holding_end"]].fillna(0.0)

        logger.info(
            "Top-%d [%d/%d] %s: %d assets, %d estimation obs, %d holding obs.",
            universe, i, len(windows), window["date"].date(),
            len(assets), len(R_fh_est), len(R_fh_hold),
        )

        # One cell per (leverage cap, strategy), plus the 1/N control once: it
        # carries no cap-sensitive weights of its own.
        cells = [
            (leverage_label, leverage_limit, role)
            for leverage_label, leverage_limit in leverages.items()
            for role in ROLES
        ]
        cells.append((CONTROL_LEVERAGE_LABEL, float("nan"), CONTROL_STRATEGY))

        for leverage_label, leverage_limit, role in cells:
            params = {**hyperparams[role], "solver": solver}
            if role != CONTROL_STRATEGY:
                params["leverage_limit"] = leverage_limit
            alpha = params.get("alpha")
            try:
                x, psi = _solve_weights(optimizers[role], params, R_fh_est, R_c_est)
                in_sample = evaluate_risk(measure, R_fh_est, R_c_est, x, psi, alpha)
                realized = evaluate_risk(measure, R_fh_hold, R_c_hold, x, psi, alpha)
            except (RuntimeError, ValueError) as exc:
                # One failed cell must not take the sweep down; the date is
                # recorded with NaN risks and drops out at aggregation.
                logger.warning(
                    "Top-%d %s %s %s (leverage %s) failed: %s",
                    universe, window["date"].date(), role, measure, leverage_label, exc,
                )
                in_sample = float("nan")
                realized = float("nan")

            rows.append(
                {
                    "measure": measure,
                    "universe": universe,
                    "leverage_label": leverage_label,
                    "leverage_limit": leverage_limit,
                    "date": window["date"],
                    "strategy": role,
                    "alpha": alpha if alpha is not None else float("nan"),
                    "n_assets": len(assets),
                    "n_obs_in_sample": len(R_fh_est),
                    "n_obs_realized": len(R_fh_hold),
                    "in_sample_risk": in_sample,
                    "realized_risk": realized,
                }
            )

    return pd.DataFrame(rows, columns=ROW_COLUMNS)


# Aggregation and plotting


def _rows_path(out_dir: Path, measure: str, universe: int) -> Path:
    return out_dir / f"in_sample_{measure}_top{universe}.csv"


def _load_rows(out_dir: Path, measure: str) -> pd.DataFrame:
    """Concatenate every per-universe CSV of this measure written so far."""
    paths = sorted(out_dir.glob(f"in_sample_{measure}_top*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No row-level CSVs for '{measure}' in {out_dir}. Run the sweep before plotting."
        )
    frames = [pd.read_csv(path, parse_dates=["date"]) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _load_all_rows(out_dir: Path) -> pd.DataFrame:
    """Row-level CSVs for every measure that has been swept so far."""
    frames = []
    for measure in MEASURES:
        paths = sorted(out_dir.glob(f"in_sample_{measure}_top*.csv"))
        frames += [pd.read_csv(path, parse_dates=["date"]) for path in paths]
    if not frames:
        raise FileNotFoundError(f"No row-level CSVs in {out_dir}. Run the sweep before plotting.")
    return pd.concat(frames, ignore_index=True)


def _overfit_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Mean overfitting ratio per measure, universe and program, with a circular
    block bootstrap interval.

    The ratio is formed date by date — the optimizer's realized / in-sample risk
    divided by 1/N's on the same two windows — before any averaging, so the
    window effect the control measures is removed per date rather than on
    average.  The mean is right-skewed at the larger universes, where a few
    quarters carry ratios above seven, which is what the interval exposes.
    """
    clean = rows.dropna(subset=["in_sample_risk", "realized_risk"])
    clean = clean[clean["in_sample_risk"] > 0]
    ratios = clean.assign(ratio=clean["realized_risk"] / clean["in_sample_risk"])

    control = ratios[ratios["strategy"] == CONTROL_STRATEGY][
        ["measure", "universe", "date", "ratio"]
    ].rename(columns={"ratio": "control_ratio"})
    optimized = ratios[ratios["strategy"] != CONTROL_STRATEGY].merge(
        control, on=["measure", "universe", "date"], how="inner"
    )
    optimized = optimized[optimized["control_ratio"] > 0]
    optimized["overfit"] = optimized["ratio"] / optimized["control_ratio"]

    out = []
    keys = ["measure", "universe", "leverage_label", "strategy"]
    for (measure, universe, leverage_label, strategy), part in optimized.groupby(keys):
        mean, lo, hi, se, block = block_bootstrap_mean(
            part.sort_values("date")["overfit"].to_numpy()
        )
        out.append(
            {
                "measure": measure,
                "universe": universe,
                "leverage_label": leverage_label,
                "strategy": strategy,
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "se": se,
                "block": block,
                "n_dates": len(part),
            }
        )
    return pd.DataFrame(out).sort_values(keys).reset_index(drop=True)


def _aggregate(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three aggregates, all formed date by date before any median is taken.

    * ``ratio_median``   — raw realized / in-sample ratio per strategy.
    * ``overfit_median`` — that ratio divided by the 1/N control's ratio on the
      same two windows: the top panel's overfitting ratio.  It is the same
      diagnostic run once per optimizer, not a comparison between them, and 1.0
      means the optimizer's risk estimate degrades no worse than 1/N's does.
    * ``gap_median``     — the one comparison between the two optimizers,
      (overlay - joint) / overlay, formed in-sample and realized.
    """
    clean = rows.dropna(subset=["in_sample_risk", "realized_risk"])
    clean = clean[clean["in_sample_risk"] > 0]

    ratios = clean.assign(ratio=clean["realized_risk"] / clean["in_sample_risk"])
    ratio_median = (
        ratios.groupby(["universe", "leverage_label", "strategy"])["ratio"]
        .agg(median="median", n_dates="size")
        .reset_index()
    )

    # Normalize per date, not median against median: the control has to be the
    # one measured on the very same estimation window and holding quarter for
    # the division to remove the window effect rather than an average of it.
    # The control is cap-independent, so it joins on (universe, date) alone.
    control_ratio = ratios[ratios["strategy"] == CONTROL_STRATEGY][
        ["universe", "date", "ratio"]
    ].rename(columns={"ratio": "control_ratio"})
    optimized = ratios[ratios["strategy"] != CONTROL_STRATEGY].merge(
        control_ratio, on=["universe", "date"], how="inner"
    )
    optimized = optimized[optimized["control_ratio"] > 0]
    optimized["overfit"] = optimized["ratio"] / optimized["control_ratio"]
    overfit_median = (
        optimized.groupby(["universe", "leverage_label", "strategy"])["overfit"]
        .agg(median="median", n_dates="size")
        .reset_index()
    )

    # Pair the two strategies date by date with a join rather than a reshape:
    # the gap is only meaningful within a date, and pivot/unstack is unreliable
    # on this stack for larger frames.
    keys = ["universe", "leverage_label", "date"]
    sides = ["in_sample_risk", "realized_risk"]
    joint = clean[clean["strategy"] == "joint"].set_index(keys)[sides]
    overlay = clean[clean["strategy"] == "overlay"].set_index(keys)[sides]
    paired = joint.join(overlay, lsuffix="_joint", rsuffix="_overlay", how="inner")
    # The overlay risk is the denominator of both gaps, so a non-positive one
    # would flip the sign rather than merely inflate it. In-sample values are
    # already positive by the filter above; the realized ones are not.
    paired = paired[paired["realized_risk_overlay"] > 0]
    paired["gap_in_sample"] = (
        paired["in_sample_risk_overlay"] - paired["in_sample_risk_joint"]
    ) / paired["in_sample_risk_overlay"]
    paired["gap_realized"] = (
        paired["realized_risk_overlay"] - paired["realized_risk_joint"]
    ) / paired["realized_risk_overlay"]

    gap_median = (
        paired.groupby(["universe", "leverage_label"])[["gap_in_sample", "gap_realized"]]
        .agg(["median", "size"])
    )
    gap_median.columns = ["gap_in_sample", "n_dates_in_sample", "gap_realized", "n_dates_realized"]
    gap_median = gap_median.reset_index()
    return ratio_median, overfit_median, gap_median


def _summary_table(
    ratio_median: pd.DataFrame,
    overfit_median: pd.DataFrame,
    gap_median: pd.DataFrame,
) -> pd.DataFrame:
    """The medians behind the figure, one row per universe and leverage cap."""
    wide = gap_median[
        ["universe", "leverage_label", "gap_in_sample", "gap_realized", "n_dates_in_sample"]
    ].rename(columns={"n_dates_in_sample": "n_dates"})

    # Plotted quantity first, then the raw ratios it is built from.
    for source, prefix in ((overfit_median, "overfit"), (ratio_median, "median_ratio")):
        for role in ROLES:
            part = source[source["strategy"] == role]
            wide = wide.merge(
                part[["universe", "leverage_label", "median"]].rename(
                    columns={"median": f"{prefix}_{role}"}
                ),
                on=["universe", "leverage_label"],
                how="left",
            )

    # The control is solved once per universe, so it joins on universe alone.
    control = ratio_median[ratio_median["strategy"] == CONTROL_STRATEGY]
    wide = wide.merge(
        control[["universe", "median"]].rename(columns={"median": "median_ratio_control"}),
        on="universe",
        how="left",
    )

    ordered = [
        "universe",
        "leverage_label",
        "overfit_joint",
        "overfit_overlay",
        "gap_in_sample",
        "gap_realized",
        "median_ratio_joint",
        "median_ratio_overlay",
        "median_ratio_control",
        "n_dates",
    ]
    return wide[ordered].sort_values(["leverage_label", "universe"]).reset_index(drop=True)


def _pad_to_reference(ax, reference: float) -> None:
    """
    Keep a reference line clearly inside the axes.

    Both panels carry one (realized = in-sample at 1.0, joint = overlay at 0.0),
    and both are natural boundaries the data can sit entirely on one side of —
    which puts the line flat on a spine, where it reads as the frame rather than
    as the level the reader is meant to compare against.
    """
    lo, hi = ax.get_ylim()
    lo, hi = min(lo, reference), max(hi, reference)
    pad = 0.08 * (hi - lo) if hi > lo else 0.1 * max(abs(reference), 1.0)
    ax.set_ylim(lo - pad, hi + pad)


def _save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("pdf", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", path)
    plt.close(fig)


def plot_overfitting(
    overfit_stats: pd.DataFrame,
    leverages: dict[str, float],
    stem: Path,
    leverage_label: str = "production",
) -> None:
    """
    One panel per risk measure, sharing a universe-size axis: the mean
    overfitting ratio of each program, its realized / in-sample risk divided by
    1/N's on the same two windows, with circular block bootstrap bands.

    The two lines are the same diagnostic run once per program rather than a
    comparison between them; each is read against 1.0, where 1.0 means the
    program's risk estimate degrades no worse than 1/N's does.
    """
    overfit_stats = overfit_stats[overfit_stats["leverage_label"] == leverage_label]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)

    universes = sorted(set(UNIVERSE_SIZES) | set(overfit_stats["universe"]))
    positions = {u: i for i, u in enumerate(universes)}

    for ax, measure in zip(axes, MEASURES):
        part = overfit_stats[overfit_stats["measure"] == measure]
        for role in ROLES:
            cells = part[part["strategy"] == role].sort_values("universe")
            if cells.empty:
                continue
            xs = [positions[u] for u in cells["universe"]]
            ax.plot(
                xs, cells["mean"], marker="o", color=ROLE_COLORS[role], label=ROLE_LABELS[role],
                zorder=3,
            )
            ax.fill_between(
                xs, cells["ci_lo"], cells["ci_hi"], color=ROLE_COLORS[role], alpha=0.2, lw=0
            )
        ax.axhline(1.0, color="grey", linewidth=1.0, zorder=0)
        ax.set_title(MEASURE_LABELS[measure], fontsize=11)
        ax.set_xlabel("Top-N stocks per economy")
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels([str(u) for u in universes])
        ax.set_xlim(-0.35, len(universes) - 0.65)
        ax.grid(True, alpha=0.3)

    _pad_to_reference(axes[0], 1.0)
    axes[0].set_ylabel("Mean overfitting ratio")
    axes[0].legend(fontsize=8, loc="upper left")
    cap = leverages.get(leverage_label)
    logger.info("Drawing overfitting panels at leverage cap %s", cap)
    fig.suptitle(
        "Out of sample degradation analysis of joint and overlay optimization "
        "across risk measures and increasing universe sizes",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, stem)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("measure", choices=MEASURES, help="Risk measure family to sweep.")
    parser.add_argument(
        "--universes",
        type=int,
        nargs="+",
        default=list(UNIVERSE_SIZES),
        choices=list(UNIVERSE_SIZES),
        help="Universe sizes to sweep (default: all three). Split across cluster jobs "
             "if wanted; the plot is built from whatever CSVs are on disk.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for the row-level CSVs, the summary table and the figure.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the sweep and rebuild the figure from the CSVs already on disk.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run the sweep and write the CSVs without producing the figure.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    if args.plot_only and args.no_plot:
        parser.error("--plot-only and --no-plot together leave nothing to do.")
    return args


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    measure = args.measure
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = _resolve_strategies(measure)
    production_leverage = {
        role: _fixed_hyperparams(name)["leverage_limit"] for role, name in strategies.items()
    }
    if len(set(production_leverage.values())) != 1:
        raise ValueError(
            f"The {measure} joint and overlay programs run at different production "
            f"leverage caps {production_leverage}; the comparison would confound the "
            f"cap with the program."
        )
    leverages = {"production": float(next(iter(production_leverage.values())))}

    if not args.plot_only:
        solver, solver_name = _pick_solver()
        logger.info(
            "Measure %s | strategies %s | leverage caps %s | solver %s",
            measure, strategies, leverages, solver_name,
        )
        for universe in args.universes:
            rows = run_universe(measure, universe, strategies, leverages, solver)
            path = _rows_path(out_dir, measure, universe)
            rows.to_csv(path, index=False)
            logger.info("Wrote %d rows to %s", len(rows), path)

    if args.no_plot:
        return

    rows = _load_rows(out_dir, measure)
    ratio_median, overfit_median, gap_median = _aggregate(rows)
    summary = _summary_table(ratio_median, overfit_median, gap_median)
    summary_path = out_dir / f"in_sample_{measure}_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote plot medians to %s", summary_path)
    print(summary.to_string(index=False))

    # The figure spans all three measures, so it is built from every CSV on disk
    # rather than from this run's measure alone.
    overfit_stats = _overfit_stats(_load_all_rows(out_dir))
    stats_path = out_dir / "overfitting_means.csv"
    overfit_stats.to_csv(stats_path, index=False)
    logger.info("Wrote overfitting means to %s", stats_path)
    print(overfit_stats.to_string(index=False))
    plot_overfitting(overfit_stats, leverages, out_dir / "overfitting_by_measure")


if __name__ == "__main__":
    main()
