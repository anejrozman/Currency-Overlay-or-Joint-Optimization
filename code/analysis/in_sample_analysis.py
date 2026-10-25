"""In-sample and realized risk of the unregularized joint, overlay and 1/N programs at every rebalancing date.

Each program is fit on the production estimation window and its weights are scored by one shared scorer,
evaluate_risk, on that window and on the following holding quarter (no costs, no drift). Writes one row per
date and program to in_sample_<measure>_top<N>.csv, which in_sample_vs_out_of_sample.py reads as its
unregularized grid cells. Also holds the helpers the gap analyses share.

Usage:
    python code/analysis/in_sample_analysis.py es --universes 10 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import universe_mask
from portfolio_backtest import config
from portfolio_backtest.backtest.backtester import Backtester
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (compute_daily_forward_premium, compute_excess_currency_returns,
                                              compute_fully_hedged_returns)

logger = logging.getLogger("in_sample_analysis")

MEASURES = ("mv", "es", "mad", "smv", "soes")
ROLES = ("joint", "overlay")
UNIVERSE_SIZES = (10, 50, 100)
CONTROL_STRATEGY = "equal_weight"
CONTROL_STRATEGY_KEY = "equally_weighted"
OUT_DIR = config.RESULTS_DIR / "in_sample_analysis"
ROW_COLUMNS = ["measure", "universe", "leverage_label", "leverage_limit", "date", "strategy", "alpha", "n_assets",
               "n_obs_in_sample", "n_obs_realized", "in_sample_risk", "realized_risk"]


def pick_solver():
    return (cp.GUROBI, "GUROBI") if "GUROBI" in cp.installed_solvers() else (cp.CLARABEL, "CLARABEL")


def resolve_strategies(measure: str) -> dict[str, str]:
    resolved = {}
    for role in ROLES:
        matches = [n for n in STRATEGY_REGISTRY if not {"reg", "mean"} & set(n.split("_"))
                   and {measure, role} <= set(n.split("_"))]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one unregularized {measure} {role} strategy, found {matches}.")
        resolved[role] = matches[0]
    return resolved


def fixed_hyperparams(strategy: str) -> dict:
    """The single value of every hyperparameter of an unregularized strategy."""
    fixed = {}
    for key, values in STRATEGY_REGISTRY[strategy]["config"].param_grid.items():
        if key.startswith("lambda"):
            raise ValueError(f"{strategy} declares the penalty '{key}'; only unregularized programs are allowed.")
        if len(values) != 1:
            raise ValueError(f"Hyperparameter '{key}' of {strategy} has candidates {values}; it must be fixed.")
        fixed[key] = values[0]
    return fixed


def load_universe_data(universe: int):
    """Fully hedged returns, excess currency returns, the daily universe mask and the asset currencies."""
    path = config.RAW_DIR / config.UNIVERSE_FILES[universe]
    local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    local = local.where(local.abs() < config.RETURN_FILTER_BACKTEST)
    spot = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    premium = compute_daily_forward_premium(load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE),
                                            config.TENOR_DAYS_3M)
    atc = build_asset_to_currency(path)
    R_c = compute_excess_currency_returns(spot, premium)
    R_fh = compute_fully_hedged_returns(local, spot, premium, atc)
    R_fh = R_fh[[c for c in R_fh.columns if c in atc]]
    common = R_fh.index.intersection(R_c.index)
    return R_fh.loc[common], R_c.loc[common], universe_mask(path, common), atc


def rebalancing_windows(R_fh: pd.DataFrame, mask: pd.DataFrame, cfg) -> list[dict]:
    """Every date the production backtester trades on, with the guards of Backtester.run."""
    probe = Backtester(optimizer=STRATEGY_REGISTRY["equally_weighted"]["optimizer"], window_years=cfg.window_years,
                       val_years=cfg.val_years, rebalancing_frequency=cfg.rebalancing_frequency)
    dates = R_fh.index
    rebalance_dates = probe._get_rebalance_dates(dates)
    windows = []
    for t, t_next in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        if pd.Timestamp(t) - pd.DateOffset(years=cfg.window_years) < dates[0]:
            continue
        try:
            splits = probe._get_estimation_and_validation_splits(dates, t, t_next)
        except ValueError:
            continue
        row = mask.loc[t] if t in mask.index else pd.Series(dtype=float)
        universe = [a for a in row[row.notna()].index if a in R_fh.columns]
        if not universe:
            continue
        try:
            assets = probe._filter_universe(
                R_fh.loc[splits["full_estimation_start"]:splits["full_estimation_end"], universe])
        except ValueError:
            continue
        windows.append({"date": t, "estimation_start": splits["full_estimation_start"],
                        "estimation_end": splits["full_estimation_end"], "holding_start": splits["holding_start"],
                        "holding_end": splits["holding_end"], "assets": assets})
    return windows


def evaluate_risk(measure: str, R_fh: pd.DataFrame, R_c: pd.DataFrame, x: np.ndarray, psi: np.ndarray,
                  alpha: float | None = None) -> float:
    """Risk of the portfolio R_fh @ x + R_c @ psi on one window, lower is better, NaN on an empty window.

    MV and SMV are reported as (semi-)deviations so all measures are linear in scale; SOES is HMCR_2.
    """
    if R_fh.empty or R_c.empty:
        return float("nan")
    if len(R_fh) != len(R_c):
        raise ValueError(f"Asset and currency windows differ in length ({len(R_fh)} vs {len(R_c)}).")
    r = R_fh.to_numpy() @ x + R_c.to_numpy() @ psi
    if r.size == 0:
        return float("nan")
    if measure in ("es", "soes") and alpha is None:
        raise ValueError(f"alpha is required to evaluate {measure.upper()}.")
    if measure == "es":
        tail = r[r <= np.quantile(r, 1.0 - alpha)]
        return float(-np.mean(tail)) if tail.size else float(-np.mean(r))
    if measure == "mv":
        return float(np.std(r, ddof=1)) if r.size > 1 else float("nan")
    if measure == "mad":
        return float(np.mean(np.abs(r - r.mean())))
    if measure == "smv":
        return float(np.sqrt(np.mean(np.minimum(r - r.mean(), 0.0) ** 2))) if r.size > 1 else float("nan")
    if measure == "soes":
        losses = -r
        best = minimize_scalar(
            lambda zeta: zeta + np.linalg.norm(np.maximum(losses - zeta, 0.0)) / ((1.0 - alpha) * np.sqrt(losses.size)),
            bounds=(float(losses.min()), float(losses.max())), method="bounded", options={"xatol": 1e-10})
        return float(best.fun)
    raise ValueError(f"Unknown measure '{measure}'.")


def solve_weights(optimizer_cls, hyperparams: dict, R_fh: pd.DataFrame, R_c: pd.DataFrame):
    """(x, psi); asset-only optimizers such as 1/N get a zero overlay, as in Backtester.run."""
    optimizer = optimizer_cls(hyperparams=hyperparams)
    optimizer.fit({"R_fh": R_fh, "R_c": R_c})
    weights = optimizer.optimize()
    if isinstance(weights, dict):
        return np.asarray(weights["x"], dtype=float).flatten(), np.asarray(weights["psi"], dtype=float).flatten()
    return np.asarray(weights, dtype=float).flatten(), np.zeros(R_c.shape[1], dtype=float)


def run_universe(measure: str, universe: int, strategies: dict[str, str], leverages: dict[str, float],
                 solver) -> pd.DataFrame:
    cfgs = {role: STRATEGY_REGISTRY[name]["config"] for role, name in strategies.items()}
    if len({(c.window_years, c.val_years, c.rebalancing_frequency) for c in cfgs.values()}) != 1:
        raise ValueError(f"The {measure} overlay and joint configs disagree on the estimation window.")
    logger.info("Loading top-%d universe data.", universe)
    R_fh, R_c, mask, _ = load_universe_data(universe)
    windows = rebalancing_windows(R_fh, mask, cfgs["joint"])
    logger.info("Top-%d: %d rebalancing dates, %s to %s.", universe, len(windows),
                windows[0]["date"].date() if windows else "-", windows[-1]["date"].date() if windows else "-")

    hyperparams = {role: fixed_hyperparams(name) for role, name in strategies.items()}
    optimizers = {role: STRATEGY_REGISTRY[name]["optimizer"] for role, name in strategies.items()}
    optimizers[CONTROL_STRATEGY] = STRATEGY_REGISTRY[CONTROL_STRATEGY_KEY]["optimizer"]
    # 1/N ignores hyperparameters; alpha only sets the confidence level of its ES score
    hyperparams[CONTROL_STRATEGY] = {k: v for k, v in hyperparams["joint"].items() if k == "alpha"}
    # 1/N cannot hit a leverage cap, so it is solved once per date
    cells = [(label, limit, role) for label, limit in leverages.items() for role in ROLES]
    cells.append(("none", float("nan"), CONTROL_STRATEGY))

    rows = []
    for i, w in enumerate(windows, start=1):
        assets = w["assets"]
        R_fh_est = R_fh.loc[w["estimation_start"]:w["estimation_end"], assets]
        R_c_est = R_c.loc[w["estimation_start"]:w["estimation_end"]]
        # as in Backtester.run, a name that stops trading contributes zero to the holding window
        R_fh_hold = R_fh.loc[w["holding_start"]:w["holding_end"], assets].fillna(0.0)
        R_c_hold = R_c.loc[w["holding_start"]:w["holding_end"]].fillna(0.0)
        logger.info("Top-%d [%d/%d] %s: %d assets, %d estimation obs, %d holding obs.", universe, i,
                    len(windows), w["date"].date(), len(assets), len(R_fh_est), len(R_fh_hold))
        for label, limit, role in cells:
            params = {**hyperparams[role], "solver": solver}
            if role != CONTROL_STRATEGY:
                params["leverage_limit"] = limit
            alpha = params.get("alpha")
            try:
                x, psi = solve_weights(optimizers[role], params, R_fh_est, R_c_est)
                in_sample = evaluate_risk(measure, R_fh_est, R_c_est, x, psi, alpha)
                realized = evaluate_risk(measure, R_fh_hold, R_c_hold, x, psi, alpha)
            except (RuntimeError, ValueError) as exc:
                logger.warning("Top-%d %s %s %s (leverage %s) failed: %s", universe, w["date"].date(), role,
                               measure, label, exc)
                in_sample = realized = float("nan")
            rows.append({"measure": measure, "universe": universe, "leverage_label": label, "leverage_limit": limit,
                         "date": w["date"], "strategy": role, "alpha": alpha if alpha is not None else float("nan"),
                         "n_assets": len(assets), "n_obs_in_sample": len(R_fh_est),
                         "n_obs_realized": len(R_fh_hold), "in_sample_risk": in_sample, "realized_risk": realized})
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measure", choices=MEASURES)
    parser.add_argument("--universes", type=int, nargs="+", default=list(UNIVERSE_SIZES), choices=UNIVERSE_SIZES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    strategies = resolve_strategies(args.measure)
    caps = {role: fixed_hyperparams(name)["leverage_limit"] for role, name in strategies.items()}
    if len(set(caps.values())) != 1:
        raise ValueError(f"The {args.measure} joint and overlay programs run at different leverage caps {caps}.")
    leverages = {"production": float(next(iter(caps.values())))}
    solver, solver_name = pick_solver()
    logger.info("Measure %s | strategies %s | leverage caps %s | solver %s", args.measure, strategies, leverages,
                solver_name)
    for universe in args.universes:
        rows = run_universe(args.measure, universe, strategies, leverages, solver)
        path = args.out_dir / f"in_sample_{args.measure}_top{universe}.csv"
        rows.to_csv(path, index=False)
        logger.info("Wrote %d rows to %s", len(rows), path)


if __name__ == "__main__":
    main()
