"""Penalty boundary sweep of the regularized overlay and joint optimizers (Appendix G).

Each penalty is swept from three decades below to two above its grid bound, all other penalties at zero, on
N_BOOTSTRAP resamples of the first estimation window of the top-10 universe. Records how far the penalized
weight block moves per grid step (d_inf) and its gross exposure, after Lucescu & Ulrych (2026, App. C).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio_backtest import config
from portfolio_backtest.backtest.backtester import Backtester
from portfolio_backtest.backtest.data_handler import DataHandler
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (compute_daily_forward_premium, compute_excess_currency_returns,
                                              compute_fully_hedged_returns)

UNIVERSE = 10
N_BOOTSTRAP = 5
POINTS_PER_DECADE = 12
ACTIVE_TOL = 1e-6
DECADES_BELOW, DECADES_ABOVE = 3, 2
SEED = 0
MEASURES = ("es", "mv", "mad", "smv", "soes")
ROLES = ("overlay", "joint")
METRICS = ("d_inf", "n_active", "gross_x", "gross_psi")


def pick_solver():
    return (cp.GUROBI, "GUROBI") if "GUROBI" in cp.installed_solvers() else (cp.CLARABEL, "CLARABEL")


def resolve_strategies(measure: str) -> dict[str, str]:
    """The regularized overlay and joint strategy of a measure, excluding the mean-risk variants."""
    resolved = {}
    for role in ROLES:
        matches = [name for name in STRATEGY_REGISTRY if {"reg", measure, role} <= set(name.split("_"))
                   and "mean" not in name.split("_")]
        if len(matches) != 1:
            raise ValueError(f"Expected one regularized {measure} {role} strategy, found {matches}.")
        resolved[role] = matches[0]
    return resolved


def split_hyperparams(strategy: str) -> tuple[dict[str, float], dict[str, object]]:
    """Grid bound of every penalty, and the single value of every other hyperparameter."""
    bounds, fixed = {}, {}
    for key, values in STRATEGY_REGISTRY[strategy]["config"].param_grid.items():
        if key.startswith("lambda"):
            bounds[key] = float(max(values))
        elif len(values) != 1:
            raise ValueError(f"Non-penalty hyperparameter '{key}' of {strategy} has candidates {values}.")
        else:
            fixed[key] = values[0]
    if not bounds:
        raise ValueError(f"{strategy} declares no lambda* penalties to sweep.")
    return bounds, fixed


def load_universe_data():
    universe_path = config.RAW_DIR / config.UNIVERSE_FILES[UNIVERSE]
    local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    local = local.where(local.abs() < config.RETURN_FILTER_BACKTEST)
    spot = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    premium = compute_daily_forward_premium(load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE),
                                            config.TENOR_DAYS_3M)
    R_c = compute_excess_currency_returns(spot, premium)
    R_fh = compute_fully_hedged_returns(local, spot, premium, build_asset_to_currency(universe_path))
    # round-trip through the DataHandler so the data match the backtest exactly
    with tempfile.TemporaryDirectory() as tmp:
        fh_path, rc_path = Path(tmp) / config.FH_RETURNS_FILE, Path(tmp) / config.RC_RETURNS_FILE
        R_fh.rename_axis("date").to_csv(fh_path)
        R_c.rename_axis("date").to_csv(rc_path)
        R_fh, _, R_c, mask = DataHandler(universe_path, fh_path, rc_path).load_and_process()
    common = R_fh.index.intersection(R_c.index)
    return R_fh.loc[common], R_c.loc[common], mask.reindex(common, method="ffill"), universe_path


def first_estimation_window(R_fh: pd.DataFrame, mask: pd.DataFrame, cfg) -> dict:
    """The first rebalancing date that passes the backtester's own guards."""
    probe = Backtester(optimizer=STRATEGY_REGISTRY["equally_weighted"]["optimizer"], window_years=cfg.window_years,
                       val_years=cfg.val_years, rebalancing_frequency=cfg.rebalancing_frequency)
    dates = R_fh.index
    rebalance_dates = probe._get_rebalance_dates(dates)
    for t, t_next in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        if pd.Timestamp(t) - pd.DateOffset(years=cfg.window_years) < dates[0]:
            continue
        try:
            splits = probe._get_estimation_and_validation_splits(dates, t, t_next)
        except ValueError:
            continue
        start, end = splits["full_estimation_start"], splits["full_estimation_end"]
        row = mask.loc[t] if t in mask.index else pd.Series(dtype=float)
        universe = [a for a in row[row.notna()].index if a in R_fh.columns]
        if not universe:
            continue
        try:
            assets = probe._filter_universe(R_fh.loc[start:end, universe])
        except ValueError:
            continue
        return {"rebalancing_date": t, "start": start, "end": end, "assets": assets}
    raise RuntimeError("No rebalancing date on the top-10 universe passes the backtester's own guards.")


def penalty_grid(bound: float) -> np.ndarray:
    log_bound = np.log10(bound)
    return np.concatenate(([0.0], np.logspace(log_bound - DECADES_BELOW, log_bound + DECADES_ABOVE,
                                              num=(DECADES_BELOW + DECADES_ABOVE) * POINTS_PER_DECADE + 1)))


def nanmean(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def reset_caches(optimizer_cls) -> None:
    """Drop the class-level compiled problems, whose cache key a bootstrap resample can collide on."""
    if hasattr(optimizer_cls, "clear_cache"):
        optimizer_cls.clear_cache()
    for attr, empty in (("_cache_key", None), ("_asset_prob", None), ("_overlay_prob", None),
                        ("_asset_solutions", {})):
        if hasattr(optimizer_cls, attr):
            setattr(optimizer_cls, attr, empty)


def solve(optimizer_cls, hyperparams: dict, R_fh: pd.DataFrame, R_c: pd.DataFrame):
    """Weights, and the penalties the optimizer clamped to a different value than requested."""
    optimizer = optimizer_cls(hyperparams=hyperparams)
    clamped = {k: float(getattr(optimizer, k)) for k, v in hyperparams.items()
               if k.startswith("lambda") and float(getattr(optimizer, k)) != float(v)}
    optimizer.fit({"R_fh": R_fh, "R_c": R_c})
    weights = optimizer.optimize()
    return np.asarray(weights["x"], dtype=float).flatten(), np.asarray(weights["psi"], dtype=float).flatten(), clamped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measure", choices=MEASURES)
    measure = parser.parse_args().measure
    solver, solver_name = pick_solver()
    strategies = resolve_strategies(measure)
    cfgs = {role: STRATEGY_REGISTRY[name]["config"] for role, name in strategies.items()}
    if len({(c.window_years, c.val_years, c.rebalancing_frequency) for c in cfgs.values()}) != 1:
        raise ValueError(f"The {measure} overlay and joint configs disagree on the estimation window.")

    R_fh, R_c, mask, universe_path = load_universe_data()
    window = first_estimation_window(R_fh, mask, cfgs["overlay"])
    R_fh_window = R_fh.loc[window["start"]:window["end"], window["assets"]]
    R_c_window = R_c.loc[window["start"]:window["end"]]
    if len(R_fh_window) != len(R_c_window):
        raise RuntimeError(f"Asset and currency windows differ in length ({len(R_fh_window)} vs {len(R_c_window)}).")
    currencies = list(R_c_window.columns)
    psi_keep = [i for i, ccy in enumerate(currencies) if ccy != "USD"]
    n_obs = len(R_fh_window)
    rng = np.random.default_rng(SEED)
    resamples = [rng.integers(0, n_obs, size=n_obs) for _ in range(N_BOOTSTRAP)]

    samples, clamped_by_strategy, specs, errors = {}, {}, {}, {}
    failures, solve_seconds = 0, 0.0
    for strategy in strategies.values():
        optimizer_cls = STRATEGY_REGISTRY[strategy]["optimizer"]
        bounds, fixed = specs[strategy] = split_hyperparams(strategy)
        grids = {penalty: penalty_grid(bound) for penalty, bound in bounds.items()}
        clamped_seen = {}
        print(f"[{measure}] {strategy}: {len(bounds)} penalties x {N_BOOTSTRAP} replicates "
              f"x {len(next(iter(grids.values())))} lambdas on {n_obs} obs, {len(window['assets'])} assets, "
              f"{len(currencies)} currencies ({solver_name})", flush=True)
        for rows in resamples:
            reset_caches(optimizer_cls)
            R_fh_boot, R_c_boot = R_fh_window.iloc[rows], R_c_window.iloc[rows]
            for penalty, grid in grids.items():
                block = "psi" if "curr" in penalty else "x"
                previous = None
                for point, lam in enumerate(grid):
                    hyperparams = {**fixed, **{name: 0.0 for name in bounds}, penalty: float(lam), "solver": solver}
                    started = time.perf_counter()
                    try:
                        x, psi, clamped = solve(optimizer_cls, hyperparams, R_fh_boot, R_c_boot)
                    except Exception as exc:  # one failed grid point must not end the sweep; reasons are recorded
                        failures += 1
                        reason = f"{type(exc).__name__}: {exc}"[:200]
                        errors[reason] = errors.get(reason, 0) + 1
                        x, psi, clamped = None, None, {}
                    solve_seconds += time.perf_counter() - started
                    for name, effective in clamped.items():
                        requested = float(hyperparams[name])
                        clamped_seen[name, requested, effective] = {"penalty": name, "requested": requested,
                                                                    "effective": effective}
                    if x is None:
                        current, values = None, [float("nan")] * 4
                    else:
                        current = x if block == "x" else psi[psi_keep]
                        d_inf = float(np.max(np.abs(current - previous))) if previous is not None else float("nan")
                        values = [d_inf, float(np.sum(np.abs(current) > ACTIVE_TOL)), float(np.sum(np.abs(x))),
                                  float(np.sum(np.abs(psi[psi_keep])))]
                    previous = current
                    record = samples.setdefault((strategy, penalty, point), {"lambda": float(lam), "block": block,
                                                                             **{m: [] for m in METRICS}})
                    for metric, value in zip(METRICS, values):
                        record[metric].append(value)
        clamped_by_strategy[strategy] = sorted(clamped_seen.values(), key=lambda e: (e["penalty"], e["requested"]))

    rows_out = []
    for (strategy, penalty, _), record in sorted(samples.items()):
        d_inf = np.asarray(record["d_inf"], dtype=float)
        rows_out.append({"strategy": strategy, "penalty": penalty, "lambda": record["lambda"],
                         "block": record["block"], "d_inf_mean": nanmean(d_inf),
                         "d_inf_std": float(np.nanstd(d_inf, ddof=1)) if np.isfinite(d_inf).sum() > 1 else np.nan,
                         **{m: nanmean(record[m]) for m in METRICS[1:]}})

    out_dir = config.RESULTS_DIR / "hyperparameter_boundary_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_dir / f"{measure}_top{UNIVERSE}.csv", index=False)
    sidecar = {
        "measure": measure, "strategies": strategies, "universe_size": UNIVERSE, "universe_file": universe_path.name,
        "rebalancing_date": str(window["rebalancing_date"].date()), "window_start": str(window["start"].date()),
        "window_end": str(window["end"].date()), "n_observations": n_obs, "n_assets": len(window["assets"]),
        "n_currencies": len(currencies), "assets": window["assets"], "currencies": currencies,
        "base_currency_excluded_from_psi": "USD", "penalty_bounds": {s: spec[0] for s, spec in specs.items()},
        "fixed_hyperparams": {s: spec[1] for s, spec in specs.items()}, "clamped_penalties": clamped_by_strategy,
        "solver": solver_name, "seed": SEED, "n_bootstrap": N_BOOTSTRAP, "points_per_decade": POINTS_PER_DECADE,
        "decades_below": DECADES_BELOW, "decades_above": DECADES_ABOVE, "active_tol": ACTIVE_TOL,
        "solve_failures": failures, "solve_errors": errors, "total_solve_seconds": round(solve_seconds, 2),
    }
    (out_dir / f"{measure}_top{UNIVERSE}.json").write_text(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
