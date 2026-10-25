"""
Penalty-boundary diagnostic for the regularized overlay and joint optimizers.

Tests the penalty grid bounds in backtest_config.py by sweeping each penalty from
three decades below its bound to two decades above it, recording how far the
affected weight block still moves per grid step and how much gross exposure it
retains. L1 penalties saturate at a corner and stop mattering outright; L2
penalties decay smoothly and never do, so gross exposure rather than d_inf is
what establishes that an L2 bound is far enough out. Replicates Appendix C /
Figure 3 of Lucescu & Ulrych (2026) with 5 bootstrap replicates instead of 100,
on the first estimation window of the top-10 universe.

Usage:
    python code/analysis/hyperparameter_boundary_analysis.py es
    python code/analysis/hyperparameter_boundary_analysis.py mv
    python code/analysis/hyperparameter_boundary_analysis.py mad
    python code/analysis/hyperparameter_boundary_analysis.py smv
    python code/analysis/hyperparameter_boundary_analysis.py soes
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
from portfolio_backtest.utils.returns import (
    compute_daily_forward_premium,
    compute_excess_currency_returns,
    compute_fully_hedged_returns,
)

UNIVERSE = 10
N_BOOTSTRAP = 5
POINTS_PER_DECADE = 12
ACTIVE_TOL = 1e-6
DECADES_BELOW = 3
DECADES_ABOVE = 2
SEED = 0
BASE_CURRENCY = "USD"

MEASURES = ("es", "mv", "mad", "smv", "soes")
ROLES = ("overlay", "joint")


def _pick_solver():
    if "GUROBI" in cp.installed_solvers():
        return cp.GUROBI, "GUROBI"
    return cp.CLARABEL, "CLARABEL"


def _resolve_strategies(measure: str) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for role in ROLES:
        matches = [
            name
            for name in STRATEGY_REGISTRY
            if {"reg", measure, role}.issubset(set(name.split("_")))
            # The mean-risk variants (*_w_mean) share the name parts but are not swept.
            and "mean" not in name.split("_")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one regularized {measure} {role} strategy in "
                f"STRATEGY_REGISTRY, found {matches}. The sweep compares the overlay "
                f"against the joint program; running on one alone is not the diagnostic."
            )
        resolved[role] = matches[0]
    return resolved


def _split_hyperparams(strategy: str) -> tuple[dict[str, float], dict[str, object]]:
    param_grid = STRATEGY_REGISTRY[strategy]["config"].param_grid
    bounds: dict[str, float] = {}
    fixed: dict[str, object] = {}
    for key, values in param_grid.items():
        if key.startswith("lambda"):
            bounds[key] = float(max(values))
        elif len(values) != 1:
            raise ValueError(
                f"Non-penalty hyperparameter '{key}' of {strategy} has {len(values)} "
                f"candidates {values}. The sweep holds every non-penalty setting fixed, "
                f"so it has no basis for choosing one."
            )
        else:
            fixed[key] = values[0]
    if not bounds:
        raise ValueError(f"{strategy} declares no lambda* penalties to sweep.")
    return bounds, fixed


def _load_universe_data():
    universe_path = config.RAW_DIR / config.UNIVERSE_FILES[UNIVERSE]

    R_local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    R_local = R_local.where(R_local.abs() < config.RETURN_FILTER_BACKTEST)
    e = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    f_raw = load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE)
    f_daily = compute_daily_forward_premium(f_raw, config.TENOR_DAYS_3M)

    asset_to_currency = build_asset_to_currency(universe_path)
    R_c = compute_excess_currency_returns(e, f_daily)
    R_fh = compute_fully_hedged_returns(R_local, e, f_daily, asset_to_currency)
    R_fh.index.name = "date"
    R_c.index.name = "date"

    with tempfile.TemporaryDirectory() as tmp:
        fh_path = Path(tmp) / config.FH_RETURNS_FILE
        rc_path = Path(tmp) / config.RC_RETURNS_FILE
        R_fh.to_csv(fh_path)
        R_c.to_csv(rc_path)
        handler = DataHandler(universe_path, fh_path, rc_path)
        R_fh, _, R_c, universe_mask = handler.load_and_process()

    common = R_fh.index.intersection(R_c.index)
    return R_fh.loc[common], R_c.loc[common], universe_mask.reindex(common, method="ffill"), universe_path


def _first_estimation_window(R_fh, universe_mask, backtest_cfg) -> dict:
    probe = Backtester(
        optimizer=STRATEGY_REGISTRY["equally_weighted"]["optimizer"],
        window_years=backtest_cfg.window_years,
        val_years=backtest_cfg.val_years,
        rebalancing_frequency=backtest_cfg.rebalancing_frequency,
    )
    dates = R_fh.index
    rebalance_dates = probe._get_rebalance_dates(dates)
    data_start = dates[0]

    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]
        if pd.Timestamp(t) - pd.DateOffset(years=backtest_cfg.window_years) < data_start:
            continue
        try:
            splits = probe._get_estimation_and_validation_splits(dates, t, rebalance_dates[i + 1])
        except ValueError:
            continue

        start = splits["full_estimation_start"]
        end = splits["full_estimation_end"]

        mask_row = universe_mask.loc[t] if t in universe_mask.index else pd.Series(dtype=float)
        current_universe = [a for a in mask_row[mask_row.notna()].index if a in R_fh.columns]
        if not current_universe:
            continue
        try:
            assets = probe._filter_universe(R_fh.loc[start:end, current_universe])
        except ValueError:
            continue

        return {"rebalancing_date": t, "start": start, "end": end, "assets": assets}

    raise RuntimeError(
        "No rebalancing date on the top-10 universe passes the backtester's own guards."
    )


def _penalty_grid(bound: float) -> np.ndarray:
    log_bound = np.log10(bound)
    positive = np.logspace(
        log_bound - DECADES_BELOW,
        log_bound + DECADES_ABOVE,
        num=(DECADES_BELOW + DECADES_ABOVE) * POINTS_PER_DECADE + 1,
    )
    return np.concatenate(([0.0], positive))


def _nanmean(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _reset_optimizer_caches(optimizer_cls) -> None:
    if hasattr(optimizer_cls, "clear_cache"):
        optimizer_cls.clear_cache()
    for attr, empty in (
        ("_cache_key", None),
        ("_asset_prob", None),
        ("_overlay_prob", None),
        ("_asset_solutions", {}),
    ):
        if hasattr(optimizer_cls, attr):
            setattr(optimizer_cls, attr, empty)


def _solve(optimizer_cls, hyperparams, R_fh_window, R_c_window):
    optimizer = optimizer_cls(hyperparams=hyperparams)
    clamped = {
        key: float(getattr(optimizer, key))
        for key, requested in hyperparams.items()
        if key.startswith("lambda") and float(getattr(optimizer, key)) != float(requested)
    }
    optimizer.fit({"R_fh": R_fh_window, "R_c": R_c_window})
    weights = optimizer.optimize()
    x = np.asarray(weights["x"], dtype=float).flatten()
    psi = np.asarray(weights["psi"], dtype=float).flatten()
    return x, psi, clamped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measure", choices=MEASURES, help="Risk measure to sweep.")
    measure = parser.parse_args().measure

    solver, solver_name = _pick_solver()
    strategies = _resolve_strategies(measure)

    backtest_cfgs = {
        role: STRATEGY_REGISTRY[name]["config"] for role, name in strategies.items()
    }
    windows = {
        (cfg.window_years, cfg.val_years, cfg.rebalancing_frequency)
        for cfg in backtest_cfgs.values()
    }
    if len(windows) != 1:
        raise ValueError(
            f"The {measure} overlay and joint configs disagree on the estimation window "
            f"{windows}. Both must sweep the same window for the results to be comparable."
        )

    R_fh, R_c, universe_mask, universe_path = _load_universe_data()
    window = _first_estimation_window(R_fh, universe_mask, backtest_cfgs["overlay"])

    R_fh_window = R_fh.loc[window["start"]:window["end"], window["assets"]]
    R_c_window = R_c.loc[window["start"]:window["end"]]
    if len(R_fh_window) != len(R_c_window):
        raise RuntimeError(
            f"Asset and currency windows differ in length ({len(R_fh_window)} vs "
            f"{len(R_c_window)}); a shared bootstrap index would misalign them."
        )

    currencies = list(R_c_window.columns)
    psi_keep = [i for i, ccy in enumerate(currencies) if ccy != BASE_CURRENCY]
    n_obs = len(R_fh_window)

    rng = np.random.default_rng(SEED)
    resamples = [rng.integers(0, n_obs, size=n_obs) for _ in range(N_BOOTSTRAP)]

    samples: dict[tuple[str, str, int], dict] = {}
    clamped_by_strategy: dict[str, list[dict]] = {}
    hyperparam_spec: dict[str, tuple[dict, dict]] = {}
    solve_failures = 0
    solve_errors: dict[str, int] = {}
    solve_seconds = 0.0

    for strategy in strategies.values():
        optimizer_cls = STRATEGY_REGISTRY[strategy]["optimizer"]
        bounds, fixed = _split_hyperparams(strategy)
        hyperparam_spec[strategy] = (bounds, fixed)
        grids = {penalty: _penalty_grid(bound) for penalty, bound in bounds.items()}
        clamped_seen: dict[tuple[str, float, float], dict] = {}

        print(
            f"[{measure}] {strategy}: {len(bounds)} penalties x {N_BOOTSTRAP} replicates "
            f"x {len(next(iter(grids.values())))} lambdas on {n_obs} obs, "
            f"{len(window['assets'])} assets, {len(currencies)} currencies "
            f"({solver_name})",
            flush=True,
        )

        for rows in resamples:
            # The compiled problems and cached step-1 solutions are held at class
            # level and keyed on a frame signature that a resample can collide on,
            # and the return matrix is baked into the problem as a constant rather
            # than a parameter. Clearing per replicate is what stops one replicate's
            # solution from being served for another's data.
            _reset_optimizer_caches(optimizer_cls)
            R_fh_boot = R_fh_window.iloc[rows]
            R_c_boot = R_c_window.iloc[rows]

            for penalty, grid in grids.items():
                block = "psi" if "curr" in penalty else "x"
                previous = None

                for point, lam in enumerate(grid):
                    hyperparams = {
                        **fixed,
                        **{name: 0.0 for name in bounds},
                        penalty: float(lam),
                        "solver": solver,
                    }
                    started = time.perf_counter()
                    try:
                        x, psi, clamped = _solve(
                            optimizer_cls, hyperparams, R_fh_boot, R_c_boot
                        )
                    except Exception as exc:
                        # Deliberately broad so one bad grid point cannot end a
                        # multi-hour sweep, but every distinct failure is recorded:
                        # otherwise an interface change would masquerade as a solver
                        # failure and quietly produce a CSV of NaNs.
                        solve_failures += 1
                        reason = f"{type(exc).__name__}: {exc}"[:200]
                        solve_errors[reason] = solve_errors.get(reason, 0) + 1
                        x, psi, clamped = None, None, {}
                    solve_seconds += time.perf_counter() - started

                    for name, effective in clamped.items():
                        clamped_seen[(name, float(hyperparams[name]), effective)] = {
                            "penalty": name,
                            "requested": float(hyperparams[name]),
                            "effective": effective,
                        }

                    if x is None:
                        current = None
                        d_inf = float("nan")
                        n_active = float("nan")
                        gross_x = float("nan")
                        gross_psi = float("nan")
                    else:
                        current = x if block == "x" else psi[psi_keep]
                        d_inf = (
                            float(np.max(np.abs(current - previous)))
                            if previous is not None
                            else float("nan")
                        )
                        n_active = float(np.sum(np.abs(current) > ACTIVE_TOL))
                        gross_x = float(np.sum(np.abs(x)))
                        gross_psi = float(np.sum(np.abs(psi[psi_keep])))
                    previous = current

                    record = samples.setdefault(
                        (strategy, penalty, point),
                        {
                            "lambda": float(lam),
                            "block": block,
                            "d_inf": [],
                            "n_active": [],
                            "gross_x": [],
                            "gross_psi": [],
                        },
                    )
                    record["d_inf"].append(d_inf)
                    record["n_active"].append(n_active)
                    record["gross_x"].append(gross_x)
                    record["gross_psi"].append(gross_psi)

        clamped_by_strategy[strategy] = sorted(
            clamped_seen.values(), key=lambda entry: (entry["penalty"], entry["requested"])
        )

    rows_out = []
    for key in sorted(samples, key=lambda k: (k[0], k[1], k[2])):
        strategy, penalty, _ = key
        record = samples[key]
        d_inf = np.asarray(record["d_inf"], dtype=float)
        rows_out.append(
            {
                "strategy": strategy,
                "penalty": penalty,
                "lambda": record["lambda"],
                "block": record["block"],
                "d_inf_mean": _nanmean(d_inf),
                "d_inf_std": (
                    float(np.nanstd(d_inf, ddof=1))
                    if np.count_nonzero(np.isfinite(d_inf)) > 1
                    else np.nan
                ),
                "n_active": _nanmean(record["n_active"]),
                "gross_x": _nanmean(record["gross_x"]),
                "gross_psi": _nanmean(record["gross_psi"]),
            }
        )

    out_dir = config.RESULTS_DIR / "hyperparameter_boundary_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{measure}_top{UNIVERSE}.csv"
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)

    sidecar = {
        "measure": measure,
        "strategies": strategies,
        "universe_size": UNIVERSE,
        "universe_file": universe_path.name,
        "rebalancing_date": str(window["rebalancing_date"].date()),
        "window_start": str(window["start"].date()),
        "window_end": str(window["end"].date()),
        "n_observations": n_obs,
        "n_assets": len(window["assets"]),
        "n_currencies": len(currencies),
        "assets": window["assets"],
        "currencies": currencies,
        "base_currency_excluded_from_psi": BASE_CURRENCY,
        "penalty_bounds": {s: spec[0] for s, spec in hyperparam_spec.items()},
        "fixed_hyperparams": {s: spec[1] for s, spec in hyperparam_spec.items()},
        "clamped_penalties": clamped_by_strategy,
        "solver": solver_name,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "points_per_decade": POINTS_PER_DECADE,
        "decades_below": DECADES_BELOW,
        "decades_above": DECADES_ABOVE,
        "active_tol": ACTIVE_TOL,
        "solve_failures": solve_failures,
        "solve_errors": solve_errors,
        "total_solve_seconds": round(solve_seconds, 2),
    }
    with open(out_dir / f"{measure}_top{UNIVERSE}.json", "w") as handle:
        json.dump(sidecar, handle, indent=2)


if __name__ == "__main__":
    main()
