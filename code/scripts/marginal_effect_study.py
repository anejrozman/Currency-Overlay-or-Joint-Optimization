"""
Marginal effect of each regularization penalty, bootstrapped over one estimation
window, for every regularized strategy of a measure. Writes one CSV per measure.

Usage:
    python code/scripts/marginal_effect_study.py es

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio_backtest import config
from portfolio_backtest.backtest.backtester import Backtester
from portfolio_backtest.backtest.data_handler import DataHandler
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY

PENALTY_PREFIX: str = "lambda"
CURRENCY_MARKER: str = "curr"
BASE_CURRENCY: str = "USD"

UNIVERSE: int = 10
N_BOOTSTRAP: int = 20
POINTS_PER_DECADE: int = 12
ACTIVE_TOL: float = 1e-6
SEED: int = 0

DECADES_BELOW: int = 3
DECADES_ABOVE: int = 2

UNIVERSE_FILE: Path = config.RAW_DIR / config.UNIVERSE_FILES[UNIVERSE]
FH_RETURNS_PATH: Path = config.PROCESSED_DIR / config.FH_RETURNS_FILE
UNHEDGED_RETURNS_PATH: Path = config.PROCESSED_DIR / config.UNHEDGED_RETURNS_FILE
RC_RETURNS_PATH: Path = config.PROCESSED_DIR / config.RC_RETURNS_FILE
OUTPUT_DIR: Path = config.RESULTS_DIR / "marginal_effect_study"


def _select_strategies(measure: str) -> list[str]:
    """Registry entries for this measure that have at least one penalty to sweep."""
    return [
        name
        for name, strategy_cfg in STRATEGY_REGISTRY.items()
        if measure in name.split("_")
        and any(k.startswith(PENALTY_PREFIX) for k in strategy_cfg["config"].param_grid)
    ]


def _split_hyperparams(param_grid: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Separate a strategy's grid into swept penalties and fixed hyperparameters."""
    penalties: dict[str, float] = {}
    fixed: dict[str, float] = {}

    for key, values in param_grid.items():
        if key.startswith(PENALTY_PREFIX):
            penalties[key] = float(max(values))
        elif len(values) == 1:
            fixed[key] = values[0]
        else:
            raise ValueError(
                f"Fixed hyperparameter '{key}' has {len(values)} candidates {values}. "
                "The study holds every non-penalty parameter constant, so it cannot "
                "choose between them."
            )

    return penalties, fixed


def _resolve_solver() -> object:
    """Prefer Gurobi, the solver the backtests ran under, else fall back to Clarabel."""
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        return cp.CLARABEL
    return cp.GUROBI


def _load_data(
    strategy_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, str], pd.DataFrame]:
    """Load returns and the universe mask exactly as run_strategy.py does."""
    handler = DataHandler(
        universe_file=UNIVERSE_FILE,
        hedged_returns_file=(
            FH_RETURNS_PATH if strategy_cfg["use_hedging"] else UNHEDGED_RETURNS_PATH
        ),
        excess_currency_returns_file=RC_RETURNS_PATH,
    )
    R_asset, asset_to_currency, R_c, universe_mask = handler.load_and_process()

    if strategy_cfg["use_hedging"]:
        common = R_asset.index.intersection(R_c.index)
        R_asset = R_asset.loc[common]
        R_c = R_c.loc[common]
        universe_mask = universe_mask.reindex(common, method="ffill")
    else:
        universe_mask = universe_mask.reindex(R_asset.index, method="ffill")
        R_c = None

    return R_asset, R_c, asset_to_currency, universe_mask


def _first_window(
    backtester: Backtester,
    R_asset: pd.DataFrame,
    universe_mask: pd.DataFrame,
) -> tuple[pd.Timestamp, dict[str, pd.Timestamp], list[str]]:
    """Find the first window the backtester would estimate on."""
    dates = R_asset.index
    rebalance_dates = backtester._get_rebalance_dates(dates)
    data_start = dates[0]

    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]
        t_next = rebalance_dates[i + 1]

        if pd.Timestamp(t) - pd.DateOffset(years=backtester.window_years) < data_start:
            continue

        try:
            splits = backtester._get_estimation_and_validation_splits(dates, t, t_next)
        except ValueError:
            continue

        mask_row = (
            universe_mask.loc[t] if t in universe_mask.index else pd.Series(dtype=float)
        )
        current_universe = [
            a for a in mask_row[mask_row.notna()].index if a in R_asset.columns
        ]
        if not current_universe:
            continue

        window = R_asset.loc[
            splits["full_estimation_start"]:splits["full_estimation_end"],
            current_universe,
        ]
        try:
            valid_assets = backtester._filter_universe(window)
        except ValueError:
            continue

        return t, splits, valid_assets

    raise RuntimeError("No rebalancing date has a complete estimation window.")


def _build_grid(bound: float) -> np.ndarray:
    """Log-spaced sweep for one penalty, prefixed with an exact zero."""
    decades = DECADES_BELOW + DECADES_ABOVE
    points = np.logspace(
        np.log10(bound) - DECADES_BELOW,
        np.log10(bound) + DECADES_ABOVE,
        decades * POINTS_PER_DECADE + 1,
    )
    return np.concatenate([[0.0], points])


def _unpack(raw_weights: np.ndarray | dict) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize optimizer output to an (assets, currencies) pair."""
    if isinstance(raw_weights, dict):
        return (
            np.asarray(raw_weights["x"], dtype=float).flatten(),
            np.asarray(raw_weights["psi"], dtype=float).flatten(),
        )
    return np.asarray(raw_weights, dtype=float).flatten(), None


def _warn_on_clamped_penalties(
    optimizer: object, requested: dict, penalties: dict[str, float], clamped: set[str]
) -> None:
    """Record penalties whose requested value the optimizer silently altered."""
    for name in penalties:
        stored = getattr(optimizer, name, None)
        if stored is None or name in clamped:
            continue
        if not np.isclose(stored, requested[name], rtol=0.0, atol=0.0):
            clamped.add(name)


def _sweep_penalty(
    penalty: str,
    bound: float,
    optimizer_cls: type,
    asset_window: pd.DataFrame,
    currency_window: pd.DataFrame | None,
    asset_return_key: str,
    C_sub: np.ndarray,
    fixed: dict,
    penalties: dict[str, float],
    currency_index: list[str],
    solver: object,
    row_draws: list[np.ndarray],
    clamped: set[str],
) -> list[dict]:
    """Vary one penalty over its grid, on each bootstrap resample of the window."""
    on_currency_block = CURRENCY_MARKER in penalty
    block = "psi" if on_currency_block else "x"
    keep = [
        i for i, c in enumerate(currency_index) if c != BASE_CURRENCY
    ] if on_currency_block else []

    grid = _build_grid(bound)
    d_inf: list[list[float]] = [[] for _ in grid]
    n_active: list[list[int]] = [[] for _ in grid]
    gross_x: list[list[float]] = [[] for _ in grid]
    gross_psi: list[list[float]] = [[] for _ in grid]

    for rows in row_draws:
        train_bundle: dict = {
            asset_return_key: asset_window.iloc[rows],
            "C": C_sub,
        }
        if currency_window is not None:
            train_bundle["R_c"] = currency_window.iloc[rows]

        previous: np.ndarray | None = None
        for i, value in enumerate(grid):
            hyperparams = {
                **fixed,
                **{name: 0.0 for name in penalties},
                penalty: float(value),
                "solver": solver,
            }

            optimizer = optimizer_cls(hyperparams=hyperparams)
            _warn_on_clamped_penalties(optimizer, hyperparams, penalties, clamped)
            optimizer.fit(train_bundle)
            x, psi = _unpack(optimizer.optimize())

            if on_currency_block:
                if psi is None:
                    raise RuntimeError(
                        f"'{penalty}' acts on the currency block, but the optimizer "
                        "returned asset weights only."
                    )
                current = psi[keep]
            else:
                current = x

            if previous is not None:
                d_inf[i].append(float(np.max(np.abs(current - previous))))
            previous = current

            n_active[i].append(int(np.sum(np.abs(current) > ACTIVE_TOL)))
            gross_x[i].append(float(np.abs(x).sum()))
            gross_psi[i].append(float(np.abs(psi).sum()) if psi is not None else np.nan)

    return [
        {
            "penalty": penalty,
            "lambda": float(value),
            "block": block,
            "d_inf_mean": float(np.mean(d_inf[i])) if d_inf[i] else np.nan,
            "d_inf_std": float(np.std(d_inf[i], ddof=1)) if len(d_inf[i]) > 1 else np.nan,
            "n_active": float(np.mean(n_active[i])),
            "gross_x": float(np.mean(gross_x[i])),
            "gross_psi": float(np.mean(gross_psi[i])),
        }
        for i, value in enumerate(grid)
    ]


def _study_strategy(
    strategy: str, solver: object, clamped: set[str]
) -> tuple[list[dict], dict]:
    """Sweep every penalty of one strategy and return its records and metadata."""
    strategy_cfg = STRATEGY_REGISTRY[strategy]
    backtest_cfg = strategy_cfg["config"]
    optimizer_cls = strategy_cfg["optimizer"]
    asset_return_key = strategy_cfg["asset_return_key"]

    penalties, fixed = _split_hyperparams(backtest_cfg.param_grid)
    R_asset, R_c, asset_to_currency, universe_mask = _load_data(strategy_cfg)

    backtester = Backtester(
        optimizer=optimizer_cls,
        param_grid={},
        window_years=backtest_cfg.window_years,
        val_years=backtest_cfg.val_years,
        rebalancing_frequency=backtest_cfg.rebalancing_frequency,
        asset_return_key=asset_return_key,
    )
    t, splits, valid_assets = _first_window(backtester, R_asset, universe_mask)

    start, end = splits["full_estimation_start"], splits["full_estimation_end"]
    asset_window = R_asset.loc[start:end, valid_assets]
    currency_window = R_c.loc[start:end] if R_c is not None else None
    currency_index = list(currency_window.columns) if currency_window is not None else []

    C = pd.DataFrame(0.0, index=R_asset.columns, columns=currency_index)
    for asset, ccy in asset_to_currency.items():
        if asset in C.index and ccy in C.columns:
            C.loc[asset, ccy] = 1.0

    n_obs = len(asset_window)
    rng = np.random.default_rng(SEED)
    row_draws = [rng.integers(0, n_obs, n_obs) for _ in range(N_BOOTSTRAP)]

    records: list[dict] = []
    for penalty, bound in penalties.items():
        for record in _sweep_penalty(
            penalty=penalty,
            bound=bound,
            optimizer_cls=optimizer_cls,
            asset_window=asset_window,
            currency_window=currency_window,
            asset_return_key=asset_return_key,
            C_sub=C.loc[valid_assets].values,
            fixed=fixed,
            penalties=penalties,
            currency_index=currency_index,
            solver=solver,
            row_draws=row_draws,
            clamped=clamped,
        ):
            records.append({"strategy": strategy, **record})

    metadata = {
        "optimizer": optimizer_cls.__name__,
        "asset_return_key": asset_return_key,
        "rebalancing_date": str(t.date()),
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "n_observations": n_obs,
        "n_assets": len(valid_assets),
        "n_currencies": len(currency_index),
        "window_years": backtest_cfg.window_years,
        "val_years": backtest_cfg.val_years,
        "rebalancing_frequency": backtest_cfg.rebalancing_frequency,
        "fixed_hyperparams": {k: float(v) for k, v in fixed.items()},
        "penalty_bounds": penalties,
    }
    return records, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Marginal effect of each regularization penalty on one window.",
    )
    parser.add_argument(
        "measure",
        choices=["es", "mv"],
        help="Measure whose regularized strategies are swept.",
    )
    measure = parser.parse_args().measure

    solver = _resolve_solver()
    clamped: set[str] = set()

    records: list[dict] = []
    runs: dict[str, dict] = {}
    for strategy in _select_strategies(measure):
        strategy_records, metadata = _study_strategy(strategy, solver, clamped)
        records.extend(strategy_records)
        runs[strategy] = metadata

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{measure}_top{UNIVERSE}.csv"
    pd.DataFrame.from_records(
        records,
        columns=[
            "strategy", "penalty", "lambda", "block",
            "d_inf_mean", "d_inf_std", "n_active", "gross_x", "gross_psi",
        ],
    ).to_csv(out_path, index=False)

    sidecar = {
        "measure": measure,
        "strategies": list(runs),
        "universe_size": UNIVERSE,
        "universe_file": UNIVERSE_FILE.name,
        "solver": str(solver),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "points_per_decade": POINTS_PER_DECADE,
        "decades_below": DECADES_BELOW,
        "decades_above": DECADES_ABOVE,
        "active_tol": ACTIVE_TOL,
        "currency_excluded_from_norm": BASE_CURRENCY,
        "clamped_penalties": sorted(clamped),
        "runs": runs,
    }
    with open(out_path.with_suffix(".json"), "w") as fh:
        json.dump(sidecar, fh, indent=2)


if __name__ == "__main__":
    main()
