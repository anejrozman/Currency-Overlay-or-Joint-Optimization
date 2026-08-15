"""
Marginal-effect study of the regularization penalties, replicating Figure 3 of
Ulrych & Lucescu (2026).

Justifies the penalty grid bounds in ``backtest_config.py``.  On a single lookback
window -- the first window the backtester itself would estimate on -- each penalty
is varied alone with the others held at zero, over a log-spaced sequence running
three decades below its grid bound to two decades above it.  For each step the
study records

    d_n = || z(lambda_n) - z(lambda_{n-1}) ||_inf

where *z* is the weight block the penalty acts on.  A bound is justified when the
curve has flattened by the time the sweep reaches it; stopping at the bound would
prove nothing, hence the two decades beyond.

The study is driven entirely by ``STRATEGY_REGISTRY``: which penalties exist,
what their bounds are, and every fixed hyperparameter are read from the strategy's
own ``BacktestConfig``.  Any strategy registered later is handled with no change
here, provided its penalty hyperparameters are named ``lambda*``.

Usage:
    python code/scripts/marginal_effect_study.py --strategy reg_es_joint
    python code/scripts/marginal_effect_study.py --strategy reg_es_asset_full --universe 50

Arguments:

    --strategy: Strategy whose penalties are swept (required)

    --universe: Top-N stocks per economy (optional, default: 10)

    --points-per-decade: Grid density, identical across penalties (optional, default: 8)

    --active-tol: Magnitude above which a weight counts as active (optional, default: 1e-6)

    --solver: auto / gurobi / clarabel (optional, default: auto)

    --out: Override the output CSV path (optional)

Writes ``results/marginal_effect_study/<strategy>_top<N>.csv`` plus a ``.json``
sidecar holding the window and the fixed hyperparameters the sweep ran under.
No plotting.

CSV columns:

    penalty        hyperparameter being swept
    lambda         its value at this grid point
    block          weight block d_inf and n_active are measured on, "x" or "psi"
    d_inf          sup-norm change in that block from the previous grid point;
                   empty at lambda = 0, which has no predecessor
    n_active       entries of that block exceeding --active-tol
    gross_x        L1 norm of the asset weights
    gross_psi      L1 norm of the currency overlay, empty for asset-only strategies
    solve_seconds  wall time of fit() plus optimize()

The currency block excludes USD throughout: its excess return column is identically
zero and psi is unconstrained, so psi_USD is unidentified and can move between
consecutive solves without meaning anything.

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio_backtest import config
from portfolio_backtest.backtest.backtester import Backtester
from portfolio_backtest.backtest.data_handler import DataHandler
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

# Penalty hyperparameters are identified by this prefix, and a penalty acting on
# the currency block by this marker. Both match the naming used across the
# optimizers: lambda_l1/lambda_l2 on the asset-only ones, and the four-way
# lambda_{l1,l2}_{asset,curr} split on the overlay and joint ones.
PENALTY_PREFIX: str = "lambda"
CURRENCY_MARKER: str = "curr"

# Decades swept below and above each penalty's grid bound.
DECADES_BELOW: int = 3
DECADES_ABOVE: int = 2

# Base currency, dropped from the currency block before differencing. Its excess
# return column is identically zero and psi carries no constraint, so its overlay
# weight is unidentified and can move arbitrarily between consecutive solves.
BASE_CURRENCY: str = "USD"

OUTPUT_DIR: Path = config.RESULTS_DIR / "marginal_effect_study"


# Strategy introspection

def _split_hyperparams(param_grid: dict) -> tuple[dict[str, float], dict[str, float]]:
    """
    Separate a strategy's grid into swept penalties and fixed hyperparameters.

    Each penalty's bound is the largest value in its grid, so the sweep is anchored
    to the same numbers the backtest searches over rather than a hardcoded copy.

    Args:
        param_grid: ``param_grid`` of the strategy's ``BacktestConfig``.

    Returns:
        Tuple of (penalty name to bound, fixed hyperparameter name to value).

    Raises:
        ValueError: If a fixed hyperparameter carries more than one candidate
                    value, which would leave the sweep's context ambiguous.
    """
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


def _resolve_solver(choice: str) -> object:
    """
    Pick the cvxpy solver, preferring the one the backtests were run under.

    Args:
        choice: ``"auto"``, ``"gurobi"`` or ``"clarabel"``.

    Returns:
        A cvxpy solver constant.
    """
    if choice == "gurobi":
        return cp.GUROBI
    if choice == "clarabel":
        return cp.CLARABEL

    try:
        import gurobipy  # noqa: F401
    except ImportError:
        logger.warning(
            "gurobipy is not installed; falling back to Clarabel. The optimizers "
            "default to Gurobi, so these results describe a slightly different "
            "program than a cluster run would."
        )
        return cp.CLARABEL
    return cp.GUROBI


# Data and window selection

def _load_data(
    strategy_cfg: dict,
    universe_path: Path,
    asset_returns_path: Path,
    excess_currency_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, str], pd.DataFrame]:
    """
    Load returns and the universe mask exactly as ``run_strategy.py`` does.

    Args:
        strategy_cfg: Registry entry for the strategy.
        universe_path: Investment universe CSV.
        asset_returns_path: Fully hedged or unhedged asset returns.
        excess_currency_path: Excess currency returns.

    Returns:
        Tuple of (asset returns, excess currency returns or ``None`` for unhedged
        strategies, asset-to-currency mapping, universe mask).
    """
    handler = DataHandler(
        universe_file=universe_path,
        hedged_returns_file=asset_returns_path,
        excess_currency_returns_file=excess_currency_path,
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
    """
    Find the first window the backtester would estimate on.

    Replicates the guards inside :meth:`Backtester.run` that surround its window
    helpers: the full-history requirement, the split failure at the end of the
    index, and the universe mask applied before the completeness filter.  Skipping
    any of them lands on a window the backtest never used, which would justify the
    bounds for a program that was never run.

    Args:
        backtester: Configured instance, used for its window helpers.
        R_asset: Asset returns.
        universe_mask: Point-in-time universe mask.

    Returns:
        Tuple of (rebalancing date, split boundaries, investable assets).

    Raises:
        RuntimeError: If no rebalancing date yields a usable window.
    """
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


# The sweep

def _build_grid(bound: float, points_per_decade: int) -> np.ndarray:
    """
    Log-spaced sweep for one penalty, prefixed with an exact zero.

    Args:
        bound: The penalty's grid bound.
        points_per_decade: Grid density.

    Returns:
        Ascending array of penalty values starting at 0.0.
    """
    decades = DECADES_BELOW + DECADES_ABOVE
    points = np.logspace(
        np.log10(bound) - DECADES_BELOW,
        np.log10(bound) + DECADES_ABOVE,
        decades * points_per_decade + 1,
    )
    return np.concatenate([[0.0], points])


def _unpack(raw_weights: np.ndarray | dict) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Normalize optimizer output to an (assets, currencies) pair.

    Args:
        raw_weights: Return value of ``optimize()`` -- a bare array for asset-only
                     optimizers, a dict for the overlay and joint ones.

    Returns:
        Tuple of (asset weights, currency overlay weights or ``None``).
    """
    if isinstance(raw_weights, dict):
        return (
            np.asarray(raw_weights["x"], dtype=float).flatten(),
            np.asarray(raw_weights["psi"], dtype=float).flatten(),
        )
    return np.asarray(raw_weights, dtype=float).flatten(), None


def _sweep_penalty(
    penalty: str,
    bound: float,
    optimizer_cls: type,
    train_bundle: dict,
    fixed: dict,
    penalties: dict[str, float],
    currency_index: list[str],
    solver: object,
    points_per_decade: int,
    active_tol: float,
) -> list[dict]:
    """
    Vary one penalty over its grid with every other penalty held at zero.

    Args:
        penalty: Hyperparameter being swept.
        bound: Its grid bound.
        optimizer_cls: Optimizer class to instantiate per solve.
        train_bundle: Estimation-window bundle passed to ``fit()``.
        fixed: Fixed hyperparameters.
        penalties: All penalty names, used to pin the others at zero.
        currency_index: Currency column order, for dropping the base currency.
        solver: cvxpy solver.
        points_per_decade: Grid density.
        active_tol: Magnitude above which a weight counts as active.

    Returns:
        One record per grid point, in ascending penalty order.
    """
    on_currency_block = CURRENCY_MARKER in penalty
    block = "psi" if on_currency_block else "x"

    keep = [
        i for i, c in enumerate(currency_index) if c != BASE_CURRENCY
    ] if on_currency_block else []

    records: list[dict] = []
    previous: np.ndarray | None = None

    for value in _build_grid(bound, points_per_decade):
        hyperparams = {
            **fixed,
            **{name: 0.0 for name in penalties},
            penalty: float(value),
            "solver": solver,
        }

        optimizer = optimizer_cls(hyperparams=hyperparams)
        _warn_on_clamped_penalties(optimizer, hyperparams, penalties)

        start = time.perf_counter()
        optimizer.fit(train_bundle)
        x, psi = _unpack(optimizer.optimize())
        solve_seconds = time.perf_counter() - start

        if on_currency_block:
            if psi is None:
                raise RuntimeError(
                    f"'{penalty}' acts on the currency block, but the optimizer "
                    "returned asset weights only."
                )
            current = psi[keep]
        else:
            current = x

        d_inf = (
            float(np.max(np.abs(current - previous))) if previous is not None else np.nan
        )
        previous = current

        records.append({
            "penalty": penalty,
            "lambda": float(value),
            "block": block,
            "d_inf": d_inf,
            "n_active": int(np.sum(np.abs(current) > active_tol)),
            "gross_x": float(np.abs(x).sum()),
            "gross_psi": float(np.abs(psi).sum()) if psi is not None else np.nan,
            "solve_seconds": solve_seconds,
        })

        logger.debug(
            "%s = %.6g -> d_inf %.3e, n_active %d, %.2fs",
            penalty, value, d_inf, records[-1]["n_active"], solve_seconds,
        )

    return records


_CLAMP_WARNED: set[str] = set()


def _warn_on_clamped_penalties(
    optimizer: object, requested: dict, penalties: dict[str, float]
) -> None:
    """
    Warn once per penalty when an optimizer silently alters a requested value.

    Some optimizers floor their L2 penalties at a small positive constant, so the
    "all other penalties at zero" premise does not hold literally.  The effect is
    negligible but should not be left implicit in the figure.

    Args:
        optimizer: Freshly constructed optimizer.
        requested: Hyperparameters it was constructed with.
        penalties: Penalty names to check.
    """
    for name in penalties:
        stored = getattr(optimizer, name, None)
        if stored is None or name in _CLAMP_WARNED:
            continue
        if not np.isclose(stored, requested[name], rtol=0.0, atol=0.0):
            _CLAMP_WARNED.add(name)
            logger.warning(
                "%s clamps %s: requested %.3g, using %.3g.",
                type(optimizer).__name__, name, requested[name], stored,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marginal effect of each regularization penalty on one window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGY_REGISTRY),
        help="Strategy whose penalties are swept.",
    )
    parser.add_argument(
        "--universe",
        type=int,
        default=10,
        choices=sorted(config.UNIVERSE_FILES),
        help="Top-N stocks per economy to include in the investment universe.",
    )
    parser.add_argument(
        "--points-per-decade",
        type=int,
        default=8,
        help="Grid density. Identical across penalties, since d_inf depends on it.",
    )
    parser.add_argument(
        "--active-tol",
        type=float,
        default=1e-6,
        help="Magnitude above which a weight counts as an active position.",
    )
    parser.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "gurobi", "clarabel"],
        help="cvxpy solver. 'auto' prefers Gurobi and falls back to Clarabel.",
    )
    parser.add_argument("--universe-file", type=Path, default=None, help="Override universe CSV.")
    parser.add_argument("--asset-returns", type=Path, default=None, help="Override asset returns CSV.")
    parser.add_argument(
        "--excess-currency-returns", type=Path, default=None,
        help="Override excess currency returns CSV.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Override the output CSV path.")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    strategy_cfg = STRATEGY_REGISTRY[args.strategy]
    backtest_cfg = strategy_cfg["config"]
    optimizer_cls = strategy_cfg["optimizer"]
    asset_return_key = strategy_cfg["asset_return_key"]

    penalties, fixed = _split_hyperparams(backtest_cfg.param_grid)
    if not penalties:
        raise SystemExit(
            f"Strategy '{args.strategy}' has no '{PENALTY_PREFIX}*' hyperparameters in "
            "its grid, so there is nothing to sweep. Pick a regularized strategy."
        )

    solver = _resolve_solver(args.solver)

    # File paths, defaulting the way run_strategy.py resolves them.
    universe_path = args.universe_file or config.RAW_DIR / config.UNIVERSE_FILES[args.universe]
    default_returns = (
        config.FH_RETURNS_FILE if strategy_cfg["use_hedging"] else config.UNHEDGED_RETURNS_FILE
    )
    asset_returns_path = args.asset_returns or config.PROCESSED_DIR / default_returns
    excess_currency_path = (
        args.excess_currency_returns or config.PROCESSED_DIR / config.RC_RETURNS_FILE
    )

    logger.info("Strategy: %s (top-%d)", args.strategy, args.universe)
    logger.info(
        "Sweeping %d penalties: %s",
        len(penalties), ", ".join(f"{k} (bound {v:g})" for k, v in penalties.items()),
    )
    logger.info("Fixed: %s | solver: %s", fixed, solver)

    R_asset, R_c, asset_to_currency, universe_mask = _load_data(
        strategy_cfg, universe_path, asset_returns_path, excess_currency_path
    )

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

    logger.info(
        "Window: %s to %s (%d days), %d assets, %d currencies. Rebalancing date %s.",
        start.date(), end.date(), len(asset_window), len(valid_assets),
        len(currency_index), t.date(),
    )

    # The estimation bundle the backtester would hand the optimizer at this date,
    # including the asset-to-currency indicator no optimizer currently reads.
    C = pd.DataFrame(0.0, index=R_asset.columns, columns=currency_index)
    for asset, ccy in asset_to_currency.items():
        if asset in C.index and ccy in C.columns:
            C.loc[asset, ccy] = 1.0

    train_bundle: dict = {asset_return_key: asset_window, "C": C.loc[valid_assets].values}
    if currency_window is not None:
        train_bundle["R_c"] = currency_window

    records: list[dict] = []
    for penalty, bound in penalties.items():
        logger.info("Sweeping %s around bound %g ...", penalty, bound)
        records.extend(
            _sweep_penalty(
                penalty=penalty,
                bound=bound,
                optimizer_cls=optimizer_cls,
                train_bundle=train_bundle,
                fixed=fixed,
                penalties=penalties,
                currency_index=currency_index,
                solver=solver,
                points_per_decade=args.points_per_decade,
                active_tol=args.active_tol,
            )
        )

    results = pd.DataFrame.from_records(records)

    out_path = args.out or OUTPUT_DIR / f"{args.strategy}_top{args.universe}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    sidecar = {
        "strategy": args.strategy,
        "optimizer": optimizer_cls.__name__,
        "universe_size": args.universe,
        "universe_file": Path(universe_path).name,
        "asset_return_key": asset_return_key,
        "solver": str(solver),
        "rebalancing_date": str(t.date()),
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "n_observations": int(len(asset_window)),
        "n_assets": len(valid_assets),
        "n_currencies": len(currency_index),
        "window_years": backtest_cfg.window_years,
        "val_years": backtest_cfg.val_years,
        "rebalancing_frequency": backtest_cfg.rebalancing_frequency,
        "fixed_hyperparams": {k: float(v) for k, v in fixed.items()},
        "penalty_bounds": penalties,
        "points_per_decade": args.points_per_decade,
        "decades_below": DECADES_BELOW,
        "decades_above": DECADES_ABOVE,
        "active_tol": args.active_tol,
        "currency_excluded_from_norm": BASE_CURRENCY,
        "total_solve_seconds": float(results["solve_seconds"].sum()),
    }
    sidecar_path = out_path.with_suffix(".json")
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)

    logger.info(
        "%d solves in %.1fs -> %s (sidecar %s)",
        len(results), results["solve_seconds"].sum(), out_path, sidecar_path.name,
    )


if __name__ == "__main__":
    main()
