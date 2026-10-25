"""Run one backtest strategy from strategy_registry.py and save its artifacts to a new results/runs directory.

--alpha and --no-demean pin one axis of the strategy's parameter grid, as the level sweep does. The
"Strategy:", "Universe:", "Config saved" and "OOS returns saved" log lines are parsed by generate_runtime_plots.py.
Set BACKTEST_SCRATCH_DIR to keep the intermediate return files of concurrent jobs apart.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from portfolio_backtest import config
from portfolio_backtest.backtest.backtester import Backtester
from portfolio_backtest.backtest.data_handler import DataHandler
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv, next_run_dir
from portfolio_backtest.utils.returns import (compute_daily_forward_premium, compute_excess_currency_returns,
                                              compute_fully_hedged_returns, compute_unhedged_returns)

log = logging.getLogger(__name__)
OVERRIDE_FLAGS = {"alpha": "--alpha", "demean": "--no-demean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-currency portfolio backtest strategy.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--strategy", required=True, choices=list(STRATEGY_REGISTRY))
    parser.add_argument("--universe", type=int, default=10, choices=sorted(config.UNIVERSE_FILES),
                        help="Top-N stocks per economy.")
    parser.add_argument("--window-years", type=int, default=None, help="Override the estimation window.")
    parser.add_argument("--val-years", type=int, default=None, help="Override the validation sub-window.")
    parser.add_argument("--no-grid", action="store_true", help="Disable the grid search (optimizer defaults).")
    parser.add_argument("--alpha", type=float, default=None, help="Pin the ES confidence level of the grid.")
    parser.add_argument("--no-demean", action="store_true", help="Pin demean to False in the grid.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def grid_with_overrides(args, param_grid: dict | None) -> tuple[dict | None, dict]:
    """Pin the overridden grid axes; naming a key the grid lacks is an error, since the optimizers would
    silently fall back to their defaults."""
    overrides = {}
    if args.alpha is not None:
        if not 0.0 < args.alpha < 1.0:
            raise SystemExit(f"--alpha must be strictly in (0, 1), got {args.alpha}.")
        overrides["alpha"] = args.alpha
    if args.no_demean:
        overrides["demean"] = False
    if not overrides:
        return param_grid, overrides
    if param_grid is None:
        raise SystemExit(f"Cannot apply {', '.join(OVERRIDE_FLAGS[k] for k in overrides)}: strategy "
                         f"'{args.strategy}' has no parameter grid (it is empty or --no-grid was passed).")
    param_grid = dict(param_grid)
    for key, value in overrides.items():
        if key not in param_grid:
            raise SystemExit(f"Strategy '{args.strategy}' has no '{key}' in its parameter grid, so "
                             f"{OVERRIDE_FLAGS[key]} would have no effect. Its grid keys are: {sorted(param_grid)}.")
        param_grid[key] = [value]
    return param_grid, overrides


def save_solver_fallbacks(optimizer_cls, run_dir: Path) -> None:
    events = getattr(optimizer_cls, "solver_fallbacks", None)
    if events is None:
        return
    if not events:
        log.info("Solver fallbacks: none.")
        return
    path = run_dir / "solver_fallbacks.csv"
    pd.DataFrame(events).to_csv(path, index=False)
    accepted = pd.Series([event["accepted"] for event in events]).value_counts().to_dict()
    log.warning("Solver fallbacks: %d solves needed one (accepted: %s). Details saved to %s", len(events),
                accepted, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), datefmt="%Y-%m-%d %H:%M:%S",
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    strategy = STRATEGY_REGISTRY[args.strategy]
    optimizer_cls = strategy["optimizer"]
    asset_return_key = strategy["asset_return_key"]
    training_return_key = strategy.get("training_return_key")
    use_hedging = strategy["use_hedging"]
    label = f"{strategy['label']} (top-{args.universe})"
    base = strategy["config"]
    window_years = args.window_years if args.window_years is not None else base.window_years
    val_years = args.val_years if args.val_years is not None else base.val_years
    param_grid, overrides = grid_with_overrides(args, None if args.no_grid else (base.param_grid or None))

    universe_path = config.RAW_DIR / config.UNIVERSE_FILES[args.universe]
    scratch = os.environ.get("BACKTEST_SCRATCH_DIR")
    scratch_dir = Path(scratch) if scratch else config.PROCESSED_DIR
    scratch_dir.mkdir(parents=True, exist_ok=True)

    run_dir = next_run_dir(optimizer_cls)
    log.info("Strategy: %s", args.strategy)
    log.info("Universe: top-%d per economy", args.universe)
    for key, value in overrides.items():
        log.info("Grid override: %s = [%r]", key, value)
    log.info("Run directory: %s", run_dir)

    # returns are written to disk and read back through the DataHandler, which builds the universe mask
    local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    spot = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    local = local.where(local.abs() < config.RETURN_FILTER_BACKTEST)
    atc = build_asset_to_currency(universe_path)
    premium = compute_daily_forward_premium(load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE),
                                            config.TENOR_DAYS_3M)
    R_train = None
    if use_hedging:
        asset_path, currency_path = scratch_dir / config.FH_RETURNS_FILE, scratch_dir / config.RC_RETURNS_FILE
        compute_fully_hedged_returns(local, spot, premium, atc).rename_axis("date").to_csv(asset_path)
        compute_excess_currency_returns(spot, premium).rename_axis("date").to_csv(currency_path)
    else:
        # the DataHandler needs a currency file, so the unhedged returns stand in for it
        asset_path = currency_path = scratch_dir / config.UNHEDGED_RETURNS_FILE
        compute_unhedged_returns(local, spot, premium, atc).rename_axis("date").to_csv(asset_path)
        if training_return_key == "R_fh":  # optimized on hedged, realized unhedged (ASSETS-ZERO)
            R_train = compute_fully_hedged_returns(local, spot, premium, atc)

    rf = pd.read_csv(config.RAW_DIR / config.RF_FILE, index_col=0, parse_dates=True)
    rf = rf["RF"].set_axis(pd.to_datetime(rf.index, format="%Y%m%d")) / 100.0

    R_asset, atc, R_c, mask = DataHandler(universe_path, asset_path, currency_path).load_and_process()
    if use_hedging:
        common = R_asset.index.intersection(R_c.index)
        R_asset, R_c, mask = R_asset.loc[common], R_c.loc[common], mask.reindex(common, method="ffill")
    else:
        R_c, mask = None, mask.reindex(R_asset.index, method="ffill")

    data = {asset_return_key: R_asset, "asset_to_currency": atc, "universe_mask": mask}
    if R_c is not None:
        data["R_c"] = R_c
        # forward premiums for the carry-conditioned currency mean of the *_w_mean optimizers
        data["F"] = premium.reindex(index=R_c.index, columns=R_c.columns)
    if R_train is not None:
        data[training_return_key] = R_train.reindex(index=R_asset.index, columns=R_asset.columns)

    run_config = {
        "strategy": args.strategy, "label": label, "run_date": datetime.now().isoformat(),
        "optimizer": optimizer_cls.__name__, "param_grid": param_grid, "grid_overrides": overrides or None,
        "window_years": window_years, "val_years": val_years, "rebalancing_frequency": base.rebalancing_frequency,
        "asset_return_key": asset_return_key, "training_return_key": training_return_key,
        "universe_file": universe_path.name, "universe_size": args.universe,
        "return_filter": config.RETURN_FILTER_BACKTEST, "rf_source": config.RF_FILE,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    log.info("Config saved to %s", run_dir / "config.json")

    backtester = Backtester(optimizer=optimizer_cls, param_grid=param_grid, window_years=window_years,
                            val_years=val_years, rebalancing_frequency=base.rebalancing_frequency,
                            asset_return_key=asset_return_key, training_return_key=training_return_key, logging=True)
    try:
        oos_returns = backtester.run(data)
    finally:
        save_solver_fallbacks(optimizer_cls, run_dir)  # also when the run fails

    oos_returns.rename("return").to_frame().rename_axis("date").to_csv(run_dir / "oos_returns.csv")
    log.info("OOS returns saved to %s", run_dir / "oos_returns.csv")
    pd.DataFrame(backtester._selected_params).to_csv(run_dir / "selected_params.csv", index=False)
    log.info("Selected params saved to %s", run_dir / "selected_params.csv")
    asset_weights, currency_weights = backtester.get_weight_records()
    asset_weights.to_csv(run_dir / "weights_assets.csv", index=False)
    currency_weights.to_csv(run_dir / "weights_currencies.csv", index=False)
    log.info("Weights saved to %s (%d asset rows) and %s (%d currency rows)", run_dir / "weights_assets.csv",
             len(asset_weights), run_dir / "weights_currencies.csv", len(currency_weights))
    metrics = backtester.compute_performance_metrics(oos_returns, label=label, rf_series=rf)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Metrics saved to %s", run_dir / "metrics.json")

    plt.figure(figsize=(10, 6))
    (1 + oos_returns).cumprod().plot(label=label)
    plt.title(f"Cumulative Returns — {label}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "cumulative_returns_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Plot saved to %s", run_dir / "cumulative_returns_plot.png")
    log.info("All run artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
