"""
Unified strategy runner for the multi-currency portfolio backtest framework.

Arguments: 

    --strategy: Which strategy to run (required, must be defined in strategy_registry.py)

    --universe: Top-N stocks per economy to include in the investment universe (optional,default: 10)

    --window-years: Override estimation window in years (optional, default: strategy config)

    --val-years: Override validation sub-window in years (optional, default: strategy config)

    --no-grid: Disable hyperparameter grid search (optional, default: strategy config)

    --log-level: Logging verbosity (optional, default: INFO)

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from pathlib import Path

from portfolio_backtest import config
from portfolio_backtest.backtest_config import BacktestConfig
from portfolio_backtest.backtest.backtester import Backtester
from portfolio_backtest.backtest.data_handler import DataHandler
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv, next_run_dir
from portfolio_backtest.utils.returns import (
    compute_daily_forward_premium,
    compute_excess_currency_returns,
    compute_fully_hedged_returns,
    compute_unhedged_returns,
)



# Argument parsing
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-currency portfolio backtest strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGY_REGISTRY),
        help="Strategy to run.",
    )
    parser.add_argument(
        "--universe",
        type=int,
        default=10,
        choices=sorted(config.UNIVERSE_FILES),
        help="Top-N stocks per economy to include in the investment universe.",
    )
    parser.add_argument(
        "--window-years",
        type=int,
        default=None,
        help="Override estimation window in years (default: strategy config).",
    )
    parser.add_argument(
        "--val-years",
        type=int,
        default=None,
        help="Override validation sub-window in years (default: strategy config).",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable hyperparameter grid search (use optimizer defaults).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None: 
    args = _parse_args()
    _configure_logging(args.log_level)
    log = logging.getLogger(__name__)

    strategy_cfg = STRATEGY_REGISTRY[args.strategy]
    optimizer_cls = strategy_cfg["optimizer"]
    asset_return_key: str = strategy_cfg["asset_return_key"]
    use_hedging: bool = strategy_cfg["use_hedging"]

    # Compose label with universe size at runtime
    label: str = f"{strategy_cfg['label']} (top-{args.universe})"

    # Build BacktestConfig: start with strategy default, apply CLI overrides
    base_cfg: BacktestConfig = strategy_cfg["config"]
    window_years: int = args.window_years if args.window_years is not None else base_cfg.window_years
    val_years: int = args.val_years if args.val_years is not None else base_cfg.val_years
    rebalancing_frequency: str = base_cfg.rebalancing_frequency
    param_grid: dict | None = None if args.no_grid else (base_cfg.param_grid or None)

    # File paths
    universe_path = config.RAW_DIR / config.UNIVERSE_FILES[args.universe]
    local_returns_path = config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE
    spot_returns_path = config.PROCESSED_DIR / config.SPOT_RETURNS_FILE
    fwd_premiums_path = config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE
    fh_returns_path = config.PROCESSED_DIR / config.FH_RETURNS_FILE
    rc_returns_path = config.PROCESSED_DIR / config.RC_RETURNS_FILE
    unhedged_returns_path = config.PROCESSED_DIR / config.UNHEDGED_RETURNS_FILE

    run_dir = next_run_dir(optimizer_cls)
    log.info("Strategy: %s", args.strategy)
    log.info("Universe: top-%d per economy", args.universe)
    log.info("Run directory: %s", run_dir)

    #Load processed data 
    R_local = load_csv(local_returns_path)
    e = load_csv(spot_returns_path)
    R_local = R_local.where(R_local.abs() < config.RETURN_FILTER_BACKTEST)

    # Build asset to currency mapping
    asset_to_currency = build_asset_to_currency(universe_path)

    # Compute returns 
    if use_hedging:
        f_raw = load_csv(fwd_premiums_path)
        f_daily_shifted = compute_daily_forward_premium(f_raw, config.TENOR_DAYS_3M)
        R_c = compute_excess_currency_returns(e, f_daily_shifted)
        R_asset = compute_fully_hedged_returns(R_local, e, f_daily_shifted, asset_to_currency)

        # Persist so DataHandler can load them back
        R_asset.index.name = "date"
        R_c.index.name = "date"
        R_asset.to_csv(fh_returns_path)
        R_c.to_csv(rc_returns_path)

        asset_returns_path = fh_returns_path
        excess_currency_path = rc_returns_path
    else:
        R_asset = compute_unhedged_returns(R_local, e, asset_to_currency)
        R_asset.index.name = "date"
        R_asset.to_csv(unhedged_returns_path)
        R_c = None

        # DataHandler needs a file for excess_currency_returns
        asset_returns_path = unhedged_returns_path
        excess_currency_path = unhedged_returns_path

    # Load risk free rate
    rf_path = config.RAW_DIR / config.RF_FILE
    rf_raw = pd.read_csv(rf_path, index_col=0, parse_dates=True)
    rf_raw.index = pd.to_datetime(rf_raw.index, format="%Y%m%d")
    rf_series: pd.Series = rf_raw["RF"] / 100.0 

    # Load via DataHandler build universe_mask 
    handler = DataHandler(
        universe_file=universe_path,
        hedged_returns_file=asset_returns_path,
        excess_currency_returns_file=excess_currency_path,
    )
    R_asset_filtered, asset_to_currency, R_c_filtered, universe_mask = handler.load_and_process()

    if use_hedging:
        common = R_asset_filtered.index.intersection(R_c_filtered.index)
        R_asset_filtered = R_asset_filtered.loc[common]
        R_c_filtered = R_c_filtered.loc[common]
        universe_mask = universe_mask.reindex(common, method="ffill")
    else:
        universe_mask = universe_mask.reindex(R_asset_filtered.index, method="ffill")
        R_c_filtered = None

    # Assemble data_bundle
    data_bundle: dict = {
        asset_return_key: R_asset_filtered,
        "asset_to_currency": asset_to_currency,
        "universe_mask": universe_mask,
    }
    if R_c_filtered is not None:
        data_bundle["R_c"] = R_c_filtered

    # Save config before running 
    run_config = {
        "strategy": args.strategy,
        "label": label,
        "run_date": datetime.now().isoformat(),
        "optimizer": optimizer_cls.__name__,
        "param_grid": param_grid,
        "window_years": window_years,
        "val_years": val_years,
        "rebalancing_frequency": rebalancing_frequency,
        "asset_return_key": asset_return_key,
        "universe_file": universe_path.name,
        "universe_size": args.universe,
        "return_filter": config.RETURN_FILTER_BACKTEST,
        "rf_source": config.RF_FILE,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    log.info("Config saved to %s", run_dir / "config.json")

    # Run backtest
    backtester = Backtester(
        optimizer=optimizer_cls,
        param_grid=param_grid,
        window_years=window_years,
        val_years=val_years,
        rebalancing_frequency=rebalancing_frequency,
        asset_return_key=asset_return_key,
        logging=True,
    )
    oos_returns = backtester.run(data_bundle)

    # Save OOS returns
    oos_df = oos_returns.rename("return").to_frame()
    oos_df.index.name = "date"
    oos_df.to_csv(run_dir / "oos_returns.csv")
    log.info("OOS returns saved to %s", run_dir / "oos_returns.csv")

    # Performance metrics
    metrics = backtester.compute_performance_metrics(
        oos_returns,
        label=label,
        rf_series=rf_series,
    )
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s", run_dir / "metrics.json")

    #save plot of cumulative returns
    cumulative_returns = (1 + oos_returns).cumprod()
    plt.figure(figsize=(10, 6))
    cumulative_returns.plot(label=label)
    plt.title(f"Cumulative Returns — {label}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = run_dir / "cumulative_returns_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Plot saved to %s", plot_path)
    log.info("All run artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
