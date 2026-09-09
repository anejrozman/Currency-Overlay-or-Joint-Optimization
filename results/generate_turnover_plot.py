"""
Asset and currency turnover per rebalancing date, one column per universe size.

Turnover is not stored in the run artifacts (metrics.json keeps only the average),
so it is rebuilt here from the saved weights and the recomputed return panel,
following Backtester.run: asset turnover is |x_t - drifted_x| and currency
turnover the traded forward notional |phi_t| excluding USD.

Usage:
    python results/generate_turnover_plot.py --optimizer-family normal --optimizer es
    python results/generate_turnover_plot.py --optimizer-family demeaned --optimizer soes
    python results/generate_turnover_plot.py --optimizer-family with-mean-term --optimizer mv

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from portfolio_backtest import config
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (
    compute_daily_forward_premium,
    compute_fully_hedged_returns,
)

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
OUT_DIR: Path = RESULTS_DIR / "figures" / "turnover"

FAMILIES: dict[str, tuple[str, str, str]] = {
    "normal": ("top_runs.csv", "", "normal"),
    "demeaned": ("top_demeaned_runs.csv", "", "demeaned"),
    "with-mean-term": ("top_runs_w_mean.csv", "_w_mean", "with_mean_term"),
}

MEASURE_TITLES: dict[str, str] = {
    "es": "Expected Shortfall",
    "soes": "Second-Order Expected Shortfall",
    "mv": "Mean-Variance",
    "smv": "Semi-Variance",
    "mad": "Mean Absolute Deviation",
}

UNIVERSES: tuple[int, ...] = (10, 50, 100)

_RETURNS: dict[str, tuple[pd.DataFrame, dict[str, str]]] = {}


def _returns(universe_file: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if universe_file not in _RETURNS:
        local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
        local = local.where(local.abs() < config.RETURN_FILTER_BACKTEST)
        spot = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
        premium = compute_daily_forward_premium(
            load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE), config.TENOR_DAYS_3M
        )
        atc = build_asset_to_currency(config.RAW_DIR / universe_file)
        returns = compute_fully_hedged_returns(local, spot, premium, atc)
        _RETURNS[universe_file] = (returns[[c for c in returns.columns if c in atc]], atc)
    return _RETURNS[universe_file]


def _strategies(measure: str, suffix: str, family: str) -> list[tuple[str, str]]:
    entries = [("equal_weight_optimizer", "Equal weight")]
    if family == "normal":
        entries.append((f"reg_{measure}_asset_full", "Assets-full"))
    for role, label in (("curr_overlay", "Overlay"), ("joint", "Joint")):
        entries.append((f"{measure}_{role}{suffix}", label))
        entries.append((f"reg_{measure}_{role}{suffix}", f"Regularized {label.lower()}"))
    return entries


def _turnover(run_dir: Path) -> tuple[pd.Series, pd.Series] | None:
    if not (run_dir / "weights_assets.csv").exists():
        return None
    universe_file = json.loads((run_dir / "config.json").read_text())["universe_file"]
    returns, atc = _returns(universe_file)

    assets = pd.read_csv(run_dir / "weights_assets.csv", parse_dates=["date"])
    overlay = pd.read_csv(run_dir / "weights_currencies.csv", parse_dates=["date"])
    psi = {t: g.set_index("currency")["psi"] for t, g in overlay.groupby("date")}

    asset_turnover, currency_turnover, dates = [], [], []
    previous, previous_date = None, None
    for date, group in assets.groupby("date"):
        x = group.set_index("asset")["weight"]
        exposure = x.groupby(x.index.map(atc)).sum()
        phi = exposure.sub(psi[date], fill_value=0.0).drop("USD", errors="ignore")
        if previous is not None:
            start = returns.index.get_loc(previous_date) + 2
            end = returns.index.get_loc(date) + 2
            growth = (1.0 + returns.iloc[start:end][previous.index].fillna(0.0)).cumprod().iloc[-1]
            drifted = growth * previous
            drifted = drifted / drifted.sum()
            asset_turnover.append(float(x.sub(drifted, fill_value=0.0).abs().sum()))
            currency_turnover.append(float(phi.abs().sum()))
            dates.append(date)
        previous, previous_date = x, date

    return pd.Series(asset_turnover, index=dates), pd.Series(currency_turnover, index=dates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-family", required=True, choices=list(FAMILIES))
    parser.add_argument("--optimizer", required=True, choices=list(MEASURE_TITLES))
    args = parser.parse_args()

    runs_file, suffix, out_name = FAMILIES[args.optimizer_family]
    table = pd.read_csv(RUNS_DIR / runs_file, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()

    fig, axes = plt.subplots(2, len(UNIVERSES), figsize=(15, 7), sharex=True)
    handles: dict[str, object] = {}
    plotted = False

    for column, universe in enumerate(UNIVERSES):
        for strategy, label in _strategies(args.optimizer, suffix, args.optimizer_family):
            row = table[table["strategy_name"] == strategy]
            if row.empty or pd.isna(row[f"top_{universe}_run"].iloc[0]):
                continue
            run = int(row[f"top_{universe}_run"].iloc[0])
            series = _turnover(RUNS_DIR / strategy / f"run_{run}")
            if series is None:
                continue
            for axis, values in zip(axes[:, column], series):
                line, = axis.plot(values.index, values.to_numpy(), linewidth=0.9, label=label)
            handles.setdefault(label, line)
            if strategy != "equal_weight_optimizer":
                plotted = True
        axes[0, column].set_title(f"top-{universe}")
        for axis in axes[:, column]:
            axis.grid(alpha=0.3)

    if not plotted:
        raise SystemExit(f"No {args.optimizer} runs with saved weights recorded in {runs_file}.")

    axes[0, 0].set_ylabel("Asset turnover")
    axes[1, 0].set_ylabel("Currency turnover")
    fig.suptitle(f"Turnover series, {MEASURE_TITLES[args.optimizer]}")
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    out_dir = OUT_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{args.optimizer}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
