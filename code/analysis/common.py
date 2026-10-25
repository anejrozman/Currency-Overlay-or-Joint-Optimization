"""Paths, run lookups and plot styling shared by the analysis scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RUNS_DIR = RESULTS_DIR / "runs"
RF_PATH = RESULTS_DIR.parent / "data" / "raw" / "us_risk_free_rates.csv"
TRADING_DAYS = 252
UNIVERSES = (10, 50, 100)
MEASURES = {"mv": "Variance", "smv": "Semi-Variance", "mad": "MAD", "es": "ES", "soes": "SOES"}


def top_runs(name: str) -> pd.DataFrame:
    """A top_runs*.csv table indexed by strategy name."""
    table = pd.read_csv(RUNS_DIR / name, skipinitialspace=True)
    table.columns = table.columns.str.strip()
    return table.set_index(table["strategy_name"].astype(str).str.strip())


def run_dir(table: pd.DataFrame, strategy: str, universe: int) -> Path | None:
    if strategy not in table.index or pd.isna(table.at[strategy, f"top_{universe}_run"]):
        return None
    return RUNS_DIR / strategy / f"run_{int(table.at[strategy, f'top_{universe}_run'])}"


def oos_returns(table: pd.DataFrame, strategy: str, universe: int) -> pd.Series:
    path = run_dir(table, strategy, universe) / "oos_returns.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)["return"].dropna()


def risk_free() -> pd.Series:
    raw = pd.read_csv(RF_PATH, index_col=0)
    return raw["RF"].set_axis(pd.to_datetime(raw.index.astype(str), format="%Y%m%d")) / 100.0


def excess_return(r: pd.Series, rf: pd.Series) -> float:
    """CAGR minus 252 times the mean daily risk-free rate, as in the Backtester."""
    cagr = (1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0
    return cagr - TRADING_DAYS * float(rf.reindex(r.index).ffill().mean())


def style_time_axis(axis) -> None:
    """Grid, spines and decade ticks of the per-date panels of Appendix G."""
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=7, length=2)
    axis.xaxis.set_major_locator(mdates.YearLocator(base=10))
    axis.xaxis.set_minor_locator(mdates.YearLocator(base=5))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def grid_legend(fig, handles: dict, order: list[str], ncol: int) -> None:
    labels = [label for label in order if label in handles]
    fig.legend([handles[label] for label in labels], labels, loc="lower center",
               ncol=ncol, fontsize=7.5, frameon=False, handlelength=2.4)


def universe_mask(universe_file, index: pd.DatetimeIndex) -> pd.DataFrame:
    """1.0 where a stock is a constituent, NaN otherwise, forward-filled from the reconstitution dates.

    Built with a groupby rather than DataHandler's pivot_table, which duplicates index labels on the
    top-100 universe file.
    """
    universe = pd.read_csv(universe_file)
    universe["ric"] = universe["ric_code"].astype(str).str.replace("<", "").str.replace(">", "")
    universe["date"] = pd.to_datetime(universe["date"])
    constituents = universe.groupby("date")["ric"].agg(set)
    mask = pd.DataFrame(float("nan"), index=constituents.index, columns=sorted(universe["ric"].unique()))
    for date, rics in constituents.items():
        mask.loc[date, list(rics)] = 1.0
    return mask.reindex(index, method="ffill")
