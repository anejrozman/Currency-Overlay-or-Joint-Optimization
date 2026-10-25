"""
Asset and currency turnover per rebalancing date (Appendix G.5).

Turnover is not stored in the run artifacts (metrics.json keeps only the average),
so it is rebuilt here from the saved weights and the recomputed return panel,
following Backtester.run: asset turnover is |x_t - drifted_x| and currency
turnover the traded forward notional |phi_t| excluding USD. As in the thesis
tables, asset turnover is reported at twice this value.

Without --optimizer, two figures are drawn, one for asset and one for currency
turnover, each with a row per risk measure and a column per universe, showing the
unregularized, regularized and mean-risk overlay and joint optimizers. With
--optimizer, the same two figures are drawn for that measure alone.

Usage:
    python code/analysis/generate_turnover_plot.py
    python code/analysis/generate_turnover_plot.py --optimizer es

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from portfolio_backtest import config
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (
    compute_daily_forward_premium,
    compute_fully_hedged_returns,
)

RESULTS_DIR: Path = Path(__file__).resolve().parents[2] / "results"
RUNS_DIR: Path = RESULTS_DIR / "runs"
OUT_DIR: Path = RESULTS_DIR / "figures" / "turnover"

# Thesis names and table order of the risk measures.
MEASURE_TITLES: dict[str, str] = {
    "mv": "Variance",
    "smv": "Semi-Variance",
    "mad": "MAD",
    "es": "ES",
    "soes": "SOES",
}

# (runs table, strategy template, legend label, colour, line style, line width). The
# overlay and joint programs keep the colours of the Appendix E runtime figures and the
# sparsity and drawdown figures; unregularized programs are dashed and the mean-risk
# programs dotted. Listed so that the legend fills one column per family.
PROGRAMS: tuple[tuple[str, str, str, str, object, float], ...] = (
    ("top_runs.csv", "{}_curr_overlay", "Overlay", "#eb6834", (0, (4, 2)), 0.7),
    ("top_runs.csv", "{}_joint", "Joint", "#2a78d6", (0, (4, 2)), 0.7),
    ("top_runs.csv", "reg_{}_curr_overlay", "Regularized Overlay", "#eda100", "-", 0.8),
    ("top_runs.csv", "reg_{}_joint", "Regularized Joint", "#e87ba4", "-", 0.8),
    ("top_runs_w_mean.csv", "{}_curr_overlay_w_mean", "Mean-Risk Overlay", "#5b3a99", (0, (1, 1.2)), 0.9),
    ("top_runs_w_mean.csv", "{}_joint_w_mean", "Mean-Risk Joint", "#0f6e6e", (0, (1, 1.2)), 0.9),
)

# Asset turnover is reported at twice the Backtester value, as in the thesis tables.
ASSET_SCALE: float = 2.0
BLOCKS: tuple[tuple[str, int, float], ...] = (("asset", 0, ASSET_SCALE), ("currency", 1, 1.0))

UNIVERSES: tuple[int, ...] = (10, 50, 100)

plt.rcParams["pdf.fonttype"] = 42

_RETURNS: dict[str, tuple[pd.DataFrame, dict[str, str]]] = {}
_TABLES: dict[str, pd.DataFrame] = {}


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


def _table(name: str) -> pd.DataFrame:
    if name not in _TABLES:
        table = pd.read_csv(RUNS_DIR / name, skipinitialspace=True)
        table.columns = [c.strip() for c in table.columns]
        table["strategy_name"] = table["strategy_name"].astype(str).str.strip()
        _TABLES[name] = table
    return _TABLES[name]


def _run_dir(runs_file: str, strategy: str, universe: int) -> Path | None:
    table = _table(runs_file)
    row = table[table["strategy_name"] == strategy]
    if row.empty or pd.isna(row[f"top_{universe}_run"].iloc[0]):
        return None
    return RUNS_DIR / strategy / f"run_{int(row[f'top_{universe}_run'].iloc[0])}"


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


def _panel(measure: str, universe: int) -> list[tuple[tuple[pd.Series, pd.Series], tuple]]:
    """The turnover series of one measure and universe, one entry per program with saved weights."""
    drawn = []
    for program in PROGRAMS:
        run_dir = _run_dir(program[0], program[1].format(measure), universe)
        series = None if run_dir is None else _turnover(run_dir)
        if series is not None:
            drawn.append((series, program))
    return drawn


def _style_axis(axis: plt.Axes, top: float) -> None:
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=7, length=2)
    axis.set_ylim(0.0, top)
    axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    axis.xaxis.set_major_locator(mdates.YearLocator(base=10))
    axis.xaxis.set_minor_locator(mdates.YearLocator(base=5))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _draw(axis: plt.Axes, panel: list, block: tuple[str, int, float], handles: dict[str, object]) -> None:
    _, index, scale = block
    for series, (_, _, label, colour, style, width) in panel:
        values = series[index] * scale
        line, = axis.plot(values.index, values.to_numpy(), color=colour, linestyle=style, linewidth=width)
        handles.setdefault(label, line)


def _legend(fig: plt.Figure, handles: dict[str, object]) -> None:
    labels = [label for _, _, label, _, _, _ in PROGRAMS if label in handles]
    # Filled column by column: unregularized, regularized and mean-risk programs.
    fig.legend([handles[label] for label in labels], labels, loc="lower center",
               ncol=3, fontsize=7.5, frameon=False, handlelength=2.4)


def _top(panels: list[list], block: tuple[str, int, float]) -> float:
    highest = max(float(series[block[1]].max()) * block[2] for panel in panels for series, _ in panel)
    return 1.05 * highest


def _plot_grid(block: tuple[str, int, float], panels: dict, out_dir: Path) -> None:
    """One figure with a row per risk measure and a column per universe."""
    measures = [m for m in MEASURE_TITLES if any(panels[(m, u)] for u in UNIVERSES)]
    fig, axes = plt.subplots(len(measures), len(UNIVERSES), figsize=(6.2, 1.64 * len(measures)),
                             sharex=True, sharey=True, squeeze=False)
    handles: dict[str, object] = {}
    top = _top([panels[(m, u)] for m in measures for u in UNIVERSES if panels[(m, u)]], block)
    for row, measure in enumerate(measures):
        for column, universe in enumerate(UNIVERSES):
            axis = axes[row, column]
            _draw(axis, panels[(measure, universe)], block, handles)
            _style_axis(axis, top)
            if row == 0:
                axis.set_title(f"top-{universe}", fontsize=8.5)
            if column == 0:
                axis.set_ylabel(MEASURE_TITLES[measure], fontsize=8.5)
    _legend(fig, handles)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{block[0]}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_measure(measure: str, block: tuple[str, int, float], panels: dict, out_dir: Path) -> None:
    """One figure for a single risk measure with a panel per universe."""
    fig, axes = plt.subplots(1, len(UNIVERSES), figsize=(6.2, 2.4), sharex=True, sharey=True)
    handles: dict[str, object] = {}
    top = _top([panels[(measure, u)] for u in UNIVERSES if panels[(measure, u)]], block)
    for axis, universe in zip(axes, UNIVERSES):
        _draw(axis, panels[(measure, universe)], block, handles)
        _style_axis(axis, top)
        axis.set_title(f"top-{universe}", fontsize=8.5)
    axes[0].set_ylabel(MEASURE_TITLES[measure], fontsize=8.5)
    _legend(fig, handles)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{measure}_{block[0]}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=list(MEASURE_TITLES),
                        help="Draw one measure alone; without it, the grid over all measures.")
    args = parser.parse_args()

    measures = list(MEASURE_TITLES) if args.optimizer is None else [args.optimizer]
    panels = {(m, u): _panel(m, u) for m in measures for u in UNIVERSES}
    if not any(panels.values()):
        raise SystemExit("No runs with saved weights recorded.")
    for block in BLOCKS:
        if args.optimizer is None:
            _plot_grid(block, panels, OUT_DIR)
        else:
            _plot_measure(args.optimizer, block, panels, OUT_DIR)


if __name__ == "__main__":
    main()
