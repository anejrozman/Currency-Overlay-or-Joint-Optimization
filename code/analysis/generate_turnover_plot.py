"""Asset and currency turnover per rebalancing date, rebuilt from the saved weights (Appendix G.5).

Follows Backtester.run: asset turnover is |x_t - drifted x_{t-1}|, reported at twice that value as in the
thesis tables, and currency turnover is |phi_t| excluding USD. With --optimizer, one measure alone.
"""

from __future__ import annotations

import argparse
import json
from functools import cache

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from common import MEASURES, RESULTS_DIR, UNIVERSES, grid_legend, run_dir, style_time_axis, top_runs
from portfolio_backtest import config
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import compute_daily_forward_premium, compute_fully_hedged_returns

OUT_DIR = RESULTS_DIR / "figures" / "turnover"
DASHED, DOTTED = (0, (4, 2)), (0, (1, 1.2))
PROGRAMS = (  # runs table, strategy template, label, colour, line style, line width
    ("top_runs.csv", "{}_curr_overlay", "Overlay", "#eb6834", DASHED, 0.7),
    ("top_runs.csv", "{}_joint", "Joint", "#2a78d6", DASHED, 0.7),
    ("top_runs.csv", "reg_{}_curr_overlay", "Regularized Overlay", "#eda100", "-", 0.8),
    ("top_runs.csv", "reg_{}_joint", "Regularized Joint", "#e87ba4", "-", 0.8),
    ("top_runs_w_mean.csv", "{}_curr_overlay_w_mean", "Mean-Risk Overlay", "#5b3a99", DOTTED, 0.9),
    ("top_runs_w_mean.csv", "{}_joint_w_mean", "Mean-Risk Joint", "#0f6e6e", DOTTED, 0.9),
)
BLOCKS = (("asset", 0, 2.0), ("currency", 1, 1.0))  # name, series index, scale
plt.rcParams["pdf.fonttype"] = 42
table = cache(top_runs)


@cache
def hedged_returns(universe_file: str) -> tuple[pd.DataFrame, dict[str, str]]:
    local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    local = local.where(local.abs() < config.RETURN_FILTER_BACKTEST)
    spot = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    premium = compute_daily_forward_premium(load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE),
                                            config.TENOR_DAYS_3M)
    atc = build_asset_to_currency(config.RAW_DIR / universe_file)
    returns = compute_fully_hedged_returns(local, spot, premium, atc)
    return returns[[c for c in returns.columns if c in atc]], atc


def turnover(path) -> tuple[pd.Series, pd.Series] | None:
    if not (path / "weights_assets.csv").exists():
        return None
    returns, atc = hedged_returns(json.loads((path / "config.json").read_text())["universe_file"])
    overlay = pd.read_csv(path / "weights_currencies.csv", parse_dates=["date"])
    psi = {t: g.set_index("currency")["psi"] for t, g in overlay.groupby("date")}
    assets, currencies, dates = [], [], []
    previous = previous_date = None
    for date, group in pd.read_csv(path / "weights_assets.csv", parse_dates=["date"]).groupby("date"):
        x = group.set_index("asset")["weight"]
        phi = x.groupby(x.index.map(atc)).sum().sub(psi[date], fill_value=0.0).drop("USD", errors="ignore")
        if previous is not None:
            start, end = returns.index.get_loc(previous_date) + 2, returns.index.get_loc(date) + 2
            drifted = (1.0 + returns.iloc[start:end][previous.index].fillna(0.0)).cumprod().iloc[-1] * previous
            assets.append(float(x.sub(drifted / drifted.sum(), fill_value=0.0).abs().sum()))
            currencies.append(float(phi.abs().sum()))
            dates.append(date)
        previous, previous_date = x, date
    return pd.Series(assets, index=dates), pd.Series(currencies, index=dates)


def panel(measure: str, universe: int) -> list:
    drawn = []
    for program in PROGRAMS:
        path = run_dir(table(program[0]), program[1].format(measure), universe)
        if path is not None and (series := turnover(path)) is not None:
            drawn.append((series, program))
    return drawn


def plot(measures: list[str], block: tuple, panels: dict, single: bool) -> None:
    name, index, scale = block
    fig, axes = plt.subplots(len(measures), len(UNIVERSES), sharex=True, sharey=True, squeeze=False,
                             figsize=(6.2, 2.4 if single else 1.64 * len(measures)))
    top = 1.05 * max(float(s[index].max()) * scale for drawn in panels.values() for s, _ in drawn)
    handles = {}
    for row, measure in enumerate(measures):
        for column, universe in enumerate(UNIVERSES):
            axis = axes[row, column]
            for series, (_, _, label, colour, style, width) in panels[measure, universe]:
                values = series[index] * scale
                handles.setdefault(label, axis.plot(values.index, values.to_numpy(), color=colour,
                                                    linestyle=style, linewidth=width)[0])
            style_time_axis(axis)
            axis.set_ylim(0.0, top)
            axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
            if row == 0:
                axis.set_title(f"top-{universe}", fontsize=8.5)
            if column == 0:
                axis.set_ylabel(MEASURES[measure], fontsize=8.5)
    grid_legend(fig, handles, [p[2] for p in PROGRAMS], ncol=3)
    if single:
        fig.tight_layout(rect=(0, 0.2, 1, 1))
    else:
        fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / (f"{measures[0]}_{name}.pdf" if single else f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=list(MEASURES), help="Draw one measure alone.")
    args = parser.parse_args()
    measures = list(MEASURES) if args.optimizer is None else [args.optimizer]
    panels = {(m, u): panel(m, u) for m in measures for u in UNIVERSES}
    if not any(panels.values()):
        raise SystemExit("No runs with saved weights recorded.")
    if args.optimizer is None:
        measures = [m for m in measures if any(panels[m, u] for u in UNIVERSES)]
    for block in BLOCKS:
        plot(measures, block, panels, single=args.optimizer is not None)
