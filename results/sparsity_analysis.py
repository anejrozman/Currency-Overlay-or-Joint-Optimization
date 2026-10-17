"""
Portfolio sparsity per rebalancing date (Appendix G.2).

The share of the investable universe in which the joint and currency overlay
optimizers, with and without regularization, hold a position: the assets whose
absolute weight exceeds TOLERANCE, and the foreign currencies (the US dollar
excluded) whose absolute currency exposure exceeds it. On the asset figures, an
up-triangle marks the rebalancing dates whose selected L1 asset penalty is one of
the two largest values of its grid, and a down-triangle does the same for the L2
asset penalty.

Without --optimizer, one figure per block (assets, currencies) is drawn with a
row per risk measure and a column per universe, as used in the thesis. With
--optimizer, one figure per block is drawn for that measure alone.

Usage:
    python results/sparsity_analysis.py --optimizer-family normal
    python results/sparsity_analysis.py --optimizer-family normal --optimizer es

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
from matplotlib.lines import Line2D

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
OUT_DIR: Path = RESULTS_DIR / "figures" / "sparsity_analysis"

FAMILIES: dict[str, tuple[str, str, str]] = {
    "normal": ("top_runs.csv", "", "normal"),
    "demeaned": ("top_demeaned_runs.csv", "", "demeaned"),
    "with-mean-term": ("top_runs_w_mean.csv", "_w_mean", "with_mean_term"),
}

# Thesis names and table order of the risk measures.
MEASURE_TITLES: dict[str, str] = {
    "mv": "Variance",
    "smv": "Semi-Variance",
    "mad": "MAD",
    "es": "ES",
    "soes": "SOES",
}

BLOCKS: tuple[tuple[str, str, str, str, bool], ...] = (
    ("assets", "weights_assets.csv", "asset", "weight", False),
    ("currencies", "weights_currencies.csv", "currency", "psi", True),
)

# (strategy role, regularized, legend label, colour, line style). The colours are the
# program slots of the palette used for the Appendix E runtime figures, so that a
# program keeps its colour across appendices; unregularized programs are dashed.
PROGRAMS: tuple[tuple[str, bool, str, str, object], ...] = (
    ("curr_overlay", False, "Overlay", "#eb6834", (0, (4, 2))),
    ("curr_overlay", True, "Regularized Overlay", "#eda100", "-"),
    ("joint", False, "Joint", "#2a78d6", (0, (4, 2))),
    ("joint", True, "Regularized Joint", "#e87ba4", "-"),
)

# Selected asset penalties marked on the regularized asset lines. The marker takes the
# colour of its line; the shape identifies the penalty.
PENALTIES: tuple[tuple[str, str, str], ...] = (
    ("lambda_l1_asset", "^", r"Large selected $L^1$ penalty"),
    ("lambda_l2_asset", "v", r"Large selected $L^2$ penalty"),
)
MARKER_EDGE: str = "#3d3d3a"
MARKER_LEGEND_FACE: str = "#8a8983"

UNIVERSES: tuple[int, ...] = (10, 50, 100)
# A position counts as held from 0.01% of the portfolio on. A numerical-zero threshold
# (1e-6) would also count the many tiny weights of the quadratic programs.
TOLERANCE: float = 1e-4
BASE_CURRENCY: str = "USD"

plt.rcParams["pdf.fonttype"] = 42


def _run_dir(table: pd.DataFrame, strategy: str, universe: int) -> Path | None:
    row = table[table["strategy_name"] == strategy]
    if row.empty or pd.isna(row[f"top_{universe}_run"].iloc[0]):
        return None
    return RUNS_DIR / strategy / f"run_{int(row[f'top_{universe}_run'].iloc[0])}"


def _held_share(run_dir: Path, block: tuple) -> pd.Series:
    _, filename, name_column, value_column, drop_base = block
    frame = pd.read_csv(run_dir / filename, parse_dates=["date"])
    if drop_base:
        frame = frame[frame[name_column] != BASE_CURRENCY]
    grouped = frame.groupby("date")[value_column]
    return grouped.apply(lambda w: (w.abs() > TOLERANCE).sum()) / grouped.size()


def _large_penalty(run_dir: Path, key: str) -> pd.DatetimeIndex:
    """Rebalancing dates whose selected penalty is one of the two largest grid values."""
    grid = json.loads((run_dir / "config.json").read_text())["param_grid"] or {}
    chosen = pd.read_csv(run_dir / "selected_params.csv", parse_dates=["date"])
    if key not in grid or key not in chosen:
        return pd.DatetimeIndex([])
    values = sorted(grid[key])
    return pd.DatetimeIndex(chosen.loc[chosen[key].isin(values[len(values) // 2:]), "date"])


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=7, length=2)
    # Slightly below zero, so that a program holding nothing stays visible above the axis.
    axis.set_ylim(-0.04, 1.05)
    axis.yaxis.set_major_locator(mticker.FixedLocator([0.0, 0.5, 1.0]))
    axis.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axis.xaxis.set_major_locator(mdates.YearLocator(base=10))
    axis.xaxis.set_minor_locator(mdates.YearLocator(base=5))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _draw(axis: plt.Axes, measure: str, universe: int, suffix: str, table: pd.DataFrame,
          block: tuple, handles: dict[str, object]) -> bool:
    """Draw the four programs of one measure and universe; return whether anything was drawn."""
    plotted = False
    for role, regularized, label, colour, style in PROGRAMS:
        strategy = f"{'reg_' if regularized else ''}{measure}_{role}{suffix}"
        run_dir = _run_dir(table, strategy, universe)
        if run_dir is None or not (run_dir / block[1]).exists():
            continue
        share = _held_share(run_dir, block)
        line, = axis.plot(share.index, share.to_numpy(), color=colour, linestyle=style,
                          linewidth=1.0 if regularized else 0.8, zorder=3 if regularized else 2)
        handles.setdefault(label, line)
        plotted = True
        if block[0] == "assets" and regularized:
            for key, marker, marker_label in PENALTIES:
                points = share.reindex(_large_penalty(run_dir, key)).dropna()
                if points.empty:
                    continue
                axis.plot(points.index, points.to_numpy(), linestyle="none", marker=marker,
                          markersize=2.6, markerfacecolor=colour, markeredgecolor=MARKER_EDGE,
                          markeredgewidth=0.3, zorder=4)
                handles.setdefault(marker_label, Line2D(
                    [], [], linestyle="none", marker=marker, markersize=4,
                    markerfacecolor=MARKER_LEGEND_FACE, markeredgecolor=MARKER_EDGE,
                    markeredgewidth=0.4))
    _style_axis(axis)
    return plotted


def _legend(fig: plt.Figure, handles: dict[str, object]) -> None:
    order = [label for _, _, label, _, _ in PROGRAMS] + [label for _, _, label in PENALTIES]
    labels = [label for label in order if label in handles]
    # Filled column by column: overlay programs, joint programs, then the penalty markers.
    columns = 3 if any(label in handles for _, _, label in PENALTIES) else 2
    fig.legend([handles[label] for label in labels], labels, loc="lower center",
               ncol=columns, fontsize=7.5, frameon=False, handlelength=2.4)


def _plot_grid(suffix: str, table: pd.DataFrame, block: tuple, out_dir: Path) -> bool:
    """One figure with a row per risk measure and a column per universe."""
    fig, axes = plt.subplots(len(MEASURE_TITLES), len(UNIVERSES), figsize=(6.2, 8.2),
                             sharex=True, sharey=True)
    handles: dict[str, object] = {}
    plotted = False
    for row, (measure, title) in enumerate(MEASURE_TITLES.items()):
        for column, universe in enumerate(UNIVERSES):
            axis = axes[row, column]
            plotted |= _draw(axis, measure, universe, suffix, table, block, handles)
            if row == 0:
                axis.set_title(f"top-{universe}", fontsize=8.5)
            if column == 0:
                axis.set_ylabel(title, fontsize=8.5)
    if not plotted:
        plt.close(fig)
        return False
    _legend(fig, handles)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{block[0]}.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_measure(measure: str, suffix: str, table: pd.DataFrame, block: tuple, out_dir: Path) -> bool:
    """One figure for a single risk measure with a panel per universe."""
    fig, axes = plt.subplots(1, len(UNIVERSES), figsize=(6.2, 2.4), sharex=True, sharey=True)
    handles: dict[str, object] = {}
    plotted = False
    for axis, universe in zip(axes, UNIVERSES):
        plotted |= _draw(axis, measure, universe, suffix, table, block, handles)
        axis.set_title(f"top-{universe}", fontsize=8.5)
    if not plotted:
        plt.close(fig)
        return False
    axes[0].set_ylabel(MEASURE_TITLES[measure], fontsize=8.5)
    _legend(fig, handles)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{measure}_{block[0]}.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-family", required=True, choices=list(FAMILIES))
    parser.add_argument("--optimizer", choices=list(MEASURE_TITLES),
                        help="Draw one measure alone; without it, the grid over all measures.")
    args = parser.parse_args()

    runs_file, suffix, out_name = FAMILIES[args.optimizer_family]
    table = pd.read_csv(RUNS_DIR / runs_file, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()

    out_dir = OUT_DIR / out_name
    if args.optimizer is None:
        written = [block[0] for block in BLOCKS if _plot_grid(suffix, table, block, out_dir)]
    else:
        written = [block[0] for block in BLOCKS
                   if _plot_measure(args.optimizer, suffix, table, block, out_dir)]
    if not written:
        raise SystemExit(f"No runs with saved weights in {runs_file}.")


if __name__ == "__main__":
    main()
