"""
Drawdown series of the backtested optimizers (Appendix G.4).

The drawdown at each trading day is the decline of the cumulative wealth from its
running maximum, whose largest value is the maximum drawdown of the performance
tables. The equally weighted portfolio is drawn in every panel as the reference.

Without --optimizer, one figure is drawn with a row per risk measure and a column
per universe, as used in the thesis. With --optimizer, one figure is drawn for that
measure alone, with a panel per universe.

Usage:
    python results/generate_drawdown_plot.py --optimizer-family normal
    python results/generate_drawdown_plot.py --optimizer-family with-mean-term
    python results/generate_drawdown_plot.py --optimizer-family normal --optimizer es

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
OUT_DIR: Path = RESULTS_DIR / "figures" / "drawdowns"

FAMILIES: dict[str, tuple[str, str]] = {
    "normal": ("top_runs.csv", "normal"),
    "demeaned": ("top_demeaned_runs.csv", "demeaned"),
    "with-mean-term": ("top_runs_w_mean.csv", "with_mean_term"),
}

# Thesis names and table order of the risk measures.
MEASURE_TITLES: dict[str, str] = {
    "mv": "Variance",
    "smv": "Semi-Variance",
    "mad": "MAD",
    "es": "ES",
    "soes": "SOES",
}

# (strategy template, legend label, colour, line style, line width) per family. The
# overlay and joint programs keep the colours of the Appendix E runtime figures and the
# sparsity figures, with the unregularized programs dashed. Zero and the mean-risk
# optimizers take colours of their own, checked for colour-vision-deficiency separation
# against the others in their figure.
PROGRAMS: dict[str, tuple[tuple[str, str, str, object, float], ...]] = {
    "normal": (
        ("{}_curr_overlay", "Overlay", "#eb6834", (0, (4, 2)), 0.8),
        ("reg_{}_curr_overlay", "Regularized Overlay", "#eda100", "-", 1.0),
        ("{}_joint", "Joint", "#2a78d6", (0, (4, 2)), 0.8),
        ("reg_{}_joint", "Regularized Joint", "#e87ba4", "-", 1.0),
        ("reg_{}_asset_zero", "Zero", "#8c5a2b", "-", 1.0),
    ),
    "demeaned": (
        ("{}_curr_overlay", "Overlay", "#eb6834", (0, (4, 2)), 0.8),
        ("reg_{}_curr_overlay", "Regularized Overlay", "#eda100", "-", 1.0),
        ("{}_joint", "Joint", "#2a78d6", (0, (4, 2)), 0.8),
        ("reg_{}_joint", "Regularized Joint", "#e87ba4", "-", 1.0),
    ),
    "with-mean-term": (
        # Dark enough to stay apart from the joint colours where they share a figure (turnover).
        ("{}_curr_overlay_w_mean", "Mean-Risk Overlay", "#5b3a99", "-", 1.0),
        ("{}_joint_w_mean", "Mean-Risk Joint", "#0f6e6e", "-", 1.0),
    ),
}
BENCHMARK: tuple[str, str, str, object, float] = (
    "equal_weight_optimizer", "EW", "#3d3d3a", (0, (3, 1.5)), 0.8)

UNIVERSES: tuple[int, ...] = (10, 50, 100)

plt.rcParams["pdf.fonttype"] = 42


def _run_dir(table: pd.DataFrame, strategy: str, universe: int) -> Path | None:
    row = table[table["strategy_name"] == strategy]
    if row.empty or pd.isna(row[f"top_{universe}_run"].iloc[0]):
        return None
    return RUNS_DIR / strategy / f"run_{int(row[f'top_{universe}_run'].iloc[0])}"


def _drawdown(table: pd.DataFrame, strategy: str, universe: int) -> pd.Series | None:
    run_dir = _run_dir(table, strategy, universe)
    if run_dir is None or not (run_dir / "oos_returns.csv").exists():
        return None
    returns = pd.read_csv(run_dir / "oos_returns.csv", index_col=0, parse_dates=True)["return"].dropna()
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def _series(measure: str, universe: int, family: str, table: pd.DataFrame) -> list[tuple[pd.Series, tuple]]:
    """The drawdown series of one panel, the optimizers of the family followed by EW."""
    drawn = []
    for program in PROGRAMS[family]:
        drawdown = _drawdown(table, program[0].format(measure), universe)
        if drawdown is not None:
            drawn.append((drawdown, program))
    if drawn:
        benchmark = _drawdown(table, BENCHMARK[0], universe)
        if benchmark is not None:
            drawn.append((benchmark, BENCHMARK))
    return drawn


def _style_axis(axis: plt.Axes, bottom: float) -> None:
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=7, length=2)
    axis.set_ylim(bottom, 0.02)
    axis.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    axis.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axis.xaxis.set_major_locator(mdates.YearLocator(base=10))
    axis.xaxis.set_minor_locator(mdates.YearLocator(base=5))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _draw(axis: plt.Axes, panel: list[tuple[pd.Series, tuple]], handles: dict[str, object]) -> None:
    for drawdown, (strategy, label, colour, style, width) in panel:
        # The benchmarks, EW and Zero, stay behind the overlay and joint programs, so that
        # their deeper drawdowns do not hide them.
        if strategy == BENCHMARK[0]:
            zorder = 1
        elif strategy.endswith("_asset_zero"):
            zorder = 2
        else:
            zorder = 3
        line, = axis.plot(drawdown.index, drawdown.to_numpy(), color=colour, linestyle=style,
                          linewidth=width, zorder=zorder)
        handles.setdefault(label, line)


def _bottom(panels: list[list[tuple[pd.Series, tuple]]]) -> float:
    """A common lower limit, the deepest drawdown rounded down to a multiple of 20%."""
    deepest = min(float(drawdown.min()) for panel in panels for drawdown, _ in panel)
    return -0.2 * math.ceil(-deepest / 0.2 - 1e-9) - 0.02


def _legend(fig: plt.Figure, family: str, handles: dict[str, object]) -> None:
    order = [label for _, label, _, _, _ in PROGRAMS[family]] + [BENCHMARK[1]]
    labels = [label for label in order if label in handles]
    # Filled column by column: overlay programs, joint programs, then Zero and EW.
    fig.legend([handles[label] for label in labels], labels, loc="lower center",
               ncol=3, fontsize=7.5, frameon=False, handlelength=2.4)


def _plot_grid(family: str, table: pd.DataFrame, out_dir: Path) -> bool:
    """One figure with a row per risk measure and a column per universe."""
    measures = [m for m in MEASURE_TITLES if any(_series(m, u, family, table) for u in UNIVERSES)]
    if not measures:
        return False
    panels = {(m, u): _series(m, u, family, table) for m in measures for u in UNIVERSES}
    fig, axes = plt.subplots(len(measures), len(UNIVERSES), figsize=(6.2, 1.64 * len(measures)),
                             sharex=True, sharey=True, squeeze=False)
    handles: dict[str, object] = {}
    bottom = _bottom([panel for panel in panels.values() if panel])
    for row, measure in enumerate(measures):
        for column, universe in enumerate(UNIVERSES):
            axis = axes[row, column]
            _draw(axis, panels[(measure, universe)], handles)
            _style_axis(axis, bottom)
            if row == 0:
                axis.set_title(f"top-{universe}", fontsize=8.5)
            if column == 0:
                axis.set_ylabel(MEASURE_TITLES[measure], fontsize=8.5)
    _legend(fig, family, handles)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "drawdowns.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_measure(measure: str, family: str, table: pd.DataFrame, out_dir: Path) -> bool:
    """One figure for a single risk measure with a panel per universe."""
    panels = [_series(measure, universe, family, table) for universe in UNIVERSES]
    if not any(panels):
        return False
    fig, axes = plt.subplots(1, len(UNIVERSES), figsize=(6.2, 2.4), sharex=True, sharey=True)
    handles: dict[str, object] = {}
    bottom = _bottom([panel for panel in panels if panel])
    for axis, universe, panel in zip(axes, UNIVERSES, panels):
        _draw(axis, panel, handles)
        _style_axis(axis, bottom)
        axis.set_title(f"top-{universe}", fontsize=8.5)
    axes[0].set_ylabel(MEASURE_TITLES[measure], fontsize=8.5)
    _legend(fig, family, handles)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{measure}.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-family", required=True, choices=list(FAMILIES))
    parser.add_argument("--optimizer", choices=list(MEASURE_TITLES),
                        help="Draw one measure alone; without it, the grid over all measures.")
    args = parser.parse_args()

    runs_file, out_name = FAMILIES[args.optimizer_family]
    table = pd.read_csv(RUNS_DIR / runs_file, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()

    out_dir = OUT_DIR / out_name
    if args.optimizer is None:
        written = _plot_grid(args.optimizer_family, table, out_dir)
    else:
        written = _plot_measure(args.optimizer, args.optimizer_family, table, out_dir)
    if not written:
        raise SystemExit(f"No runs recorded in {runs_file}.")


if __name__ == "__main__":
    main()
