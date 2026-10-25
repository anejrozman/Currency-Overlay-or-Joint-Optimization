"""Drawdown series of the top runs, a row per risk measure and a column per universe (Appendix G.4).

With --optimizer, one figure for that measure alone.
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from common import MEASURES, RESULTS_DIR, UNIVERSES, grid_legend, run_dir, style_time_axis, top_runs

OUT_DIR = RESULTS_DIR / "figures" / "drawdowns"
FAMILIES = {"normal": ("top_runs.csv", "normal"), "demeaned": ("top_demeaned_runs.csv", "demeaned"),
            "with-mean-term": ("top_runs_w_mean.csv", "with_mean_term")}
DASHED = (0, (4, 2))
OVERLAY_JOINT = (  # strategy template, label, colour, line style, line width
    ("{}_curr_overlay", "Overlay", "#eb6834", DASHED, 0.8),
    ("reg_{}_curr_overlay", "Regularized Overlay", "#eda100", "-", 1.0),
    ("{}_joint", "Joint", "#2a78d6", DASHED, 0.8),
    ("reg_{}_joint", "Regularized Joint", "#e87ba4", "-", 1.0),
)
PROGRAMS = {
    "normal": OVERLAY_JOINT + (("reg_{}_asset_zero", "Zero", "#8c5a2b", "-", 1.0),),
    "demeaned": OVERLAY_JOINT,
    "with-mean-term": (("{}_curr_overlay_w_mean", "Mean-Risk Overlay", "#5b3a99", "-", 1.0),
                       ("{}_joint_w_mean", "Mean-Risk Joint", "#0f6e6e", "-", 1.0)),
}
BENCHMARK = ("equal_weight_optimizer", "EW", "#3d3d3a", (0, (3, 1.5)), 0.8)
plt.rcParams["pdf.fonttype"] = 42


def drawdown(table: pd.DataFrame, strategy: str, universe: int) -> pd.Series | None:
    path = run_dir(table, strategy, universe)
    if path is None or not (path / "oos_returns.csv").exists():
        return None
    wealth = (1.0 + pd.read_csv(path / "oos_returns.csv", index_col=0, parse_dates=True)["return"].dropna()).cumprod()
    return wealth / wealth.cummax() - 1.0


def panel(measure: str, universe: int, family: str, table: pd.DataFrame) -> list:
    """(drawdown, program) pairs of the family's optimizers, followed by EW."""
    drawn = [(d, p) for p in PROGRAMS[family] if (d := drawdown(table, p[0].format(measure), universe)) is not None]
    if drawn and (ew := drawdown(table, BENCHMARK[0], universe)) is not None:
        drawn.append((ew, BENCHMARK))
    return drawn


def draw(axis, drawn: list, handles: dict, bottom: float) -> None:
    for series, (strategy, label, colour, style, width) in drawn:
        # EW and Zero stay behind the overlay and joint programs
        zorder = 1 if strategy == BENCHMARK[0] else 2 if strategy.endswith("_asset_zero") else 3
        line, = axis.plot(series.index, series.to_numpy(), color=colour, linestyle=style, linewidth=width,
                          zorder=zorder)
        handles.setdefault(label, line)
    style_time_axis(axis)
    axis.set_ylim(bottom, 0.02)
    axis.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    axis.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))


def plot(measures: list[str], family: str, table: pd.DataFrame, path, single: bool) -> bool:
    panels = {(m, u): panel(m, u, family, table) for m in measures for u in UNIVERSES}
    if not any(panels.values()):
        return False
    fig, axes = plt.subplots(len(measures), len(UNIVERSES), sharex=True, sharey=True, squeeze=False,
                             figsize=(6.2, 2.4 if single else 1.64 * len(measures)))
    deepest = min(float(d.min()) for drawn in panels.values() for d, _ in drawn)
    bottom = -0.2 * math.ceil(-deepest / 0.2 - 1e-9) - 0.02
    handles = {}
    for row, measure in enumerate(measures):
        for column, universe in enumerate(UNIVERSES):
            axis = axes[row, column]
            draw(axis, panels[measure, universe], handles, bottom)
            if row == 0:
                axis.set_title(f"top-{universe}", fontsize=8.5)
            if column == 0:
                axis.set_ylabel(MEASURES[measure], fontsize=8.5)
    grid_legend(fig, handles, [p[1] for p in PROGRAMS[family]] + [BENCHMARK[1]], ncol=3)
    if single:
        fig.tight_layout(rect=(0, 0.2, 1, 1))
    else:
        fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-family", required=True, choices=list(FAMILIES))
    parser.add_argument("--optimizer", choices=list(MEASURES), help="Draw one measure alone.")
    args = parser.parse_args()
    runs_file, out_name = FAMILIES[args.optimizer_family]
    table = top_runs(runs_file)
    if args.optimizer is None:
        measures = [m for m in MEASURES if any(panel(m, u, args.optimizer_family, table) for u in UNIVERSES)]
        path = OUT_DIR / out_name / "drawdowns.pdf"
    else:
        measures, path = [args.optimizer], OUT_DIR / out_name / f"{args.optimizer}.pdf"
    if not measures or not plot(measures, args.optimizer_family, table, path, single=args.optimizer is not None):
        raise SystemExit(f"No runs recorded in {runs_file}.")
