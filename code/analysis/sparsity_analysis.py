"""Share of assets and foreign currencies held per rebalancing date (Appendix G.2).

A position counts as held above TOLERANCE. On the asset figures, triangles mark dates whose selected L1 (up)
or L2 (down) asset penalty is in the upper half of its grid. With --optimizer, one measure alone.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.lines import Line2D

from common import MEASURES, RESULTS_DIR, UNIVERSES, grid_legend, run_dir, style_time_axis, top_runs

OUT_DIR = RESULTS_DIR / "figures" / "sparsity_analysis"
FAMILIES = {"normal": ("top_runs.csv", "", "normal"), "demeaned": ("top_demeaned_runs.csv", "", "demeaned"),
            "with-mean-term": ("top_runs_w_mean.csv", "_w_mean", "with_mean_term")}
BLOCKS = (  # name, weights file, name column, value column, drop USD
    ("assets", "weights_assets.csv", "asset", "weight", False),
    ("currencies", "weights_currencies.csv", "currency", "psi", True),
)
PROGRAMS = (  # role, regularized, label, colour, line style
    ("curr_overlay", False, "Overlay", "#eb6834", (0, (4, 2))),
    ("curr_overlay", True, "Regularized Overlay", "#eda100", "-"),
    ("joint", False, "Joint", "#2a78d6", (0, (4, 2))),
    ("joint", True, "Regularized Joint", "#e87ba4", "-"),
)
PENALTIES = (("lambda_l1_asset", "^", r"Large selected $L^1$ penalty"),
             ("lambda_l2_asset", "v", r"Large selected $L^2$ penalty"))
MARKER_EDGE = "#3d3d3a"
TOLERANCE = 1e-4
plt.rcParams["pdf.fonttype"] = 42


def held_share(path, block: tuple) -> pd.Series:
    _, filename, name, value, drop_usd = block
    frame = pd.read_csv(path / filename, parse_dates=["date"])
    if drop_usd:
        frame = frame[frame[name] != "USD"]
    grouped = frame.groupby("date")[value]
    return grouped.apply(lambda w: (w.abs() > TOLERANCE).sum()) / grouped.size()


def large_penalty(path, key: str) -> pd.DatetimeIndex:
    grid = json.loads((path / "config.json").read_text())["param_grid"] or {}
    chosen = pd.read_csv(path / "selected_params.csv", parse_dates=["date"])
    if key not in grid or key not in chosen:
        return pd.DatetimeIndex([])
    values = sorted(grid[key])
    return pd.DatetimeIndex(chosen.loc[chosen[key].isin(values[len(values) // 2:]), "date"])


def draw(axis, measure: str, universe: int, suffix: str, table: pd.DataFrame, block: tuple, handles: dict) -> bool:
    plotted = False
    for role, regularized, label, colour, style in PROGRAMS:
        path = run_dir(table, f"{'reg_' if regularized else ''}{measure}_{role}{suffix}", universe)
        if path is None or not (path / block[1]).exists():
            continue
        share = held_share(path, block)
        line, = axis.plot(share.index, share.to_numpy(), color=colour, linestyle=style,
                          linewidth=1.0 if regularized else 0.8, zorder=3 if regularized else 2)
        handles.setdefault(label, line)
        plotted = True
        if block[0] == "assets" and regularized:
            for key, marker, marker_label in PENALTIES:
                points = share.reindex(large_penalty(path, key)).dropna()
                if points.empty:
                    continue
                axis.plot(points.index, points.to_numpy(), linestyle="none", marker=marker, markersize=2.6,
                          markerfacecolor=colour, markeredgecolor=MARKER_EDGE, markeredgewidth=0.3, zorder=4)
                handles.setdefault(marker_label, Line2D([], [], linestyle="none", marker=marker, markersize=4,
                                                        markerfacecolor="#8a8983", markeredgecolor=MARKER_EDGE,
                                                        markeredgewidth=0.4))
    style_time_axis(axis)
    axis.set_ylim(-0.04, 1.05)  # a program holding nothing stays visible above the axis
    axis.yaxis.set_major_locator(mticker.FixedLocator([0.0, 0.5, 1.0]))
    axis.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    return plotted


def plot(measures: list[str], suffix: str, table: pd.DataFrame, block: tuple, out_dir, single: bool) -> bool:
    fig, axes = plt.subplots(len(measures), len(UNIVERSES), sharex=True, sharey=True, squeeze=False,
                             figsize=(6.2, 2.4) if single else (6.2, 8.2))
    handles = {}
    plotted = False
    for row, measure in enumerate(measures):
        for column, universe in enumerate(UNIVERSES):
            plotted |= draw(axes[row, column], measure, universe, suffix, table, block, handles)
            if row == 0:
                axes[row, column].set_title(f"top-{universe}", fontsize=8.5)
            if column == 0 and not single:
                axes[row, column].set_ylabel(MEASURES[measure], fontsize=8.5)
    if not plotted:
        plt.close(fig)
        return False
    if single:
        axes[0, 0].set_ylabel(MEASURES[measures[0]], fontsize=8.5)
    order = [p[2] for p in PROGRAMS] + [p[2] for p in PENALTIES]
    grid_legend(fig, handles, order, ncol=3 if any(p[2] in handles for p in PENALTIES) else 2)
    if single:
        fig.tight_layout(rect=(0, 0.16, 1, 1))
    else:
        fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / (f"{measures[0]}_{block[0]}.pdf" if single else f"{block[0]}.pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-family", required=True, choices=list(FAMILIES))
    parser.add_argument("--optimizer", choices=list(MEASURES), help="Draw one measure alone.")
    args = parser.parse_args()
    runs_file, suffix, out_name = FAMILIES[args.optimizer_family]
    table = top_runs(runs_file)
    measures = list(MEASURES) if args.optimizer is None else [args.optimizer]
    written = [b for b in BLOCKS
               if plot(measures, suffix, table, b, OUT_DIR / out_name, single=args.optimizer is not None)]
    if not written:
        raise SystemExit(f"No runs with saved weights in {runs_file}.")
