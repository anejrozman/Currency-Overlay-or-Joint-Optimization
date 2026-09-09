"""
Portfolio sparsity per rebalancing date, one panel per universe size.

The share of available names carrying a non-zero weight, for the joint and
currency overlay programs with and without regularization.  On the asset figure
a red up-triangle marks dates whose selected L1 asset penalty fell in the top
half of its grid, and a blue down-triangle does the same for the L2 penalty.

Usage:
    python results/sparsity_analysis.py --optimizer-family normal --optimizer es

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

MEASURE_TITLES: dict[str, str] = {
    "es": "Expected Shortfall",
    "soes": "Second-Order Expected Shortfall",
    "mv": "Mean-Variance",
    "smv": "Semi-Variance",
    "mad": "Mean Absolute Deviation",
}

BLOCKS: tuple[tuple[str, str, str, str, str, bool], ...] = (
    ("assets", "weights_assets.csv", "asset", "weight",
     "Assets held, share of universe", False),
    ("currencies", "weights_currencies.csv", "currency", "psi",
     "Currencies hedged, share of universe", True),
)

PENALTIES: tuple[tuple[str, str, str, str], ...] = (
    ("lambda_l1_asset", "^", "#d62728", "L1 penalty in top half of grid"),
    ("lambda_l2_asset", "v", "#1f77b4", "L2 penalty in top half of grid"),
)

UNIVERSES: tuple[int, ...] = (10, 50, 100)
COLORS: dict[str, str] = {"curr_overlay": "#eb6834", "joint": "#1baf7a"}
WEIGHT_TOLERANCE: float = 1e-6
BASE_CURRENCY: str = "USD"

plt.rcParams["pdf.fonttype"] = 42


def _strategies(measure: str, suffix: str) -> list[tuple[str, str, str, bool]]:
    entries = []
    for role, label in (("joint", "Joint"), ("curr_overlay", "Currency overlay")):
        entries.append((f"{measure}_{role}{suffix}", label, COLORS[role], False))
        entries.append((f"reg_{measure}_{role}{suffix}", f"Regularized {label.lower()}",
                        COLORS[role], True))
    return entries


def _active_share(run_dir: Path, block: tuple) -> pd.Series:
    _, filename, name_column, value_column, _, drop_base = block
    frame = pd.read_csv(run_dir / filename, parse_dates=["date"])
    if drop_base:
        frame = frame[frame[name_column] != BASE_CURRENCY]
    grouped = frame.groupby("date")[value_column]
    return grouped.apply(lambda w: (w.abs() > WEIGHT_TOLERANCE).sum()) / grouped.size()


def _high_penalty(run_dir: Path, key: str) -> pd.DatetimeIndex:
    grid = json.loads((run_dir / "config.json").read_text())["param_grid"] or {}
    chosen = pd.read_csv(run_dir / "selected_params.csv", parse_dates=["date"])
    if key not in grid or key not in chosen:
        return pd.DatetimeIndex([])
    values = sorted(grid[key])
    return pd.DatetimeIndex(chosen.loc[chosen[key].isin(values[len(values) // 2:]), "date"])


def _plot(measure: str, suffix: str, table: pd.DataFrame, block: tuple, out_dir: Path) -> bool:
    key, filename, _, _, y_label, _ = block
    fig, axes = plt.subplots(1, len(UNIVERSES), figsize=(11, 3.9), sharex=True, sharey=True)
    handles: dict[str, object] = {}
    plotted = False

    for axis, universe in zip(axes, UNIVERSES):
        for strategy, label, color, regularized in _strategies(measure, suffix):
            row = table[table["strategy_name"] == strategy]
            if row.empty or pd.isna(row[f"top_{universe}_run"].iloc[0]):
                continue
            run_dir = RUNS_DIR / strategy / f"run_{int(row[f'top_{universe}_run'].iloc[0])}"
            if not (run_dir / filename).exists():
                continue
            share = _active_share(run_dir, block)
            line, = axis.plot(share.index, share.to_numpy(), color=color,
                              linestyle="-" if regularized else (0, (4, 2)),
                              linewidth=1.2 if regularized else 1.0)
            handles.setdefault(label, line)
            plotted = True
            if key == "assets" and regularized:
                for penalty, marker, marker_color, marker_label in PENALTIES:
                    points = share.reindex(_high_penalty(run_dir, penalty)).dropna()
                    if points.empty:
                        continue
                    axis.plot(points.index, points.to_numpy(), linestyle="none",
                              marker=marker, markersize=3.5, color=marker_color, zorder=5)
                    handles.setdefault(marker_label, Line2D(
                        [], [], linestyle="none", marker=marker, markersize=3.5,
                        color=marker_color))
        axis.set_title(f"top-{universe}", fontsize=10)
        axis.grid(alpha=0.3)

    if not plotted:
        plt.close(fig)
        return False

    axes[0].set_ylabel(y_label, fontsize=9)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    fig.suptitle(f"Portfolio sparsity, {MEASURE_TITLES[measure]}, {key}", fontsize=12)
    markers = [penalty[3] for penalty in PENALTIES]
    ordered = [k for k in handles if k not in markers] + [k for k in handles if k in markers]
    fig.legend([handles[k] for k in ordered], ordered, loc="lower center",
               ncol=len(ordered) if len(ordered) < 5 else 3, fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{measure}_{key}.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-family", required=True, choices=list(FAMILIES))
    parser.add_argument("--optimizer", required=True, choices=list(MEASURE_TITLES))
    args = parser.parse_args()

    runs_file, suffix, out_name = FAMILIES[args.optimizer_family]
    table = pd.read_csv(RUNS_DIR / runs_file, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()

    written = [
        block[0]
        for block in BLOCKS
        if _plot(args.optimizer, suffix, table, block, OUT_DIR / out_name)
    ]
    if not written:
        raise SystemExit(f"No {args.optimizer} runs with saved weights in {runs_file}.")


if __name__ == "__main__":
    main()
