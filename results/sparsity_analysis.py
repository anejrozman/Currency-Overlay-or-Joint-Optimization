"""
Portfolio sparsity over the backtest, one figure per weight block.

For a given risk measure the four multi-currency strategies (joint and currency
overlay, each with and without regularization) are drawn side by side for every
universe size, one panel per universe. The vertical axis is the share of the
names available at that rebalancing date that carry a non-zero weight, and it is
shared across the panels, so a top-10 and a top-100 portfolio can be read against
each other and the widening gap between the regularized and unregularized
strategies is not an artefact of the universe size.

Runs are taken from results/runs/top_runs.csv. Runs that predate the weight
export in code/scripts/run_strategy.py carry no weights_*.csv and are reported as
skipped.

Usage:
    python results/sparsity_analysis.py es
    python results/sparsity_analysis.py mv

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
TOP_RUNS_FILE: Path = RUNS_DIR / "top_runs.csv"
OUT_DIR: Path = RESULTS_DIR / "figures" / "sparsity_analysis"

STRATEGY_COLORS: dict[str, str] = {
    "overlay": "#eb6834",
    "joint":   "#1baf7a",
}
STRATEGY_LABELS: dict[str, str] = {
    "overlay": "Currency overlay",
    "joint":   "Joint",
}
FAMILY_ORDER: list[str] = ["joint", "overlay"]

# The regularized runs are what the figure is about, so they get the solid line
# and sit on top of their unregularized counterpart where the two coincide.
REGULARIZED_LINESTYLE: object = "-"
PLAIN_LINESTYLE: object = (0, (4, 2))
REGULARIZED_LINEWIDTH: float = 1.2
PLAIN_LINEWIDTH: float = 1.0

TEXT_PRIMARY: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
GRID_COLOR: str = "#d9d8d4"
LEGEND_FACE_COLOR: str = "#ffffff"

MEASURE_TITLES: dict[str, str] = {
    "es": "Expected Shortfall",
    "mv": "Mean-Variance",
}

# Keyed by universe size rather than panel position, so a universe added later
# does not silently rename the existing panels.
UNIVERSE_TITLES: dict[int, str] = {
    10:  "Small Universe",
    50:  "Medium Universe",
    100: "Large Universe",
}

# Solver output is dense noise around the exact zeros the penalties produce: in a
# top-50 overlay run only five of fifty thousand weights land strictly between
# this floor and zero, so the cut is insensitive to its exact value.
WEIGHT_TOLERANCE: float = 1e-6

# The overlay is a vector of forward positions against the base currency, which
# therefore never carries one and would otherwise cap every panel below one.
BASE_CURRENCY: str = "USD"

plt.rcParams["pdf.fonttype"] = 42


@dataclass(frozen=True)
class Block:
    """One weight block of the portfolio, as exported by run_strategy.py."""

    key: str
    filename: str
    name_column: str
    value_column: str
    y_label: str
    title_noun: str
    drop_base_currency: bool


BLOCKS: tuple[Block, ...] = (
    Block(
        key="assets",
        filename="weights_assets.csv",
        name_column="asset",
        value_column="weight",
        y_label="Assets held, share of universe",
        title_noun="asset holdings",
        drop_base_currency=False,
    ),
    Block(
        key="currencies",
        filename="weights_currencies.csv",
        name_column="currency",
        value_column="psi",
        y_label="Currencies hedged, share of universe",
        title_noun="currency positions",
        drop_base_currency=True,
    ),
)


def _read_top_runs() -> tuple[list[int], list[tuple[str, dict[int, int | None]]]]:
    with open(TOP_RUNS_FILE, newline="") as fh:
        rows = [[cell.strip() for cell in row] for row in csv.reader(fh)]

    header = next(row for row in rows if row and row[0])
    universes: list[int] = []
    columns: list[int] = []
    for idx, name in enumerate(header):
        match = re.fullmatch(r"top_(\d+)_run", name)
        if match:
            universes.append(int(match.group(1)))
            columns.append(idx)

    entries: list[tuple[str, dict[int, int | None]]] = []
    for row in rows[rows.index(header) + 1:]:
        if not row or not row[0]:
            continue
        runs: dict[int, int | None] = {}
        for universe, col in zip(universes, columns):
            cell = row[col] if col < len(row) else ""
            runs[universe] = int(cell) if cell else None
        entries.append((row[0], runs))

    return universes, entries


def _classify(strategy: str) -> tuple[str | None, str | None, bool]:
    tokens = strategy.split("_")
    measure = next((m for m in MEASURE_TITLES if m in tokens), None)
    family = next((f for f in STRATEGY_COLORS if f in tokens), None)
    return measure, family, tokens[0] == "reg"


def _run_label(run_dir: Path, fallback: str) -> str:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return fallback
    label = json.loads(config_path.read_text()).get("label")
    return re.sub(r"\s*\(top-\d+\)\s*$", "", label) if label else fallback


def _active_share(run_dir: Path, block: Block) -> pd.Series:
    """Share of the names recorded at each rebalancing date that are held.

    The backtester writes one row per name in the point-in-time universe, zeros
    included, so the row count is the denominator rather than a count of files.
    """
    frame = pd.read_csv(run_dir / block.filename, parse_dates=["date"])
    if block.drop_base_currency:
        frame = frame[frame[block.name_column] != BASE_CURRENCY]

    grouped = frame.groupby("date")[block.value_column]
    available = grouped.size()
    active = grouped.apply(lambda block_weights: (block_weights.abs() > WEIGHT_TOLERANCE).sum())
    return active / available


def _collect_series(
    entries: list[tuple[str, dict[int, int | None]]],
    universe: int,
    measure: str,
    block: Block,
    skipped: list[str],
) -> list[dict]:
    series: list[dict] = []

    for strategy, runs in entries:
        strategy_measure, family, regularized = _classify(strategy)
        if strategy_measure != measure or family is None:
            continue

        run_number = runs.get(universe)
        if run_number is None:
            continue

        run_dir = RUNS_DIR / strategy / f"run_{run_number}"
        if not (run_dir / block.filename).exists():
            skipped.append(f"{strategy} run_{run_number} (top-{universe})")
            continue

        series.append({
            "label": _run_label(run_dir, strategy),
            "share": _active_share(run_dir, block),
            "family": family,
            "regularized": regularized,
            "color": STRATEGY_COLORS[family],
            "linestyle": REGULARIZED_LINESTYLE if regularized else PLAIN_LINESTYLE,
            "linewidth": REGULARIZED_LINEWIDTH if regularized else PLAIN_LINEWIDTH,
            "zorder": 4.0 if regularized else 3.0,
        })

    # Fixed order, so the legend reads joint before overlay and unregularized
    # before its regularized counterpart no matter how top_runs.csv is sorted.
    series.sort(key=lambda spec: (FAMILY_ORDER.index(spec["family"]), spec["regularized"]))
    return series


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _plot(panels: dict[int, list[dict]], measure: str, block: Block) -> Path:
    universes = sorted(panels)
    fig, axes = plt.subplots(
        1, len(universes),
        figsize=(3.6 * len(universes), 3.9),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[0]

    # Keyed by the sort order rather than insertion order, so the legend reads the
    # same way whether or not every strategy has a run in the leftmost panel.
    legend_handles: dict[tuple[int, bool, str], plt.Line2D] = {}

    for ax, universe in zip(axes, universes):
        series = panels[universe]

        for spec in series:
            line, = ax.plot(
                spec["share"].index,
                spec["share"].to_numpy(),
                color=spec["color"],
                linestyle=spec["linestyle"],
                linewidth=spec["linewidth"],
                zorder=spec["zorder"],
                solid_capstyle="round",
                dash_capstyle="round",
            )
            legend_handles.setdefault(
                (FAMILY_ORDER.index(spec["family"]), spec["regularized"], spec["label"]),
                line,
            )

        ax.set_title(
            UNIVERSE_TITLES.get(universe, f"Top-{universe}"),
            color=TEXT_PRIMARY, fontsize=10, pad=10,
        )
        if not series:
            ax.text(
                0.5, 0.5, "no weights recorded",
                transform=ax.transAxes, ha="center", va="center",
                color=TEXT_SECONDARY, fontsize=8.5,
            )
        _style_axes(ax)

    axes[0].set_ylabel(block.y_label, color=TEXT_SECONDARY, fontsize=9)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    top = max(
        (spec["share"].max() for series in panels.values() for spec in series),
        default=1.0,
    )
    axes[0].set_ylim(-0.03 * top, min(1.05, 1.10 * top))

    fig.suptitle(
        f"Portfolio sparsity, {MEASURE_TITLES[measure]} strategies, {block.title_noun}",
        color=TEXT_PRIMARY, fontsize=12, y=0.995,
    )

    ordered = sorted(legend_handles)
    legend = fig.legend(
        [legend_handles[key] for key in ordered], [key[-1] for key in ordered],
        loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=2, frameon=True, fontsize=9, labelcolor=TEXT_SECONDARY,
        handlelength=2.4, facecolor=LEGEND_FACE_COLOR, edgecolor=GRID_COLOR,
        framealpha=1.0, borderpad=0.6,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.tight_layout(rect=(0, 0.175, 1, 0.965))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{measure}_{block.key}.pdf"
    fig.savefig(path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "measure",
        choices=sorted(MEASURE_TITLES),
        help="Optimizer family to plot.",
    )
    measure = parser.parse_args().measure

    universes, entries = _read_top_runs()
    for block in BLOCKS:
        skipped: list[str] = []
        panels = {
            universe: _collect_series(entries, universe, measure, block, skipped)
            for universe in universes
        }
        if skipped:
            print(
                f"[{block.key}] no {block.filename} in: " + ", ".join(skipped),
                file=sys.stderr,
            )
        if not any(panels.values()):
            print(
                f"[{block.key}] nothing to plot for '{measure}'; rerun the "
                f"strategies so that {block.filename} is written.",
                file=sys.stderr,
            )
            continue
        print(f"Wrote {_plot(panels, measure, block)}")


if __name__ == "__main__":
    main()
