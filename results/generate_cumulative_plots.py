"""
Cumulative out-of-sample wealth plots, one figure per universe size.

Usage:
    python results/generate_cumulative_plots.py es
    python results/generate_cumulative_plots.py soes --variant w_mean
    python results/generate_cumulative_plots.py smv --variant demeaned

The --variant flag selects which top-runs table to read and where the
figures are written:

    base      runs/top_runs.csv          -> figures/cumulative_returns/
    w_mean    runs/top_runs_w_mean.csv   -> figures/cumulative_returns/w_mean/
    demeaned  runs/top_demeaned_runs.csv -> figures/cumulative_returns/demeaned/

Not every measure exists in every variant; combinations without runs are
skipped silently.

TODO: implement factor and general measures in the future.

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
OUT_DIR: Path = RESULTS_DIR / "figures" / "cumulative_returns"

VARIANTS: dict[str, dict[str, str]] = {
    "base":     {"file": "top_runs.csv",          "subdir": "",         "suffix": ""},
    "w_mean":   {"file": "top_runs_w_mean.csv",   "subdir": "w_mean",   "suffix": " with mean term"},
    "demeaned": {"file": "top_demeaned_runs.csv", "subdir": "demeaned", "suffix": " (demeaned)"},
}

BENCHMARK_STRATEGY: str = "equal_weight_optimizer"

STRATEGY_COLORS: dict[str, str] = {
    "asset":   "#2a78d6",
    "overlay": "#eb6834",
    "joint":   "#1baf7a",
}
STRATEGY_LINEWIDTH: float = 1.1
BENCHMARK_LINEWIDTH: float = 1.0

BENCHMARK_COLOR: str = "#0b0b0b"
TEXT_PRIMARY: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
GRID_COLOR: str = "#d9d8d4"
LEGEND_FACE_COLOR: str = "#ffffff"

MEASURE_TITLES: dict[str, str] = {
    "es":   "Expected Shortfall",
    "soes": "Second-Order Expected Shortfall",
    "mv":   "Mean-Variance",
    "smv":  "Semi-Variance",
    "mad":  "Mean Absolute Deviation",
}
plt.rcParams["pdf.fonttype"] = 42


def _read_top_runs(
    path: Path,
) -> tuple[list[int], list[tuple[str, dict[int, int | None]]]]:
    with open(path, newline="") as fh:
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
    return measure, family, tokens[0] == "reg" and "zero" not in tokens


def _run_label(run_dir: Path, fallback: str) -> str:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return fallback
    label = json.loads(config_path.read_text()).get("label")
    return re.sub(r"\s*\(top-\d+\)\s*$", "", label) if label else fallback


def _load_wealth(run_dir: Path) -> pd.Series:
    returns = pd.read_csv(
        run_dir / "oos_returns.csv", index_col=0, parse_dates=True
    )["return"].dropna()
    wealth = (1.0 + returns).cumprod()
    anchor = pd.Series([1.0], index=[returns.index[0] - pd.offsets.BDay(1)])
    return pd.concat([anchor, wealth])


def _collect_series(
    entries: list[tuple[str, dict[int, int | None]]],
    universe: int,
    measure: str,
) -> list[dict]:
    series: list[dict] = [] 
    benchmark: dict | None = None

    for strategy, runs in entries:
        is_benchmark = strategy == BENCHMARK_STRATEGY
        strategy_measure, family, regularized_hedged = _classify(strategy)
        if not is_benchmark and not (
            regularized_hedged
            and strategy_measure == measure
            and family in STRATEGY_COLORS
        ):
            continue

        run_number = runs.get(universe)
        if run_number is None:
            continue

        run_dir = RUNS_DIR / strategy / f"run_{run_number}"
        if not (run_dir / "oos_returns.csv").exists():
            continue

        if is_benchmark:
            color, linestyle = BENCHMARK_COLOR, ":"
            width, zorder = BENCHMARK_LINEWIDTH, 100.0
        else:
            color, linestyle = STRATEGY_COLORS[family], "-"
            width, zorder = STRATEGY_LINEWIDTH, 2.0 + len(series)

        spec = {
            "label": _run_label(run_dir, strategy),
            "wealth": _load_wealth(run_dir),
            "color": color,
            "linestyle": linestyle,
            "linewidth": width,
            "zorder": zorder,
        }
        if is_benchmark:
            benchmark = spec
        else:
            series.append(spec)

    if not series:
        return []
    if benchmark is not None:
        series.append(benchmark)
    return series


def _align_series(series: list[dict]) -> None:
    common = series[0]["wealth"].index
    for spec in series[1:]:
        common = common.intersection(spec["wealth"].index)
    for spec in series:
        aligned = spec["wealth"].reindex(common).ffill()
        spec["wealth"] = aligned / aligned.iloc[0]


def _plot(
    series: list[dict], measure: str, universe: int, variant: dict[str, str]
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for spec in series:
        ax.plot(
            spec["wealth"].index,
            spec["wealth"].to_numpy(),
            label=spec["label"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=spec["linewidth"],
            zorder=spec["zorder"],
            solid_capstyle="round",
            dash_capstyle="round",
        )

    ax.set_title(
        f"Cumulative performance, regularized {MEASURE_TITLES[measure]} "
        f"strategies{variant['suffix']}, top-{universe}",
        color=TEXT_PRIMARY, fontsize=12, loc="center", pad=12,
    )
    ax.set_xlabel("")
    ax.axhline(1.0, color=GRID_COLOR, linewidth=1.0, zorder=1)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)

    if len(series) > 1:
        legend = ax.legend(
            loc="upper left", frameon=True, fontsize=9,
            labelcolor=TEXT_SECONDARY, ncol=1, handlelength=2.4,
            facecolor=LEGEND_FACE_COLOR, edgecolor=GRID_COLOR,
            framealpha=1.0, borderpad=0.6,
        )
        legend.get_frame().set_linewidth(0.6)
        legend.set_zorder(200)

    fig.tight_layout()

    out_dir = OUT_DIR / variant["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / f"{measure}_top{universe}.pdf",
        format="pdf", dpi=600, bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "measure",
        choices=sorted(MEASURE_TITLES),
        help="Optimizer family to plot.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="base",
        help="Top-runs table to read (default: base).",
    )
    args = parser.parse_args()
    measure, variant = args.measure, VARIANTS[args.variant]

    universes, entries = _read_top_runs(RUNS_DIR / variant["file"])
    for universe in universes:
        series = _collect_series(entries, universe, measure)
        if not series:
            continue
        _align_series(series)
        _plot(series, measure, universe, variant)


if __name__ == "__main__":
    main()
