"""
Cumulative out-of-sample wealth plots, one figure per universe size.

Usage:
    python results/generate_cumulative_plots.py --measure es
    python results/generate_cumulative_plots.py --measure mv
    TODO: implement --measure mad, measure factor, measure general in the future.

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

BENCHMARK_STRATEGY: str = "equal_weight_optimizer"

FAMILY_COLORS: dict[str, str] = {
    "asset":   "#2a78d6",  # blue
    "overlay": "#eb6834",  # orange
    "joint":   "#1baf7a",  # aqua
}
VARIANT_STYLES: dict[str, str] = {
    "full":  "-",
    "zero":  "--",
    "plain": "-",
    "reg":   "--",
}
FALLBACK_COLORS: list[str] = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

BENCHMARK_COLOR: str = "#0b0b0b"
TEXT_PRIMARY: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
GRID_COLOR: str = "#d9d8d4"

MEASURE_TITLES: dict[str, str] = {
    "es": "Expected Shortfall",
    "mv": "Mean-Variance",
}
plt.rcParams["pdf.fonttype"] = 42


def _read_top_runs(path: Path) -> tuple[list[int], list[tuple[str, dict[int, int | None]]]]:
    with open(path, newline="") as fh:
        rows = [[cell.strip() for cell in row] for row in csv.reader(fh)]

    header = next((r for r in rows if r and r[0]), None)
    if header is None:
        raise ValueError(f"{path} is empty.")

    universes: list[int] = []
    columns: list[int] = []
    for idx, name in enumerate(header):
        match = re.fullmatch(r"top_(\d+)_run", name)
        if match:
            universes.append(int(match.group(1)))
            columns.append(idx)
    if not universes:
        raise ValueError(f"No 'top_<N>_run' columns found in the header of {path}.")

    entries: list[tuple[str, dict[int, int | None]]] = []
    for row in rows[rows.index(header) + 1:]:
        if not row or not row[0]:
            continue
        strategy = row[0]
        runs: dict[int, int | None] = {}
        for universe, col in zip(universes, columns):
            cell = row[col] if col < len(row) else ""
            runs[universe] = int(cell) if cell else None
        entries.append((strategy, runs))

    return universes, entries


def _classify(strategy: str) -> tuple[str | None, str | None, str | None]:
    tokens = strategy.split("_")
    measure = next((m for m in ("es", "mv") if m in tokens), None)
    regularized = tokens[0] == "reg"

    if "asset" in tokens:
        family = "asset"
        variant = next((v for v in ("full", "zero") if v in tokens), None)
    elif "overlay" in tokens:
        family = "overlay"
        variant = "reg" if regularized else "plain"
    elif "joint" in tokens:
        family = "joint"
        variant = "reg" if regularized else "plain"
    else:
        family = variant = None

    return measure, family, variant

def _run_label(run_dir: Path, fallback: str) -> str:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return fallback
    try:
        label = json.loads(config_path.read_text()).get("label")
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read %s; falling back to '%s'.", config_path, fallback)
        return fallback
    if not label:
        return fallback
    return re.sub(r"\s*\(top-\d+\)\s*$", "", label)


def _check_universe(run_dir: Path, expected: int, strategy: str) -> None:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return
    try:
        actual = json.loads(config_path.read_text()).get("universe_size")
    except (json.JSONDecodeError, OSError):
        return
    if actual is not None and int(actual) != expected:
        logger.warning(
            "%s %s was run on the top-%s universe but is listed under top_%d_run.",
            strategy, run_dir.name, actual, expected,
        )


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
    fallbacks = iter(FALLBACK_COLORS)
    series: list[dict] = []
    benchmark: dict | None = None

    for strategy, runs in entries:
        is_benchmark = strategy == BENCHMARK_STRATEGY
        strategy_measure, family, variant = _classify(strategy)
        if not is_benchmark and strategy_measure != measure:
            continue

        run_number = runs.get(universe)
        if run_number is None:
            logger.warning(
                "%s has no run listed for top-%d; skipping.", strategy, universe
            )
            continue

        run_dir = RUNS_DIR / strategy / f"run_{run_number}"
        if not (run_dir / "oos_returns.csv").exists():
            logger.warning(
                "%s run_%d (top-%d) has no oos_returns.csv; skipping.",
                strategy, run_number, universe,
            )
            continue

        _check_universe(run_dir, universe, strategy)

        if is_benchmark:
            color, linestyle, width, zorder = BENCHMARK_COLOR, ":", 0.8, 3.0
        elif family in FAMILY_COLORS and variant in VARIANT_STYLES:
            color, linestyle, width, zorder = (
                FAMILY_COLORS[family], VARIANT_STYLES[variant], 0.9, 2.0
            )
        else:
            color = next(fallbacks, FALLBACK_COLORS[-1])
            linestyle, width, zorder = "-", 0.9, 2.0
            logger.warning(
                "%s does not match the naming convention; using a fallback colour.",
                strategy,
            )

        spec = {
            "label": _run_label(run_dir, strategy),
            "wealth": _load_wealth(run_dir),
            "color": color,
            "linestyle": linestyle,
            "linewidth": width,
            "zorder": zorder,
            "is_benchmark": is_benchmark,
        }
        if is_benchmark:
            benchmark = spec
        else:
            series.append(spec)

    if benchmark is not None:
        series.append(benchmark)
    return series


# Plotting

def _plot(series: list[dict], measure: str, universe: int, out_dir: Path) -> None:
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
        )

    ax.set_title(
        f"Cumulative performance, {MEASURE_TITLES[measure]} strategies, top-{universe}",
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
        ax.legend(
            loc="upper left", frameon=False, fontsize=9,
            labelcolor=TEXT_SECONDARY, ncol=1, handlelength=2.4,
        )

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{measure}_top{universe}.pdf"
    fig.savefig(path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    logger.info("top-%d: %d series -> %s", universe, len(series), path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative out-of-sample wealth, one figure per universe size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--measure",
        required=True,
        choices=["es", "mv"],
        help="Risk measure family to plot. The benchmark is always included.",
    )
    parser.add_argument(
        "--top-runs",
        type=Path,
        default=RUNS_DIR / "top_runs.csv",
        help="Path to the run lookup table.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURES_DIR / "cumulative_returns",
        help="Directory to write figures into.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-8s  %(message)s",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    universes, entries = _read_top_runs(args.top_runs)
    logger.info(
        "Plotting %s strategies for universes: %s",
        args.measure.upper(), ", ".join(f"top-{u}" for u in universes),
    )

    for universe in universes:
        series = _collect_series(entries, universe, args.measure)
        if not series:
            logger.warning("top-%d: nothing to plot; skipping figure.", universe)
            continue
        if all(spec["is_benchmark"] for spec in series):
            logger.warning(
                "top-%d: no %s strategy had a usable run, so the figure shows the "
                "benchmark alone — do not use it as a comparison plot.",
                universe, args.measure.upper(),
            )
        _plot(series, args.measure, universe, args.out_dir)


if __name__ == "__main__":
    main()
