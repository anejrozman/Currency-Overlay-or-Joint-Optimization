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

STRATEGY_COLORS: dict[str, str] = {
    "asset":   "#2a78d6",  # blue   — ASSETS-FULL
    "overlay": "#eb6834",  # orange — currency overlay
    "joint":   "#1baf7a",  # aqua   — joint
}
FALLBACK_COLORS: list[str] = ["#eda100", "#e87ba4", "#008300", "#7a5cc4", "#e34948"]

STRATEGY_LINEWIDTH: float = 1.1
BENCHMARK_LINEWIDTH: float = 1.0

# Settings that must agree across every run shown in the same figure for the
# comparison to be meaningful.  ``None`` on either side is skipped, so the
# benchmark (no param_grid) never trips the check.
COMPARED_SETTINGS: tuple[str, ...] = (
    "window_years",
    "val_years",
    "rebalancing_frequency",
    "alpha",
    "leverage_limit",
)

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


def _classify(strategy: str) -> tuple[str | None, str | None, str | None, bool]:
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

    return measure, family, variant, regularized


def _is_regularized_hedged(variant: str | None, regularized: bool) -> bool:
    """
    Whether a strategy is the regularized, fully hedged variant of its family.

    ``asset``/``zero`` is ASSETS-ZERO, which holds no forward positions and so
    is unhedged; the overlay and joint families hedge by construction, leaving
    regularization as the only criterion for them.
    """
    return regularized and variant != "zero"

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


def _read_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read %s.", config_path)
        return {}


def _run_settings(config: dict) -> dict:
    """Extract the settings compared across the runs in one figure."""
    grid = config.get("param_grid") or {}
    return {
        "window_years":          config.get("window_years"),
        "val_years":             config.get("val_years"),
        "rebalancing_frequency": config.get("rebalancing_frequency"),
        "alpha":                 grid.get("alpha"),
        "leverage_limit":        grid.get("leverage_limit"),
    }


def _run_number(run_dir: Path) -> int | None:
    match = re.fullmatch(r"run_(\d+)", run_dir.name)
    return int(match.group(1)) if match else None


def _check_universe(run_dir: Path, expected: int, strategy: str, config: dict) -> None:
    actual = config.get("universe_size")
    if actual is not None and int(actual) != expected:
        logger.warning(
            "%s %s was run on the top-%s universe but is listed under top_%d_run.",
            strategy, run_dir.name, actual, expected,
        )


def _check_is_latest(run_dir: Path, strategy: str, universe: int) -> None:
    """
    Warn when a newer run on the same universe exists than the one listed in
    ``top_runs.csv`` — the lookup table is maintained by hand and goes stale.
    """
    listed = _run_number(run_dir)
    if listed is None:
        return
    newer = []
    for candidate in run_dir.parent.glob("run_*"):
        number = _run_number(candidate)
        if number is None or number <= listed:
            continue
        if not (candidate / "oos_returns.csv").exists():
            continue
        if _read_config(candidate).get("universe_size") == universe:
            newer.append(number)
    if newer:
        logger.warning(
            "%s (top-%d) is plotted from run_%d, but newer top-%d runs exist: %s. "
            "Update top_runs.csv if one of those supersedes it.",
            strategy, universe, listed, universe,
            ", ".join(f"run_{n}" for n in sorted(newer)),
        )


def _check_settings(series: list[dict], universe: int) -> None:
    """Warn when the runs in one figure were not produced under the same setup."""
    reference: dict[str, object] = {}
    reference_label: dict[str, str] = {}
    for spec in series:
        for key in COMPARED_SETTINGS:
            value = spec["settings"].get(key)
            if value is None:
                continue
            if key not in reference:
                reference[key] = value
                reference_label[key] = spec["label"]
            elif value != reference[key]:
                logger.warning(
                    "top-%d: %s uses %s=%s but %s uses %s — the curves are not "
                    "directly comparable.",
                    universe, spec["label"], key, value,
                    reference_label[key], reference[key],
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
        strategy_measure, family, variant, regularized = _classify(strategy)
        if not is_benchmark:
            if strategy_measure != measure:
                continue
            if not _is_regularized_hedged(variant, regularized):
                logger.debug(
                    "%s is not a regularized hedged strategy; skipping.", strategy
                )
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

        config = _read_config(run_dir)
        _check_universe(run_dir, universe, strategy, config)
        _check_is_latest(run_dir, strategy, universe)

        # Each series gets its own zorder so the drawing order stops depending on
        # the row order in top_runs.csv.  Previously every strategy shared
        # zorder=2.0 and the first row was buried under all the others.
        if is_benchmark:
            color, linestyle = BENCHMARK_COLOR, ":"
            width, zorder = BENCHMARK_LINEWIDTH, 100.0
        elif family in STRATEGY_COLORS:
            color, linestyle = STRATEGY_COLORS[family], "-"
            width, zorder = STRATEGY_LINEWIDTH, 2.0 + len(series)
        else:
            color = next(fallbacks, FALLBACK_COLORS[-1])
            linestyle = "-"
            width, zorder = STRATEGY_LINEWIDTH, 2.0 + len(series)
            logger.warning(
                "%s does not match the naming convention; using a fallback colour.",
                strategy,
            )

        spec = {
            "label": _run_label(run_dir, strategy),
            "wealth": _load_wealth(run_dir),
            "settings": _run_settings(config),
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


def _align_series(series: list[dict], universe: int) -> list[dict]:
    """
    Put every curve on the common trading-day index and re-anchor it at 1.0.

    ``_load_wealth`` normalises each run to 1.0 at *its own* first out-of-sample
    date.  When two runs cover different date ranges — different ``window_years``
    or a first rebalance skipped by the universe filter — that silently compares
    curves anchored at different points in time.
    """
    common = series[0]["wealth"].index
    for spec in series[1:]:
        common = common.intersection(spec["wealth"].index)
    if len(common) < 2:
        raise ValueError(
            f"top-{universe}: the selected runs share fewer than two common dates."
        )

    for spec in series:
        wealth = spec["wealth"]
        if not wealth.index.equals(common):
            logger.error(
                "top-%d: %s covers %s to %s, outside the common window %s to %s; "
                "it has been re-anchored at the common start.",
                universe, spec["label"],
                wealth.index[0].date(), wealth.index[-1].date(),
                common[0].date(), common[-1].date(),
            )
        aligned = wealth.reindex(common).ffill()
        spec["wealth"] = aligned / aligned.iloc[0]

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
            dash_capstyle="round",
        )

    ax.set_title(
        f"Cumulative performance, regularized {MEASURE_TITLES[measure]} strategies, "
        f"top-{universe}",
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
        help=(
            "Risk measure family to plot. Only the regularized, fully hedged "
            "variant of each optimizer family is drawn; the benchmark is always "
            "included."
        ),
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
        _check_settings(series, universe)
        series = _align_series(series, universe)
        _plot(series, args.measure, universe, args.out_dir)


if __name__ == "__main__":
    main()
