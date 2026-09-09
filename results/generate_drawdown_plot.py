"""
Drawdown time series, one panel per universe size.

Usage:
    python results/generate_drawdown_plot.py --optimizer-family normal --optimizer es
    python results/generate_drawdown_plot.py --optimizer-family demeaned --optimizer soes
    python results/generate_drawdown_plot.py --optimizer-family with-mean-term --optimizer mv

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
OUT_DIR: Path = RESULTS_DIR / "figures" / "drawdowns"

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

UNIVERSES: tuple[int, ...] = (10, 50, 100)


def _strategies(measure: str, suffix: str, family: str) -> list[tuple[str, str]]:
    entries = [("equal_weight_optimizer", "Equal weight")]
    if family == "normal":
        entries.append((f"reg_{measure}_asset_full", "Assets-full"))
    for role, label in (("curr_overlay", "Overlay"), ("joint", "Joint")):
        entries.append((f"{measure}_{role}{suffix}", label))
        entries.append((f"reg_{measure}_{role}{suffix}", f"Regularized {label.lower()}"))
    return entries


def _drawdown(table: pd.DataFrame, strategy: str, universe: int) -> pd.Series | None:
    row = table[table["strategy_name"] == strategy]
    if row.empty:
        return None
    run = row[f"top_{universe}_run"].iloc[0]
    if pd.isna(run):
        return None
    path = RUNS_DIR / strategy / f"run_{int(run)}" / "oos_returns.csv"
    if not path.exists():
        return None
    returns = pd.read_csv(path, index_col=0, parse_dates=True)["return"].dropna()
    wealth = (1.0 + returns).cumprod()
    return (wealth / wealth.cummax() - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-family", required=True, choices=list(FAMILIES))
    parser.add_argument("--optimizer", required=True, choices=list(MEASURE_TITLES))
    args = parser.parse_args()

    runs_file, suffix, out_name = FAMILIES[args.optimizer_family]
    table = pd.read_csv(RUNS_DIR / runs_file, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()

    fig, axes = plt.subplots(1, len(UNIVERSES), figsize=(15, 4.5), sharey=True)
    handles: dict[str, object] = {}
    plotted = False

    for ax, universe in zip(axes, UNIVERSES):
        for strategy, label in _strategies(args.optimizer, suffix, args.optimizer_family):
            drawdown = _drawdown(table, strategy, universe)
            if drawdown is None:
                continue
            line, = ax.plot(drawdown.index, drawdown.to_numpy(), linewidth=0.8, label=label)
            handles.setdefault(label, line)
            if strategy != "equal_weight_optimizer":
                plotted = True
        ax.set_title(f"top-{universe}")
        ax.grid(alpha=0.3)

    if not plotted:
        raise SystemExit(f"No {args.optimizer} runs recorded in {runs_file}.")

    axes[0].set_ylabel("Drawdown (%)")
    fig.suptitle(f"Drawdown series, {MEASURE_TITLES[args.optimizer]}")
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out_dir = OUT_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{args.optimizer}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
