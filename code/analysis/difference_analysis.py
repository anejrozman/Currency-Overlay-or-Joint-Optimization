"""
Difference between the regularized joint and the regularized currency overlay
out-of-sample return series, one figure per universe size.

Usage:
    python code/analysis/difference_analysis.py es
    python code/analysis/difference_analysis.py mv

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR: Path = Path(__file__).resolve().parents[2] / "results"
RUNS_DIR: Path = RESULTS_DIR / "runs"
TOP_RUNS_FILE: Path = RUNS_DIR / "top_runs.csv"
OUT_DIR: Path = RESULTS_DIR / "figures" / "difference_analysis"

UNIVERSES: tuple[int, ...] = (10, 50, 100)
MEASURE_TITLES: dict[str, str] = {"es": "Expected Shortfall", "mv": "Mean-Variance"}

SERIES_COLOR: str = "#2a78d6"
TEXT_PRIMARY: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
GRID_COLOR: str = "#d9d8d4"
plt.rcParams["pdf.fonttype"] = 42


def _returns(strategy: str, run: int) -> pd.Series | None:
    path = RUNS_DIR / strategy / f"run_{run}" / "oos_returns.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)["return"].dropna()


def _plot(diff: pd.Series, measure: str, universe: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(diff.index, diff.to_numpy(), color=SERIES_COLOR, linewidth=0.5)
    axes[0].axhline(0.0, color=TEXT_PRIMARY, linewidth=0.8)
    axes[0].set_ylabel("Return difference")
    axes[1].hist(diff.to_numpy(), bins=100, color=SERIES_COLOR)
    axes[1].set_xlabel("Return difference")
    axes[1].set_ylabel("Frequency")

    for ax in axes:
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)

    fig.suptitle(
        f"Joint minus currency overlay, regularized {MEASURE_TITLES[measure]}, "
        f"top-{universe}",
        color=TEXT_PRIMARY, fontsize=12,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / f"{measure}_top{universe}.pdf",
        format="pdf", dpi=600, bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", choices=sorted(MEASURE_TITLES))
    measure = parser.parse_args().measure

    top_runs = pd.read_csv(TOP_RUNS_FILE, skipinitialspace=True)
    top_runs.columns = [name.strip() for name in top_runs.columns]
    top_runs["strategy_name"] = top_runs["strategy_name"].str.strip()
    top_runs = top_runs.set_index("strategy_name")

    joint_name = f"reg_{measure}_joint"
    overlay_name = f"reg_{measure}_curr_overlay"

    for universe in UNIVERSES:
        column = f"top_{universe}_run"
        joint = _returns(joint_name, int(top_runs.at[joint_name, column]))
        overlay = _returns(overlay_name, int(top_runs.at[overlay_name, column]))
        if joint is None or overlay is None:
            print(f"skipping top-{universe}: missing out-of-sample returns")
            continue
        _plot((joint - overlay).dropna(), measure, universe)


if __name__ == "__main__":
    main()
