"""
Realized risk measure of the regularized joint and the regularized currency
overlay strategies over all non-overlapping blocks of consecutive out-of-sample
days, one figure per universe size.  Both strategies are evaluated on the same
time indices, so the per-block differences are paired.

Usage:
    python code/analysis/bootstrap_risk_measure_analysis.py es
    python code/analysis/bootstrap_risk_measure_analysis.py mv

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

RESULTS_DIR: Path = Path(__file__).resolve().parents[2] / "results"
RUNS_DIR: Path = RESULTS_DIR / "runs"
TOP_RUNS_FILE: Path = RUNS_DIR / "top_runs.csv"
OUT_DIR: Path = RESULTS_DIR / "figures" / "bootstrap_risk_measure_analysis"

UNIVERSES: tuple[int, ...] = (10, 50, 100)
MEASURE_TITLES: dict[str, str] = {"es": "Expected Shortfall", "mv": "Mean-Variance"}
RISK_LABELS: dict[str, str] = {
    "es": "ES$_{80}$ (daily, %)",
    "mv": "Volatility (ann., %)",
}

BLOCK_DAYS: int = 30
ES_ALPHA: float = 0.8
TRADING_DAYS: int = 252
CONFIDENCE_LEVEL: float = 0.95

JOINT_COLOR: str = "#1baf7a"
OVERLAY_COLOR: str = "#eb6834"
DIFF_COLOR: str = "#2a78d6"
TEXT_PRIMARY: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
GRID_COLOR: str = "#d9d8d4"
plt.rcParams["pdf.fonttype"] = 42


def _returns(strategy: str, run: int) -> pd.Series | None:
    path = RUNS_DIR / strategy / f"run_{run}" / "oos_returns.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)["return"].dropna()


def _risk(blocks: np.ndarray, measure: str) -> np.ndarray:
    if measure == "es":
        tail = max(1, int(round((1.0 - ES_ALPHA) * BLOCK_DAYS)))
        return -np.sort(blocks, axis=1)[:, :tail].mean(axis=1) * 100.0
    return blocks.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS) * 100.0


def _percentile_lines(
    ax: plt.Axes, values: np.ndarray, color: str, height: float
) -> None:
    for level in (2.5, 97.5):
        bound = float(np.percentile(values, level))
        ax.axvline(bound, color=color, linestyle="--", linewidth=0.9)
        ax.text(
            bound, height, f" {level:g}%: {bound:.2f} ",
            transform=ax.get_xaxis_transform(), rotation=90,
            va="top", ha="right", fontsize=7, color=color,
        )


def _stats_text(difference: np.ndarray) -> str:
    wins = int((difference < 0.0).sum())
    interval = binomtest(wins, difference.size).proportion_ci(
        confidence_level=CONFIDENCE_LEVEL
    )
    q1, q3 = np.percentile(difference, [25.0, 75.0])
    return (
        f"Joint better: {wins / difference.size:.1%} "
        f"[{interval.low:.1%}, {interval.high:.1%}]\n"
        f"Mean: {difference.mean():+.3f}\n"
        f"Median: {np.median(difference):+.3f}\n"
        f"IQR: [{q1:+.3f}, {q3:+.3f}]"
    )


def _plot(
    joint: np.ndarray, overlay: np.ndarray, measure: str, universe: int
) -> None:
    difference = joint - overlay
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bins = np.histogram_bin_edges(np.concatenate([joint, overlay]), bins=40)
    axes[0].hist(joint, bins=bins, color=JOINT_COLOR, alpha=0.55, label="Joint")
    axes[0].hist(
        overlay, bins=bins, color=OVERLAY_COLOR, alpha=0.55, label="Currency overlay"
    )
    _percentile_lines(axes[0], joint, JOINT_COLOR, 0.97)
    _percentile_lines(axes[0], overlay, OVERLAY_COLOR, 0.72)
    axes[0].set_xlabel(RISK_LABELS[measure])
    axes[0].legend(
        loc="upper right", frameon=True, fontsize=9, labelcolor=TEXT_SECONDARY,
        facecolor="#ffffff", edgecolor=GRID_COLOR, framealpha=1.0,
    )

    axes[1].hist(difference, bins=40, color=DIFF_COLOR, alpha=0.75)
    axes[1].axvline(0.0, color=TEXT_PRIMARY, linewidth=1.0)
    axes[1].set_xlabel(f"{RISK_LABELS[measure]}, joint $-$ overlay")
    axes[1].text(
        0.98, 0.98, _stats_text(difference), transform=axes[1].transAxes,
        ha="right", va="top", fontsize=9, color=TEXT_SECONDARY,
        bbox={"facecolor": "#ffffff", "edgecolor": GRID_COLOR, "linewidth": 0.6},
    )

    for ax in axes:
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)

    fig.suptitle(
        f"Regularized {MEASURE_TITLES[measure]} risk over {joint.size} "
        f"non-overlapping {BLOCK_DAYS}-day blocks, top-{universe}",
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

        aligned = pd.concat([joint, overlay], axis=1, join="inner").dropna()
        n_blocks = len(aligned) // BLOCK_DAYS
        blocks = aligned.to_numpy()[: n_blocks * BLOCK_DAYS].reshape(
            n_blocks, BLOCK_DAYS, 2
        )
        _plot(
            _risk(blocks[:, :, 0], measure),
            _risk(blocks[:, :, 1], measure),
            measure,
            universe,
        )


if __name__ == "__main__":
    main()
