"""
Panel of marginal-effect curves from the sweeps in results/marginal_effect_study.

One panel per penalty: the sup-norm step d_inf against lambda on log-log axes,
with the penalty's grid bound marked. Reads the CSV written by
code/scripts/marginal_effect_study.py and its sidecar.

Usage:
    python results/plot_marginal_effect.py
    python results/plot_marginal_effect.py --input results/marginal_effect_study/reg_es_joint_top10.csv

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent
STUDY_DIR = RESULTS_DIR / "marginal_effect_study"
OUT_DIR = RESULTS_DIR / "figures" / "marginal_effect_study"

LINE_COLOR = "#2a78d6"
BOUND_COLOR = "#52514e"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d4"

plt.rcParams["pdf.fonttype"] = 42


def _plot(csv_path: Path, out_dir: Path) -> Path:
    data = pd.read_csv(csv_path)
    sidecar = csv_path.with_suffix(".json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    bounds = meta.get("penalty_bounds", {})

    penalties = list(dict.fromkeys(data["penalty"]))
    ncols = min(2, len(penalties))
    nrows = math.ceil(len(penalties) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)

    for ax, penalty in zip(axes.flat, penalties):
        # lambda = 0 has no predecessor and no place on a log axis.
        panel = data[(data["penalty"] == penalty) & (data["lambda"] > 0)].dropna(subset=["d_inf"])
        ax.plot(panel["lambda"], panel["d_inf"], color=LINE_COLOR, linewidth=1.2,
                marker="o", markersize=2.5)

        bound = bounds.get(penalty)
        if bound:
            ax.axvline(bound, color=BOUND_COLOR, linewidth=1.0, linestyle="--")
            ax.annotate("grid bound", xy=(bound, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -10), textcoords="offset points",
                        color=BOUND_COLOR, fontsize=7.5)

        # Past the bound the solver often returns an identical solution, making
        # d_inf exactly 0. A log axis would drop those points, hiding the very
        # flattening the figure is meant to show; symlog keeps them visible.
        positive = panel["d_inf"][panel["d_inf"] > 0]
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=positive.min() if len(positive) else 1e-16)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{penalty}  ({panel['block'].iloc[0]})", color=TEXT_PRIMARY,
                     fontsize=10, loc="left", pad=8)
        ax.set_xlabel(r"$\lambda$", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel(r"$\|z(\lambda_n)-z(\lambda_{n-1})\|_\infty$",
                      color=TEXT_SECONDARY, fontsize=9)
        ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)
        ax.tick_params(which="both", colors=TEXT_SECONDARY, labelsize=8, length=0)

    for ax in axes.flat[len(penalties):]:
        ax.set_visible(False)

    subtitle = ""
    if meta:
        subtitle = (f"{meta['strategy']}, top-{meta['universe_size']}, "
                    f"{meta['window_start']} to {meta['window_end']}, "
                    f"{meta['n_assets']} assets, {meta['solver']}")
    fig.suptitle(f"Marginal effect of each penalty\n{subtitle}" if subtitle
                 else "Marginal effect of each penalty",
                 color=TEXT_PRIMARY, fontsize=11)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}.pdf"
    fig.savefig(out_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot marginal-effect sweeps.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Sweep CSV. Defaults to every CSV in the study directory.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    args = parser.parse_args()

    inputs = [args.input] if args.input else sorted(STUDY_DIR.glob("*.csv"))
    if not inputs:
        raise SystemExit(f"No sweep CSVs found in {STUDY_DIR}.")

    for csv_path in inputs:
        print(f"{csv_path.name} -> {_plot(csv_path, args.out_dir)}")


if __name__ == "__main__":
    main()
