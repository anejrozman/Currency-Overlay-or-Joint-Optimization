"""Penalty boundary figures from the sweep of hyperparameter_boundary_analysis.py (Appendix G).

For each penalty, the marginal effect d_n and the gross exposure of the affected weight block as lambda
passes the largest value of its grid.
"""

from __future__ import annotations

import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from common import RESULTS_DIR

DATA_DIR = RESULTS_DIR / "hyperparameter_boundary_analysis"
OUT_DIR = RESULTS_DIR / "figures" / "boundary_plots"
# label, colour, line width, zorder; the thinner joint line stays visible on top where both coincide
STYLES = {"overlay": ("Regularized Overlay", "#eda100", 2.0, 3.0), "joint": ("Regularized Joint", "#e87ba4", 1.0, 4.0)}
TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR, BOUND_COLOR = "#0b0b0b", "#52514e", "#d9d8d4", "#8a8983"
BOUND_STYLE = {"color": BOUND_COLOR, "linewidth": 1.0, "linestyle": (0, (4, 3))}
MEASURES = ("es", "mad", "mv", "smv", "soes")
PENALTIES = {"lambda_l1_asset": r"$\lambda_1^{a}$", "lambda_l2_asset": r"$\lambda_2^{a}$",
             "lambda_l1_curr": r"$\lambda_1^{c}$", "lambda_l2_curr": r"$\lambda_2^{c}$"}
NOISE_FLOOR = 1e-10  # solver tolerance, drawn as zero
plt.rcParams["pdf.fonttype"] = 42


def format_bound(value: float) -> str:
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    if exponent == 0:
        return f"${mantissa:g}$"
    if abs(mantissa - 1.0) < 1e-9:
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:g}\times10^{{{exponent}}}$"


def load(measure: str, universe: int) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = pd.read_csv(DATA_DIR / f"{measure}_top{universe}.csv")
    sidecar = json.loads((DATA_DIR / f"{measure}_top{universe}.json").read_text())
    bounds = {}
    for penalty in frame["penalty"].unique():
        values = {float(b[penalty]) for b in sidecar["penalty_bounds"].values() if penalty in b}
        if len(values) != 1:
            raise ValueError(f"Strategies disagree on the bound for {penalty}: {sorted(values)}.")
        bounds[penalty] = values.pop()
    return frame, bounds


def draw(frame: pd.DataFrame, bounds: dict[str, float], column: str, log_y: bool, y_label: str):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.8), sharex=False, sharey=True)
    for ax, penalty in zip(axes.flat, PENALTIES):
        bound = bounds[penalty]
        for strategy, group in frame[frame["penalty"] == penalty].groupby("strategy"):
            group = group[group["lambda"] > 0].sort_values("lambda")
            label, colour, width, zorder = STYLES["overlay" if "overlay" in strategy.split("_") else "joint"]
            x, y = group["lambda"].to_numpy(), group[column].to_numpy()
            if log_y:
                spread = group["d_inf_std"].to_numpy()
                # a band reaching zero would render as a spike on the log axis
                ax.fill_between(x, np.maximum(y - spread, NOISE_FLOOR), np.maximum(y + spread, NOISE_FLOOR),
                                where=y - spread > 0.0, color=colour, alpha=0.16, linewidth=0, zorder=2)
                y = np.maximum(y, NOISE_FLOOR)
            ax.plot(x, y, label=label, color=colour, linewidth=width, solid_capstyle="round", zorder=zorder)
        ax.axvline(bound, zorder=5, **BOUND_STYLE)
        ax.text(bound, 1.015, format_bound(bound), transform=blended_transform_factory(ax.transData, ax.transAxes),
                ha="center", va="bottom", color=BOUND_COLOR, fontsize=7, zorder=6)
        ax.set_xscale("log")
        ax.set_xlim(bound * 7e-4, bound * 1.5e2)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(PENALTIES[penalty], color=TEXT_PRIMARY, fontsize=9, pad=16)
        ax.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=7.5, length=0)
        ax.set_xlabel(r"$\lambda$", color=TEXT_SECONDARY, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=8.5)
    if log_y:
        axes.flat[0].set_ylim(NOISE_FLOOR / 3.0, 3.0)
    return fig, axes


def save(fig, axes, filename: str) -> None:
    handles = {}
    for handle, label in zip(*axes.flat[0].get_legend_handles_labels()):
        handles.setdefault(label, handle)
    handles["Largest grid value"] = plt.Line2D([], [], **BOUND_STYLE)
    legend = fig.legend(list(handles.values()), list(handles), loc="lower center", bbox_to_anchor=(0.5, 0.018),
                        ncol=3, frameon=True, fontsize=8, labelcolor=TEXT_SECONDARY, handlelength=2.4,
                        facecolor="#ffffff", edgecolor=GRID_COLOR, framealpha=1.0, borderpad=0.6)
    legend.get_frame().set_linewidth(0.6)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / filename, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", choices=MEASURES)
    measure = parser.parse_args().measure
    universes = [int(m.group(1)) for p in sorted(DATA_DIR.glob(f"{measure}_top*.csv"))
                 if (m := re.fullmatch(rf"{measure}_top(\d+)", p.stem))]
    if not universes:
        raise SystemExit(f"No sweep found for '{measure}' in {DATA_DIR}. Run "
                         f"code/analysis/hyperparameter_boundary_analysis.py {measure} first.")
    for universe in universes:
        frame, bounds = load(measure, universe)
        save(*draw(frame, bounds, "d_inf_mean", True, r"Marginal effect $d_n$"), f"{measure}_top{universe}_dinf.pdf")
        frame["gross"] = np.where(frame["penalty"].str.contains("curr"), frame["gross_psi"], frame["gross_x"])
        fig, axes = draw(frame, bounds, "gross", False, "Portfolio exposure")
        # the joint currency exposure can exceed the asset leverage limit of 2
        top = max(2.25, 1.08 * float(np.nanmax(frame["gross"])))
        axes.flat[0].set_ylim(-0.04 * top, top)
        save(fig, axes, f"{measure}_top{universe}_gross.pdf")
