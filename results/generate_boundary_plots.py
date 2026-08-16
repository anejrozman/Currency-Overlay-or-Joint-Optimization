"""
Penalty boundary diagnostics, one figure pair per universe size.

Reads the sweep written by code/scripts/hyperparameter_boundary_analysis.py and
draws, for each penalty, how far the affected weight block still moves as the
penalty approaches and passes its configured grid bound. Lambda is shown relative
to that bound, so the bound sits at 1.0 in every panel.

Usage:
    python results/generate_boundary_plots.py es
    python results/generate_boundary_plots.py mv

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

RESULTS_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = RESULTS_DIR / "hyperparameter_boundary_analysis"
OUT_DIR: Path = RESULTS_DIR / "figures" / "boundary_plots"

STRATEGY_COLORS: dict[str, str] = {
    "overlay": "#eb6834",
    "joint":   "#1baf7a",
}
STRATEGY_LABELS: dict[str, str] = {
    "overlay": "Currency overlay",
    "joint":   "Joint",
}
# Asymmetric widths so the two series stay legible where they coincide exactly:
# the thinner joint line sits on top of an overlay fringe rather than erasing it.
STRATEGY_LINEWIDTH: dict[str, float] = {
    "overlay": 2.3,
    "joint":   1.2,
}
STRATEGY_ZORDER: dict[str, float] = {
    "overlay": 3.0,
    "joint":   4.0,
}
BOUND_LINEWIDTH: float = 1.0

TEXT_PRIMARY: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
GRID_COLOR: str = "#d9d8d4"
BOUND_COLOR: str = "#8a8983"
LEGEND_FACE_COLOR: str = "#ffffff"

MEASURE_TITLES: dict[str, str] = {
    "es": "Expected Shortfall",
    "mv": "Mean-Variance",
}

# Panel order: asset block on the top row, currency block on the bottom row.
PENALTY_ORDER: list[str] = [
    "lambda_l1_asset",
    "lambda_l2_asset",
    "lambda_l1_curr",
    "lambda_l2_curr",
]
PENALTY_LABELS: dict[str, str] = {
    "lambda_l1_asset": r"$\lambda_1^{a}$",
    "lambda_l2_asset": r"$\lambda_2^{a}$",
    "lambda_l1_curr":  r"$\lambda_1^{c}$",
    "lambda_l2_curr":  r"$\lambda_2^{c}$",
}

# Anything smaller is at solver tolerance, so it is graphed as zero rather than
# stretching the shared log axis across another five empty decades.
NOISE_FLOOR: float = 1e-10

plt.rcParams["pdf.fonttype"] = 42


def _family(strategy: str) -> str:
    return "overlay" if "overlay" in strategy.split("_") else "joint"


def _format_bound(value: float) -> str:
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    if exponent == 0:
        return f"${mantissa:g}$"
    if abs(mantissa - 1.0) < 1e-9:
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:g}\times10^{{{exponent}}}$"


def _discover(measure: str) -> list[int]:
    universes = []
    for path in sorted(DATA_DIR.glob(f"{measure}_top*.csv")):
        match = re.fullmatch(rf"{measure}_top(\d+)", path.stem)
        if match:
            universes.append(int(match.group(1)))
    return universes


def _load(measure: str, universe: int) -> tuple[pd.DataFrame, dict, dict[str, float]]:
    frame = pd.read_csv(DATA_DIR / f"{measure}_top{universe}.csv")
    sidecar = json.loads((DATA_DIR / f"{measure}_top{universe}.json").read_text())

    bounds: dict[str, float] = {}
    for penalty in frame["penalty"].unique():
        values = {
            float(per_strategy[penalty])
            for per_strategy in sidecar["penalty_bounds"].values()
            if penalty in per_strategy
        }
        if len(values) != 1:
            raise ValueError(
                f"Strategies disagree on the bound for {penalty}: {sorted(values)}. "
                f"A single bound line per panel would be ambiguous."
            )
        bounds[penalty] = values.pop()
    return frame, sidecar, bounds


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)


def _draw_panels(
    frame: pd.DataFrame,
    bounds: dict[str, float],
    value_column: str,
    log_y: bool,
    y_label: str,
) -> tuple[plt.Figure, np.ndarray]:
    # Each penalty keeps its own absolute lambda axis, so the panels do not share x.
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=False, sharey=True)

    for ax, penalty in zip(axes.flat, PENALTY_ORDER):
        panel = frame[frame["penalty"] == penalty]
        bound = bounds[penalty]

        for strategy, group in panel.groupby("strategy"):
            group = group[group["lambda"] > 0].sort_values("lambda")
            family = _family(strategy)
            x = group["lambda"].to_numpy()
            y = group[value_column].to_numpy()

            if log_y:
                spread = group["d_inf_std"].to_numpy()
                lower = y - spread
                # On a log axis a non-positive lower edge renders as a spike to the
                # floor, which is an artefact of the scale rather than a wider
                # interval. Draw the band only where it excludes zero.
                ax.fill_between(
                    x,
                    np.maximum(lower, NOISE_FLOOR),
                    np.maximum(y + spread, NOISE_FLOOR),
                    where=lower > 0.0,
                    color=STRATEGY_COLORS[family], alpha=0.16, linewidth=0, zorder=2,
                )
                y = np.maximum(y, NOISE_FLOOR)

            ax.plot(
                x, y,
                label=STRATEGY_LABELS[family],
                color=STRATEGY_COLORS[family],
                linewidth=STRATEGY_LINEWIDTH[family],
                solid_capstyle="round",
                zorder=STRATEGY_ZORDER[family],
            )

        ax.axvline(
            bound, color=BOUND_COLOR, linewidth=BOUND_LINEWIDTH,
            linestyle=(0, (4, 3)), zorder=5,
        )
        # Annotated just outside the axes, the only region no panel's data reaches
        # in either figure, and spelled out because a bound such as 5e-3 falls
        # between decade ticks and is awkward to read off a log axis.
        ax.text(
            bound, 1.015, _format_bound(bound),
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", va="bottom",
            color=BOUND_COLOR, fontsize=7.5, zorder=6,
        )
        ax.set_xscale("log")
        ax.set_xlim(bound * 7e-4, bound * 1.5e2)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(
            PENALTY_LABELS[penalty],
            color=TEXT_PRIMARY, fontsize=9.5, pad=20,
        )
        _style_axes(ax)

    for ax in axes.flat:
        ax.set_xlabel(r"$\lambda$", color=TEXT_SECONDARY, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=9)

    if log_y:
        axes.flat[0].set_ylim(NOISE_FLOOR / 3.0, 3.0)
    return fig, axes


def _finish(
    fig: plt.Figure,
    axes: np.ndarray,
    measure: str,
    filename: str,
) -> None:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    bound_handle = plt.Line2D(
        [], [], color=BOUND_COLOR, linewidth=BOUND_LINEWIDTH, linestyle=(0, (4, 3))
    )
    seen["Configured grid bound"] = bound_handle

    fig.suptitle(
        f"Hyperparameter Analysis, regularized {MEASURE_TITLES[measure]} strategies",
        color=TEXT_PRIMARY, fontsize=12, y=0.995,
    )

    legend = fig.legend(
        list(seen.values()), list(seen),
        loc="lower center", bbox_to_anchor=(0.5, 0.018),
        ncol=3, frameon=True, fontsize=9, labelcolor=TEXT_SECONDARY,
        handlelength=2.4, facecolor=LEGEND_FACE_COLOR, edgecolor=GRID_COLOR,
        framealpha=1.0, borderpad=0.6,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.tight_layout(rect=(0, 0.085, 1, 0.985))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / filename, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "measure",
        choices=sorted(MEASURE_TITLES),
        help="Optimizer family to plot.",
    )
    measure = parser.parse_args().measure

    universes = _discover(measure)
    if not universes:
        raise SystemExit(
            f"No sweep found for '{measure}' in {DATA_DIR}. Run "
            f"code/scripts/hyperparameter_boundary_analysis.py {measure} first."
        )

    for universe in universes:
        frame, _, bounds = _load(measure, universe)

        fig, axes = _draw_panels(
            frame, bounds, "d_inf_mean", log_y=True,
            y_label=r"$d_\infty$, change in block per grid step",
        )
        _finish(
            fig, axes, measure,
            filename=f"{measure}_top{universe}_dinf.pdf",
        )

        frame = frame.assign(
            gross=np.where(
                frame["penalty"].str.contains("curr"),
                frame["gross_psi"],
                frame["gross_x"],
            )
        )
        fig, axes = _draw_panels(
            frame, bounds, "gross", log_y=False,
            y_label="Portfolio exposure",
        )
        axes.flat[0].set_ylim(-0.08, 2.25)
        _finish(
            fig, axes, measure,
            filename=f"{measure}_top{universe}_gross.pdf",
        )


if __name__ == "__main__":
    main()
