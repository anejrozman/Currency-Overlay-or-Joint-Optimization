"""
Long share of gross weight per rebalancing date, one column per universe size.

Each line is sum(w+) / sum(|w|) for one strategy: the fraction of gross exposure
held long.  The short share is one minus it, so a single line carries both.  The
top row uses the asset weights, the bottom row the currency overlay psi.

Usage:
    python code/analysis/generate_weight_fraction_plot.py --optimizer-family normal --optimizer es
    python code/analysis/generate_weight_fraction_plot.py --optimizer-family demeaned --optimizer soes
    python code/analysis/generate_weight_fraction_plot.py --optimizer-family with-mean-term --optimizer mv

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR: Path = Path(__file__).resolve().parents[2] / "results"
RUNS_DIR: Path = RESULTS_DIR / "runs"
OUT_DIR: Path = RESULTS_DIR / "figures" / "long_short_split"

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


def _long_share(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("date")[column]
    gross = grouped.apply(lambda w: w.abs().sum())
    longs = grouped.apply(lambda w: w[w > 0].sum())
    return (longs / gross).where(gross > 0).dropna()


def _shares(run_dir: Path) -> tuple[pd.Series, pd.Series] | None:
    if not (run_dir / "weights_assets.csv").exists():
        return None
    assets = pd.read_csv(run_dir / "weights_assets.csv", parse_dates=["date"])
    overlay = pd.read_csv(run_dir / "weights_currencies.csv", parse_dates=["date"])
    return _long_share(assets, "weight"), _long_share(overlay, "psi")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-family", required=True, choices=list(FAMILIES))
    parser.add_argument("--optimizer", required=True, choices=list(MEASURE_TITLES))
    args = parser.parse_args()

    runs_file, suffix, out_name = FAMILIES[args.optimizer_family]
    table = pd.read_csv(RUNS_DIR / runs_file, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()

    fig, axes = plt.subplots(2, len(UNIVERSES), figsize=(15, 7), sharex=True, sharey=True)
    handles: dict[str, object] = {}
    plotted = False

    for column, universe in enumerate(UNIVERSES):
        for strategy, label in _strategies(args.optimizer, suffix, args.optimizer_family):
            row = table[table["strategy_name"] == strategy]
            if row.empty or pd.isna(row[f"top_{universe}_run"].iloc[0]):
                continue
            run = int(row[f"top_{universe}_run"].iloc[0])
            shares = _shares(RUNS_DIR / strategy / f"run_{run}")
            if shares is None:
                continue
            for axis, values in zip(axes[:, column], shares):
                if values.empty:
                    continue
                line, = axis.plot(values.index, values.to_numpy(), linewidth=0.9, label=label)
                handles.setdefault(label, line)
            if strategy != "equal_weight_optimizer":
                plotted = True
        axes[0, column].set_title(f"top-{universe}")
        for axis in axes[:, column]:
            axis.grid(alpha=0.3)

    if not plotted:
        raise SystemExit(f"No {args.optimizer} runs with saved weights recorded in {runs_file}.")

    axes[0, 0].set_ylabel("Assets, long share of gross")
    axes[1, 0].set_ylabel("Currencies, long share of gross")
    axes[0, 0].set_ylim(0.0, 1.0)
    fig.suptitle(f"Long share of gross weight, {MEASURE_TITLES[args.optimizer]}")
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    out_dir = OUT_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{args.optimizer}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
