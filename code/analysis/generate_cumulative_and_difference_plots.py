"""Cumulative wealth of the joint optimizers and their cumulative return difference to the overlay (Chapter 5)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results"
MEASURES = {"mv": ("Variance", "tab:blue"), "smv": ("Semi-Variance", "tab:green"), "mad": ("MAD", "tab:orange"),
            "es": ("ES", "tab:red"), "soes": ("SOES", "tab:purple")}
TYPES = {
    "unregularized": ("top_runs.csv", "{}_{}"),
    "regularized": ("top_runs.csv", "reg_{}_{}"),
    "w_mean": ("top_runs_w_mean.csv", "{}_{}_w_mean"),
}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=TYPES, required=True)
kind = parser.parse_args().type
file, pattern = TYPES[kind]
runs = pd.read_csv(RESULTS / "runs" / file, skipinitialspace=True, index_col=0)
out = RESULTS / "figures" / "cumulative_and_difference"
out.mkdir(parents=True, exist_ok=True)


def returns(strategy, n):
    path = RESULTS / "runs" / strategy / f"run_{runs.at[strategy, f'top_{n}_run']}" / "oos_returns.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)["return"]


for n in (10, 50, 100):
    names = [pattern.format(m, p) for m in MEASURES for p in ("joint", "curr_overlay")]
    r = pd.DataFrame({s: returns(s, n) for s in names + ["equal_weight_optimizer"]}).dropna()
    r = pd.concat([pd.DataFrame(0.0, [r.index[0] - pd.offsets.BDay()], r.columns), r])
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 7), height_ratios=(2, 1),
                                      layout="constrained")
    for m, (label, color) in MEASURES.items():
        joint, overlay = r[pattern.format(m, "joint")], r[pattern.format(m, "curr_overlay")]
        top.plot((1 + joint).cumprod(), color=color, linewidth=1, label=label)
        bottom.plot((joint - overlay).cumsum(), color=color, linewidth=1)
    top.plot((1 + r["equal_weight_optimizer"]).cumprod(), color="black", linestyle="--", linewidth=1, label="EW")
    bottom.axhline(0, color="black", linewidth=0.8)
    top.set(ylabel="Cumulative wealth")
    bottom.set(ylabel="Return difference (joint-overlay)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=False)
    fig.savefig(out / f"{kind}_top{n}.pdf", bbox_inches="tight")
    plt.close(fig)
