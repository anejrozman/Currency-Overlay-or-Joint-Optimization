"""Wall-clock time per rebalancing date of the backtests, read from the Euler job logs (Appendix E.3).

A step is the time between two "Backtest progress" lines. Per program and universe the steps are averaged over
all finished logs at the main ES level 0.8; a running job is used only while its program has no finished log.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import MEASURES, RESULTS_DIR, RUNS_DIR, UNIVERSES, top_runs

LOGS_DIR = RESULTS_DIR.parent / "logs"
OUT_DIR = RESULTS_DIR / "figures" / "runtime"
RESOLUTION_SECONDS = 1.0
MAIN_LEVEL = 0.8
MATCH_SECONDS = 300
# script names of the June runs that wrote into a strategy's run directory
ALIASES = {"assets_full": "reg_es_asset_full", "assets_zero": "reg_es_asset_zero"}
PROGRAMS = [  # strategy pattern, label, colour, line style
    ("reg_{}_asset_full", "Assets-Full", "#1baf7a", "-"),
    ("{}_curr_overlay", "Overlay", "#eb6834", "--"),
    ("reg_{}_curr_overlay", "Regularized Overlay", "#eda100", "-"),
    ("{}_joint", "Joint", "#2a78d6", "--"),
    ("reg_{}_joint", "Regularized Joint", "#e87ba4", "-"),
]

STAMP = r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+INFO\s+"
CONFIG_RE = re.compile(STAMP + r"__main__\s+Config saved to .*/results/runs/([^/]+)/(run_\d+)/config\.json")
STRATEGY_RE = re.compile(STAMP + r"__main__\s+Strategy: (\S+)")
UNIVERSE_RE = re.compile(STAMP + r"__main__\s+Universe: top-(\d+)")
PROGRESS_RE = re.compile(STAMP + r"\S+backtester\s+Backtest progress: .*\((\d+)/(\d+)\)")
END_RE = re.compile(STAMP + r"__main__\s+OOS returns saved")
LEVEL_TAG_RE = re.compile(r"_a([\d.]+)_u\d+_\d+\.err$")


def level(log: dict) -> float:
    """Confidence level from the job tag of the ES-level runs, else from the run config."""
    if tag := LEVEL_TAG_RE.search(log["file"]):
        return float(tag.group(1)) / 100
    path = RUNS_DIR / str(log["run_strategy"]) / str(log["run"]) / "config.json"
    if log["run"] and path.exists():
        config = json.loads(path.read_text())
        if abs((datetime.fromisoformat(config["run_date"]) - log["saved"]).total_seconds()) < MATCH_SECONDS:
            alpha = (config.get("param_grid") or {}).get("alpha")
            if alpha and len(alpha) == 1:
                return float(alpha[0])
    return MAIN_LEVEL


def parse_log(path: Path) -> dict | None:
    saved = run_strategy = run = strategy = universe = end = periods = None
    stamps = []
    for line in path.read_text(errors="ignore").splitlines():
        if (m := STRATEGY_RE.match(line)) and strategy is None:
            strategy = ALIASES.get(m.group(2), m.group(2))
        elif (m := UNIVERSE_RE.match(line)) and universe is None:
            universe = int(m.group(2))
        elif (m := CONFIG_RE.match(line)) and saved is None:
            saved, run_strategy, run = datetime.fromisoformat(m.group(1)), m.group(2), m.group(3)
        elif m := PROGRESS_RE.match(line):
            stamps.append(datetime.fromisoformat(m.group(1)))
            periods = int(m.group(3))
        elif m := END_RE.match(line):
            end = datetime.fromisoformat(m.group(1))
    if strategy is None or universe is None or not stamps:
        return None
    bounds = stamps + [end] if end is not None else stamps
    return {"file": path.name, "saved": saved, "strategy": strategy, "run_strategy": run_strategy, "run": run,
            "universe": universe, "periods": periods, "complete": end is not None,
            "steps": np.array([(b - a).total_seconds() for a, b in zip(bounds[:-1], bounds[1:])])}


def step_series(strategy: str, universe: int, top: pd.DataFrame, logs: list[dict]):
    """Mean step time per rebalancing date and the number of logs averaged."""
    if strategy not in top.index:
        return None
    run = f"run_{int(top.at[strategy, f'top_{universe}_run'])}"
    dates = pd.to_datetime(pd.read_csv(RUNS_DIR / strategy / run / "selected_params.csv")["date"])
    candidates = [log for log in logs if log["strategy"] == strategy and log["universe"] == universe]
    series = []
    for log in [log for log in candidates if log["complete"]] or candidates:
        steps = log["steps"]
        # progress is logged at the start of each period; the rebalancing dates are the
        # len(dates) periods before the final one, earlier and later periods do no work
        first = log["periods"] - len(dates) - 1
        work, late = steps[first:first + len(dates)], steps[first + len(dates):]
        if steps[:first].sum() > 2 * RESOLUTION_SECONDS or (len(late) and late.max() > 2 * RESOLUTION_SECONDS):
            raise RuntimeError(f"{log['file']}: work steps do not line up with the {len(dates)} rebalancing dates.")
        series.append(pd.Series(work, index=dates[:len(work)]).reindex(dates))
    return (pd.concat(series, axis=1).mean(axis=1), len(series)) if series else None


def main() -> None:
    top = top_runs("top_runs.csv")
    logs = [log for f in sorted(LOGS_DIR.glob("*.err")) if (log := parse_log(f)) is not None
            and abs(level(log) - MAIN_LEVEL) < 1e-9]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

    records = []
    for universe in UNIVERSES:
        curves = {}
        for measure in MEASURES:
            for pattern, label, _, _ in PROGRAMS:
                if (result := step_series(pattern.format(measure), universe, top, logs)) is None:
                    continue
                curves[measure, label] = result[0]
                records.append({"universe": universe, "measure": measure, "program": label,
                                "strategy": pattern.format(measure), "logs": result[1],
                                "median_s": result[0].median(), "max_s": result[0].max()})

        fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
        handles = {}
        for axis, (measure, title) in zip(axes.flat, MEASURES.items()):
            for _, label, colour, style in PROGRAMS:
                if (series := curves.get((measure, label))) is not None:
                    handles.setdefault(label, axis.plot(series.index, series.clip(lower=RESOLUTION_SECONDS).to_numpy(),
                                                        color=colour, linestyle=style, linewidth=1.2, label=label)[0])
            if not any((measure, label) in curves for _, label, _, _ in PROGRAMS):
                axis.text(0.5, 0.5, "No logged runs", transform=axis.transAxes, ha="center", va="center", color="0.4")
            axis.set_title(title)
            axis.set_yscale("log")
        axes.flat[-1].axis("off")
        labels = [label for _, label, _, _ in PROGRAMS if label in handles]
        axes.flat[-1].legend([handles[label] for label in labels], labels, loc="center", frameon=False)
        for axis in axes[:, 0]:
            axis.set_ylabel("Seconds per rebalancing date")
        axes[0, 2].tick_params(labelbottom=True)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"top{universe}.pdf", bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(records)
    summary.to_csv(OUT_DIR / "step_times_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
