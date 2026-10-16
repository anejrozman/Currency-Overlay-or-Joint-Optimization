"""
Runtime per rebalancing date of the backtests, one figure per universe size (Appendix E.3).

The Euler job logs in master_thesis/logs/ timestamp one "Backtest progress" line per
rebalancing period, so the time between two consecutive lines is the wall-clock time of
one rebalancing step: grid search on the estimation window, final refit, CVXPY
compilation and scoring. Solver times are not logged.

For every program and universe the step times are averaged over all complete backtest
logs of that program, whatever their date or hyperparameter grid, except the runs of the
ES-level analysis, whose confidence level differs from the 0.8 of the main backtests.
A job that is still running is used, for the dates it has reached, only while its
program has no finished log. The program is read from the log's "Strategy:" line (the
June runs of the asset program used the old script name assets_full) and the universe
from its "Universe:" line.

Usage:
    python results/generate_runtime_plots.py

Author: Anej Rozman
"""

from __future__ import annotations

import glob
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
LOGS_DIR: Path = RESULTS_DIR.parent / "logs"
OUT_DIR: Path = RESULTS_DIR / "figures" / "runtime"

UNIVERSES: tuple[int, ...] = (10, 50, 100)
RESOLUTION_SECONDS = 1.0
MAIN_LEVEL = 0.8
MATCH_SECONDS = 300
# Script names of the June runs that wrote into a strategy's run directory.
STRATEGY_ALIASES: dict[str, str] = {"assets_full": "reg_es_asset_full", "assets_zero": "reg_es_asset_zero"}

MEASURES: dict[str, str] = {
    "mv": "Variance",
    "smv": "Semi-Variance",
    "mad": "MAD",
    "es": "ES",
    "soes": "SOES",
}
# (strategy pattern, legend label, colour, line style). Colours are the first five slots
# of the validated categorical palette, assigned so that the lines lying close together
# (Overlay and Joint, Assets-Full and Regularized Overlay) get clearly distinct hues;
# unregularized programs are dashed as a second cue.
PROGRAMS: list[tuple[str, str, str, str]] = [
    ("reg_{}_asset_full", "Assets-Full", "#1baf7a", "-"),
    ("{}_curr_overlay", "Overlay", "#eb6834", "--"),
    ("reg_{}_curr_overlay", "Regularized Overlay", "#eda100", "-"),
    ("{}_joint", "Joint", "#2a78d6", "--"),
    ("reg_{}_joint", "Regularized Joint", "#e87ba4", "-"),
]

TIMESTAMP = r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+INFO\s+"
CONFIG_RE = re.compile(TIMESTAMP + r"__main__\s+Config saved to .*/results/runs/([^/]+)/(run_\d+)/config\.json")
STRATEGY_RE = re.compile(TIMESTAMP + r"__main__\s+Strategy: (\S+)")
UNIVERSE_RE = re.compile(TIMESTAMP + r"__main__\s+Universe: top-(\d+)")
PROGRESS_RE = re.compile(TIMESTAMP + r"\S+backtester\s+Backtest progress: .*\((\d+)/(\d+)\)")
END_RE = re.compile(TIMESTAMP + r"__main__\s+OOS returns saved")
LEVEL_TAG_RE = re.compile(r"_a([\d.]+)_u\d+_\d+\.err$")


def _top_runs() -> pd.DataFrame:
    table = pd.read_csv(RUNS_DIR / "top_runs.csv", skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()
    return table.set_index("strategy_name")


def _config(strategy: str, run: str) -> dict | None:
    path = RUNS_DIR / strategy / run / "config.json"
    return json.loads(path.read_text()) if path.exists() else None


def _level(log: dict) -> float:
    """Confidence level of a run: from the job tag of the ES-level runs, else from its config."""
    tag = LEVEL_TAG_RE.search(log["file"])
    if tag:
        return float(tag.group(1)) / 100
    config = _config(log["run_strategy"], log["run"]) if log["run"] else None
    if config and abs((datetime.fromisoformat(config["run_date"]) - log["saved"]).total_seconds()) < MATCH_SECONDS:
        alpha = (config.get("param_grid") or {}).get("alpha")
        if alpha and len(alpha) == 1:
            return float(alpha[0])
    return MAIN_LEVEL


def _parse_log(path: Path) -> dict | None:
    """Program, universe and step durations of one backtest log, finished or still running."""
    saved = run_strategy = run = strategy = universe = end = periods = None
    stamps: list[datetime] = []
    for line in path.read_text(errors="ignore").splitlines():
        if (m := STRATEGY_RE.match(line)) and strategy is None:
            strategy = STRATEGY_ALIASES.get(m.group(2), m.group(2))
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
    # A running job has no end time yet; its last period is still in progress.
    bounds = stamps + [end] if end is not None else stamps
    steps = np.array([(b - a).total_seconds() for a, b in zip(bounds[:-1], bounds[1:])])
    return {"file": path.name, "saved": saved, "strategy": strategy, "run_strategy": run_strategy, "run": run,
            "universe": universe, "steps": steps, "periods": periods, "complete": end is not None}


def _backtest_logs() -> list[dict]:
    logs = []
    for file in sorted(glob.glob(str(LOGS_DIR / "*.err"))):
        log = _parse_log(Path(file))
        if log is not None and abs(_level(log) - MAIN_LEVEL) < 1e-9:
            logs.append(log)
    return logs


def _step_series(strategy: str, universe: int, top: pd.DataFrame, logs: list[dict]) -> tuple[pd.Series, int] | None:
    """Mean step time per rebalancing date over all logs of the program and universe."""
    if strategy not in top.index:
        return None
    top_run = f"run_{int(top.at[strategy, f'top_{universe}_run'])}"
    dates = pd.to_datetime(pd.read_csv(RUNS_DIR / strategy / top_run / "selected_params.csv")["date"])
    candidates = [log for log in logs if log["strategy"] == strategy and log["universe"] == universe]
    # Finished logs only; a running job counts only while the program has no finished log.
    finished = [log for log in candidates if log["complete"]]
    series = []
    for log in finished or candidates:
        steps = log["steps"]
        # Progress is logged at the start of each period. The final period and the periods
        # before the first full window do no work; the rebalancing dates are the len(dates)
        # periods before the final one.
        first = log["periods"] - len(dates) - 1
        work = steps[first:first + len(dates)]
        late = steps[first + len(dates):]
        if steps[:first].sum() > 2 * RESOLUTION_SECONDS or (len(late) and late.max() > 2 * RESOLUTION_SECONDS):
            raise RuntimeError(f"{log['file']}: work steps do not line up with the {len(dates)} rebalancing dates.")
        series.append(pd.Series(work, index=dates[:len(work)]).reindex(dates))
    if not series:
        return None
    return pd.concat(series, axis=1).mean(axis=1), len(series)


def main() -> None:
    top = _top_runs()
    logs = _backtest_logs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

    records = []
    for universe in UNIVERSES:
        curves = {}
        for measure in MEASURES:
            for pattern, label, _, _ in PROGRAMS:
                strategy = pattern.format(measure)
                result = _step_series(strategy, universe, top, logs)
                if result is None:
                    continue
                curves[(measure, label)] = result[0]
                records.append({"universe": universe, "measure": measure, "program": label, "strategy": strategy,
                                "logs": result[1], "median_s": result[0].median(), "max_s": result[0].max()})

        fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
        handles: dict[str, object] = {}
        for axis, (measure, title) in zip(axes.flat, MEASURES.items()):
            for _, label, colour, style in PROGRAMS:
                series = curves.get((measure, label))
                if series is None:
                    continue
                line, = axis.plot(series.index, series.clip(lower=RESOLUTION_SECONDS).to_numpy(),
                                  color=colour, linestyle=style, linewidth=1.2, label=label)
                handles.setdefault(label, line)
            if not any((measure, label) in curves for _, label, _, _ in PROGRAMS):
                axis.text(0.5, 0.5, "No logged runs", transform=axis.transAxes, ha="center", va="center",
                          color="0.4")
            axis.set_title(title)
            axis.set_yscale("log")
        legend_axis = axes.flat[-1]
        legend_axis.axis("off")
        legend_axis.legend([handles[label] for _, label, _, _ in PROGRAMS if label in handles],
                           [label for _, label, _, _ in PROGRAMS if label in handles],
                           loc="center", frameon=False)
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
