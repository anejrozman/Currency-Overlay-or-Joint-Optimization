"""Fill results/runs/top_runs_<measure>_at_different_levels.csv from the level-sweep runs on disk.

The sweep jobs claim run numbers in scheduler order, so the level of a run is read from its config.json: a
sweep run carries "grid_overrides" with an "alpha" key. Rows are named <strategy>_level_<L> and point into
results/runs/<strategy>/. A cell with several finished runs keeps the newest.

Usage:
    python code/scripts/collect_level_runs.py --es             # print only
    python code/scripts/collect_level_runs.py --soes --write   # rewrite the SOES table
    python code/scripts/collect_level_runs.py --es --demean    # the demean=True arm
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from portfolio_backtest import config
from portfolio_backtest.utils.levels import SWEEP_LEVELS, SWEEP_MEASURES, format_level, level_of_alpha

LEVELS = SWEEP_LEVELS
UNIVERSES = (10, 50, 100)
REQUIRED_ARTIFACTS = ("oos_returns.csv", "metrics.json")  # a run directory exists from the moment a job starts


def collect(strategies, demean: bool) -> tuple[dict, list[str]]:
    """(strategy, level, universe) -> run number of the finished sweep runs, and notes on judgement calls."""
    found, notes = {}, []
    for strategy in strategies:
        strategy_dir = config.RUNS_DIR / strategy
        if not strategy_dir.is_dir():
            notes.append(f"{strategy}: no run directory at {strategy_dir}")
            continue
        for run_dir in sorted(strategy_dir.glob("run_*")):
            if not (run_dir / "config.json").is_file():
                continue
            run_config = json.loads((run_dir / "config.json").read_text())
            overrides = run_config.get("grid_overrides") or {}
            if "alpha" not in overrides or bool(overrides.get("demean", True)) != demean:
                continue
            level = level_of_alpha(float(overrides["alpha"]), LEVELS)
            if level is None:
                notes.append(f"{run_dir.name}: alpha={overrides['alpha']} is not one of the sweep levels, skipped")
                continue
            universe = int(run_config["universe_size"])
            missing = [f for f in REQUIRED_ARTIFACTS if not (run_dir / f).is_file()]
            if missing:
                notes.append(f"{strategy}/{run_dir.name} (level {format_level(level)}, top-{universe}): "
                             f"incomplete, missing {', '.join(missing)} -- left blank")
                continue
            found.setdefault((strategy, level, universe), []).append(
                (run_config.get("run_date", ""), int(run_dir.name.removeprefix("run_"))))
    resolved = {}
    for (strategy, level, universe), candidates in found.items():
        candidates.sort()
        resolved[strategy, level, universe] = candidates[-1][1]
        if len(candidates) > 1:
            notes.append(f"{strategy} level {format_level(level)} top-{universe}: {len(candidates)} runs match, kept "
                         f"the newest (run_{candidates[-1][1]}), ignored "
                         f"{', '.join(f'run_{n}' for _, n in candidates[:-1])}")
    return resolved, notes


def preserved_rows(path: Path, strategies) -> list[list[str]]:
    """Rows of the existing table that are not sweep rows, such as the benchmark, kept verbatim."""
    if not path.is_file():
        return [["equal_weight_optimizer", "1", "2", "3"]]
    generated = {f"{s}_level_{format_level(lvl)}" for s in strategies for lvl in LEVELS}
    with open(path, newline="") as fh:
        rows = [[cell.strip() for cell in row] for row in csv.reader(fh)]
    return [r for r in rows if r and r[0] and r[0] != "strategy_name" and r[0] not in generated]


def render(resolved: dict, path: Path, strategies) -> str:
    """The table, column-aligned like the other top-runs files."""
    header = ["strategy_name"] + [f"top_{u}_run" for u in UNIVERSES]
    kept = preserved_rows(path, strategies)
    width = max(max(len(f"{s}_level_{format_level(lvl)}") for s in strategies for lvl in LEVELS),
                len(header[0]), *(len(r[0]) for r in kept or [[""]]))

    def line(name: str, cells: list[str]) -> str:
        return (f"{name + ',':<{width + 1}}" + " " + ", ".join(f"{c:>3}" for c in cells)).rstrip()

    blocks = [[line(header[0], header[1:])]]
    if kept:
        blocks.append([line(r[0], (r[1:] + ["", "", ""])[:3]) for r in kept])
    for s in strategies:
        blocks.append([line(f"{s}_level_{format_level(lvl)}", [str(resolved.get((s, lvl, u), "")) for u in UNIVERSES])
                       for lvl in LEVELS])
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill in a level-sweep table from the runs on disk.")
    measure = parser.add_mutually_exclusive_group(required=True)
    measure.add_argument("--es", dest="measure", action="store_const", const="es")
    measure.add_argument("--soes", dest="measure", action="store_const", const="soes")
    parser.add_argument("--write", action="store_true", help="Rewrite the table (default: print only).")
    parser.add_argument("--demean", action="store_true", help="Collect the demean=True arm.")
    args = parser.parse_args()
    sweep = SWEEP_MEASURES[args.measure]
    strategies = (sweep.overlay, sweep.joint)  # overlay first, as the rows are rendered

    path = config.RUNS_DIR / sweep.runs_file
    resolved, notes = collect(strategies, args.demean)
    table = render(resolved, path, strategies)
    print(table)
    print(f"{len(resolved)}/{len(strategies) * len(LEVELS) * len(UNIVERSES)} {sweep.key} cells filled "
          f"(demean={args.demean}).")
    for note in notes:
        print(f"  note: {note}")
    if args.write:
        path.write_text(table)
        print(f"Wrote {path}")
    elif resolved:
        print("Nothing written. Re-run with --write to update the file.")


if __name__ == "__main__":
    main()
