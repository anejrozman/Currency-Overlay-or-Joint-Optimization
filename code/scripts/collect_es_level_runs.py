"""
Fill in results/runs/top_runs_es_at_different_levels.csv from the runs on disk.

The ES-level sweep (code/scripts/euler/run_es_levels.sh) submits 36 concurrent
jobs, and next_run_dir() claims a run number with an atomic mkdir when each job
starts. Which number ends up holding which level is therefore decided by the
scheduler, and all 18 overlay runs share results/runs/es_curr_overlay/ with the
runs that were already there (the joint runs likewise share es_joint/). This
script reads it back off disk instead of asking anyone to open 36 config.json
files by hand.

A run belongs to the sweep when its config.json carries "grid_overrides" with an
"alpha" key. That marker, not the alpha value itself, is what separates a swept
run from an ordinary one: the level-80 cell has the same alpha as every default
run in the directory, so matching on the value alone would collide with them.

Reading the table it writes
---------------------------
The row name carries the strategy AND the level, but the run directory is named
after the optimizer file only. To find a run, strip the "_level_NN" suffix from
the row name and use that as the directory:

    row es_joint_level_95, cell top_100_run = 21
        -> results/runs/es_joint/run_21

Usage:
    python code/scripts/collect_es_level_runs.py            # print, change nothing
    python code/scripts/collect_es_level_runs.py --write     # rewrite the CSV
    python code/scripts/collect_es_level_runs.py --demean    # sweep run with demean=True

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from portfolio_backtest import config

STRATEGIES: tuple[str, ...] = ("es_curr_overlay", "es_joint")
LEVELS: tuple[int, ...] = (50, 60, 70, 80, 90, 95)
UNIVERSES: tuple[int, ...] = (10, 50, 100)
TABLE_FILE: str = "top_runs_es_at_different_levels.csv"

# Artifacts a finished run always writes. A run directory exists from the moment the
# job starts (next_run_dir creates it to claim the number), so its presence says
# nothing about whether the backtest completed -- these files do.
REQUIRED_ARTIFACTS: tuple[str, ...] = ("oos_returns.csv", "metrics.json")


def _level_of(alpha: float) -> int | None:
    """Map a pinned alpha onto one of the sweep's integer levels, or None."""
    level = round(alpha * 100)
    return level if level in LEVELS and abs(alpha * 100 - level) < 1e-9 else None


def _collect(demean: bool) -> tuple[dict[tuple[str, int, int], int], list[str]]:
    """
    Scan the run directories for sweep runs matching *demean*.

    Returns the (strategy, level, universe) -> run number mapping plus a list of
    human-readable notes about anything that needed a judgement call.
    """
    found: dict[tuple[str, int, int], list[tuple[str, int]]] = {}
    notes: list[str] = []

    for strategy in STRATEGIES:
        strategy_dir = config.RUNS_DIR / strategy
        if not strategy_dir.is_dir():
            notes.append(f"{strategy}: no run directory at {strategy_dir}")
            continue

        for run_dir in sorted(strategy_dir.glob("run_*")):
            config_path = run_dir / "config.json"
            if not config_path.is_file():
                continue
            run_config = json.loads(config_path.read_text())

            overrides = run_config.get("grid_overrides") or {}
            if "alpha" not in overrides:
                continue  # not a sweep run
            if bool(overrides.get("demean", True)) != demean:
                continue  # the other demean arm of the sweep

            level = _level_of(float(overrides["alpha"]))
            if level is None:
                notes.append(
                    f"{run_dir.name}: alpha={overrides['alpha']} is not one of the "
                    f"sweep levels {LEVELS}, skipped"
                )
                continue

            universe = int(run_config["universe_size"])
            missing = [f for f in REQUIRED_ARTIFACTS if not (run_dir / f).is_file()]
            if missing:
                notes.append(
                    f"{strategy}/{run_dir.name} (level {level}, top-{universe}): "
                    f"incomplete, missing {', '.join(missing)} -- left blank"
                )
                continue

            run_number = int(run_dir.name.removeprefix("run_"))
            found.setdefault((strategy, level, universe), []).append(
                (run_config.get("run_date", ""), run_number)
            )

    # One cell, several runs: keep the most recent and say which ones were dropped,
    # so a re-submitted job silently superseding an older one stays visible.
    resolved: dict[tuple[str, int, int], int] = {}
    for key, candidates in found.items():
        candidates.sort()
        strategy, level, universe = key
        resolved[key] = candidates[-1][1]
        if len(candidates) > 1:
            dropped = ", ".join(f"run_{n}" for _, n in candidates[:-1])
            notes.append(
                f"{strategy} level {level} top-{universe}: {len(candidates)} runs "
                f"match, kept the newest (run_{resolved[key]}), ignored {dropped}"
            )

    return resolved, notes


def _preserved_rows(path: Path) -> list[list[str]]:
    """
    Rows of the existing table that the sweep does not own (the benchmark row).

    The file is maintained by hand as well as by this script, so anything that is
    not a generated sweep row is carried through verbatim rather than dropped.
    """
    if not path.is_file():
        return [["equal_weight_optimizer", "1", "2", "3"]]

    generated = {f"{s}_level_{l}" for s in STRATEGIES for l in LEVELS}
    preserved: list[list[str]] = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            cells = [cell.strip() for cell in row]
            if not cells or not cells[0]:
                continue
            if cells[0] == "strategy_name" or cells[0] in generated:
                continue
            preserved.append(cells)
    return preserved


def _render(resolved: dict[tuple[str, int, int], int], path: Path) -> str:
    """Render the table, matching the layout of the other top-runs files."""
    header = ["strategy_name"] + [f"top_{u}_run" for u in UNIVERSES]
    name_width = max(
        len(f"{s}_level_{l}") for s in STRATEGIES for l in LEVELS
    )
    name_width = max(name_width, len(header[0]), *(len(r[0]) for r in _preserved_rows(path) or [[""]]))

    def line(name: str, cells: list[str]) -> str:
        padded = f"{name + ',':<{name_width + 1}}"
        return padded + " " + ", ".join(f"{c:>3}" for c in cells)

    blocks: list[list[str]] = [[line(header[0], header[1:])]]

    preserved = [line(row[0], (row[1:] + ["", "", ""])[:3]) for row in _preserved_rows(path)]
    if preserved:
        blocks.append(preserved)

    for strategy in STRATEGIES:
        block: list[str] = []
        for level in LEVELS:
            cells = [
                str(resolved.get((strategy, level, u), "")) for u in UNIVERSES
            ]
            block.append(line(f"{strategy}_level_{level}", cells))
        blocks.append(block)

    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill in the ES-level sweep table from the runs on disk.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"Rewrite results/runs/{TABLE_FILE} (default: print only).",
    )
    parser.add_argument(
        "--demean",
        action="store_true",
        help="Collect the demean=True arm of the sweep (default: demean=False).",
    )
    args = parser.parse_args()

    path = config.RUNS_DIR / TABLE_FILE
    resolved, notes = _collect(demean=args.demean)
    table = _render(resolved, path)

    expected = len(STRATEGIES) * len(LEVELS) * len(UNIVERSES)
    print(table)
    print(f"{len(resolved)}/{expected} cells filled (demean={args.demean}).")
    for note in notes:
        print(f"  note: {note}")

    if args.write:
        path.write_text(table)
        print(f"Wrote {path}")
    elif resolved:
        print("Nothing written. Re-run with --write to update the file.")


if __name__ == "__main__":
    main()
