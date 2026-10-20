"""
Does the joint program's in-sample edge over the currency overlay depend on where
in the loss distribution the risk measure looks?

The same measurement in_sample_vs_out_of_sample.py makes, with the risk measure
held fixed and its confidence level swept instead: 50, 60 and 70 percent, then a
2.5-point grid from 72.5 to 97.5 (the sweep is
portfolio_backtest.utils.levels.SWEEP_LEVELS).  The sibling scripts vary the
measure across panels; this one varies the level, so where they ask whether the
joint program's advantage is a property of the functional, this one asks whether
it is a property of the tail depth -- ES at 0.50 is a whole-lower-half average,
ES at 0.95 is a tail average, and the overlay's sequential split could plausibly
cost more at one than at the other.

--es and --soes pick which measure is held fixed: Expected Shortfall, or its
second-order variant HMCR_2.  Both are swept over the same levels, so their
panels can be read side by side, and both write to their own table and results
directory, so neither run can overwrite the other.  Neither is a default; a
84-cell sweep is too expensive to start by accident on the wrong measure.  What
each measure is called and where its artifacts go is in levels.SWEEP_MEASURES.

At every quarterly rebalancing date the unregularized joint and overlay
programs are fit on the ten-year estimation window and scored twice by one shared
empirical scorer: on that window (f_in, the ceiling) and on the following quarter
(f_out).  The signed gap g = f_overlay - f_joint is positive when joint is better.
Per (level, universe) cell the mean of each gap gets a circular block bootstrap CI
(10000 resamples, block length by Politis-White), alongside the fraction of dates
with g < 0.  A supporting table reports each program's realized risk over equal
weight's, the absolute penalty both pay where their difference does not.

Two regressions turn the two sweeps into slopes.  Per level, the relative gap is
regressed on the currency share s = 14 / (14 + 14 * top_N) across the three
universes -- the collapse the sibling scripts measure, now resolved by level.  Per
universe, it is regressed on the level itself across the fourteen levels, which is the
slope this script exists for.  Both use block-bootstrapped standard errors.

Four things differ from in_sample_vs_out_of_sample.py, each forced by the sweep:

  * The scorer's alpha follows the cell.  Level 95 weights are scored at 95 and
    level 50 weights at 50, so each cell is measured against the functional its
    programs actually minimized -- the same principle by which the sibling scripts
    score each measure by its own functional.  The relative gap is dimensionless,
    so the panels stay comparable despite being fourteen different functionals.
  * Hyperparameters come from the production runs, not from the registry grid.
    isa._fixed_hyperparams reads the registry, where ALPHA is fixed at 0.80 and
    DEMEAN at True; both are wrong for every cell here, since the sweep pinned
    alpha per run and demean to False (--alpha / --no-demean in run_strategy.py).
    The per-date settings recorded in each run's selected_params.csv are what
    production actually held, so those are read, and each is checked against the
    cell it is claimed to belong to -- see _require_level.
  * Scoring is on raw returns, with no centering.  The swept runs set
    demean = False, so their objective is the raw measure that also rewards a high
    mean; centering the scoring windows, as in_sample_vs_out_of_sample_demeaned.py
    does, would measure an object these programs were not minimizing.
  * The currency-count arm is dropped, as in the de-meaned and mean-bearing
    scripts.  Every cell carries the full K = 14 overlay block.

Two out-of-sample horizons, because one is not enough at the top of the sweep.
f_out is scored on the one quarter actually held, which is what makes it
comparable with the three sibling analyses.  Measured on the top-10 cells the
windows are 2610 trading days in sample and 65 out, so the tail each side averages
over thins sharply with the level:

    level    50    60    70    80    90    95  97.5
    in     1305  1044   783   522   261   131    65
    out      32    26    20    13     6     3     2

At the top of the sweep that quarterly f_out is an extreme order statistic rather
than an expectation: the per-date gap turns heavy-tailed and the intervals widen
for reasons that have nothing to do with either program.  So each cell also scores
the following year (~252 trading days) as ``gap_out_1y``, which lifts the level
97.5 tail from about two observations to six.  The quarterly horizon stays primary
and is what the figures draw; the annual one is the arm to read when the quarterly
figure disagrees with the production backtests at the top end, as it does at 90
and 95.  Its windows overlap by three quarters between consecutive rebalancing
dates, so Politis-White selects a longer block and its intervals are wider than
the date count alone suggests.

Both horizons start at the same rebalance, so the annual window contains the
quarterly one rather than following it; the annual arm is the same decision judged
over a longer horizon, not a later decision.  A date whose annual window is
truncated by the end of the data is left unscored on that arm rather than scored
on a stub.  Every summary row reports the median realized tail count of each
horizon, and any date whose f_out could not be relativized is counted and logged
rather than silently dropped.

Only the unregularized programs were swept across levels, so there is no --family
flag: results/runs/top_runs.csv records the regularized runs at alpha = 0.80
alone.  The shards still carry a "family" column, fixed at "unregularized", so
ivo.summarise, ivo.regressions and ivo.overfitting_table apply unchanged.

Each cell writes its own shard, gaps_unregularized_level<L>_top<N>.csv, the moment
it finishes, and the tables and figures are built by globbing every shard in the
output directory.  Whole levels are spelled without a decimal point and half levels
with one (format_level), so cells solved before the dense grid existed keep their
filenames.  A shard written before the annual horizon was added carries none of its
columns; _read_shards fills them with NaN and says which cells are affected, so a
directory holding both generations still tabulates as one frame.

Two things follow from the sharding, both of which the cluster submission in
euler/run_gap_analysis_levels.sh depends on.  A cell that has landed is
durable: a job killed by its wall-clock limit keeps what it already solved.  And
the 42 cells can be solved by 42 concurrent jobs sharing one output directory,
because each writes a filename no other job touches -- there is no shared file to
race on and no merge step.  Re-solving a cell overwrites its own shard.

Every cell needs a filled entry in the measure's own
results/runs/top_runs_<measure>_at_different_levels.csv, written by
collect_level_runs.py, whose run directory carries selected_params.csv; a cell filled in when the backtest
was submitted is not yet enough.  A level missing any of its requested universes is
skipped whole, since the per-level collapse regression contrasts a level's
universes against each other.  That drop is scoped to what one invocation asks
for, which is why the cluster submission runs one cell per job: --levels 95
--universes 10 solves that cell regardless of what else has landed.

The figures split the two sweeps.  gap_vs_level draws every level of the sweep
against the level axis, one panel per universe.  gap_vs_universe draws one panel
per level against the universe axis and is restricted to the round levels
(ROUND_LEVELS), because fourteen panels do not fit on a page; every level is still
tabulated in both.  Level 50 is excluded from both figures and kept in the tables
(PLOT_EXCLUDE_LEVELS).

Usage:
    python code/scripts/in_sample_vs_out_of_sample_at_diff_levels.py --es
    python code/scripts/in_sample_vs_out_of_sample_at_diff_levels.py --soes --levels 72.5 --universes 10
    python code/scripts/in_sample_vs_out_of_sample_at_diff_levels.py --soes --plot-only

One cell per job on the cluster, then the tables and figures once they land:
    code/scripts/euler/run_gap_analysis_levels.sh --soes
    python code/scripts/in_sample_vs_out_of_sample_at_diff_levels.py --soes --plot-only

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import in_sample_analysis as isa
import in_sample_vs_out_of_sample as ivo
from portfolio_backtest import config
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.bootstrap import block_bootstrap_mean
from portfolio_backtest.utils.levels import (
    ROUND_LEVELS,
    SWEEP_LEVELS,
    SWEEP_MEASURES,
    SweepMeasure,
    alpha_of,
    format_level,
    parse_level,
)

logger = logging.getLogger("in_sample_vs_out_of_sample_at_diff_levels")

FAMILY = "unregularized"
LEVELS = SWEEP_LEVELS
UNIVERSES = (10, 50, 100)

# Trading days in the secondary out-of-sample window. f_out is scored on the one
# quarter actually held, which is what makes it comparable with the three sibling
# analyses; f_out_1y repeats the measurement on the following year so the top of
# the level sweep has a populated tail to be judged on. At level 97.5 the quarterly
# tail is ~2 observations and the annual one ~6. The annual windows of consecutive
# rebalancing dates overlap by three quarters, which inflates the serial dependence
# the block bootstrap has to absorb; Politis-White picks a longer block in response,
# so the intervals stay honest but are wider than the date count alone suggests.
WINDOW_1Y_DAYS = config.TRADING_DAYS
# Below this many trading days the annual window is too truncated to score, which
# happens only for the last rebalancing dates where the data runs out.
WINDOW_1Y_MIN_DAYS = config.TRADING_DAYS // 2

# Levels solved and tabulated but left off the figures, mirroring PLOT_EXCLUDE in
# in_sample_vs_out_of_sample.py. ES at 50% averages the whole lower half rather
# than a tail, which puts it at the edge of what the measure is for; it is also
# the level where the realized gap behaves least like the rest of the sweep
# (top-100: in-sample +5.7%, realized -3.7%, the widest interval in the study and
# the one date lost to the relative gap's denominator). It stays in gaps_*.csv,
# gap_summary.csv and both regressions -- only the figures drop it.
PLOT_EXCLUDE_LEVELS: tuple[float, ...] = (50.0,)
PLOT_LEVELS = tuple(lvl for lvl in LEVELS if lvl not in PLOT_EXCLUDE_LEVELS)

# The by-universe figure draws one panel per level, which the full 14-level sweep
# cannot fit on a page; the round levels carry its shape and the dense grid is what
# the by-level figure exists to show. Every level is still tabulated in both.
UNIVERSE_PANEL_LEVELS = tuple(lvl for lvl in ROUND_LEVELS if lvl in LEVELS)

# Which measure is being swept, and the four things that follow from it. Set by
# _configure() from --es/--soes before anything reads them; module globals rather
# than threaded arguments because nearly every helper below needs one of them and
# the script runs exactly one measure per invocation.
#
# MEASURE     the key isa.evaluate_risk dispatches on
# STRATEGIES  the swept pair, by the role each plays in the gap
# RUNS_FILE   the sweep's own table. top_runs.csv and top_demeaned_runs.csv both
#             point at alpha = 0.80 runs, so reusing either would score one
#             level's weights under fourteen different level labels.
# OUT_DIR     where this measure's shards, tables and figures land. Separate per
#             measure: the shard filenames carry the level and universe but not
#             the measure, so one shared directory would have ES and SOES cells
#             overwriting each other.
MEASURE: str
MEASURE_LABEL: str
MEASURE_SHORT: str
STRATEGIES: dict[str, str]
RUNS_FILE: Path
OUT_DIR: Path

SEED = ivo.SEED


def _configure(measure: SweepMeasure) -> None:
    """Point the module at one measure of the sweep."""
    global MEASURE, MEASURE_LABEL, MEASURE_SHORT, STRATEGIES, RUNS_FILE, OUT_DIR
    MEASURE = measure.key
    MEASURE_LABEL = measure.label
    MEASURE_SHORT = measure.short
    STRATEGIES = measure.strategies
    RUNS_FILE = config.RUNS_DIR / measure.runs_file
    OUT_DIR = config.RESULTS_DIR / measure.out_dir


# Run lookup


def _row_name(role: str, level: int) -> str:
    """
    The table row holding *role*'s run numbers at *level*.

    The rows are named strategy-plus-level while the run directories are named
    after the optimizer file alone, because the sweep reused the existing
    directories rather than making one per level.  _run_dir is the other half of
    this mapping and the two must stay in step.
    """
    return f"{STRATEGIES[role]}_level_{format_level(level)}"


def _run_dir(role: str, run: int) -> Path:
    return config.RUNS_DIR / STRATEGIES[role] / f"run_{run}"


def _top_run(role: str, level: int, universe: int) -> int:
    row_name = _row_name(role, level)
    # The table is written by collect_level_runs.py after the backtests land, so a
    # measure whose sweep has not been collected yet has no file at all. Say that,
    # rather than letting pandas raise a bare FileNotFoundError on a path whose
    # provenance is two scripts away.
    if not RUNS_FILE.is_file():
        raise ValueError(
            f"No sweep table at {RUNS_FILE}. Submit the backtests with "
            f"euler/run_levels.sh --{MEASURE}, then record them with "
            f"collect_level_runs.py --{MEASURE} --write."
        )
    table = pd.read_csv(RUNS_FILE)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()
    row = table[table["strategy_name"] == row_name]
    if row.empty:
        raise ValueError(f"{row_name} has no entry in {RUNS_FILE.name}.")
    # A blank cell is how the table records a production run that has not landed;
    # without this it would surface as int(nan) far from the cause. The table is
    # column-aligned, so an unfilled cell reads back as a run of spaces rather than
    # as NaN -- pandas only calls a field missing when it is genuinely empty, and
    # the name padding before it is part of the field.
    run = row[f"top_{universe}_run"].iloc[0]
    if isinstance(run, str) and not run.strip():
        run = None
    if run is None or pd.isna(run):
        raise ValueError(
            f"{row_name} has no top-{universe} run recorded in {RUNS_FILE.name}. "
            f"Every cell reuses the settings production ran with, so this cell "
            f"cannot be solved until that backtest lands."
        )
    return int(run)


def _has_run(role: str, level: int, universe: int) -> bool:
    """
    Whether this cell's per-date settings can actually be read.

    A filled cell is not sufficient.  The table points at a run directory from the
    moment the job is submitted -- next_run_dir creates it to claim the number --
    so checking the cell alone would let _available() clear a level whose runs are
    still in flight, and the sweep would then die on a bare FileNotFoundError
    partway through grid_arm, after earlier cells had already been solved.  The
    file on disk is the real precondition, so that is what is tested.
    """
    try:
        run = _top_run(role, level, universe)
    except ValueError:
        return False
    return (_run_dir(role, run) / "selected_params.csv").exists()


# Hyperparameters


def _require_level(label: str, level: int, params: dict) -> None:
    """
    Guard that the settings about to be solved really are this cell's.

    The table is maintained by a collector but read by row name, and the row name
    carries the level while the directory does not -- es_joint_level_95 and
    es_joint_level_50 resolve to run directories that are siblings in the same
    folder.  A transposed or mistyped run number therefore parses perfectly well
    and yields a selected_params.csv whose only symptom is a level label that does
    not match the alpha inside it.  Checking the recorded alpha against the cell
    turns that into an error at the point of use.

    demean is checked for the same reason it is checked in the de-meaned script:
    it is a manual switch, not a searched hyperparameter, so a run predating the
    sweep would leave this script measuring the de-meaned programs under raw-ES
    scoring.
    """
    if "alpha" not in params:
        raise ValueError(f"{label} records no 'alpha'; cannot verify it is level {format_level(level)}.")
    alpha = float(params["alpha"])
    if abs(alpha - alpha_of(level)) > 1e-9:
        raise ValueError(
            f"{label} was run at alpha={alpha}, not {alpha_of(level)} as the "
            f"level-{format_level(level)} "
            f"cell requires. Check {RUNS_FILE.name}: its rows are named "
            f"<strategy>_level_<L> but point at run directories that are siblings, so a "
            f"transposed run number is not otherwise detectable."
        )
    if "demean" not in params:
        raise ValueError(
            f"{label} records no 'demean' setting. This script scores the raw "
            f"measure, which matches the sweep's --no-demean runs; use "
            f"in_sample_vs_out_of_sample_demeaned.py for the de-meaned programs."
        )
    if bool(params["demean"]):
        raise ValueError(
            f"{label} was run with demean=True. The level sweep was submitted with "
            f"--no-demean and this script scores the raw measure accordingly; a "
            f"de-meaned run belongs to in_sample_vs_out_of_sample_demeaned.py."
        )


def _hyperparams(role: str, level: int, universe: int) -> dict:
    """
    Per-date settings of one cell, keyed by rebalancing date.

    Read from the production run rather than from the registry grid.  The
    unregularized programs declare no penalties, so their grid is a singleton
    and one dict would serve every date -- but the singleton in the registry is
    alpha = 0.80, demean = True, which is one cell of the fourteen this sweep
    covers and the wrong one for the other thirteen.  What production held is recorded per date, so that is what is read, and
    every date is checked against the cell.
    """
    run = _top_run(role, level, universe)
    run_dir = _run_dir(role, run)
    chosen = pd.read_csv(run_dir / "selected_params.csv", parse_dates=["date"])
    drop = {"date", "n_assets", "val_score"}
    keys = [c for c in chosen.columns if c not in drop]
    table = {row["date"]: {k: row[k] for k in keys} for row in chosen.to_dict("records")}
    for params in table.values():
        _require_level(
            f"{STRATEGIES[role]} {run_dir.name} (level {format_level(level)}, top-{universe})",
            level, params,
        )
    return table


def _available(levels, universes) -> list[int]:
    """
    Levels whose requested cells can all be solved.

    The whole level is dropped rather than the missing cell, because the per-level
    collapse regression contrasts a level's universes against each other and a
    single-universe level would leave it with no contrast to regress on.  Scoped
    to what this invocation asks for, so one-cell cluster jobs are unaffected.
    """
    keep = []
    for level in levels:
        missing = [
            f"{_row_name(role, level)} top-{universe}"
            for universe in universes
            for role in STRATEGIES
            if not _has_run(role, level, universe)
        ]
        if missing:
            logger.warning(
                "Skipping level %s: no usable production run for %s (no entry in %s, "
                "or the backtest has not written selected_params.csv yet).",
                format_level(level), ", ".join(missing), RUNS_FILE.name,
            )
        else:
            keep.append(level)
    return keep


# The sweep


def _sweep(level, R_fh, R_c, windows, solver, universe) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Solve and score both programs plus the 1/N control at every rebalancing date.

    Returns the per-(date, strategy) scores and a per-date frame of window sizes,
    the second of which carries the realized tail counts that make the level's
    out-of-sample noise legible.
    """
    alpha = alpha_of(level)
    tables = {role: _hyperparams(role, level, universe) for role in STRATEGIES}
    names = {**STRATEGIES, "equal_weight": isa.CONTROL_STRATEGY_KEY}
    calendar = R_fh.index

    rows, sizes = [], []
    for i, window in enumerate(windows, start=1):
        date = window["date"]
        try:
            chosen = {role: table[date] for role, table in tables.items()}
        except KeyError:
            raise ValueError(
                f"level {level}: no production settings recorded for {date.date()}. "
                f"The diagnostic and the production run must share rebalancing dates."
            ) from None
        # The control optimizes nothing, so it takes the scoring alpha and nothing
        # else; passing demean or the leverage cap to 1/N would be meaningless.
        chosen["equal_weight"] = {k: v for k, v in chosen["joint"].items() if k == "alpha"}

        assets = window["assets"]
        R_fh_est = R_fh.loc[window["estimation_start"] : window["estimation_end"], assets]
        R_c_est = R_c.loc[window["estimation_start"] : window["estimation_end"]]
        R_fh_hold = R_fh.loc[window["holding_start"] : window["holding_end"], assets].fillna(0.0)
        R_c_hold = R_c.loc[window["holding_start"] : window["holding_end"]].fillna(0.0)

        # The annual window starts where the quarterly one does and runs a year,
        # so the two share their opening quarter rather than being disjoint: this
        # is the same holding period judged over a longer horizon, not a different
        # one. Truncated at the end of the data, where fewer than half a year
        # remains it is not scored at all.
        start_pos = calendar.searchsorted(window["holding_start"])
        span = calendar[start_pos : start_pos + WINDOW_1Y_DAYS]
        long_enough = len(span) >= WINDOW_1Y_MIN_DAYS
        R_fh_1y = R_fh.loc[span, assets].fillna(0.0)
        R_c_1y = R_c.loc[span].fillna(0.0)
        logger.debug("[%d/%d] %s: %d assets", i, len(windows), date.date(), len(assets))

        # Tail counts implied by this window and this level. np.quantile takes the
        # whole lower tail including the interpolated boundary, so these are the
        # counts evaluate_risk actually averages over, to within one observation.
        sizes.append(
            {
                "date": date,
                "n_assets": len(assets),
                "n_days_in": len(R_fh_est),
                "n_days_out": len(R_fh_hold),
                "n_tail_in": max(1, round(len(R_fh_est) * (1.0 - alpha))),
                "n_tail_out": max(1, round(len(R_fh_hold) * (1.0 - alpha))),
                "n_days_out_1y": len(span) if long_enough else 0,
                "n_tail_out_1y": max(1, round(len(span) * (1.0 - alpha))) if long_enough else 0,
            }
        )

        for role, name in names.items():
            params = {**chosen[role], "solver": solver}
            try:
                x, psi = isa._solve_weights(
                    STRATEGY_REGISTRY[name]["optimizer"], params, R_fh_est, R_c_est
                )
                f_in = isa.evaluate_risk(MEASURE, R_fh_est, R_c_est, x, psi, alpha)
                f_out = isa.evaluate_risk(MEASURE, R_fh_hold, R_c_hold, x, psi, alpha)
                f_out_1y = (
                    isa.evaluate_risk(MEASURE, R_fh_1y, R_c_1y, x, psi, alpha)
                    if long_enough
                    else float("nan")
                )
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "level %s %s %s failed: %s", format_level(level), date.date(), role, exc
                )
                f_in = f_out = f_out_1y = float("nan")
            rows.append(
                {
                    "date": date,
                    "strategy": role,
                    "n_assets": len(assets),
                    "in_sample_risk": f_in,
                    "realized_risk": f_out,
                    "realized_risk_1y": f_out_1y,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(sizes)


def _pair_annual(rows: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the annual-horizon gap to the quarterly *paired* frame.

    ivo._pair pivots one realized column, so the second horizon is paired by
    handing it the same frame with realized_risk_1y in that slot and renaming what
    comes back. Joined left onto the quarterly frame: the quarter is the primary
    measurement and decides which dates the cell has, while a date whose annual
    window was truncated at the end of the data, or whose annual overlay risk was
    not strictly positive, simply carries NaN in the annual columns rather than
    dropping out of the cell.
    """
    annual = rows.dropna(subset=["in_sample_risk", "realized_risk_1y"]).copy()
    if annual.empty:
        for col in ("gap_out_1y", "gap_out_1y_rel", "out_joint_1y", "out_overlay_1y", "out_ew_1y"):
            paired[col] = float("nan")
        return paired
    annual["realized_risk"] = annual["realized_risk_1y"]
    annual_paired = ivo._pair(annual, ["date"])
    keep = annual_paired[
        ["date", "gap_out", "gap_out_rel", "out_joint", "out_overlay", "out_ew"]
    ].rename(columns={
        "gap_out": "gap_out_1y",
        "gap_out_rel": "gap_out_1y_rel",
        "out_joint": "out_joint_1y",
        "out_overlay": "out_overlay_1y",
        "out_ew": "out_ew_1y",
    })
    return paired.merge(keep, on="date", how="left")


def _shard_path(out_dir: Path, level: float, universe: int) -> Path:
    """One cell's rows.  Unique per cell, so concurrent jobs never share a file."""
    return out_dir / f"gaps_{FAMILY}_level{format_level(level)}_top{universe}.csv"


def _read_shards(out_dir: Path) -> pd.DataFrame:
    """
    Every cell that has landed in *out_dir*, concatenated and de-duplicated.

    Reads back from disk rather than reusing what this process just solved, so a
    job that swept one cell still builds its tables against the cells its siblings
    finished, and --plot-only needs no sweep at all.

    Re-solving a cell overwrites its own shard, so running the same level twice
    needs no handling.  What does need handling is a cell reaching the directory
    under a second filename -- a hand-made backup, a rename, a copy from another
    out-dir.  The glob would count it twice: the means survive, but every
    bootstrap interval tightens as though the cell carried twice the evidence, and
    nothing in the output would say so.  Shards are read newest-wins and
    de-duplicated on the cell key, with a warning naming the files.
    """
    paths = sorted(out_dir.glob("gaps_*.csv"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(
            f"No gaps_*.csv shards in {out_dir}. Solve at least one cell before "
            f"building the tables and figures."
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["shard"] = path.name
        frames.append(frame)
    gaps = pd.concat(frames, ignore_index=True)

    # Shards written before the annual horizon existed carry none of its columns.
    # They are filled with NaN here rather than guarded at each use site, so a
    # directory holding both generations tabulates as one frame in which those
    # cells simply have no annual measurement. Re-solving a cell adds it.
    annual_columns = (
        "gap_out_1y", "gap_out_1y_rel",
        "out_joint_1y", "out_overlay_1y", "out_ew_1y",
        "n_days_out_1y", "n_tail_out_1y",
    )
    missing = [c for c in annual_columns if c not in gaps.columns]
    for column in missing:
        gaps[column] = float("nan")
    if missing:
        stale = sorted(
            f"level {format_level(lvl)} top-{int(uni)}"
            for lvl, uni in gaps.loc[
                gaps[annual_columns[0]].isna(), ["level", "universe"]
            ].drop_duplicates().itertuples(index=False)
        )
        logger.warning(
            "%d cell(s) predate the annual out-of-sample horizon and carry no "
            "gap_out_1y: %s. Re-solve them (run_gap_analysis_levels.sh --force) "
            "to fill that arm.",
            len(stale), ", ".join(stale),
        )

    keys = ["family", "measure", "level", "universe", "date"]
    clashing = gaps.duplicated(keys, keep=False)
    if clashing.any():
        logger.warning(
            "Duplicate cells across shards: %s. One cell is present under more than "
            "one filename, which would double its weight in every bootstrap interval. "
            "Keeping whichever shard was written most recently; delete the stale one.",
            ", ".join(sorted(gaps.loc[clashing, "shard"].unique())),
        )
        # Paths were read oldest-first, so the newest solve is the last occurrence.
        gaps = gaps.drop_duplicates(keys, keep="last")
    return gaps.drop(columns="shard").sort_values(keys).reset_index(drop=True)


def grid_arm(solver, levels, universes, out_dir: Path) -> int:
    """
    The 18 cells, six levels by three universes.  Nothing is read from
    results/in_sample_analysis/: those caches hold the alpha = 0.80 sweep and
    would silently contradict five of the six levels here.

    Each cell is written to its own shard as soon as it finishes rather than
    accumulated for one write at the end, so a job that dies partway -- a SLURM
    wall-clock limit, most likely -- keeps every cell it completed.

    Returns:
        The number of cells solved and written.
    """
    written = 0
    for level in _available(levels, universes):
        for universe in universes:
            logger.info("Grid level %s top-%d", format_level(level), universe)
            R_fh, R_c, mask, _ = ivo._load(universe)
            windows = isa._rebalancing_windows(
                R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"]
            )
            rows, sizes = _sweep(level, R_fh, R_c, windows, solver, universe)
            scored = rows.dropna(subset=["in_sample_risk", "realized_risk"])
            paired = ivo._pair(scored, ["date"])
            paired = _pair_annual(rows, paired)

            # _pair drops any date whose overlay risk is not strictly positive,
            # because it is the denominator of the relative gap. Raw ES can be
            # negative at the shallow levels -- ES_50 over a quarter that rallied
            # is the mean of its worst half, which need not be a loss -- so the
            # count is logged rather than left to be inferred from n_dates.
            dropped = scored["date"].nunique() - paired["date"].nunique()
            if dropped:
                logger.warning(
                    "level %s top-%d: %d of %d dates dropped, overlay risk not "
                    "strictly positive (the raw measure can be negative at shallow levels).",
                    format_level(level), universe, dropped, scored["date"].nunique(),
                )

            paired = paired.merge(
                sizes[[
                    "date", "n_days_in", "n_days_out", "n_tail_in", "n_tail_out",
                    "n_days_out_1y", "n_tail_out_1y",
                ]],
                on="date", how="left",
            )
            paired["measure"] = MEASURE
            paired["level"] = level
            paired["universe"] = universe
            paired["n_curr"] = ivo.N_CURRENCIES
            paired["arm"] = "grid"
            paired["draw"] = 0
            paired["family"] = FAMILY

            path = _shard_path(out_dir, level, universe)
            # Written through a per-process temp file and renamed into place, so a
            # shard is never observed half-written. Two jobs solving the same cell
            # at once then resolve to one whole file rather than an interleaved one.
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            paired.to_csv(tmp, index=False)
            tmp.replace(path)
            logger.info("Wrote %d rows to %s", len(paired), path)
            written += 1
    return written


# Tables


def summarise(gaps: pd.DataFrame) -> pd.DataFrame:
    """
    ivo.summarise applied per level, with the realized tail counts carried through.

    Called on one level at a time rather than on the whole frame because
    ivo.summarise groups by (family, measure, universe, n_curr, arm, draw) and the
    level is none of those -- pooling the six levels into one group would average
    six different functionals.
    """
    parts = []
    for level, part in gaps.groupby("level"):
        summary = ivo.summarise(part)
        summary.insert(0, "level", level)
        # ivo.summarise knows only the quarterly columns, so the annual horizon is
        # bootstrapped here with the same estimator rather than by widening it.
        for col in ("gap_out_1y", "gap_out_1y_rel"):
            stats = {}
            for universe, cell in part.groupby("universe"):
                values = cell[col].to_numpy() if col in cell else np.array([])
                mean, lo, hi, se, block = block_bootstrap_mean(values)
                stats[universe] = (mean, lo, hi, se, block)
            for i, suffix in enumerate(("mean", "lo", "hi", "se", "block")):
                summary[f"{col}_{suffix}"] = summary["universe"].map(
                    {u: v[i] for u, v in stats.items()}
                )
        # Median rather than mean: the count is an integer per quarter and varies
        # only with the trading-day count of that quarter.
        tails = part.groupby("universe")[["n_tail_in", "n_tail_out", "n_tail_out_1y"]].median()
        for col in ("n_tail_in", "n_tail_out", "n_tail_out_1y"):
            summary[col] = summary["universe"].map(tails[col]).astype(float)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["level", "universe"]
    ).reset_index(drop=True)


def collapse_regressions(gaps: pd.DataFrame) -> pd.DataFrame:
    """
    Relative gap on the currency share across universes, one regression per level.

    The sibling scripts' regression, resolved by level.  A level carrying fewer
    than two universes is skipped: the contrast s - mean(s) is then identically
    zero and the slope would come back as 0/0 rather than as an error.
    """
    parts = []
    for level, part in gaps.groupby("level"):
        if part["universe"].nunique() < 2:
            logger.warning(
                "Level %s has one universe; skipping its collapse regression "
                "(a single point has no slope).", format_level(level),
            )
            continue
        reg = ivo.regressions(part)
        reg.insert(0, "level", level)
        parts.append(reg)
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["level", "family", "measure", "outcome", "slope"])
    )


def level_regressions(gaps: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Relative gap on the ES level across levels, one regression per universe.

    The slope this script exists for: whether the joint program's edge grows or
    shrinks as the measure looks further into the tail.  Built exactly as
    ivo.regressions builds its own -- the per-date OLS slope against a centered
    regressor, resampled by circular block bootstrap -- with the level in place of
    the currency share.  Restricted to dates every level shares, so the slope is a
    within-date contrast across levels and no level's different sample of
    rebalancing dates can masquerade as a level effect.

    The regressor is the level in percent (50 ... 95), so a slope is a change in
    the relative gap per percentage point of confidence level.
    """
    rows = []
    for universe, part in gaps.groupby("universe"):
        cells = {lvl: cell.set_index("date") for lvl, cell in part.groupby("level")}
        levels = sorted(cells)
        if len(levels) < 2:
            logger.warning(
                "Top-%d has one level; skipping its level regression "
                "(a single point has no slope).", universe,
            )
            continue
        dates = sorted(set.intersection(*(set(cells[lvl].index) for lvl in levels)))
        if not dates:
            logger.warning(
                "Top-%d levels share no rebalancing dates; skipping its level "
                "regression.", universe,
            )
            continue
        x = np.array([float(lvl) for lvl in levels])
        contrast = x - x.mean()
        denom = float((contrast**2).sum())
        for col in ("gap_in_rel", "gap_out_rel"):
            Y = np.column_stack([cells[lvl].loc[dates, col].to_numpy() for lvl in levels])
            slope, lo, hi, se, block = block_bootstrap_mean(Y @ contrast / denom, seed)
            rows.append(
                {
                    "family": FAMILY,
                    "measure": MEASURE,
                    "universe": universe,
                    "outcome": col,
                    "slope_per_level_pct": slope,
                    "intercept": float(Y.mean() - slope * x.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "se": se,
                    "block": block,
                    "levels": ",".join(format_level(lvl) for lvl in levels),
                    "n_dates": len(dates),
                    "n_obs": int(Y.size),
                }
            )
    return pd.DataFrame(rows)


def overfitting_table(gaps: pd.DataFrame) -> pd.DataFrame:
    """ivo.overfitting_table applied per level, for the same reason summarise is."""
    parts = []
    for level, part in gaps.groupby("level"):
        table = ivo.overfitting_table(part)
        table.insert(0, "level", level)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


# Figures


def _style_gap_axis(ax) -> None:
    ax.axhline(0.0, color="grey", lw=1.0, zorder=0)
    ax.grid(True, alpha=0.3)


def _draw(ax, line: pd.DataFrame, xcol: str, styles: dict) -> None:
    for col, (color, _) in styles.items():
        if line.empty:
            continue
        ax.plot(line[xcol], 100 * line[f"{col}_mean"], marker="s", ms=5, color=color, zorder=3)
        ax.fill_between(
            line[xcol], 100 * line[f"{col}_lo"], 100 * line[f"{col}_hi"], color=color, alpha=0.2, lw=0
        )


def _legend(ax, styles: dict) -> None:
    handles = [
        Line2D([], [], color=color, marker="s", label=label)
        for color, label in styles.values()
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)


def _save(fig, stem: Path) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", stem.with_suffix(f".{ext}"))
    plt.close(fig)


STYLES = {"gap_in_rel": ("C0", "In-sample"), "gap_out_rel": ("C3", "Out-of-sample")}


def plot_vs_universe(summary: pd.DataFrame, stem: Path) -> None:
    """
    One panel per level, x = universe size: the sibling scripts' figure, with the
    ES level where they put the risk measure.  A level with no cell yet draws an
    empty, labelled panel rather than being dropped, so the figure keeps its shape
    as the cells land.
    """
    if summary.empty:
        logger.warning("No cells to plot.")
        return
    fig, axes = plt.subplots(
        1, len(UNIVERSE_PANEL_LEVELS), figsize=(4.33 * len(UNIVERSE_PANEL_LEVELS), 4.4),
        sharex=True, sharey=True,
    )
    for ax, level in zip(axes, UNIVERSE_PANEL_LEVELS):
        line = summary[summary["level"] == level].sort_values("universe")
        _draw(ax, line, "universe", STYLES)
        if line.empty:
            ax.text(
                0.5, 0.5, "not yet solved", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="grey",
            )
        _style_gap_axis(ax)
        ax.set_title(f"{MEASURE_SHORT} at {format_level(level)}%", fontsize=11)
        ax.set_xlabel("Universe size")
        # Log spacing keeps 10/50/100 legible, but the decade ticks a log axis
        # picks by default (10, 100) would drop the 50, so label the three
        # universes explicitly and drop the minor ticks between them.
        ax.set_xscale("log")
        ax.set_xticks(list(UNIVERSES))
        ax.set_xticklabels([str(u) for u in UNIVERSES])
        ax.set_xticks([], minor=True)
        ax.set_xlim(min(UNIVERSES) * 0.8, max(UNIVERSES) * 1.25)

    axes[0].set_ylabel("Risk reduction, joint vs sequential (%)")
    _legend(axes[0], STYLES)
    _save(fig, stem)


def plot_vs_level(summary: pd.DataFrame, stem: Path) -> None:
    """
    One panel per universe, x = ES level: the swept dimension on the x-axis, which
    is the view plot_vs_universe can only show across panels.
    """
    if summary.empty:
        logger.warning("No cells to plot.")
        return
    fig, axes = plt.subplots(
        1, len(UNIVERSES), figsize=(4.7 * len(UNIVERSES), 4.4), sharex=True, sharey=True
    )
    for ax, universe in zip(axes, UNIVERSES):
        line = summary[
            (summary["universe"] == universe) & summary["level"].isin(PLOT_LEVELS)
        ].sort_values("level")
        _draw(ax, line, "level", STYLES)
        if line.empty:
            ax.text(
                0.5, 0.5, "not yet solved", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="grey",
            )
        _style_gap_axis(ax)
        ax.set_title(f"Top-{universe} universe", fontsize=11)
        ax.set_xlabel(f"{MEASURE_SHORT} confidence level (%)")
        # Every swept level is drawn, but only the round ones are labelled: 13
        # ticks at 2.5-point spacing would overprint each other.
        ax.set_xticks([lvl for lvl in PLOT_LEVELS if lvl in ROUND_LEVELS])
        ax.set_xticklabels(
            [format_level(lvl) for lvl in PLOT_LEVELS if lvl in ROUND_LEVELS]
        )
        ax.set_xticks(list(PLOT_LEVELS), minor=True)
        ax.set_xlim(min(PLOT_LEVELS) - 3, max(PLOT_LEVELS) + 3)

    axes[0].set_ylabel("Risk reduction, joint vs sequential (%)")
    _legend(axes[0], STYLES)
    _save(fig, stem)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    measure = parser.add_mutually_exclusive_group(required=True)
    measure.add_argument(
        "--es", dest="measure", action="store_const", const="es",
        help="Sweep Expected Shortfall.",
    )
    measure.add_argument(
        "--soes", dest="measure", action="store_const", const="soes",
        help="Sweep second-order Expected Shortfall.",
    )
    parser.add_argument(
        "--levels", nargs="+", type=parse_level, default=list(LEVELS),
        metavar="LEVEL",
        help="Confidence levels in percent, e.g. 72.5 (default: the whole sweep). "
             "Every cell is re-solved, so scope this to what actually needs "
             "rebuilding. parse_level rejects anything not in the sweep.",
    )
    parser.add_argument(
        "--universes", nargs="+", type=int, default=list(UNIVERSES), choices=list(UNIVERSES),
        help="Universe sizes to solve (default: all three). Concurrent jobs can share "
             "one --out-dir: each cell writes its own shard, so there is nothing to merge.",
    )
    parser.add_argument(
        "--sweep-only", action="store_true",
        help="Write the cell shards and stop, skipping the tables and figures. Use for a "
             "job covering one cell, whose regressions would have nothing to contrast.",
    )
    # Resolved against the chosen measure in main(); it cannot be defaulted here
    # because the parser is built before --es/--soes has been read.
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Override where the shards, tables and figures go "
             "(default: the measure's own results/ directory).",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Build the tables and figures from the shards already in --out-dir, "
             "solving nothing.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure(SWEEP_MEASURES[args.measure])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.out_dir is None:
        args.out_dir = OUT_DIR
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Sweeping %s into %s", MEASURE_LABEL, args.out_dir)

    if not args.plot_only:
        solver = isa._pick_solver()[0]
        solved = grid_arm(solver, args.levels, args.universes, args.out_dir)
        if solved == 0:
            logger.warning(
                "Nothing solved for %s at levels %s on %s; no shard written.",
                MEASURE, [format_level(lvl) for lvl in args.levels], args.universes,
            )
            return
        if args.sweep_only:
            return

    # Read back from disk rather than from what this process solved: a job that
    # swept one cell still tabulates against every sibling cell that has landed.
    gaps = _read_shards(args.out_dir)
    gaps["s"] = [ivo.currency_share(u, k) for u, k in zip(gaps["universe"], gaps["n_curr"])]
    summary = summarise(gaps)
    collapse = collapse_regressions(gaps)
    by_level = level_regressions(gaps)
    overfit = overfitting_table(gaps)

    tables = (
        ("gap_summary", summary),
        ("regression_collapse", collapse),
        ("regression_level", by_level),
        ("overfitting", overfit),
    )
    for name, table in tables:
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        logger.info("Wrote %s", args.out_dir / f"{name}.csv")

    plot_vs_universe(summary, args.out_dir / "gap_vs_universe")
    plot_vs_level(summary, args.out_dir / "gap_vs_level")

    print(summary.to_string(index=False))
    print()
    print(collapse.to_string(index=False))
    print()
    print(by_level.to_string(index=False))
    print()
    print(overfit.to_string(index=False))


if __name__ == "__main__":
    main()
