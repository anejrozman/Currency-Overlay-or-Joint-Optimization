"""
Does the joint program's in-sample edge over the currency overlay survive out of
sample once a mean term is put back into the objective?

The same measurement in_sample_vs_out_of_sample.py makes, for the mean-bearing
programs (``*_w_mean``).  Those minimize a mean-risk functional

    Risk(w) - (1 / gamma) * (mu_a' x + mu_c' psi),

where mu_a is the economy-pooled, James-Stein shrunk asset mean and mu_c is the
carry mean conditioned on the forward premium.  This script asks whether the
joint program's advantage over the overlay -- and its decay in the universe size
-- looks the same once the objective rewards return as well as penalizing risk.

At every quarterly rebalancing date the joint and overlay programs of one measure
are fit on the ten-year estimation window and scored twice by one shared
empirical scorer: on that window (f_in, the ceiling) and on the following quarter
(f_out).  The signed gap g = f_overlay - f_joint is positive when joint is
better.  Per (measure, universe) cell the mean of each gap gets a circular block
bootstrap CI (10000 resamples, block length by Politis-White), alongside the
fraction of dates with g < 0.  Pooled per measure, the relative gap is regressed
on the currency share s = 14 / (14 + 14 * top_N) with block-bootstrapped standard
errors, turning the collapse into one slope.  A supporting table reports each
program's realized risk over equal weight's.

Three things differ from in_sample_vs_out_of_sample.py, each forced by the mean
term:

  * The scorer subtracts the realized mean.  f = Risk(r) - mean(r), the
    gamma-free yardstick these optimizers' own ``score()`` methods use to select
    hyperparameters.  Scoring on the risk alone would count any risk a program
    took in order to capture return as pure loss, measuring an edge neither
    program was pursuing.  The coefficient is 1.0 rather than 1 / gamma for a
    reason the gap makes unavoidable: gamma is selected per date in the
    unregularized family, and joint and overlay routinely select different
    values, so a gamma-scaled f_overlay and f_joint would be two different
    functionals and their difference would not be a gap at all.  All five
    measures return degree-one quantities on the 1e-3 to 1e-2 daily scale against
    a mean near 1e-4, so f stays positive and the relative gap keeps its
    denominator.
  * Both families reuse the hyperparameters production selected, per date.  The
    regularized ones do so for the same reason as everywhere else -- a 256-point
    penalty grid is far too large to re-search here.  The unregularized ones must
    too, which is new: gamma is a searched hyperparameter for them, so
    isa._fixed_hyperparams would reject their grid outright, and fixing gamma at
    one value for the diagnostic would score weights no backtest ever held.  Both
    families therefore need, for every cell, a run recorded in
    results/runs/top_runs_w_mean.csv whose directory carries selected_params.csv
    -- a cell filled in when the backtest was submitted is not yet enough.
  * The currency-count arm is dropped, as in the de-meaned script.  Every cell
    carries the full K = 14 overlay block.

The bundle these optimizers need is wider than isa._solve_weights builds.  They
read the asset-to-economy indicator ``"C"`` for the pooled asset mean, and the
forward-premium frame ``"F"`` for the carry mean, so this module assembles both
per window rather than reusing that helper.

Each cell writes its own shard, gaps_<family>_<measure>_top<N>.csv, the moment it
finishes, and the tables and figures are built by globbing every shard in the
output directory.  Two things follow, both of which a cluster submission depends
on.  A cell that has landed is durable: a job killed by its wall-clock limit
keeps what it already solved.  And the cells can be solved by concurrent jobs
sharing one output directory, because each writes a filename no other job
touches -- there is no shared file to race on and no merge step.  Re-solving a
cell overwrites its own shard and nothing else.

A measure whose ``*_w_mean`` optimizers are not registered yet is warned about
and skipped; all five pairs are registered as of 2026-09-09, so what gates a
measure now is whether its production runs have landed.  A measure missing any
of its requested cells is skipped whole, since regressions() contrasts a
measure's universes against each other and a single-universe cell has nothing to
regress on.  That drop is scoped to what one invocation asks for, which is why
the cluster submission runs one cell per job: --measures smv --universes 10
solves that cell even while smv's top-100 backtest is still in flight.

Usage:
    python code/scripts/in_sample_vs_out_of_sample_w_mean.py
    python code/scripts/in_sample_vs_out_of_sample_w_mean.py --family regularized
    python code/scripts/in_sample_vs_out_of_sample_w_mean.py --measures es --universes 10
    python code/scripts/in_sample_vs_out_of_sample_w_mean.py --plot-only

One cell per job on the cluster, then the tables and figures once they land:
    code/scripts/euler/run_gap_analysis_w_mean.sh --family unregularized
    python code/scripts/in_sample_vs_out_of_sample_w_mean.py --plot-only

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import json
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
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (
    compute_daily_forward_premium,
    compute_excess_currency_returns,
    compute_fully_hedged_returns,
)

logger = logging.getLogger("in_sample_vs_out_of_sample_w_mean")

MEASURES = ("mv", "es", "mad", "smv", "soes")
FAMILIES = ("unregularized", "regularized")
UNIVERSES = (10, 50, 100)

# The mean-bearing production runs are recorded in their own table: top_runs.csv
# points at the mean-free runs and top_demeaned_runs.csv at the de-meaned ones,
# and reusing hyperparameters from either would score weights no mean-bearing
# backtest ever held.
RUNS_FILE = config.RUNS_DIR / "top_runs_w_mean.csv"
OUT_DIR = config.RESULTS_DIR / "in_sample_vs_out_of_sample_w_mean"

MEASURE_LABELS = {
    "es": "Expected Shortfall",
    "mv": "Volatility",
    "mad": "Mean Absolute Deviation",
    "smv": "Semi-Deviation",
    "soes": "Second-Order Expected Shortfall",
}


# Strategy resolution


def _resolve_strategies(measure: str, family: str) -> dict[str, str]:
    """
    The mean-bearing joint and overlay strategies of one measure and family.

    Requires ``mean`` among the name parts, which is what separates these from
    the mean-free programs in_sample_vs_out_of_sample.py resolves; that module
    excludes them symmetrically.

    Raises:
        ValueError: If either role does not resolve to exactly one strategy --
                    which is how a measure whose ``*_w_mean`` optimizers have not
                    been written yet surfaces, and is caught by _available().
    """
    want_reg = family == "regularized"
    resolved = {}
    for role in ("joint", "overlay"):
        parts = {measure, role, "mean"}
        matches = [
            name
            for name in STRATEGY_REGISTRY
            if ("reg" in name.split("_")) == want_reg
            and parts.issubset(set(name.split("_")))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {family} {measure} {role} mean-bearing "
                f"strategy, found {matches}."
            )
        resolved[role] = matches[0]
    return resolved


# Hyperparameters


def _top_run(strategy: str, universe: int) -> int:
    table = pd.read_csv(RUNS_FILE)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()
    row = table[table["strategy_name"] == strategy]
    if row.empty:
        raise ValueError(f"{strategy} has no entry in {RUNS_FILE.name}.")
    # A blank cell is how the table records a production run that has not landed;
    # without this it would surface as int(nan) far from the cause.
    run = row[f"top_{universe}_run"].iloc[0]
    if pd.isna(run):
        raise ValueError(
            f"{strategy} has no top-{universe} run recorded in {RUNS_FILE.name}. "
            f"Both families here reuse the hyperparameters production selected, so "
            f"this cell cannot be solved until that backtest lands."
        )
    return int(run)


def _has_run(strategy: str, universe: int) -> bool:
    """
    Whether *strategy*'s top-*universe* hyperparameters can actually be read.

    A filled cell is not sufficient.  top_runs_w_mean.csv records the run each
    universe was launched under, so a cell points at a run directory from the
    moment the job is submitted -- before the backtest has written the
    selected_params.csv this script reuses.  Checking the cell alone would let
    _available() clear a measure whose runs have not landed, and the sweep would
    then die on a bare FileNotFoundError partway through grid_arm, after the
    cells before it had already been solved.  The file on disk is the real
    precondition, so that is what is tested.
    """
    try:
        run = _top_run(strategy, universe)
    except ValueError:
        return False
    return (config.RUNS_DIR / strategy / f"run_{run}" / "selected_params.csv").exists()


def _require_mean_run(strategy: str, run_dir: Path) -> None:
    """
    Guard that the run being read really is a mean-bearing one.

    top_runs_w_mean.csv is maintained by hand, and a run number copied from
    top_runs.csv would point at a mean-free backtest whose selected_params.csv
    parses perfectly well -- the diagnostic would then reuse hyperparameters from
    a program with no mean term at all and report it under a _w_mean filename.
    The optimizer class name recorded in config.json settles it unambiguously.
    """
    path = run_dir / "config.json"
    if not path.exists():
        raise ValueError(f"{run_dir} has no config.json; cannot verify it is mean-bearing.")
    with open(path) as handle:
        optimizer = json.load(handle).get("optimizer", "")
    if "WithMean" not in optimizer:
        raise ValueError(
            f"{strategy} points at {run_dir.name}, whose optimizer is '{optimizer}' "
            f"-- not a mean-bearing program. Check {RUNS_FILE.name}: this script "
            f"measures the *_w_mean optimizers, and the mean-free ones are covered "
            f"by in_sample_vs_out_of_sample.py."
        )


def _hyperparams(strategy: str, universe: int) -> dict:
    """
    Per-date hyperparameters, keyed by rebalancing date.

    Both families read them from the production run, unlike the two sibling
    scripts where the unregularized branch reads a fixed singleton off the grid.
    That branch is not available here: gamma is a searched hyperparameter of the
    unregularized mean-bearing programs, so isa._fixed_hyperparams would reject
    their grid, and pinning gamma to one value for the diagnostic would score
    weights the backtest never held.

    The regularized programs fix gamma rather than searching it, so their
    selected_params.csv carries no gamma column and the optimizer's own default
    applies -- which is the value production ran with, exactly as intended.
    """
    run_dir = config.RUNS_DIR / strategy / f"run_{_top_run(strategy, universe)}"
    _require_mean_run(strategy, run_dir)
    chosen = pd.read_csv(run_dir / "selected_params.csv", parse_dates=["date"])
    drop = {"date", "n_assets", "val_score"}
    keys = [c for c in chosen.columns if c not in drop]
    return {row["date"]: {k: row[k] for k in keys} for row in chosen.to_dict("records")}


def _available(measures, universes, family: str) -> list[str]:
    """
    Measures this family can actually be solved for.

    Two ways a measure drops out.  Its ``*_w_mean`` optimizers may not be written
    yet, which is the ordinary case while the rollout is in progress.  Or its
    production runs may not have landed on every requested universe -- either a
    blank cell in top_runs_w_mean.csv, or a filled cell naming a run whose
    backtest is still in flight and has written no selected_params.csv.  Either
    way the whole measure is dropped rather than the missing cell, because
    regressions() contrasts a measure's universes against each other and a
    single-universe cell would leave it with no contrast to regress on.
    """
    keep = []
    for measure in measures:
        try:
            names = _resolve_strategies(measure, family)
        except ValueError:
            logger.warning(
                "Skipping %s %s: no mean-bearing optimizers registered for it yet.",
                family, measure,
            )
            continue
        missing = [
            f"{name} top-{universe}"
            for universe in universes
            for name in names.values()
            if not _has_run(name, universe)
        ]
        if missing:
            logger.warning(
                "Skipping %s %s: no usable mean-bearing production run for %s "
                "(no entry in %s, or the backtest has not written "
                "selected_params.csv yet).",
                family, measure, ", ".join(missing), RUNS_FILE.name,
            )
        else:
            keep.append(measure)
    return keep


# Data


def _load(universe: int):
    """
    Returns R_fh, R_c, the universe mask, the asset-to-currency map and the daily
    forward premiums, all on one shared calendar.

    ivo._load returns the first four; F is assembled here because the mean-bearing
    optimizers condition their currency mean on it, and it must be aligned to
    R_c's calendar and column order exactly as the runner aligns it.
    """
    path = config.RAW_DIR / config.UNIVERSE_FILES[universe]
    R_local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    R_local = R_local.where(R_local.abs() < config.RETURN_FILTER_BACKTEST)
    e = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    f_daily = compute_daily_forward_premium(
        load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE), config.TENOR_DAYS_3M
    )
    atc = build_asset_to_currency(path)
    R_c = compute_excess_currency_returns(e, f_daily)
    R_fh = compute_fully_hedged_returns(R_local, e, f_daily, atc)
    R_fh = R_fh[[c for c in R_fh.columns if c in atc]]
    common = R_fh.index.intersection(R_c.index)
    R_fh, R_c = R_fh.loc[common], R_c.loc[common]
    F = f_daily.reindex(index=R_c.index, columns=R_c.columns)
    return R_fh, R_c, isa._build_universe_mask(path, common), atc, F


def _indicator(assets, currencies, atc) -> np.ndarray:
    """
    The bundle's ``"C"``: a (len(assets), len(currencies)) one-hot matrix in the
    currency column order, which is what economy_ids_from_indicator validates and
    argmaxes over.
    """
    C = pd.DataFrame(0.0, index=list(assets), columns=list(currencies))
    for asset in assets:
        ccy = atc.get(asset)
        if ccy in C.columns:
            C.loc[asset, ccy] = 1.0
    return C.to_numpy()


def _solve_weights(optimizer_cls, hyperparams, R_fh_w, R_c_w, C, F_w):
    """
    isa._solve_weights with the two extra bundle keys the mean-bearing optimizers
    read.  The equal-weight control ignores both, so the same call serves it.
    """
    optimizer = optimizer_cls(hyperparams=hyperparams)
    optimizer.fit({"R_fh": R_fh_w, "R_c": R_c_w, "C": C, "F": F_w})
    weights = optimizer.optimize()
    if isinstance(weights, dict):
        x = np.asarray(weights["x"], dtype=float).flatten()
        psi = np.asarray(weights["psi"], dtype=float).flatten()
    else:
        # Asset-only optimizers (the 1/N control) carry no overlay, exactly as
        # Backtester.run treats a bare weight array.
        x = np.asarray(weights, dtype=float).flatten()
        psi = np.zeros(R_c_w.shape[1], dtype=float)
    return x, psi


def _score(measure, R_fh_w, R_c_w, x, psi, alpha) -> float:
    """
    The shared scorer: realized risk minus the realized mean of the same
    portfolio return series, at a reference risk aversion of one.

    One functional for the joint and the overlay alike, whatever gamma either
    selected -- which is what makes f_overlay - f_joint a gap rather than a
    comparison of two different objectives.  See the module docstring.
    """
    risk = isa.evaluate_risk(measure, R_fh_w, R_c_w, x, psi, alpha)
    r = R_fh_w.to_numpy() @ x + R_c_w.to_numpy() @ psi
    return risk - float(np.mean(r)) if r.size else float("nan")


# The sweep


def _sweep(measure, R_fh, R_c, F, atc, windows, solver, family, universe) -> pd.DataFrame:
    names = _resolve_strategies(measure, family)
    tables = {role: _hyperparams(name, universe) for role, name in names.items()}
    names["equal_weight"] = isa.CONTROL_STRATEGY_KEY

    rows = []
    for i, window in enumerate(windows, start=1):
        date = window["date"]
        try:
            chosen = {role: table[date] for role, table in tables.items()}
        except KeyError:
            raise ValueError(
                f"{family} {measure}: no production hyperparameters recorded for "
                f"{date.date()}. The diagnostic and the production run must share "
                f"rebalancing dates."
            ) from None
        chosen["equal_weight"] = {k: v for k, v in chosen["joint"].items() if k == "alpha"}

        assets = window["assets"]
        R_fh_est = R_fh.loc[window["estimation_start"] : window["estimation_end"], assets]
        R_c_est = R_c.loc[window["estimation_start"] : window["estimation_end"]]
        F_est = F.loc[window["estimation_start"] : window["estimation_end"]]
        R_fh_hold = R_fh.loc[window["holding_start"] : window["holding_end"], assets].fillna(0.0)
        R_c_hold = R_c.loc[window["holding_start"] : window["holding_end"]].fillna(0.0)
        # One indicator per window: the three programs share the asset set.
        C = _indicator(assets, R_c.columns, atc)
        logger.debug("[%d/%d] %s: %d assets", i, len(windows), date.date(), len(assets))

        for role, name in names.items():
            params = {**chosen[role], "solver": solver}
            alpha = params.get("alpha")
            try:
                x, psi = _solve_weights(
                    STRATEGY_REGISTRY[name]["optimizer"], params, R_fh_est, R_c_est, C, F_est
                )
                f_in = _score(measure, R_fh_est, R_c_est, x, psi, alpha)
                f_out = _score(measure, R_fh_hold, R_c_hold, x, psi, alpha)
            except (RuntimeError, ValueError) as exc:
                logger.warning("%s %s %s failed: %s", measure, date.date(), role, exc)
                f_in = f_out = float("nan")
            rows.append(
                {
                    "date": date,
                    "strategy": role,
                    "n_assets": len(assets),
                    "in_sample_risk": f_in,
                    "realized_risk": f_out,
                }
            )
    return pd.DataFrame(rows)


def _shard_path(out_dir: Path, family: str, measure: str, universe: int) -> Path:
    """One cell's rows.  Unique per cell, so concurrent jobs never share a file."""
    return out_dir / f"gaps_{family}_{measure}_top{universe}.csv"


def _read_shards(out_dir: Path) -> pd.DataFrame:
    """
    Every cell that has landed in *out_dir*, concatenated and de-duplicated.

    Reads back from disk rather than reusing what this process just solved, so a
    job that swept one cell still builds its tables against the cells its
    siblings finished, and --plot-only needs no sweep at all.

    Re-solving a cell overwrites its own shard, so running the same family twice
    needs no handling.  What does need handling is a cell reaching the directory
    under a second filename -- a hand-made backup, a rename, a copy from another
    out-dir.  The glob would count it twice: the means survive, but every
    bootstrap interval tightens as though the cell carried twice the evidence,
    and nothing in the output says so.  Shards are read newest-wins and
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

    keys = ["family", "measure", "universe", "date"]
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


def grid_arm(solver, family: str, measures, universes, out_dir: Path) -> int:
    """
    The main cells, one per (measure, universe).  Nothing is read from
    results/in_sample_analysis/: those caches were written for the mean-free
    programs and would silently contradict this sweep.

    Each cell is written to its own shard as soon as it finishes rather than
    accumulated for one write at the end, so a job that dies partway -- a SLURM
    wall-clock limit, most likely -- keeps every cell it completed.

    Returns:
        The number of cells solved and written.
    """
    written = 0
    for measure in _available(measures, universes, family):
        for universe in universes:
            logger.info("Grid %s top-%d %s", family, universe, measure)
            R_fh, R_c, mask, atc, F = _load(universe)
            windows = isa._rebalancing_windows(
                R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"]
            )
            rows = _sweep(measure, R_fh, R_c, F, atc, windows, solver, family, universe)
            paired = ivo._pair(rows.dropna(subset=["in_sample_risk", "realized_risk"]), ["date"])
            paired["measure"] = measure
            paired["universe"] = universe
            paired["n_curr"] = ivo.N_CURRENCIES
            paired["arm"] = "grid"
            paired["draw"] = 0
            paired["family"] = family
            path = _shard_path(out_dir, family, measure, universe)
            # Written through a per-process temp file and renamed into place, so a
            # shard is never observed half-written. Two jobs solving the same cell
            # at once then resolve to one whole file rather than an interleaved one.
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            paired.to_csv(tmp, index=False)
            tmp.replace(path)
            logger.info("Wrote %d rows to %s", len(paired), path)
            written += 1
    return written


# Output


def plot(summary: pd.DataFrame, stem: Path, family: str) -> None:
    """
    One panel per measure, drawn for *family* alone so the regularized and
    unregularized programs never share a figure.  A measure with no cell yet
    draws an empty, labelled panel rather than being dropped, so the figure keeps
    the same shape as the rollout fills in.
    """
    summary = summary[summary["family"] == family]
    if summary.empty:
        logger.warning("No %s cells to plot.", family)
        return
    fig, axes = plt.subplots(
        1, len(MEASURES), figsize=(4.33 * len(MEASURES), 4.4), sharex=True, sharey=True
    )
    styles = {"gap_in_rel": ("C0", "In-sample"), "gap_out_rel": ("C3", "Out-of-sample")}

    for ax, measure in zip(axes, MEASURES):
        line = summary[summary["measure"] == measure].sort_values("universe")
        for col, (color, label) in styles.items():
            if line.empty:
                continue
            ax.plot(
                line["universe"], 100 * line[f"{col}_mean"],
                marker="s", ms=5, color=color, zorder=3,
            )
            ax.fill_between(
                line["universe"], 100 * line[f"{col}_lo"], 100 * line[f"{col}_hi"],
                color=color, alpha=0.2, lw=0,
            )
        if line.empty:
            ax.text(
                0.5, 0.5, "not yet solved", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="grey",
            )
        ax.axhline(0.0, color="grey", lw=1.0, zorder=0)
        ax.set_title(MEASURE_LABELS[measure], fontsize=11)
        ax.set_xlabel("Universe size")
        # Log spacing keeps 10/50/100 legible, but the decade ticks a log axis
        # picks by default (10, 100) would drop the 50, so label the three
        # universes explicitly and drop the minor ticks between them.
        ax.set_xscale("log")
        ax.set_xticks(list(UNIVERSES))
        ax.set_xticklabels([str(u) for u in UNIVERSES])
        ax.set_xticks([], minor=True)
        ax.set_xlim(min(UNIVERSES) * 0.8, max(UNIVERSES) * 1.25)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Risk reduction, joint vs sequential (%)")
    handles = [
        Line2D([], [], color=color, marker="s", label=label)
        for color, label in styles.values()
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)
    qualifier = "" if family == "unregularized" else f", {family} programs"
    fig.suptitle(
        "Joint vs overlay optimization across increasing investable universe sizes "
        f"(mean-bearing{qualifier})",
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", stem.with_suffix(f".{ext}"))
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", default="unregularized", choices=list(FAMILIES))
    parser.add_argument(
        "--measures", nargs="+", default=list(MEASURES), choices=list(MEASURES),
        help="Measures to solve (default: all). Every cell is re-solved, so scope "
             "this to what actually needs rebuilding. A measure whose *_w_mean "
             "optimizers are not registered yet is warned about and skipped.",
    )
    parser.add_argument(
        "--universes", nargs="+", type=int, default=list(UNIVERSES), choices=list(UNIVERSES),
        help="Universe sizes to solve (default: all three). Concurrent jobs can share "
             "one --out-dir: each cell writes its own shard, so there is nothing to merge.",
    )
    parser.add_argument(
        "--sweep-only", action="store_true",
        help="Write the cell shards and stop, skipping the tables and figures. Use for a "
             "job covering one cell, whose regression would have no universe contrast.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Build the tables and figures from the shards already in --out-dir, "
             "solving nothing.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        solver = isa._pick_solver()[0]
        solved = grid_arm(solver, args.family, args.measures, args.universes, args.out_dir)
        if solved == 0:
            logger.warning(
                "Nothing solved for %s %s on %s; no shard written.",
                args.family, args.measures, args.universes,
            )
            return
        if args.sweep_only:
            return

    # Read back from disk rather than from what this process solved: a job that
    # swept one cell still tabulates against every sibling cell that has landed.
    gaps = _read_shards(args.out_dir)
    gaps["s"] = [ivo.currency_share(u, k) for u, k in zip(gaps["universe"], gaps["n_curr"])]
    summary = ivo.summarise(gaps)
    reg = ivo.regressions(gaps)
    overfit = ivo.overfitting_table(gaps)

    for name, table in (("gap_summary", summary), ("regression", reg), ("overfitting", overfit)):
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        logger.info("Wrote %s", args.out_dir / f"{name}.csv")

    # One figure per family, never a shared one.
    for family in sorted(gaps["family"].unique()):
        suffix = "" if family == "unregularized" else f"_{family}"
        plot(summary, args.out_dir / f"gap_vs_universe{suffix}", family)
    print(summary.to_string(index=False))
    print()
    print(reg.to_string(index=False))
    print()
    print(overfit.to_string(index=False))


if __name__ == "__main__":
    main()
