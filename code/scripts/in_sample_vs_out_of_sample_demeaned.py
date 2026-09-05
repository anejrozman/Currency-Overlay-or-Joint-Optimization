"""
Does the joint program's in-sample edge over the currency overlay survive out of
sample once the mean is taken out of the objective?

The same measurement in_sample_vs_out_of_sample.py makes, restricted to the two
risk measures the de-meaning switch applies to.  ES and its second-order variant
(HMCR_2) are the only measures here that are not deviation measures: volatility,
MAD and semi-variance subtract the sample mean before measuring risk, so their
objectives are location-free, while raw ES and SOES also reward a high mean.
With ``DEMEAN`` set in backtest_config.py the four ES/SOES joint and overlay
programs center their return matrices columnwise before solving, which turns
their objectives into pure deviation measures.  This script asks whether the
joint program's advantage over the overlay -- and its decay in the universe size
-- looks the same once that mean term is gone.

At every quarterly rebalancing date the joint and overlay programs of one measure
are fit on the ten-year estimation window and scored twice by one shared
empirical scorer: on that window (f_in, the ceiling) and on the following quarter
(f_out).  The signed gap g = f_overlay - f_joint is positive when joint is
better.  Per (measure, universe) cell the mean of each gap gets a circular block
bootstrap CI (10000 resamples, block length by Politis-White), alongside the
fraction of dates with g < 0.  Pooled per measure, the relative gap is regressed
on the currency share s = 14 / (14 + 14 * top_N) with block-bootstrapped standard
errors, turning the collapse into one slope.  A supporting table reports each
program's realized risk over equal weight's, the absolute penalty both pay where
their difference does not.

Three things differ from the non-de-meaned script, each forced by the de-meaning:

  * The scorer centers too.  Both scoring windows are centered columnwise by
    their own means before the risk is read off, so f_in and f_out measure the
    object the de-meaned programs actually minimize.  Centering the columns is
    the same operation as subtracting the mean of the resulting portfolio return
    series -- the map from returns to r = R_fh @ x + R_c @ psi is linear -- and
    it is exactly what the optimizers' own ``score()`` methods do on their
    validation windows: the location a window deviates from is that window's own
    mean, not the training window's.  Scoring these weights with the raw ES
    instead would reward a high realized mean as if it were low risk.
  * Nothing is read from results/in_sample_analysis/.  Those caches were written
    on 2026-08-19, before ``DEMEAN`` was flipped on (61a5b05, 2026-09-04), so
    they hold the raw-ES sweep and would silently contradict this one.  Both
    families are solved here.
  * The currency-count arm is dropped.  It moved the currency share without
    moving the asset count; this script covers the three universe sizes alone,
    so every cell carries the full K = 14 overlay block.

Each cell writes its own shard, gaps_<family>_<measure>_top<N>.csv, the moment it
finishes, and the tables and figure are built by globbing every shard in the
output directory.  Two things follow, both of which the cluster submission in
euler/run_gap_analysis_demeaned.sh depends on.  A cell that has landed is durable:
a job killed by its wall-clock limit keeps what it already solved, rather than
losing the sweep as a single end-of-run write would.  And the six cells can be
solved by six concurrent jobs sharing one output directory, because each writes a
filename no other job touches -- there is no shared file to race on and no merge
step.  Re-solving a cell overwrites its own shard and nothing else.

--family regularized needs, for each measure, a production run of both its joint
and its overlay strategy plus a filled cell in results/runs/top_demeaned_runs.csv
(not top_runs.csv, which records the raw-ES runs), because the per-date penalties
are reused from what production selected rather than re-searched over the
256-point grid.  A measure missing any of its six cells is warned about and
skipped whole, since regressions() contrasts a measure's universes against each
other and a single-universe cell has nothing to regress on.

Usage:
    python code/scripts/in_sample_vs_out_of_sample_demeaned.py
    python code/scripts/in_sample_vs_out_of_sample_demeaned.py --family regularized
    python code/scripts/in_sample_vs_out_of_sample_demeaned.py --measures soes --universes 10
    python code/scripts/in_sample_vs_out_of_sample_demeaned.py --plot-only

One cell per job on the cluster, then the tables and figure once they land:
    code/scripts/euler/run_gap_analysis_demeaned.sh --family unregularized
    python code/scripts/in_sample_vs_out_of_sample_demeaned.py --plot-only

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

import in_sample_analysis as isa
import in_sample_vs_out_of_sample as ivo
from portfolio_backtest import config
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY

logger = logging.getLogger("in_sample_vs_out_of_sample_demeaned")

# The de-meaning switch applies to the ES and SOES joint/overlay programs alone;
# the other three measures demean by construction and have nothing to compare.
MEASURES = ("es", "soes")
FAMILIES = ("unregularized", "regularized")
UNIVERSES = (10, 50, 100)

# The de-meaned production runs are recorded in their own table: top_runs.csv
# still points at the raw-ES runs, and reusing penalties from those would score
# weights no de-meaned backtest ever held.
RUNS_FILE = config.RUNS_DIR / "top_demeaned_runs.csv"
OUT_DIR = config.RESULTS_DIR / "in_sample_vs_out_of_sample_demeaned"

MEASURE_LABELS = {
    "es": "Expected Shortfall",
    "soes": "Second-Order Expected Shortfall",
}


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
            f"{strategy} has no top-{universe} run recorded in {RUNS_FILE.name}. The "
            f"regularized arm reuses the penalties that production selected, so it "
            f"cannot run on this cell until that backtest lands."
        )
    return int(run)


def _has_run(strategy: str, universe: int) -> bool:
    try:
        _top_run(strategy, universe)
    except ValueError:
        return False
    return True


def _require_demeaned(strategy: str, params: dict) -> None:
    """
    Guard that the weights about to be solved are the de-meaned ones.

    ``demean`` is a manual switch, not a searched hyperparameter, so a stale
    ``DEMEAN = False`` in backtest_config.py or a production run predating the
    flip would leave this script silently measuring the raw programs under a
    de-meaned filename.  Cheap to check, and the failure it prevents is invisible
    in the output.
    """
    if "demean" not in params:
        raise ValueError(
            f"{strategy} declares no 'demean' setting. This script measures the "
            f"de-meaned ES and SOES programs; use in_sample_vs_out_of_sample.py "
            f"for the raw ones."
        )
    if not bool(params["demean"]):
        raise ValueError(
            f"{strategy} is configured with demean=False. Set DEMEAN = True in "
            f"backtest_config.py (unregularized), or point {RUNS_FILE.name} at a "
            f"de-meaned production run (regularized)."
        )


def _hyperparams(strategy: str, universe: int, family: str) -> dict:
    """
    Per-date hyperparameters, keyed by rebalancing date.

    The unregularized programs declare no penalties, so their production grid is a
    singleton and one dict serves every date.  The regularized programs select
    penalties by validation at each date over a 256-point grid, which is far too
    large to re-search here; we reuse the selections the de-meaned production
    backtest made and recorded, so the diagnostic scores the weights production
    actually held.
    """
    if family == "unregularized":
        fixed = isa._fixed_hyperparams(strategy)
        _require_demeaned(strategy, fixed)
        return fixed
    run_dir = config.RUNS_DIR / strategy / f"run_{_top_run(strategy, universe)}"
    chosen = pd.read_csv(run_dir / "selected_params.csv", parse_dates=["date"])
    drop = {"date", "n_assets", "val_score"}
    keys = [c for c in chosen.columns if c not in drop]
    table = {row["date"]: {k: row[k] for k in keys} for row in chosen.to_dict("records")}
    for params in table.values():
        _require_demeaned(f"{strategy} ({run_dir.name})", params)
    return table


def _available(measures, universes, family: str) -> list[str]:
    """
    Measures this family can actually be solved for.

    The regularized arm reuses the penalties production selected per date, so a
    measure whose backtest has not landed cannot be solved at all -- the runs
    table records that as a blank cell.  The whole measure is dropped rather than
    the missing cell, because regressions() contrasts a measure's universes
    against each other and a single-universe cell would leave it with no contrast
    to regress on.
    """
    if family == "unregularized":
        return list(measures)
    keep = []
    for measure in measures:
        missing = [
            f"{name} top-{universe}"
            for universe in universes
            for name in ivo._resolve_strategies(measure, family).values()
            if not _has_run(name, universe)
        ]
        if missing:
            logger.warning(
                "Skipping %s %s: no de-meaned production run recorded for %s.",
                family, measure, ", ".join(missing),
            )
        else:
            keep.append(measure)
    return keep


# The sweep


def _center(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Columnwise centering, the scoring counterpart of what the optimizers do
    before solving.  Linear in the weights, so centering the columns and then
    forming r = R_fh @ x + R_c @ psi gives exactly r minus its own mean.
    """
    return frame - frame.mean(axis=0)


def _sweep(measure, R_fh, R_c, windows, solver, family, universe) -> pd.DataFrame:
    names = ivo._resolve_strategies(measure, family)
    tables = {role: _hyperparams(name, universe, family) for role, name in names.items()}
    names["equal_weight"] = isa.CONTROL_STRATEGY_KEY

    rows = []
    for i, window in enumerate(windows, start=1):
        date = window["date"]
        try:
            chosen = {
                role: (table if family == "unregularized" else table[date])
                for role, table in tables.items()
            }
        except KeyError:
            raise ValueError(
                f"{family} {measure}: no production hyperparameters recorded for {date.date()}. "
                f"The diagnostic and the production run must share rebalancing dates."
            ) from None
        chosen["equal_weight"] = {k: v for k, v in chosen["joint"].items() if k == "alpha"}

        assets = window["assets"]
        R_fh_est = R_fh.loc[window["estimation_start"] : window["estimation_end"], assets]
        R_c_est = R_c.loc[window["estimation_start"] : window["estimation_end"]]
        R_fh_hold = R_fh.loc[window["holding_start"] : window["holding_end"], assets].fillna(0.0)
        R_c_hold = R_c.loc[window["holding_start"] : window["holding_end"]].fillna(0.0)
        # Solved on the raw frames -- the optimizers center internally -- and
        # scored on the centered ones. Centered once per window rather than once
        # per role: the three programs share both scoring windows.
        scoring = {
            "in": (_center(R_fh_est), _center(R_c_est)),
            "out": (_center(R_fh_hold), _center(R_c_hold)),
        }
        logger.debug("[%d/%d] %s: %d assets", i, len(windows), date.date(), len(assets))

        for role, name in names.items():
            params = {**chosen[role], "solver": solver}
            alpha = params.get("alpha")
            try:
                x, psi = isa._solve_weights(
                    STRATEGY_REGISTRY[name]["optimizer"], params, R_fh_est, R_c_est
                )
                f_in = isa.evaluate_risk(measure, *scoring["in"], x, psi, alpha)
                f_out = isa.evaluate_risk(measure, *scoring["out"], x, psi, alpha)
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
    Every cell that has landed in *out_dir*, concatenated.

    Reads back from disk rather than reusing what this process just solved, so a
    job that swept one cell still builds its tables against the cells its
    siblings finished, and --plot-only needs no sweep at all.
    """
    paths = sorted(out_dir.glob("gaps_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No gaps_*.csv shards in {out_dir}. Solve at least one cell before "
            f"building the tables and figure."
        )
    return pd.concat(
        [pd.read_csv(path, parse_dates=["date"]) for path in paths], ignore_index=True
    )


def grid_arm(solver, family: str, measures, universes, out_dir: Path) -> int:
    """
    The six main cells, two measures by three universes.  Both families are
    solved here: the in_sample_analysis caches predate the de-meaning flip, so
    there is nothing to read back.

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
            R_fh, R_c, mask, _ = ivo._load(universe)
            windows = isa._rebalancing_windows(
                R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"]
            )
            rows = _sweep(measure, R_fh, R_c, windows, solver, family, universe)
            paired = ivo._pair(rows.dropna(subset=["in_sample_risk", "realized_risk"]), ["date"])
            paired["measure"] = measure
            paired["universe"] = universe
            paired["n_curr"] = ivo.N_CURRENCIES
            paired["arm"] = "grid"
            paired["draw"] = 0
            paired["family"] = family
            path = _shard_path(out_dir, family, measure, universe)
            paired.to_csv(path, index=False)
            logger.info("Wrote %d rows to %s", len(paired), path)
            written += 1
    return written


# Output


def plot(summary: pd.DataFrame, stem: Path, family: str) -> None:
    summary = summary[summary["family"] == family]
    if summary.empty:
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
                line["universe"], line[f"{col}_mean"],
                marker="s", ms=5, color=color, zorder=3,
            )
            ax.fill_between(
                line["universe"], line[f"{col}_lo"], line[f"{col}_hi"],
                color=color, alpha=0.2, lw=0,
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

    axes[0].set_ylabel("Mean (overlay - joint) / overlay")
    handles = [
        Line2D([], [], color=color, marker="s", label=label)
        for color, label in styles.values()
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)
    qualifier = "" if family == "unregularized" else f", {family} programs"
    fig.suptitle(
        "Joint vs overlay optimization across increasing investable universe sizes "
        f"(de-meaned{qualifier})",
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
        help="Measures to solve (default: both). Every cell is re-solved, so scope "
             "this to what actually needs rebuilding.",
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
        solved = grid_arm(
            solver, args.family, args.measures, args.universes, args.out_dir
        )
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
