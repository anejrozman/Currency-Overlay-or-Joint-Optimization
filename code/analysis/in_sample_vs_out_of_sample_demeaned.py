"""The gap analysis of in_sample_vs_out_of_sample.py for the de-meaned ES and SOES programs (Chapter 6).

Both scoring windows are centered columnwise, so the scorer measures what the de-meaned programs minimize.
Every cell is solved here (the in_sample_analysis caches predate the de-meaning) and written to its own shard
gaps_<family>_<measure>_top<N>.csv, so concurrent cluster jobs can share one output directory; the tables and
figure are built from all shards on disk. Regularized cells reuse the penalties of top_demeaned_runs.csv.

Usage:
    python code/analysis/in_sample_vs_out_of_sample_demeaned.py [--family regularized] [--measures soes]
    python code/analysis/in_sample_vs_out_of_sample_demeaned.py --plot-only
On the cluster: code/scripts/euler/run_gap_analysis_demeaned.sh --family unregularized
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import in_sample_analysis as isa
import in_sample_vs_out_of_sample as ivo
from common import top_runs
from portfolio_backtest import config
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY

logger = logging.getLogger("in_sample_vs_out_of_sample_demeaned")

MEASURES = ("es", "soes")
FAMILIES = ("unregularized", "regularized")
UNIVERSES = (10, 50, 100)
RUNS_FILE = "top_demeaned_runs.csv"
OUT_DIR = config.RESULTS_DIR / "in_sample_vs_out_of_sample_demeaned"
MEASURE_LABELS = {"es": "Expected Shortfall", "soes": "Second-Order Expected Shortfall"}


def top_run(strategy: str, universe: int) -> int:
    table = top_runs(RUNS_FILE)
    if strategy not in table.index:
        raise ValueError(f"{strategy} has no entry in {RUNS_FILE}.")
    run = table.at[strategy, f"top_{universe}_run"]
    if pd.isna(run):
        raise ValueError(f"{strategy} has no top-{universe} run recorded in {RUNS_FILE}.")
    return int(run)


def has_run(strategy: str, universe: int) -> bool:
    """Whether the run's selected_params.csv exists; the table names a run before it has finished."""
    try:
        run = top_run(strategy, universe)
    except ValueError:
        return False
    return (config.RUNS_DIR / strategy / f"run_{run}" / "selected_params.csv").exists()


def require_demeaned(strategy: str, params: dict) -> None:
    if "demean" not in params:
        raise ValueError(f"{strategy} declares no 'demean' setting; use in_sample_vs_out_of_sample.py instead.")
    if not bool(params["demean"]):
        raise ValueError(f"{strategy} is configured with demean=False. Set DEMEAN = True in backtest_config.py "
                         f"(unregularized), or point {RUNS_FILE} at a de-meaned production run (regularized).")


def hyperparams(strategy: str, universe: int, family: str) -> dict:
    if family == "unregularized":
        fixed = isa.fixed_hyperparams(strategy)
        require_demeaned(strategy, fixed)
        return fixed
    run_dir = config.RUNS_DIR / strategy / f"run_{top_run(strategy, universe)}"
    chosen = pd.read_csv(run_dir / "selected_params.csv", parse_dates=["date"])
    keys = [c for c in chosen.columns if c not in {"date", "n_assets", "val_score"}]
    table = {row["date"]: {k: row[k] for k in keys} for row in chosen.to_dict("records")}
    for params in table.values():
        require_demeaned(f"{strategy} ({run_dir.name})", params)
    return table


def read_shards(out_dir: Path) -> pd.DataFrame:
    """All shards, oldest first; a cell present in two files keeps the newest, since a duplicate would
    double its weight in every bootstrap interval."""
    paths = sorted(out_dir.glob("gaps_*.csv"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No gaps_*.csv shards in {out_dir}. Solve at least one cell first.")
    gaps = pd.concat([pd.read_csv(p, parse_dates=["date"]).assign(shard=p.name) for p in paths], ignore_index=True)
    keys = ["family", "measure", "universe", "date"]
    clashing = gaps.duplicated(keys, keep=False)
    if clashing.any():
        logger.warning("Duplicate cells across shards: %s. Keeping the most recent; delete the stale one.",
                       ", ".join(sorted(gaps.loc[clashing, "shard"].unique())))
        gaps = gaps.drop_duplicates(keys, keep="last")
    return gaps.drop(columns="shard").sort_values(keys).reset_index(drop=True)


def grid_arm(solver, family: str, measures, universes, out_dir: Path) -> int:
    """Solve the cells and write each shard as it finishes, so a job killed partway keeps its cells."""
    written = 0
    for measure in ivo.available(measures, universes, family, has_run=has_run):
        for universe in universes:
            logger.info("Grid %s top-%d %s", family, universe, measure)
            R_fh, R_c, mask, _ = isa.load_universe_data(universe)
            windows = isa.rebalancing_windows(R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"])
            rows = ivo.sweep(measure, R_fh, R_c, windows, solver, family, universe, hyperparams, demean=True)
            paired = ivo.tag(rows, measure, universe, ivo.N_CURRENCIES, "grid").assign(family=family)
            path = out_dir / f"gaps_{family}_{measure}_top{universe}.csv"
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # never observed half-written
            paired.to_csv(tmp, index=False)
            tmp.replace(path)
            logger.info("Wrote %d rows to %s", len(paired), path)
            written += 1
    return written


def plot(summary: pd.DataFrame, stem: Path, family: str) -> None:
    summary = summary[summary["family"] == family]
    if summary.empty:
        return
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(4.33 * len(MEASURES), 4.4), sharex=True, sharey=True)
    styles = {"gap_in_rel": ("C0", "In-sample"), "gap_out_rel": ("C3", "Out-of-sample")}
    for ax, measure in zip(axes, MEASURES):
        line = summary[summary["measure"] == measure].sort_values("universe")
        for col, (color, _) in styles.items():
            if line.empty:
                continue
            ax.plot(line["universe"], 100 * line[f"{col}_mean"], marker="s", ms=5, color=color, zorder=3)
            ax.fill_between(line["universe"], 100 * line[f"{col}_lo"], 100 * line[f"{col}_hi"], color=color,
                            alpha=0.2, lw=0)
        ax.axhline(0.0, color="grey", lw=1.0, zorder=0)
        ax.set_title(MEASURE_LABELS[measure], fontsize=11)
        ivo.universe_axis(ax)
    axes[0].set_ylabel("Risk reduction, joint vs sequential (%)")
    axes[0].legend(handles=[Line2D([], [], color=c, marker="s", label=label) for c, label in styles.values()],
                   fontsize=7, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    ivo.save(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default="unregularized", choices=FAMILIES)
    parser.add_argument("--measures", nargs="+", default=list(MEASURES), choices=MEASURES)
    parser.add_argument("--universes", nargs="+", type=int, default=list(UNIVERSES), choices=UNIVERSES)
    parser.add_argument("--sweep-only", action="store_true", help="Write the shards and skip the tables and figures.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--plot-only", action="store_true", help="Build the tables and figures from the shards.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        if grid_arm(isa.pick_solver()[0], args.family, args.measures, args.universes, args.out_dir) == 0:
            logger.warning("Nothing solved for %s %s on %s; no shard written.", args.family, args.measures,
                           args.universes)
            return
        if args.sweep_only:
            return

    gaps = read_shards(args.out_dir)
    gaps["s"] = [ivo.currency_share(u, k) for u, k in zip(gaps["universe"], gaps["n_curr"])]
    tables = {"gap_summary": ivo.summarise(gaps), "regression": ivo.regressions(gaps),
              "overfitting": ivo.overfitting_table(gaps)}
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        logger.info("Wrote %s", args.out_dir / f"{name}.csv")
    for family in sorted(gaps["family"].unique()):
        plot(tables["gap_summary"], args.out_dir / f"gap_vs_universe{'' if family == 'unregularized' else '_' + family}",
             family)
    print("\n\n".join(t.to_string(index=False) for t in tables.values()))


if __name__ == "__main__":
    main()
