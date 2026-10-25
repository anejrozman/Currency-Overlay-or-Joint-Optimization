"""The gap analysis of in_sample_vs_out_of_sample.py with ES or SOES held fixed and its level swept (Chapter 6).

Each cell (level x universe) reuses the per-date settings of the level-sweep production runs, which are checked
against the cell's alpha and demean=False, and is scored on raw returns at its own alpha. Besides the held
quarter, each date is also scored on the following year (gap_out_1y), since at high levels the quarterly tail
holds only a few observations. Cells are written to their own shards as in the de-meaned script. Two
regressions: the relative gap on the currency share per level, and on the level per universe.

Usage:
    python code/analysis/in_sample_vs_out_of_sample_at_diff_levels.py --es
    python code/analysis/in_sample_vs_out_of_sample_at_diff_levels.py --soes --levels 72.5 --universes 10
    python code/analysis/in_sample_vs_out_of_sample_at_diff_levels.py --soes --plot-only
On the cluster: code/scripts/euler/run_gap_analysis_levels.sh --soes
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import in_sample_analysis as isa
import in_sample_vs_out_of_sample as ivo
from portfolio_backtest import config
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.bootstrap import block_bootstrap_mean
from portfolio_backtest.utils.levels import (ROUND_LEVELS, SWEEP_LEVELS, SWEEP_MEASURES, SweepMeasure, alpha_of,
                                             format_level, parse_level)

logger = logging.getLogger("in_sample_vs_out_of_sample_at_diff_levels")

FAMILY = "unregularized"
LEVELS = SWEEP_LEVELS
UNIVERSES = (10, 50, 100)
WINDOW_1Y_DAYS = config.TRADING_DAYS
WINDOW_1Y_MIN_DAYS = config.TRADING_DAYS // 2  # shorter annual windows at the end of the data are not scored
PLOT_LEVELS = tuple(lvl for lvl in LEVELS if lvl != 50.0)  # level 50 is tabulated but not drawn
UNIVERSE_PANEL_LEVELS = tuple(lvl for lvl in ROUND_LEVELS if lvl in LEVELS)
ANNUAL_COLUMNS = ("gap_out_1y", "gap_out_1y_rel", "out_joint_1y", "out_overlay_1y", "out_ew_1y",
                  "n_days_out_1y", "n_tail_out_1y")
STYLES = {"gap_in_rel": ("C0", "In-sample"), "gap_out_rel": ("C3", "Out-of-sample")}
SEED = ivo.SEED

# set by configure() from --es / --soes
MEASURE: str
MEASURE_LABEL: str
MEASURE_SHORT: str
STRATEGIES: dict[str, str]
RUNS_FILE: Path
OUT_DIR: Path


def configure(measure: SweepMeasure) -> None:
    global MEASURE, MEASURE_LABEL, MEASURE_SHORT, STRATEGIES, RUNS_FILE, OUT_DIR
    MEASURE, MEASURE_LABEL, MEASURE_SHORT = measure.key, measure.label, measure.short
    STRATEGIES = measure.strategies
    RUNS_FILE = config.RUNS_DIR / measure.runs_file
    OUT_DIR = config.RESULTS_DIR / measure.out_dir


def row_name(role: str, level) -> str:
    """Table rows carry the level, while all levels share the strategy's run directory."""
    return f"{STRATEGIES[role]}_level_{format_level(level)}"


def run_dir(role: str, run: int) -> Path:
    return config.RUNS_DIR / STRATEGIES[role] / f"run_{run}"


def top_run(role: str, level, universe: int) -> int:
    name = row_name(role, level)
    if not RUNS_FILE.is_file():
        raise ValueError(f"No sweep table at {RUNS_FILE}. Submit the backtests with euler/run_levels.sh --{MEASURE}, "
                         f"then record them with collect_level_runs.py --{MEASURE} --write.")
    table = pd.read_csv(RUNS_FILE)
    table.columns = [c.strip() for c in table.columns]
    row = table[table["strategy_name"].astype(str).str.strip() == name]
    if row.empty:
        raise ValueError(f"{name} has no entry in {RUNS_FILE.name}.")
    run = row[f"top_{universe}_run"].iloc[0]
    # the table is column-aligned, so an unfilled cell reads back as spaces
    if (isinstance(run, str) and not run.strip()) or pd.isna(run):
        raise ValueError(f"{name} has no top-{universe} run recorded in {RUNS_FILE.name}.")
    return int(run)


def has_run(role: str, level, universe: int) -> bool:
    """Whether the run's selected_params.csv exists; the table names a run before it has finished."""
    try:
        run = top_run(role, level, universe)
    except ValueError:
        return False
    return (run_dir(role, run) / "selected_params.csv").exists()


def require_level(label: str, level, params: dict) -> None:
    """A transposed run number in the table is only detectable through the recorded alpha."""
    if "alpha" not in params:
        raise ValueError(f"{label} records no 'alpha'; cannot verify it is level {format_level(level)}.")
    alpha = float(params["alpha"])
    if abs(alpha - alpha_of(level)) > 1e-9:
        raise ValueError(f"{label} was run at alpha={alpha}, not {alpha_of(level)} as the level-"
                         f"{format_level(level)} cell requires. Check {RUNS_FILE.name}.")
    if "demean" not in params:
        raise ValueError(f"{label} records no 'demean' setting; this script scores the raw measure.")
    if bool(params["demean"]):
        raise ValueError(f"{label} was run with demean=True; use in_sample_vs_out_of_sample_demeaned.py.")


def hyperparams(role: str, level, universe: int) -> dict:
    """Per-date settings from the production run; the registry's alpha = 0.80 fits only one level."""
    path = run_dir(role, top_run(role, level, universe))
    chosen = pd.read_csv(path / "selected_params.csv", parse_dates=["date"])
    keys = [c for c in chosen.columns if c not in {"date", "n_assets", "val_score"}]
    table = {row["date"]: {k: row[k] for k in keys} for row in chosen.to_dict("records")}
    for params in table.values():
        require_level(f"{STRATEGIES[role]} {path.name} (level {format_level(level)}, top-{universe})", level, params)
    return table


def available(levels, universes) -> list:
    """Levels with every requested cell; the collapse regression needs all universes of a level."""
    keep = []
    for level in levels:
        missing = [f"{row_name(role, level)} top-{u}" for u in universes for role in STRATEGIES
                   if not has_run(role, level, u)]
        if missing:
            logger.warning("Skipping level %s: no usable production run for %s.", format_level(level),
                           ", ".join(missing))
        else:
            keep.append(level)
    return keep


def sweep(level, R_fh, R_c, windows, solver, universe) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores per date and program, and per-date window sizes and tail counts."""
    alpha = alpha_of(level)
    tables = {role: hyperparams(role, level, universe) for role in STRATEGIES}
    names = {**STRATEGIES, "equal_weight": isa.CONTROL_STRATEGY_KEY}
    rows, sizes = [], []
    for w in windows:
        date = w["date"]
        try:
            chosen = {role: table[date] for role, table in tables.items()}
        except KeyError:
            raise ValueError(f"level {level}: no production settings recorded for {date.date()}.") from None
        chosen["equal_weight"] = {k: v for k, v in chosen["joint"].items() if k == "alpha"}
        assets = w["assets"]
        R_fh_est = R_fh.loc[w["estimation_start"]:w["estimation_end"], assets]
        R_c_est = R_c.loc[w["estimation_start"]:w["estimation_end"]]
        R_fh_hold = R_fh.loc[w["holding_start"]:w["holding_end"], assets].fillna(0.0)
        R_c_hold = R_c.loc[w["holding_start"]:w["holding_end"]].fillna(0.0)
        # the annual window starts with the held quarter
        start = R_fh.index.searchsorted(w["holding_start"])
        span = R_fh.index[start:start + WINDOW_1Y_DAYS]
        long_enough = len(span) >= WINDOW_1Y_MIN_DAYS
        R_fh_1y, R_c_1y = R_fh.loc[span, assets].fillna(0.0), R_c.loc[span].fillna(0.0)
        tail = lambda n: max(1, round(n * (1.0 - alpha)))
        sizes.append({"date": date, "n_assets": len(assets), "n_days_in": len(R_fh_est),
                      "n_days_out": len(R_fh_hold), "n_tail_in": tail(len(R_fh_est)),
                      "n_tail_out": tail(len(R_fh_hold)), "n_days_out_1y": len(span) if long_enough else 0,
                      "n_tail_out_1y": tail(len(span)) if long_enough else 0})
        for role, name in names.items():
            try:
                x, psi = isa.solve_weights(STRATEGY_REGISTRY[name]["optimizer"], {**chosen[role], "solver": solver},
                                           R_fh_est, R_c_est)
                f_in = isa.evaluate_risk(MEASURE, R_fh_est, R_c_est, x, psi, alpha)
                f_out = isa.evaluate_risk(MEASURE, R_fh_hold, R_c_hold, x, psi, alpha)
                f_out_1y = isa.evaluate_risk(MEASURE, R_fh_1y, R_c_1y, x, psi, alpha) if long_enough else float("nan")
            except (RuntimeError, ValueError) as exc:
                logger.warning("level %s %s %s failed: %s", format_level(level), date.date(), role, exc)
                f_in = f_out = f_out_1y = float("nan")
            rows.append({"date": date, "strategy": role, "n_assets": len(assets), "in_sample_risk": f_in,
                         "realized_risk": f_out, "realized_risk_1y": f_out_1y})
    return pd.DataFrame(rows), pd.DataFrame(sizes)


def pair_annual(rows: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Attach the annual gap to the quarterly pairs; dates without one carry NaN."""
    annual = rows.dropna(subset=["in_sample_risk", "realized_risk_1y"]).copy()
    if annual.empty:
        return paired.assign(**{c: float("nan") for c in ANNUAL_COLUMNS[:5]})
    annual["realized_risk"] = annual["realized_risk_1y"]
    keep = ivo.pair(annual, ["date"])[["date", "gap_out", "gap_out_rel", "out_joint", "out_overlay", "out_ew"]]
    keep.columns = ["date", *ANNUAL_COLUMNS[:5]]
    return paired.merge(keep, on="date", how="left")


def read_shards(out_dir: Path) -> pd.DataFrame:
    """All shards, oldest first, a duplicated cell keeping the newest; shards from before the annual
    horizon get NaN annual columns."""
    paths = sorted(out_dir.glob("gaps_*.csv"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No gaps_*.csv shards in {out_dir}. Solve at least one cell first.")
    gaps = pd.concat([pd.read_csv(p, parse_dates=["date"]).assign(shard=p.name) for p in paths], ignore_index=True)
    missing = [c for c in ANNUAL_COLUMNS if c not in gaps.columns]
    for column in missing:
        gaps[column] = float("nan")
    if missing:
        stale = sorted(f"level {format_level(lvl)} top-{int(uni)}" for lvl, uni in gaps.loc[
            gaps[ANNUAL_COLUMNS[0]].isna(), ["level", "universe"]].drop_duplicates().itertuples(index=False))
        logger.warning("%d cell(s) predate the annual horizon and carry no gap_out_1y: %s. Re-solve them "
                       "(run_gap_analysis_levels.sh --force).", len(stale), ", ".join(stale))
    keys = ["family", "measure", "level", "universe", "date"]
    clashing = gaps.duplicated(keys, keep=False)
    if clashing.any():
        logger.warning("Duplicate cells across shards: %s. Keeping the most recent; delete the stale one.",
                       ", ".join(sorted(gaps.loc[clashing, "shard"].unique())))
        gaps = gaps.drop_duplicates(keys, keep="last")
    return gaps.drop(columns="shard").sort_values(keys).reset_index(drop=True)


def grid_arm(solver, levels, universes, out_dir: Path) -> int:
    written = 0
    for level in available(levels, universes):
        for universe in universes:
            logger.info("Grid level %s top-%d", format_level(level), universe)
            R_fh, R_c, mask, _ = isa.load_universe_data(universe)
            windows = isa.rebalancing_windows(R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"])
            rows, sizes = sweep(level, R_fh, R_c, windows, solver, universe)
            scored = rows.dropna(subset=["in_sample_risk", "realized_risk"])
            paired = pair_annual(rows, ivo.pair(scored, ["date"]))
            # raw ES can be negative at shallow levels, and pair() drops those dates
            if dropped := scored["date"].nunique() - paired["date"].nunique():
                logger.warning("level %s top-%d: %d of %d dates dropped, overlay risk not strictly positive.",
                               format_level(level), universe, dropped, scored["date"].nunique())
            paired = paired.merge(sizes[["date", "n_days_in", "n_days_out", "n_tail_in", "n_tail_out",
                                         "n_days_out_1y", "n_tail_out_1y"]], on="date", how="left")
            paired = paired.assign(measure=MEASURE, level=level, universe=universe, n_curr=ivo.N_CURRENCIES,
                                   arm="grid", draw=0, family=FAMILY)
            path = out_dir / f"gaps_{FAMILY}_level{format_level(level)}_top{universe}.csv"
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # never observed half-written
            paired.to_csv(tmp, index=False)
            tmp.replace(path)
            logger.info("Wrote %d rows to %s", len(paired), path)
            written += 1
    return written


def per_level(table_fn, gaps: pd.DataFrame) -> list[pd.DataFrame]:
    """table_fn applied to each level alone, with the level as first column."""
    parts = []
    for level, part in gaps.groupby("level"):
        table = table_fn(part)
        table.insert(0, "level", level)
        parts.append(table)
    return parts


def summarise(gaps: pd.DataFrame) -> pd.DataFrame:
    """ivo.summarise per level, plus the annual gap and the median tail counts."""
    parts = []
    for summary, (_, part) in zip(per_level(ivo.summarise, gaps), gaps.groupby("level")):
        for col in ("gap_out_1y", "gap_out_1y_rel"):
            stats = {u: block_bootstrap_mean(cell[col].to_numpy() if col in cell else np.array([]))
                     for u, cell in part.groupby("universe")}
            for i, suffix in enumerate(("mean", "lo", "hi", "se", "block")):
                summary[f"{col}_{suffix}"] = summary["universe"].map({u: v[i] for u, v in stats.items()})
        tails = part.groupby("universe")[["n_tail_in", "n_tail_out", "n_tail_out_1y"]].median()
        for col in tails:
            summary[col] = summary["universe"].map(tails[col]).astype(float)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True).sort_values(["level", "universe"]).reset_index(drop=True)


def collapse_regressions(gaps: pd.DataFrame) -> pd.DataFrame:
    """ivo.regressions per level, skipping levels with a single universe."""
    keep = []
    for level, part in gaps.groupby("level"):
        if part["universe"].nunique() < 2:
            logger.warning("Level %s has one universe; skipping its collapse regression.", format_level(level))
        else:
            keep.append(part)
    parts = per_level(ivo.regressions, pd.concat(keep)) if keep else []
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=["level", "family", "measure", "outcome", "slope"]))


def level_regressions(gaps: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Per universe, the per-date slope of the relative gap on the level in percent, on dates all levels share."""
    rows = []
    for universe, part in gaps.groupby("universe"):
        cells = {lvl: cell.set_index("date") for lvl, cell in part.groupby("level")}
        levels = sorted(cells)
        if len(levels) < 2:
            logger.warning("Top-%d has one level; skipping its level regression.", universe)
            continue
        dates = sorted(set.intersection(*(set(cells[lvl].index) for lvl in levels)))
        if not dates:
            logger.warning("Top-%d levels share no rebalancing dates; skipping its level regression.", universe)
            continue
        x = np.array([float(lvl) for lvl in levels])
        contrast = x - x.mean()
        denom = float((contrast**2).sum())
        for col in ("gap_in_rel", "gap_out_rel"):
            Y = np.column_stack([cells[lvl].loc[dates, col].to_numpy() for lvl in levels])
            slope, lo, hi, se, block = block_bootstrap_mean(Y @ contrast / denom, seed)
            rows.append({"family": FAMILY, "measure": MEASURE, "universe": universe, "outcome": col,
                         "slope_per_level_pct": slope, "intercept": float(Y.mean() - slope * x.mean()),
                         "ci_lo": lo, "ci_hi": hi, "se": se, "block": block,
                         "levels": ",".join(format_level(lvl) for lvl in levels), "n_dates": len(dates),
                         "n_obs": int(Y.size)})
    return pd.DataFrame(rows)


def plot_panels(summary: pd.DataFrame, stem: Path, panels, select, xcol: str, title, style_x, width: float) -> None:
    """One panel per entry of panels; an unsolved panel stays in place, labelled."""
    if summary.empty:
        logger.warning("No cells to plot.")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), 4.4), sharex=True, sharey=True)
    for ax, key in zip(axes, panels):
        line = select(summary, key)
        for col, (color, _) in STYLES.items():
            if line.empty:
                continue
            ax.plot(line[xcol], 100 * line[f"{col}_mean"], marker="s", ms=5, color=color, zorder=3)
            ax.fill_between(line[xcol], 100 * line[f"{col}_lo"], 100 * line[f"{col}_hi"], color=color, alpha=0.2,
                            lw=0)
        if line.empty:
            ax.text(0.5, 0.5, "not yet solved", transform=ax.transAxes, ha="center", va="center", fontsize=9,
                    color="grey")
        ax.axhline(0.0, color="grey", lw=1.0, zorder=0)
        ax.grid(True, alpha=0.3)
        ax.set_title(title(key), fontsize=11)
        style_x(ax)
    axes[0].set_ylabel("Risk reduction, joint vs sequential (%)")
    axes[0].legend(handles=[Line2D([], [], color=c, marker="s", label=label) for c, label in STYLES.values()],
                   fontsize=7, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    ivo.save(fig, stem)


def level_axis(ax) -> None:
    """All swept levels as minor ticks, only the round ones labelled."""
    ax.set_xlabel(f"{MEASURE_SHORT} confidence level (%)")
    ax.set_xticks([lvl for lvl in PLOT_LEVELS if lvl in ROUND_LEVELS])
    ax.set_xticklabels([format_level(lvl) for lvl in PLOT_LEVELS if lvl in ROUND_LEVELS])
    ax.set_xticks(list(PLOT_LEVELS), minor=True)
    ax.set_xlim(min(PLOT_LEVELS) - 3, max(PLOT_LEVELS) + 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    measure = parser.add_mutually_exclusive_group(required=True)
    measure.add_argument("--es", dest="measure", action="store_const", const="es")
    measure.add_argument("--soes", dest="measure", action="store_const", const="soes")
    parser.add_argument("--levels", nargs="+", type=parse_level, default=list(LEVELS), metavar="LEVEL",
                        help="Confidence levels in percent, e.g. 72.5 (default: the whole sweep).")
    parser.add_argument("--universes", nargs="+", type=int, default=list(UNIVERSES), choices=UNIVERSES)
    parser.add_argument("--sweep-only", action="store_true", help="Write the shards and skip the tables and figures.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: the measure's own results directory.")
    parser.add_argument("--plot-only", action="store_true", help="Build the tables and figures from the shards.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()
    configure(SWEEP_MEASURES[args.measure])
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args.out_dir = args.out_dir or OUT_DIR
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Sweeping %s into %s", MEASURE_LABEL, args.out_dir)

    if not args.plot_only:
        if grid_arm(isa.pick_solver()[0], args.levels, args.universes, args.out_dir) == 0:
            logger.warning("Nothing solved for %s at levels %s on %s; no shard written.", MEASURE,
                           [format_level(lvl) for lvl in args.levels], args.universes)
            return
        if args.sweep_only:
            return

    gaps = read_shards(args.out_dir)
    gaps["s"] = [ivo.currency_share(u, k) for u, k in zip(gaps["universe"], gaps["n_curr"])]
    summary = summarise(gaps)
    tables = {"gap_summary": summary, "regression_collapse": collapse_regressions(gaps),
              "regression_level": level_regressions(gaps),
              "overfitting": pd.concat(per_level(ivo.overfitting_table, gaps), ignore_index=True)}
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        logger.info("Wrote %s", args.out_dir / f"{name}.csv")
    plot_panels(summary, args.out_dir / "gap_vs_universe", UNIVERSE_PANEL_LEVELS,
                lambda s, lvl: s[s["level"] == lvl].sort_values("universe"), "universe",
                lambda lvl: f"{MEASURE_SHORT} at {format_level(lvl)}%", ivo.universe_axis, 4.33)
    plot_panels(summary, args.out_dir / "gap_vs_level", UNIVERSES,
                lambda s, u: s[(s["universe"] == u) & s["level"].isin(PLOT_LEVELS)].sort_values("level"), "level",
                lambda u: f"Top-{u} universe", level_axis, 4.7)
    print("\n\n".join(t.to_string(index=False) for t in tables.values()))


if __name__ == "__main__":
    main()
