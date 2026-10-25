"""Does the joint program's in-sample edge over the currency overlay survive out of sample (Chapter 6)?

At every rebalancing date the joint and overlay programs of one risk measure are scored on the estimation
window (in) and the following quarter (out); the gap g = f_overlay - f_joint is positive when joint is better.
Per cell the mean gap gets a circular block bootstrap interval. The grid arm covers the 15 measure x universe
cells, unregularized ones read from in_sample_analysis.py's output; the currency arm cuts the overlay block to
the K largest economies on an intact asset universe. Regularized cells reuse the penalties production selected.

Usage:
    python code/analysis/in_sample_vs_out_of_sample.py [--arms grid currency] [--family regularized]
    python code/analysis/in_sample_vs_out_of_sample.py --plot-only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import in_sample_analysis as isa
from common import top_runs
from portfolio_backtest import config
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.bootstrap import block_bootstrap_mean

logger = logging.getLogger("in_sample_vs_out_of_sample")

MEASURES = ("mv", "es", "mad", "smv", "soes")
FAMILIES = ("unregularized", "regularized")
UNIVERSES = (10, 50, 100)
N_CURRENCIES = 14
CURRENCY_ARM = {50: (10, 6), 10: (10, 6, 3)}
SEED = 0
SRC_DIR = config.RESULTS_DIR / "in_sample_analysis"
OUT_DIR = config.RESULTS_DIR / "in_sample_vs_out_of_sample"
MEASURE_LABELS = {"es": "Expected Shortfall", "mv": "Volatility", "mad": "Mean Absolute Deviation",
                  "smv": "Semi-Deviation", "soes": "Second-Order Expected Shortfall"}
# deviation and tail measures on separate panels; SOES runs an order of magnitude higher
MEASURE_PANELS = (("Deviation measures", ("mv", "mad", "smv"), "lower right"),
                  ("Tail measures", ("es", "soes"), "lower right"))
MEASURE_COLORS = {"mv": "C0", "mad": "C1", "smv": "C2", "es": "C3", "soes": "C4"}
MEASURE_MARKERS = {"mv": "o", "mad": "s", "smv": "^", "es": "D", "soes": "v"}


def sorted_frame(rows: list[dict], by: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame.sort_values(by).reset_index(drop=True) if not frame.empty else frame


def currency_share(top_n: int, n_curr: int = N_CURRENCIES) -> float:
    return n_curr / (n_curr + N_CURRENCIES * top_n)


def resolve_strategies(measure: str, family: str) -> dict[str, str]:
    """The joint and overlay strategy of a family, excluding the mean-risk (*_w_mean) programs."""
    resolved = {}
    for role in ("joint", "overlay"):
        matches = [n for n in STRATEGY_REGISTRY if ("reg" in n.split("_")) == (family == "regularized")
                   and "mean" not in n.split("_") and {measure, role} <= set(n.split("_"))]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {family} {measure} {role} strategy, found {matches}.")
        resolved[role] = matches[0]
    return resolved


def top_run(strategy: str, universe: int) -> int:
    table = top_runs("top_runs.csv")
    if strategy not in table.index:
        raise ValueError(f"{strategy} has no entry in top_runs.csv.")
    run = table.at[strategy, f"top_{universe}_run"]
    if pd.isna(run):
        raise ValueError(f"{strategy} has no top-{universe} run recorded in top_runs.csv, so its selected "
                         f"penalties cannot be reused.")
    return int(run)


def hyperparams(strategy: str, universe: int, family: str) -> dict:
    """One dict for an unregularized strategy, else the penalties production selected, keyed by date."""
    if family == "unregularized":
        return isa.fixed_hyperparams(strategy)
    chosen = pd.read_csv(config.RUNS_DIR / strategy / f"run_{top_run(strategy, universe)}" / "selected_params.csv",
                         parse_dates=["date"])
    keys = [c for c in chosen.columns if c not in {"date", "n_assets", "val_score"}]
    return {row["date"]: {k: row[k] for k in keys} for row in chosen.to_dict("records")}


def has_run(strategy: str, universe: int) -> bool:
    try:
        top_run(strategy, universe)
    except ValueError:
        return False
    return True


def available(measures, universes, family: str, has_run=has_run) -> list[str]:
    """Measures with a production run on every universe; regressions need all universes of a measure."""
    if family == "unregularized":
        return list(measures)
    keep = []
    for measure in measures:
        missing = [f"{name} top-{u}" for u in universes for name in resolve_strategies(measure, family).values()
                   if not has_run(name, u)]
        if missing:
            logger.warning("Skipping %s %s: no production run recorded for %s.", family, measure, ", ".join(missing))
        else:
            keep.append(measure)
    return keep


def pair(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Joint, overlay and 1/N risks side by side with the absolute and relative gaps."""
    parts = [rows[rows["strategy"] == strategy].set_index(keys)[["in_sample_risk", "realized_risk"]]
             .rename(columns={"in_sample_risk": f"in_{tag}", "realized_risk": f"out_{tag}"})
             for strategy, tag in (("joint", "joint"), ("overlay", "overlay"), ("equal_weight", "ew"))]
    paired = parts[0].join(parts[1], how="inner").join(parts[2], how="left").reset_index()
    paired = paired[(paired["in_overlay"] > 0) & (paired["out_overlay"] > 0)]
    paired["gap_in"] = paired["in_overlay"] - paired["in_joint"]
    paired["gap_out"] = paired["out_overlay"] - paired["out_joint"]
    paired["gap_in_rel"] = paired["gap_in"] / paired["in_overlay"]
    paired["gap_out_rel"] = paired["gap_out"] / paired["out_overlay"]
    return paired


def center(frame: pd.DataFrame) -> pd.DataFrame:
    return frame - frame.mean(axis=0)


def sweep(measure, R_fh, R_c, windows, solver, family, universe, hyperparams=hyperparams,
          demean: bool = False) -> pd.DataFrame:
    """Joint, overlay and 1/N risk per date. With demean, the scoring windows are centered columnwise,
    which scores the object the de-meaned programs minimize; the programs are still solved on raw returns."""
    names = resolve_strategies(measure, family)
    tables = {role: hyperparams(name, universe, family) for role, name in names.items()}
    names["equal_weight"] = isa.CONTROL_STRATEGY_KEY
    rows = []
    for w in windows:
        date = w["date"]
        try:
            chosen = {role: table if family == "unregularized" else table[date] for role, table in tables.items()}
        except KeyError:
            raise ValueError(f"{family} {measure}: no production hyperparameters recorded for {date.date()}.") from None
        chosen["equal_weight"] = {k: v for k, v in chosen["joint"].items() if k == "alpha"}
        assets = w["assets"]
        R_fh_est = R_fh.loc[w["estimation_start"]:w["estimation_end"], assets]
        R_c_est = R_c.loc[w["estimation_start"]:w["estimation_end"]]
        R_fh_hold = R_fh.loc[w["holding_start"]:w["holding_end"], assets].fillna(0.0)
        R_c_hold = R_c.loc[w["holding_start"]:w["holding_end"]].fillna(0.0)
        score_in, score_out = (R_fh_est, R_c_est), (R_fh_hold, R_c_hold)
        if demean:
            score_in, score_out = tuple(map(center, score_in)), tuple(map(center, score_out))
        for role, name in names.items():
            params = {**chosen[role], "solver": solver}
            alpha = params.get("alpha")
            try:
                x, psi = isa.solve_weights(STRATEGY_REGISTRY[name]["optimizer"], params, R_fh_est, R_c_est)
                f_in = isa.evaluate_risk(measure, *score_in, x, psi, alpha)
                f_out = isa.evaluate_risk(measure, *score_out, x, psi, alpha)
            except (RuntimeError, ValueError) as exc:
                logger.warning("%s %s %s failed: %s", measure, date.date(), role, exc)
                f_in = f_out = float("nan")
            rows.append({"date": date, "strategy": role, "n_assets": len(assets), "in_sample_risk": f_in,
                         "realized_risk": f_out})
    return pd.DataFrame(rows)


def tag(rows: pd.DataFrame, measure: str, universe: int, n_curr: int, arm: str) -> pd.DataFrame:
    paired = pair(rows.dropna(subset=["in_sample_risk", "realized_risk"]), ["date"])
    return paired.assign(measure=measure, universe=universe, n_curr=n_curr, arm=arm, draw=0)


def grid_arm(solver, family: str, measures, universes) -> pd.DataFrame:
    """The measure x universe cells; unregularized ones are read from in_sample_analysis.py's output."""
    frames = []
    for measure in available(measures, universes, family):
        for universe in universes:
            if family == "unregularized":
                path = SRC_DIR / f"in_sample_{measure}_top{universe}.csv"
                if not path.exists():
                    logger.warning("Missing %s; skipping.", path)
                    continue
                rows = pd.read_csv(path, parse_dates=["date"])
            else:
                logger.info("Grid %s top-%d %s", family, universe, measure)
                R_fh, R_c, mask, _ = isa.load_universe_data(universe)
                windows = isa.rebalancing_windows(R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"])
                rows = sweep(measure, R_fh, R_c, windows, solver, family, universe)
            frames.append(tag(rows, measure, universe, N_CURRENCIES, "grid"))
    return pd.concat(frames, ignore_index=True)


def currency_arm(solver, family: str, measures, universes) -> pd.DataFrame:
    """Overlay block cut to the K largest economies by constituent count; regularized cells reuse the
    penalties selected on the full K = 14 problem, an approximation."""
    frames = []
    measures = available(measures, universes, family)
    for universe, counts in CURRENCY_ARM.items():
        if universe not in universes:
            continue
        R_fh, R_c, mask, atc = isa.load_universe_data(universe)
        windows = isa.rebalancing_windows(R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"])
        order = [c for c in pd.Series(list(atc.values())).value_counts().index if c in R_c.columns]
        for n_curr in counts:
            keep = order[:n_curr] if "USD" in order[:n_curr] else ["USD"] + order[:n_curr - 1]
            logger.info("Currency arm %s top-%d K=%d: %s", family, universe, n_curr, keep)
            for measure in measures:
                rows = sweep(measure, R_fh, R_c[keep], windows, solver, family, universe)
                frames.append(tag(rows, measure, universe, n_curr, "currency"))
    return pd.concat(frames, ignore_index=True)


def summarise(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["family", "measure", "universe", "n_curr", "arm", "draw"]
    for values, part in gaps.groupby(keys):
        row = dict(zip(keys, values))
        row.update(s=currency_share(row["universe"], row["n_curr"]), n_dates=part["date"].nunique(), n_obs=len(part))
        for side in ("in", "out"):
            for col in (f"gap_{side}", f"gap_{side}_rel"):
                stats = block_bootstrap_mean(part[col].to_numpy())
                row.update({f"{col}_{k}": v for k, v in zip(("mean", "lo", "hi", "se", "block"), stats)})
            row[f"frac_joint_better_{side}"] = float((part[f"gap_{side}"] > 0).mean())
        rows.append(row)
    return sorted_frame(rows, ["family", "measure", "arm", "s", "draw"])


def regressions(grid: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Slope of the relative gap on the currency share, as a block-bootstrapped mean of per-date contrasts."""
    rows = []
    for (family, measure), part in grid.groupby(["family", "measure"]):
        cells = {u: cell.set_index("date") for u, cell in part.groupby("universe")}
        universes = sorted(cells)
        dates = sorted(set.intersection(*(set(cells[u].index) for u in universes)))
        s = np.array([currency_share(u) for u in universes])
        contrast = s - s.mean()
        denom = float((contrast**2).sum())
        for col in ("gap_in_rel", "gap_out_rel"):
            Y = np.column_stack([cells[u].loc[dates, col].to_numpy() for u in universes])
            slope, lo, hi, se, block = block_bootstrap_mean(Y @ contrast / denom, seed)
            rows.append({"family": family, "measure": measure, "outcome": col, "slope": slope,
                         "intercept": float(Y.mean() - slope * s.mean()), "ci_lo": lo, "ci_hi": hi, "se": se,
                         "block": block, "n_dates": len(dates), "n_obs": int(Y.size)})
    return pd.DataFrame(rows)


def overfitting_table(grid: pd.DataFrame) -> pd.DataFrame:
    """Realized risk over 1/N's, and the realized/in-sample ratio over 1/N's, per program."""
    rows = []
    for (family, measure, universe), cell in grid[(grid["out_ew"] > 0) & (grid["in_ew"] > 0)].groupby(
            ["family", "measure", "universe"]):
        row = {"family": family, "measure": measure, "universe": universe, "n_dates": len(cell)}
        ew_decay = (cell["out_ew"] / cell["in_ew"]).to_numpy()
        for t in ("joint", "overlay"):
            for name, values in ((f"{t}_over_ew", cell[f"out_{t}"] / cell["out_ew"]),
                                 (f"{t}_overfit", (cell[f"out_{t}"] / cell[f"in_{t}"]).to_numpy() / ew_decay)):
                mean, lo, hi, _, _ = block_bootstrap_mean(np.asarray(values))
                row.update({name: mean, f"{name}_lo": lo, f"{name}_hi": hi})
        rows.append(row)
    return sorted_frame(rows, ["family", "measure", "universe"])


def delta_epsilon_table(grid: pd.DataFrame) -> pd.DataFrame:
    """Paired bootstrap of g_in_rel - g_out_rel, the in-sample advantage that was not realized."""
    rows = []
    for (family, measure, universe), cell in grid.groupby(["family", "measure", "universe"]):
        paired = (cell["gap_in_rel"] - cell["gap_out_rel"]).to_numpy()
        mean, lo, hi, se, block = block_bootstrap_mean(paired, SEED)
        rows.append({"family": family, "measure": measure, "universe": universe, "delta_eps_mean": mean,
                     "delta_eps_lo": lo, "delta_eps_hi": hi, "delta_eps_se": se, "delta_eps_block": block,
                     "n_dates": cell["date"].nunique(), "frac_negative": float((paired < 0).mean())})
    return sorted_frame(rows, ["family", "measure", "universe"])


def universe_axis(ax) -> None:
    """Log-spaced universe axis labelled at 10, 50 and 100."""
    ax.set_xlabel("Universe size")
    ax.set_xscale("log")
    ax.set_xticks(list(UNIVERSES))
    ax.set_xticklabels([str(u) for u in UNIVERSES])
    ax.set_xticks([], minor=True)
    ax.set_xlim(min(UNIVERSES) * 0.8, max(UNIVERSES) * 1.25)
    ax.grid(True, alpha=0.3)


def save(fig, stem: Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", stem.with_suffix(f".{ext}"))
    plt.close(fig)


def plot_delta_epsilon(table: pd.DataFrame, stem: Path, family: str) -> None:
    table = table[table["family"] == family]
    if table.empty:
        return
    fig, axes = plt.subplots(1, len(MEASURE_PANELS), figsize=(11.0, 4.4), sharex=True)
    for ax, (title, measures, legend_loc) in zip(axes, MEASURE_PANELS):
        for measure in measures:
            line = table[table["measure"] == measure].sort_values("universe")
            if line.empty:
                continue
            color = MEASURE_COLORS[measure]
            ax.plot(line["universe"], 100 * line["delta_eps_mean"], marker=MEASURE_MARKERS[measure], ms=5,
                    color=color, label=MEASURE_LABELS[measure], zorder=3)
            ax.fill_between(line["universe"], 100 * line["delta_eps_lo"], 100 * line["delta_eps_hi"],
                            color=color, alpha=0.15, lw=0)
        ax.axhline(0.0, color="grey", lw=1.0, zorder=0)
        ax.set_title(title, fontsize=11)
        universe_axis(ax)
        ax.legend(fontsize=8, loc=legend_loc, framealpha=0.9)
        ax.set_ylabel(r"$\tilde{g}^{\mathrm{IS}} - \tilde{g}^{\mathrm{OOS}}$ (pp)")
    fig.tight_layout()
    save(fig, stem)


def plot(summary: pd.DataFrame, stem: Path, family: str) -> None:
    """Grid-arm gaps against universe size; deviation measures on top, tail measures centred below."""
    summary = summary[summary["family"] == family]
    if summary.empty:
        return
    fig = plt.figure(figsize=(13.0, 8.6))
    grid = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.55)
    styles = {"gap_in_rel": ("C0", "In-sample"), "gap_out_rel": ("C3", "Out-of-sample")}
    first_axes = []
    for r, (measures, starts) in enumerate(((("mv", "mad", "smv"), (0, 2, 4)), (("es", "soes"), (1, 3)))):
        row_axes = []
        for c, measure in zip(starts, measures):
            ax = fig.add_subplot(grid[r, c:c + 2], sharey=row_axes[0] if row_axes else None)
            row_axes.append(ax)
            line = summary[(summary["measure"] == measure) & (summary["arm"] == "grid")].sort_values("universe")
            for col, (color, _) in styles.items():
                if line.empty:
                    continue
                ax.plot(line["universe"], 100 * line[f"{col}_mean"], marker="s", ms=5, color=color, zorder=3)
                ax.fill_between(line["universe"], 100 * line[f"{col}_lo"], 100 * line[f"{col}_hi"], color=color,
                                alpha=0.2, lw=0)
            ax.axhline(0.0, color="grey", lw=1.0, zorder=0)
            ax.set_title(MEASURE_LABELS[measure], fontsize=11)
            universe_axis(ax)
            ax.tick_params(labelleft=True)
        row_axes[0].set_ylabel("Risk reduction (%)")
        first_axes.append(row_axes[0])
    handles = [Line2D([], [], color=color, marker="s", label=label) for color, label in styles.values()]
    first_axes[0].legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)
    save(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=["grid", "currency"], choices=["grid", "currency"])
    parser.add_argument("--family", default="unregularized", choices=FAMILIES)
    parser.add_argument("--measures", nargs="+", default=list(MEASURES), choices=MEASURES)
    parser.add_argument("--universes", nargs="+", type=int, default=list(UNIVERSES), choices=UNIVERSES,
                        help="Split across jobs with distinct --out-dir, then merge their gaps.csv.")
    parser.add_argument("--sweep-only", action="store_true", help="Write gaps.csv and skip the tables and figures.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gaps_path = args.out_dir / "gaps.csv"

    def read_gaps() -> pd.DataFrame:
        gaps = pd.read_csv(gaps_path, parse_dates=["date"])
        return gaps if "family" in gaps.columns else gaps.assign(family="unregularized")

    if args.plot_only:
        gaps = read_gaps()
    else:
        solver = isa.pick_solver()[0]
        arms = {"grid": grid_arm, "currency": currency_arm}
        gaps = pd.concat([arms[a](solver, args.family, args.measures, args.universes) for a in args.arms],
                         ignore_index=True)
        gaps["family"] = args.family
        if gaps_path.exists():
            old = read_gaps()
            stale = ((old["family"] == args.family) & old["arm"].isin(args.arms)
                     & old["measure"].isin(args.measures) & old["universe"].isin(args.universes))
            gaps = pd.concat([old[~stale], gaps], ignore_index=True)
        gaps.to_csv(gaps_path, index=False)
        logger.info("Wrote %d rows to %s", len(gaps), gaps_path)
        if args.sweep_only:
            return

    gaps["s"] = [currency_share(u, k) for u, k in zip(gaps["universe"], gaps["n_curr"])]
    summary = summarise(gaps)
    grid = gaps[gaps["arm"] == "grid"]
    if grid.empty:
        logger.warning("No 'grid' rows in %s, so the regression and overfitting tables are empty.", gaps_path)
    tables = {"gap_summary": summary, "regression": regressions(grid), "overfitting": overfitting_table(grid),
              "delta_epsilon": delta_epsilon_table(grid)}
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        logger.info("Wrote %s", args.out_dir / f"{name}.csv")
    for family in sorted(gaps["family"].unique()):
        suffix = "" if family == "unregularized" else f"_{family}"
        plot(summary, args.out_dir / f"gap_vs_currency_share{suffix}", family)
        plot_delta_epsilon(tables["delta_epsilon"], args.out_dir / f"delta_epsilon{suffix}", family)
    print("\n\n".join(t.to_string(index=False) for t in tables.values()))


if __name__ == "__main__":
    main()
