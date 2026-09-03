"""
Does the joint program's in-sample edge over the currency overlay survive out of sample,
and does it collapse as the currency share of the decision vector falls?

At every quarterly rebalancing date the unregularized joint and overlay programs of one
risk measure are fit on the ten-year estimation window and scored twice by one shared
empirical scorer: on that window (f_in, the ceiling) and on the following quarter
(f_out).  The signed gap g = f_overlay - f_joint is positive when joint is better.  Per
(measure, universe) cell the mean of each gap gets a circular block bootstrap CI (10000
resamples, block length by Politis-White), alongside the fraction of dates with g < 0.

Pooled per measure, the relative gap is regressed on the currency share
s = K / (K + 14 * top_N) with block-bootstrapped standard errors, turning the collapse
into one slope.  A currency-count arm then moves s without moving the asset count: the
asset universe is held intact and the tradable overlay block is cut to the K largest
economies by constituent count, K = 10, 6 on top-50 and K = 10, 6, 3 on top-10.  Because
s = K / (K + 14N), top-10 at K = 3 lands on s = 0.021 against the top-50 grid cell's
0.020 with 140 securities rather than 700, which separates the currency share from the
asset count.  A supporting table reports each program's realized risk over equal weight's,
the absolute penalty both pay where their difference does not.

The main cells (one per measure x universe) are read from results/in_sample_analysis/
rather than re-solved; a measure with no cached in_sample_<measure>_top<N>.csv is
warned about and skipped.

--family regularized additionally needs, for each measure, a production run of both
its joint and its overlay strategy plus a row in results/runs/top_runs.csv, because
the per-date penalties are reused from what production selected rather than
re-searched over the 256-point grid.  reg_smv_joint and reg_smv_curr_overlay have
neither yet, so the regularized arm is available for smv only once those cluster
runs land and their top_runs.csv rows are added.

Usage:
    python code/scripts/in_sample_vs_out_of_sample.py
    python code/scripts/in_sample_vs_out_of_sample.py --arms currency
    python code/scripts/in_sample_vs_out_of_sample.py --plot-only

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import in_sample_analysis as isa
from portfolio_backtest import config
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.bootstrap import block_bootstrap_mean
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (
    compute_daily_forward_premium,
    compute_excess_currency_returns,
    compute_fully_hedged_returns,
)

logger = logging.getLogger("in_sample_vs_out_of_sample")

MEASURES = ("mv", "es", "mad", "smv")
FAMILIES = ("unregularized", "regularized")
UNIVERSES = (10, 50, 100)
N_CURRENCIES = 14
CURRENCY_ARM = {50: (10, 6), 10: (10, 6, 3)}
SEED = 0

SRC_DIR = config.RESULTS_DIR / "in_sample_analysis"
OUT_DIR = config.RESULTS_DIR / "in_sample_vs_out_of_sample"
MEASURE_LABELS = {
    "es": "Expected Shortfall",
    "mv": "Volatility",
    "mad": "Mean Absolute Deviation",
    "smv": "Semi-Deviation",
}
# The figure draws the grid arm alone, so these three are no longer read by
# plot().  They are kept because the currency arm is still solved and written to
# gaps.csv / gap_summary.csv, and a future figure of that arm would want them.
# One marker per universe the currency-count arm runs on, so the two arms stay
# distinguishable where they meet at the same currency share.
CURRENCY_MARKERS = {50: "s", 10: "D"}
CURRENCY_LABELS = {50: "Top-50 universe, reduced K", 10: "Top-10 universe, reduced K"}
# Cells solved but left off the figure; (universe, K).
PLOT_EXCLUDE = {(10, 3)}


def _sorted(rows: list[dict], by: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame.sort_values(by).reset_index(drop=True) if not frame.empty else frame


def currency_share(top_n: int, n_curr: int = N_CURRENCIES) -> float:
    return n_curr / (n_curr + N_CURRENCIES * top_n)


def _resolve_strategies(measure: str, family: str) -> dict[str, str]:
    want_reg = family == "regularized"
    resolved = {}
    for role in ("joint", "overlay"):
        parts = {measure, role}
        matches = [
            name
            for name in STRATEGY_REGISTRY
            if ("reg" in name.split("_")) == want_reg and parts.issubset(set(name.split("_")))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {family} {measure} {role} strategy, found {matches}."
            )
        resolved[role] = matches[0]
    return resolved


def _top_run(strategy: str, universe: int) -> int:
    table = pd.read_csv(config.RUNS_DIR / "top_runs.csv")
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()
    row = table[table["strategy_name"] == strategy]
    if row.empty:
        raise ValueError(f"{strategy} has no entry in top_runs.csv.")
    return int(row[f"top_{universe}_run"].iloc[0])


def _hyperparams(strategy: str, universe: int, family: str) -> dict:
    """
    Per-date hyperparameters, keyed by rebalancing date.

    The unregularized programs declare no penalties, so their production grid is a
    singleton and one dict serves every date.  The regularized programs select
    penalties by validation at each date over a 256-point grid, which is far too
    large to re-search here; we reuse the selections the production backtest made
    and recorded, so the diagnostic scores the weights production actually held.
    """
    if family == "unregularized":
        return isa._fixed_hyperparams(strategy)
    path = config.RUNS_DIR / strategy / f"run_{_top_run(strategy, universe)}" / "selected_params.csv"
    chosen = pd.read_csv(path, parse_dates=["date"])
    drop = {"date", "n_assets", "val_score"}
    keys = [c for c in chosen.columns if c not in drop]
    return {row["date"]: {k: row[k] for k in keys} for row in chosen.to_dict("records")}


def _pair(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    parts = []
    for strategy, tag in (("joint", "joint"), ("overlay", "overlay"), ("equal_weight", "ew")):
        part = rows[rows["strategy"] == strategy].set_index(keys)[
            ["in_sample_risk", "realized_risk"]
        ]
        parts.append(
            part.rename(columns={"in_sample_risk": f"in_{tag}", "realized_risk": f"out_{tag}"})
        )
    paired = parts[0].join(parts[1], how="inner").join(parts[2], how="left").reset_index()
    paired = paired[(paired["in_overlay"] > 0) & (paired["out_overlay"] > 0)]
    # Signed so that POSITIVE means the joint program is better. These are risk
    # measures, where lower is better, so the joint's advantage is the overlay's
    # risk minus the joint's -- not the other way round.
    paired["gap_in"] = paired["in_overlay"] - paired["in_joint"]
    paired["gap_out"] = paired["out_overlay"] - paired["out_joint"]
    paired["gap_in_rel"] = paired["gap_in"] / paired["in_overlay"]
    paired["gap_out_rel"] = paired["gap_out"] / paired["out_overlay"]
    return paired


def _load(universe: int):
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
    return R_fh.loc[common], R_c.loc[common], isa._build_universe_mask(path, common), atc


def _sweep(measure, R_fh, R_c, windows, solver, family, universe) -> pd.DataFrame:
    names = _resolve_strategies(measure, family)
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
        logger.debug("[%d/%d] %s: %d assets", i, len(windows), date.date(), len(assets))

        for role, name in names.items():
            params = {**chosen[role], "solver": solver}
            alpha = params.get("alpha")
            try:
                x, psi = isa._solve_weights(
                    STRATEGY_REGISTRY[name]["optimizer"], params, R_fh_est, R_c_est
                )
                f_in = isa.evaluate_risk(measure, R_fh_est, R_c_est, x, psi, alpha)
                f_out = isa.evaluate_risk(measure, R_fh_hold, R_c_hold, x, psi, alpha)
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


def grid_arm(solver, family: str) -> pd.DataFrame:
    """
    The 9 main cells.  The unregularized ones were already solved and scored by
    in_sample_analysis.py, so they are read back rather than recomputed; the
    regularized ones have no such cache and are solved here.
    """
    frames = []
    for measure in MEASURES:
        for universe in UNIVERSES:
            if family == "unregularized":
                path = SRC_DIR / f"in_sample_{measure}_top{universe}.csv"
                if not path.exists():
                    logger.warning("Missing %s; skipping.", path)
                    continue
                rows = pd.read_csv(path, parse_dates=["date"])
            else:
                logger.info("Grid %s top-%d %s", family, universe, measure)
                R_fh, R_c, mask, _ = _load(universe)
                windows = isa._rebalancing_windows(
                    R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"]
                )
                rows = _sweep(measure, R_fh, R_c, windows, solver, family, universe)
            paired = _pair(rows.dropna(subset=["in_sample_risk", "realized_risk"]), ["date"])
            paired["measure"] = measure
            paired["universe"] = universe
            paired["n_curr"] = N_CURRENCIES
            paired["arm"] = "grid"
            paired["draw"] = 0
            frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def currency_arm(solver, family: str) -> pd.DataFrame:
    """
    Tradable overlay block cut to the K largest economies by constituent count,
    with the asset universe left intact.  Regularized cells reuse the penalties
    production selected on the full K = 14 problem, which is an approximation:
    those penalties were not chosen for a reduced currency block.
    """
    frames = []
    for universe, counts in CURRENCY_ARM.items():
        R_fh, R_c, mask, atc = _load(universe)
        windows = isa._rebalancing_windows(R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"])
        order = [c for c in pd.Series(list(atc.values())).value_counts().index if c in R_c.columns]
        for n_curr in counts:
            keep = order[:n_curr]
            if "USD" not in keep:
                keep = ["USD"] + order[: n_curr - 1]
            logger.info("Currency arm %s top-%d K=%d: %s", family, universe, n_curr, keep)
            for measure in MEASURES:
                rows = _sweep(measure, R_fh, R_c[keep], windows, solver, family, universe)
                paired = _pair(rows.dropna(subset=["in_sample_risk", "realized_risk"]), ["date"])
                paired["measure"] = measure
                paired["universe"] = universe
                paired["n_curr"] = n_curr
                paired["arm"] = "currency"
                paired["draw"] = 0
                frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def summarise(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, measure, universe, n_curr, arm, draw), part in gaps.groupby(
        ["family", "measure", "universe", "n_curr", "arm", "draw"]
    ):
        row = {
            "family": family,
            "measure": measure,
            "universe": universe,
            "n_curr": n_curr,
            "arm": arm,
            "draw": draw,
            "s": currency_share(universe, n_curr),
            "n_dates": part["date"].nunique(),
            "n_obs": len(part),
        }
        for side in ("in", "out"):
            for scale in ("", "_rel"):
                col = f"gap_{side}{scale}"
                mean, lo, hi, se, block = block_bootstrap_mean(part[col].to_numpy())
                row[f"{col}_mean"] = mean
                row[f"{col}_lo"] = lo
                row[f"{col}_hi"] = hi
                row[f"{col}_se"] = se
                row[f"{col}_block"] = block
            row[f"frac_joint_better_{side}"] = float((part[f"gap_{side}"] > 0).mean())
        rows.append(row)
    return _sorted(rows, ["family", "measure", "arm", "s", "draw"])


def regressions(grid: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
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
            rows.append(
                {
                    "family": family,
                    "measure": measure,
                    "outcome": col,
                    "slope": slope,
                    "intercept": float(Y.mean() - slope * s.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "se": se,
                    "block": block,
                    "n_dates": len(dates),
                    "n_obs": int(Y.size),
                }
            )
    return pd.DataFrame(rows)


def overfitting_table(grid: pd.DataFrame) -> pd.DataFrame:
    part = grid[(grid["out_ew"] > 0) & (grid["in_ew"] > 0)]
    rows = []
    for (family, measure, universe), cell in part.groupby(["family", "measure", "universe"]):
        row = {"family": family, "measure": measure, "universe": universe, "n_dates": len(cell)}
        ew_decay = (cell["out_ew"] / cell["in_ew"]).to_numpy()
        for tag in ("joint", "overlay"):
            mean, lo, hi, _, _ = block_bootstrap_mean((cell[f"out_{tag}"] / cell["out_ew"]).to_numpy())
            row[f"{tag}_over_ew"] = mean
            row[f"{tag}_over_ew_lo"] = lo
            row[f"{tag}_over_ew_hi"] = hi
            decay = (cell[f"out_{tag}"] / cell[f"in_{tag}"]).to_numpy() / ew_decay
            mean, lo, hi, _, _ = block_bootstrap_mean(decay)
            row[f"{tag}_overfit"] = mean
            row[f"{tag}_overfit_lo"] = lo
            row[f"{tag}_overfit_hi"] = hi
        rows.append(row)
    return _sorted(rows, ["family", "measure", "universe"])


def plot(summary: pd.DataFrame, stem: Path, family: str) -> None:
    summary = summary[summary["family"] == family]
    if summary.empty:
        return
    fig, axes = plt.subplots(
        1, len(MEASURES), figsize=(4.33 * len(MEASURES), 4.4), sharex=True, sharey=True
    )
    styles = {"gap_in_rel": ("C0", "In-sample"), "gap_out_rel": ("C3", "Out-of-sample")}

    for ax, measure in zip(axes, MEASURES):
        part = summary[summary["measure"] == measure]
        # Grid arm only: one point per universe size, plotted against the asset
        # universe rather than the currency share.  The currency arm is still
        # solved and summarised into gap_summary.csv, it is simply not drawn.
        line = part[part["arm"] == "grid"].sort_values("universe")
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
    qualifier = "" if family == "unregularized" else f" ({family} programs)"
    fig.suptitle(
        "Joint vs overlay optimization across risk measures and "
        f"increasing investable universe sizes{qualifier}",
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
    parser.add_argument(
        "--arms", nargs="+", default=["grid", "currency"], choices=["grid", "currency"]
    )
    parser.add_argument("--family", default="unregularized", choices=list(FAMILIES))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--plot-only", action="store_true")
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
    gaps_path = args.out_dir / "gaps.csv"

    if args.plot_only:
        gaps = pd.read_csv(gaps_path, parse_dates=["date"])
        if "family" not in gaps.columns:
            gaps["family"] = "unregularized"
    else:
        solver = isa._pick_solver()[0]
        builders = {
            "grid": lambda: grid_arm(solver, args.family),
            "currency": lambda: currency_arm(solver, args.family),
        }
        gaps = pd.concat([builders[arm]() for arm in args.arms], ignore_index=True)
        gaps["family"] = args.family
        if gaps_path.exists():
            old = pd.read_csv(gaps_path, parse_dates=["date"])
            if "family" not in old.columns:
                old["family"] = "unregularized"
            stale = (old["family"] == args.family) & old["arm"].isin(args.arms)
            gaps = pd.concat([old[~stale], gaps], ignore_index=True)
        gaps.to_csv(gaps_path, index=False)
        logger.info("Wrote %d rows to %s", len(gaps), gaps_path)

    gaps["s"] = [currency_share(u, k) for u, k in zip(gaps["universe"], gaps["n_curr"])]
    summary = summarise(gaps)
    grid = gaps[gaps["arm"] == "grid"]
    if grid.empty:
        logger.warning(
            "No 'grid' rows in %s, so the regression and overfitting tables are empty. "
            "They are built from the 9 main cells; merge this arm into the shared gaps.csv "
            "and rerun with --plot-only to get them.",
            gaps_path,
        )
    reg = regressions(grid)
    overfit = overfitting_table(grid)

    for name, table in (("gap_summary", summary), ("regression", reg), ("overfitting", overfit)):
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        logger.info("Wrote %s", args.out_dir / f"{name}.csv")

    for family in sorted(gaps["family"].unique()):
        suffix = "" if family == "unregularized" else f"_{family}"
        plot(summary, args.out_dir / f"gap_vs_currency_share{suffix}", family)
    print(summary.to_string(index=False))
    print()
    print(reg.to_string(index=False))
    print()
    print(overfit.to_string(index=False))


if __name__ == "__main__":
    main()
