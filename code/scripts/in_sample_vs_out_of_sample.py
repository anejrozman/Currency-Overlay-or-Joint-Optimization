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
its joint and its overlay strategy plus a filled cell in results/runs/top_runs.csv,
because the per-date penalties are reused from what production selected rather than
re-searched over the 256-point grid.  reg_soes_joint and reg_soes_curr_overlay have
finished on top-10 alone, so their top-50 and top-100 cells are blank and the
regularized arm covers soes only once those cluster runs land.

Usage:
    python code/scripts/in_sample_vs_out_of_sample.py
    python code/scripts/in_sample_vs_out_of_sample.py --arms grid
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

MEASURES = ("mv", "es", "mad", "smv", "soes")
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
    "soes": "Second-Order Expected Shortfall",
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

# The delta-epsilon figure splits on the taxonomy the mechanism argument uses:
# deviation measures see every observation, tail measures only the tail.  They are
# drawn on separate axes because second-order Expected Shortfall runs an order of
# magnitude above the deviation measures and would flatten them onto one scale.
MEASURE_PANELS = (
    ("Deviation measures", ("mv", "mad", "smv"), "lower right"),
    ("Tail measures", ("es", "soes"), "lower right"),
)
MEASURE_COLORS = {"mv": "C0", "mad": "C1", "smv": "C2", "es": "C3", "soes": "C4"}
MEASURE_MARKERS = {"mv": "o", "mad": "s", "smv": "^", "es": "D", "soes": "v"}


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
            if ("reg" in name.split("_")) == want_reg
            # The mean-bearing programs (*_w_mean) are a separate family with their
            # own script; without this they match here too and every measure raises
            # on two candidates. Restores the behaviour this had before they existed.
            and "mean" not in name.split("_")
            and parts.issubset(set(name.split("_")))
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
    # A blank cell is how top_runs.csv records a production run that has not
    # landed; without this it would surface as int(nan) far from the cause.
    run = row[f"top_{universe}_run"].iloc[0]
    if pd.isna(run):
        raise ValueError(
            f"{strategy} has no top-{universe} run recorded in top_runs.csv. The "
            f"regularized arm reuses the penalties that production selected, so it "
            f"cannot run on this cell until that backtest lands."
        )
    return int(run)


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


def _available(measures, universes, family: str) -> list[str]:
    """
    Measures this family can actually be solved for.

    The regularized arm reuses the penalties production selected per date, so a
    measure whose backtest has not landed cannot be solved at all -- top_runs.csv
    records that as a blank cell.  The whole measure is dropped rather than the
    missing cell, because regressions() contrasts a measure's universes against
    each other and a single-universe cell would leave it with no contrast to
    regress on.  Mirrors how the unregularized branch of grid_arm warns and
    skips a measure with no cached in_sample_<measure>_top<N>.csv.
    """
    if family == "unregularized":
        return list(measures)
    keep = []
    for measure in measures:
        missing = [
            f"{name} top-{universe}"
            for universe in universes
            for name in _resolve_strategies(measure, family).values()
            if not _has_run(name, universe)
        ]
        if missing:
            logger.warning(
                "Skipping %s %s: no production run recorded for %s.",
                family, measure, ", ".join(missing),
            )
        else:
            keep.append(measure)
    return keep


def _has_run(strategy: str, universe: int) -> bool:
    try:
        _top_run(strategy, universe)
    except ValueError:
        return False
    return True


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


def grid_arm(solver, family: str, measures, universes) -> pd.DataFrame:
    """
    The 9 main cells.  The unregularized ones were already solved and scored by
    in_sample_analysis.py, so they are read back rather than recomputed; the
    regularized ones have no such cache and are solved here.
    """
    frames = []
    for measure in _available(measures, universes, family):
        for universe in universes:
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


def currency_arm(solver, family: str, measures, universes) -> pd.DataFrame:
    """
    Tradable overlay block cut to the K largest economies by constituent count,
    with the asset universe left intact.  Regularized cells reuse the penalties
    production selected on the full K = 14 problem, which is an approximation:
    those penalties were not chosen for a reduced currency block.
    """
    frames = []
    measures = _available(measures, universes, family)
    for universe, counts in CURRENCY_ARM.items():
        if universe not in universes:
            continue
        R_fh, R_c, mask, atc = _load(universe)
        windows = isa._rebalancing_windows(R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"])
        order = [c for c in pd.Series(list(atc.values())).value_counts().index if c in R_c.columns]
        for n_curr in counts:
            keep = order[:n_curr]
            if "USD" not in keep:
                keep = ["USD"] + order[: n_curr - 1]
            logger.info("Currency arm %s top-%d K=%d: %s", family, universe, n_curr, keep)
            for measure in measures:
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


def delta_epsilon_table(grid: pd.DataFrame) -> pd.DataFrame:
    """
    The shortfall term of the decomposition, estimated pairwise.

    g_oos = g_is - delta_eps identically at every rebalancing date, so delta_eps is
    observed per date and needs no separate model.  Bootstrapping the paired
    difference rather than differencing two marginal intervals is what makes this
    worth reporting on its own: both gaps move together across dates -- they share a
    universe, a window and a draw of returns -- so the pairing removes that common
    variation.  For Expected Shortfall at top-100 the marginal intervals on g_is and
    g_oos overlap enough to leave the difference ambiguous, while the paired interval
    is [+3.29, +5.03] percentage points.

    Positive delta_eps means the in-sample gap overstated what the joint program went
    on to realize.  The sign is not assumed anywhere; it is read off the data.
    """
    rows = []
    for (family, measure, universe), cell in grid.groupby(["family", "measure", "universe"]):
        paired = (cell["gap_in_rel"] - cell["gap_out_rel"]).to_numpy()
        mean, lo, hi, se, block = block_bootstrap_mean(paired, SEED)
        rows.append(
            {
                "family": family,
                "measure": measure,
                "universe": universe,
                "delta_eps_mean": mean,
                "delta_eps_lo": lo,
                "delta_eps_hi": hi,
                "delta_eps_se": se,
                "delta_eps_block": block,
                "n_dates": cell["date"].nunique(),
                # Share of dates on which the realized gap exceeded the in-sample one.
                "frac_negative": float((paired < 0).mean()),
            }
        )
    return _sorted(rows, ["family", "measure", "universe"])


def plot_delta_epsilon(table: pd.DataFrame, stem: Path, family: str) -> None:
    """
    Delta-epsilon against universe size, one line per risk measure.

    Two panels, not one: see MEASURE_PANELS.  Values are percentage points of the
    sequential program's risk, since both gaps entering the difference are relative.
    """
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
            ax.plot(
                line["universe"], 100 * line["delta_eps_mean"],
                marker=MEASURE_MARKERS[measure], ms=5, color=color,
                label=MEASURE_LABELS[measure], zorder=3,
            )
            ax.fill_between(
                line["universe"], 100 * line["delta_eps_lo"], 100 * line["delta_eps_hi"],
                color=color, alpha=0.15, lw=0,
            )
        ax.axhline(0.0, color="grey", lw=1.0, zorder=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Universe size")
        ax.set_xscale("log")
        ax.set_xticks(list(UNIVERSES))
        ax.set_xticklabels([str(u) for u in UNIVERSES])
        ax.set_xticks([], minor=True)
        ax.set_xlim(min(UNIVERSES) * 0.8, max(UNIVERSES) * 1.25)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc=legend_loc, framealpha=0.9)
        # Labelled on both panels: the two carry the same quantity on deliberately
        # different scales, and a single left-hand label invites reading the right
        # panel off the left axis.
        ax.set_ylabel("In-sample gap not realized, $\\Delta\\epsilon$ (pp)")
    qualifier = "" if family == "unregularized" else f" ({family} programs)"
    fig.suptitle(
        "Part of the joint program's in-sample advantage that is not realized "
        f"out of sample{qualifier}",
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", stem.with_suffix(f".{ext}"))
    plt.close(fig)


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
                line["universe"], 100 * line[f"{col}_mean"],
                marker="s", ms=5, color=color, zorder=3,
            )
            ax.fill_between(
                line["universe"], 100 * line[f"{col}_lo"], 100 * line[f"{col}_hi"],
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

    axes[0].set_ylabel("Risk reduction, joint vs sequential (%)")
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
    parser.add_argument(
        "--measures", nargs="+", default=list(MEASURES), choices=list(MEASURES),
        help="Measures to solve (default: all). The regularized arm re-solves rather "
             "than reading a cache, so scope it to what actually needs rebuilding.",
    )
    parser.add_argument(
        "--universes", nargs="+", type=int, default=list(UNIVERSES), choices=list(UNIVERSES),
        help="Universe sizes to solve (default: all three). Split across concurrent "
             "jobs with distinct --out-dir, then merge their gaps.csv.",
    )
    parser.add_argument(
        "--sweep-only", action="store_true",
        help="Write gaps.csv and stop, skipping the tables and figures. Use for a job "
             "covering one slice, whose regression would have no universe contrast.",
    )
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
            "grid": lambda: grid_arm(solver, args.family, args.measures, args.universes),
            "currency": lambda: currency_arm(solver, args.family, args.measures, args.universes),
        }
        gaps = pd.concat([builders[arm]() for arm in args.arms], ignore_index=True)
        gaps["family"] = args.family
        if gaps_path.exists():
            old = pd.read_csv(gaps_path, parse_dates=["date"])
            if "family" not in old.columns:
                old["family"] = "unregularized"
            stale = (
                (old["family"] == args.family)
                & old["arm"].isin(args.arms)
                & old["measure"].isin(args.measures)
                & old["universe"].isin(args.universes)
            )
            gaps = pd.concat([old[~stale], gaps], ignore_index=True)
        gaps.to_csv(gaps_path, index=False)
        logger.info("Wrote %d rows to %s", len(gaps), gaps_path)
        if args.sweep_only:
            return

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

    delta_eps = delta_epsilon_table(grid)

    for name, table in (
        ("gap_summary", summary), ("regression", reg),
        ("overfitting", overfit), ("delta_epsilon", delta_eps),
    ):
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        logger.info("Wrote %s", args.out_dir / f"{name}.csv")

    for family in sorted(gaps["family"].unique()):
        suffix = "" if family == "unregularized" else f"_{family}"
        plot(summary, args.out_dir / f"gap_vs_currency_share{suffix}", family)
        plot_delta_epsilon(delta_eps, args.out_dir / f"delta_epsilon{suffix}", family)
    print(summary.to_string(index=False))
    print()
    print(reg.to_string(index=False))
    print()
    print(overfit.to_string(index=False))
    print()
    print(delta_eps.to_string(index=False))


if __name__ == "__main__":
    main()
