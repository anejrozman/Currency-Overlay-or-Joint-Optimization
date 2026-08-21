"""
Does the joint program's in-sample edge over the currency overlay survive out of sample,
and does it collapse as the currency share of the decision vector falls?

At every quarterly rebalancing date the unregularized joint and overlay programs of one
risk measure are fit on the ten-year estimation window and scored twice by one shared
empirical scorer: on that window (f_in, the ceiling) and on the following quarter
(f_out).  The signed gap g = f_joint - f_overlay is negative when joint is better.  Per
(measure, universe) cell the mean of each gap gets a circular block bootstrap CI (10000
resamples, block length by Politis-White), alongside the fraction of dates with g < 0.

Pooled per measure, the relative gap is regressed on the currency share
s = K / (K + 14 * top_N) with block-bootstrapped standard errors, turning the collapse
into one slope.  Two mechanism arms move s without moving the asset count: a
currency-count arm (top-50 universe intact, tradable overlay block cut to K = 10, 6 by
economy size) and a shuffle arm (top-10, 20 draws permuting which currency each security
is hedged with, bucket sizes preserved).  Under --shuffle-mode rehedge the security keeps
its true local return and its true FX exposure but is hedged with the permuted currency,
R = R_u + f_perm - e_perm, leaving residual exposure e_true - e_perm: the one-hedge-per-
security sparsity and the 14-currency block survive, only their alignment with the true
returns does not.  Under --shuffle-mode label the permutation instead selects which
forward premium enters R_fh, which moves returns by well under a percent and is retained
only as a null control.  A supporting table reports each program's realized risk over
equal weight's, the absolute penalty both pay where their difference does not.

The main 9 cells are read from results/in_sample_analysis/ rather than re-solved.

Usage:
    python code/scripts/in_sample_vs_out_of_sample.py
    python code/scripts/in_sample_vs_out_of_sample.py --arms grid currency shuffle
    python code/scripts/in_sample_vs_out_of_sample.py --plot-only

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import in_sample_analysis as isa
from portfolio_backtest import config
from portfolio_backtest.strategy_registry import STRATEGY_REGISTRY
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (
    compute_daily_forward_premium,
    compute_excess_currency_returns,
    compute_fully_hedged_returns,
    compute_unhedged_returns,
)

logger = logging.getLogger("in_sample_vs_out_of_sample")

MEASURES = ("mv", "es", "mad")
UNIVERSES = (10, 50, 100)
N_CURRENCIES = 14
CURRENCY_COUNTS = (10, 6)
CURRENCY_UNIVERSE = 50
SHUFFLE_UNIVERSE = 10
SHUFFLE_DRAWS = 20
N_BOOT = 10000
SEED = 0

SRC_DIR = config.RESULTS_DIR / "in_sample_analysis"
OUT_DIR = config.RESULTS_DIR / "in_sample_vs_out_of_sample"
MEASURE_LABELS = {"es": "ES(80%)", "mv": "Volatility", "mad": "MAD"}


def currency_share(top_n: int, n_curr: int = N_CURRENCIES) -> float:
    return n_curr / (n_curr + N_CURRENCIES * top_n)


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
    paired["gap_in"] = paired["in_joint"] - paired["in_overlay"]
    paired["gap_out"] = paired["out_joint"] - paired["out_overlay"]
    paired["gap_in_rel"] = paired["gap_in"] / paired["in_overlay"]
    paired["gap_out_rel"] = paired["gap_out"] / paired["out_overlay"]
    return paired


def _load(universe: int, rng: np.random.Generator | None = None, mode: str = "label"):
    path = config.RAW_DIR / config.UNIVERSE_FILES[universe]
    R_local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    R_local = R_local.where(R_local.abs() < config.RETURN_FILTER_BACKTEST)
    e = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    f_daily = compute_daily_forward_premium(
        load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE), config.TENOR_DAYS_3M
    )
    atc = build_asset_to_currency(path)
    perm = atc
    if rng is not None:
        keys = list(atc)
        values = [atc[k] for k in keys]
        rng.shuffle(values)
        perm = dict(zip(keys, values))
    R_c = compute_excess_currency_returns(e, f_daily)
    if rng is not None and mode == "rehedge":
        R_u = compute_unhedged_returns(R_local, e, f_daily, atc)
        cols = [c for c in R_u.columns if c in perm]
        idx = R_u.index.intersection(e.index).intersection(f_daily.index)
        swap = pd.DataFrame(
            {c: f_daily.loc[idx, perm[c]].to_numpy() - e.loc[idx, perm[c]].to_numpy() for c in cols},
            index=idx,
        )
        R_fh = R_u.loc[idx, cols] + swap
    else:
        R_fh = compute_fully_hedged_returns(R_local, e, f_daily, perm)
    R_fh = R_fh[[c for c in R_fh.columns if c in atc]]
    common = R_fh.index.intersection(R_c.index)
    return R_fh.loc[common], R_c.loc[common], isa._build_universe_mask(path, common), atc


def _sweep(measure: str, R_fh, R_c, windows, solver) -> pd.DataFrame:
    names = dict(isa._resolve_strategies(measure))
    fixed = {role: isa._fixed_hyperparams(name) for role, name in names.items()}
    names["equal_weight"] = isa.CONTROL_STRATEGY_KEY
    fixed["equal_weight"] = {k: v for k, v in fixed["joint"].items() if k == "alpha"}

    rows = []
    for i, window in enumerate(windows, start=1):
        assets = window["assets"]
        R_fh_est = R_fh.loc[window["estimation_start"] : window["estimation_end"], assets]
        R_c_est = R_c.loc[window["estimation_start"] : window["estimation_end"]]
        R_fh_hold = R_fh.loc[window["holding_start"] : window["holding_end"], assets].fillna(0.0)
        R_c_hold = R_c.loc[window["holding_start"] : window["holding_end"]].fillna(0.0)
        logger.debug("[%d/%d] %s: %d assets", i, len(windows), window["date"].date(), len(assets))

        for role, name in names.items():
            params = {**fixed[role], "solver": solver}
            alpha = params.get("alpha")
            try:
                x, psi = isa._solve_weights(
                    STRATEGY_REGISTRY[name]["optimizer"], params, R_fh_est, R_c_est
                )
                f_in = isa.evaluate_risk(measure, R_fh_est, R_c_est, x, psi, alpha)
                f_out = isa.evaluate_risk(measure, R_fh_hold, R_c_hold, x, psi, alpha)
            except (RuntimeError, ValueError) as exc:
                logger.warning("%s %s %s failed: %s", measure, window["date"].date(), role, exc)
                f_in = f_out = float("nan")
            rows.append(
                {
                    "date": window["date"],
                    "strategy": role,
                    "n_assets": len(assets),
                    "in_sample_risk": f_in,
                    "realized_risk": f_out,
                }
            )
    return pd.DataFrame(rows)


def grid_arm() -> pd.DataFrame:
    frames = []
    for measure in MEASURES:
        for universe in UNIVERSES:
            path = SRC_DIR / f"in_sample_{measure}_top{universe}.csv"
            if not path.exists():
                logger.warning("Missing %s; skipping.", path)
                continue
            rows = pd.read_csv(path, parse_dates=["date"])
            rows = rows.dropna(subset=["in_sample_risk", "realized_risk"])
            paired = _pair(rows, ["date"])
            paired["measure"] = measure
            paired["universe"] = universe
            paired["n_curr"] = N_CURRENCIES
            paired["arm"] = "grid"
            paired["draw"] = 0
            frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def currency_arm(solver) -> pd.DataFrame:
    R_fh, R_c, mask, atc = _load(CURRENCY_UNIVERSE)
    windows = isa._rebalancing_windows(R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"])
    order = [c for c in pd.Series(list(atc.values())).value_counts().index if c in R_c.columns]

    frames = []
    for n_curr in CURRENCY_COUNTS:
        keep = order[:n_curr]
        if "USD" not in keep:
            keep = ["USD"] + order[: n_curr - 1]
        logger.info("Currency arm K=%d: %s", n_curr, keep)
        for measure in MEASURES:
            rows = _sweep(measure, R_fh, R_c[keep], windows, solver)
            paired = _pair(rows.dropna(subset=["in_sample_risk", "realized_risk"]), ["date"])
            paired["measure"] = measure
            paired["universe"] = CURRENCY_UNIVERSE
            paired["n_curr"] = n_curr
            paired["arm"] = "currency"
            paired["draw"] = 0
            frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def shuffle_arm(solver, mode: str = "label") -> pd.DataFrame:
    frames = []
    for draw in range(SHUFFLE_DRAWS):
        rng = np.random.default_rng(SEED + draw)
        R_fh, R_c, mask, _ = _load(SHUFFLE_UNIVERSE, rng=rng, mode=mode)
        windows = isa._rebalancing_windows(R_fh, mask, STRATEGY_REGISTRY["mv_joint"]["config"])
        logger.info("Shuffle draw %d/%d", draw + 1, SHUFFLE_DRAWS)
        for measure in MEASURES:
            rows = _sweep(measure, R_fh, R_c, windows, solver)
            paired = _pair(rows.dropna(subset=["in_sample_risk", "realized_risk"]), ["date"])
            paired["measure"] = measure
            paired["universe"] = SHUFFLE_UNIVERSE
            paired["n_curr"] = N_CURRENCIES
            paired["arm"] = f"shuffle_{mode}"
            paired["draw"] = draw
            frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def politis_white(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 8 or np.allclose(x, 0.0):
        return 1
    k_n = max(5, int(np.sqrt(np.log10(n))))
    m_max = int(np.ceil(np.sqrt(n))) + k_n
    b_max = np.ceil(min(3.0 * np.sqrt(n), n / 3.0))
    acov = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(m_max + 1)])
    rho = acov / acov[0]
    crit = 2.0 * np.sqrt(np.log10(n) / n)
    m = m_max
    for k in range(1, m_max - k_n + 1):
        if np.all(np.abs(rho[k : k + k_n]) < crit):
            m = k - 1
            break
    M = int(min(max(2 * max(m, 1), 2), m_max))
    lags = np.arange(-M, M + 1)
    r = rho[np.abs(lags)]
    flat_top = np.clip(2.0 * (1.0 - np.abs(lags) / M), 0.0, 1.0)
    G = np.sum(flat_top * np.abs(lags) * r)
    D = (4.0 / 3.0) * np.sum(flat_top * r) ** 2
    if D <= 0 or not np.isfinite(G):
        return 1
    b = (2.0 * G**2 / D) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return int(np.clip(round(b), 1, b_max))


def _circular_index(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]


def block_bootstrap_mean(x: np.ndarray, seed: int = SEED) -> tuple[float, float, float, float, int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    nan = float("nan")
    if x.size < 8:
        return (float(np.mean(x)) if x.size else nan), nan, nan, nan, 1
    block = politis_white(x)
    rng = np.random.default_rng(seed)
    means = x[_circular_index(x.size, block, N_BOOT, rng)].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi), float(means.std(ddof=1)), block


def summarise(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (measure, universe, n_curr, arm, draw), part in gaps.groupby(
        ["measure", "universe", "n_curr", "arm", "draw"]
    ):
        row = {
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
            row[f"frac_neg_{side}"] = float((part[f"gap_{side}"] < 0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["measure", "arm", "s", "draw"]).reset_index(drop=True)


def regressions(grid: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for measure, part in grid.groupby("measure"):
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
    for (measure, universe), cell in part.groupby(["measure", "universe"]):
        row = {"measure": measure, "universe": universe, "n_dates": len(cell)}
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
    return pd.DataFrame(rows).sort_values(["measure", "universe"]).reset_index(drop=True)


def plot(summary: pd.DataFrame, reg: pd.DataFrame, stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
    styles = {"gap_in_rel": ("C0", "In-sample"), "gap_out_rel": ("C3", "Out-of-sample")}

    for ax, measure in zip(axes, MEASURES):
        part = summary[summary["measure"] == measure]
        line = part[part["arm"] == "grid"].sort_values("s")
        arm = part[part["arm"] == "currency"].sort_values("s")
        shuffled = part[part["arm"].str.startswith("shuffle")]
        for col, (color, label) in styles.items():
            if not line.empty:
                ax.plot(
                    line["s"], line[f"{col}_mean"], marker="o", color=color, label=label, zorder=3
                )
                ax.fill_between(
                    line["s"], line[f"{col}_lo"], line[f"{col}_hi"], color=color, alpha=0.2, lw=0
                )
            if not arm.empty:
                ax.errorbar(
                    arm["s"],
                    arm[f"{col}_mean"],
                    yerr=[
                        arm[f"{col}_mean"] - arm[f"{col}_lo"],
                        arm[f"{col}_hi"] - arm[f"{col}_mean"],
                    ],
                    fmt="s",
                    ms=6,
                    color=color,
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=2.5,
                    zorder=4,
                )
            if not shuffled.empty:
                ax.scatter(
                    shuffled["s"],
                    shuffled[f"{col}_mean"],
                    marker="o",
                    s=70,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.0,
                    alpha=0.8,
                    zorder=5,
                )
        slope = reg[(reg["measure"] == measure) & (reg["outcome"] == "gap_out_rel")]
        subtitle = f"  (slope {slope['slope'].iloc[0]:.3f})" if not slope.empty else ""
        ax.axhline(0.0, color="grey", lw=1.0, zorder=0)
        ax.set_title(f"{MEASURE_LABELS[measure]}{subtitle}", fontsize=11)
        ax.set_xlabel("Currency share  s = K / (K + 14N)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean (joint - overlay) / overlay")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "The joint-over-overlay ceiling against the currency share of the decision vector\n"
        "circles: top-10/50/100 grid   squares: currency-count arm   hollow: shuffled labels",
        fontsize=11,
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
        "--arms", nargs="+", default=["grid", "currency"], choices=["grid", "currency", "shuffle"]
    )
    parser.add_argument("--shuffle-mode", default="label", choices=["label", "rehedge"])
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
    else:
        solver = isa._pick_solver()[0]
        builders = {
            "grid": grid_arm,
            "currency": lambda: currency_arm(solver),
            "shuffle": lambda: shuffle_arm(solver, args.shuffle_mode),
        }
        gaps = pd.concat([builders[arm]() for arm in args.arms], ignore_index=True)
        if gaps_path.exists():
            replaced = {f"shuffle_{args.shuffle_mode}" if a == "shuffle" else a for a in args.arms}
            old = pd.read_csv(gaps_path, parse_dates=["date"])
            gaps = pd.concat([old[~old["arm"].isin(replaced)], gaps], ignore_index=True)
        gaps.to_csv(gaps_path, index=False)
        logger.info("Wrote %d rows to %s", len(gaps), gaps_path)

    gaps["s"] = [currency_share(u, k) for u, k in zip(gaps["universe"], gaps["n_curr"])]
    summary = summarise(gaps)
    grid = gaps[gaps["arm"] == "grid"]
    reg = regressions(grid)
    overfit = overfitting_table(grid)

    for name, table in (("gap_summary", summary), ("regression", reg), ("overfitting", overfit)):
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        logger.info("Wrote %s", args.out_dir / f"{name}.csv")

    plot(summary, reg, args.out_dir / "gap_vs_currency_share")
    print(summary.to_string(index=False))
    print()
    print(reg.to_string(index=False))
    print()
    print(overfit.to_string(index=False))


if __name__ == "__main__":
    main()
