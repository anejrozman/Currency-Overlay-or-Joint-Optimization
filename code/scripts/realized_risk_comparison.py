"""
Does regularization or a mean term change the joint-versus-overlay verdict?

The in-sample gap framework of in_sample_vs_out_of_sample.py cannot answer this.
That framework rests on the joint program's in-sample risk being a ceiling, which
requires the scorer to be the objective; regularized and mean-bearing programs
minimize Risk + penalty and Risk - mean/gamma respectively, with penalties and
risk-aversion selected per rebalancing date and routinely differing between the
joint and overlay arms.  Empirically the ceiling is violated on 30.0% of
regularized date-cells and 22.2% of mean-bearing ones, against 0.6% at solver
tolerance for the unregularized programs.

This script therefore asks the question using out-of-sample quantities only.  For
each (family, risk measure, universe) it takes the realized return series of the
joint and overlay production backtests -- net of transaction costs, with drifted
weights -- evaluates the measure's own risk functional on each, and reports the
relative reduction

    (rho_overlay - rho_joint) / rho_overlay,

positive when the joint program realizes less risk.  No in-sample quantity enters
anywhere, so nothing here depends on the nesting argument.

Confidence intervals come from a circular block bootstrap in which the same
resampled indices are applied to both series, preserving the dependence between
the two strategies; the block length is chosen per cell by the Politis-White
rule.

Usage:
    python code/scripts/realized_risk_comparison.py
    python code/scripts/realized_risk_comparison.py --n-boot 9999

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
from scipy.optimize import minimize_scalar

import in_sample_vs_out_of_sample as ivo
from portfolio_backtest import config
from portfolio_backtest.utils.bootstrap import circular_index, politis_white

logger = logging.getLogger("realized_risk_comparison")

MEASURES = ("mv", "mad", "smv", "es", "soes")
UNIVERSES = (10, 50, 100)
ES_ALPHA = 0.80

# Each family is defined by the runs table naming its production backtests and by
# how its strategies are spelled in the registry. The de-meaned family is covered
# by the gap analysis instead, its nesting argument being intact.
FAMILIES: dict[str, dict] = {
    "unregularized": {"table": "top_runs.csv", "regularized": False, "mean": False},
    "regularized":   {"table": "top_runs.csv", "regularized": True,  "mean": False},
    "mean-bearing":  {"table": "top_runs_w_mean.csv", "regularized": False, "mean": True},
}

MEASURE_LABELS = {
    "es": "Expected Shortfall", "mv": "Volatility",
    "mad": "Mean Absolute Deviation", "smv": "Semi-Deviation",
    "soes": "Second-Order Expected Shortfall",
}
FAMILY_STYLE = {
    "unregularized": ("C0", "o", "Unregularized"),
    "regularized":   ("C3", "s", "Regularized"),
    "mean-bearing":  ("C2", "^", "With mean term"),
}

OUT_DIR = config.RESULTS_DIR / "realized_risk_comparison"


def _risk(r: np.ndarray, measure: str, alpha: float = ES_ALPHA) -> float:
    """
    The measure's own risk functional, evaluated on a realized return series.

    Mirrors in_sample_analysis.evaluate_risk, which takes weights and return
    matrices; here the portfolio return series is already formed. Volatility and
    semi-deviation are reported on the linear scale (standard deviation rather
    than variance) so that relative reductions are comparable across measures.
    """
    if r.size < 2:
        return float("nan")
    if measure == "mv":
        return float(np.std(r, ddof=1))
    if measure == "mad":
        return float(np.mean(np.abs(r - r.mean())))
    if measure == "smv":
        return float(np.sqrt(np.mean(np.minimum(r - r.mean(), 0.0) ** 2)))
    if measure == "es":
        tail = r[r <= np.quantile(r, 1.0 - alpha)]
        return float(-tail.mean()) if tail.size else float("nan")
    if measure == "soes":
        # HMCR_2 (Krokhmal 2007): min over zeta of
        # zeta + ||max(loss - zeta, 0)||_2 / ((1-alpha) sqrt(T)).
        losses = -r
        tail_prob = 1.0 - alpha
        best = minimize_scalar(
            lambda z: z + np.linalg.norm(np.maximum(losses - z, 0.0))
            / (tail_prob * np.sqrt(losses.size)),
            bounds=(float(losses.min()), float(losses.max())),
            method="bounded", options={"xatol": 1e-10},
        )
        return float(best.fun)
    raise ValueError(f"Unknown measure '{measure}'.")


def _strategies(measure: str, family: str) -> dict[str, str]:
    """Registry names of the joint and overlay programs of one family."""
    spec = FAMILIES[family]
    want_reg = spec["regularized"]
    want_mean = spec["mean"]
    resolved = {}
    for role in ("joint", "overlay"):
        parts = {measure, role} | ({"mean"} if want_mean else set())
        matches = [
            name for name in ivo.STRATEGY_REGISTRY
            if ("reg" in name.split("_")) == want_reg
            and ("mean" in name.split("_")) == want_mean
            and parts.issubset(set(name.split("_")))
        ]
        if len(matches) != 1:
            raise ValueError(f"{family} {measure} {role}: found {matches}")
        resolved[role] = matches[0]
    return resolved


def _run_number(table: pd.DataFrame, strategy: str, universe: int) -> int | None:
    row = table[table["strategy_name"] == strategy]
    if row.empty:
        return None
    value = row[f"top_{universe}_run"].iloc[0]
    if isinstance(value, str) and not value.strip():
        return None
    return None if pd.isna(value) else int(value)


def _returns(strategy: str, run: int) -> pd.Series | None:
    """
    Realized out-of-sample returns of one run, or None if it has not landed.

    A runs table records the run number a backtest was launched under, so a cell
    can name a directory that exists but holds no results yet -- or, as with
    soes_joint_w_mean, no directory at all. The file on disk is the precondition.
    """
    path = config.RUNS_DIR / strategy / f"run_{run}" / "oos_returns.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)["return"].dropna()


def compare(n_boot: int, seed: int) -> pd.DataFrame:
    tables = {
        name: None for name in {spec["table"] for spec in FAMILIES.values()}
    }
    for name in tables:
        t = pd.read_csv(config.RUNS_DIR / name)
        t.columns = [c.strip() for c in t.columns]
        t["strategy_name"] = t["strategy_name"].astype(str).str.strip()
        tables[name] = t

    rng = np.random.default_rng(seed)
    rows = []
    for family, spec in FAMILIES.items():
        table = tables[spec["table"]]
        for measure in MEASURES:
            try:
                names = _strategies(measure, family)
            except ValueError:
                logger.warning("No %s %s pair registered; skipping.", family, measure)
                continue
            for universe in UNIVERSES:
                runs = {r: _run_number(table, n, universe) for r, n in names.items()}
                if any(v is None for v in runs.values()):
                    logger.warning(
                        "%s %s top-%d: run not recorded; skipping.",
                        family, measure, universe,
                    )
                    continue
                rj_s, ro_s = (_returns(names[r], runs[r]) for r in ("joint", "overlay"))
                if rj_s is None or ro_s is None:
                    absent = [names[r] for r, s_ in (("joint", rj_s), ("overlay", ro_s))
                              if s_ is None]
                    logger.warning(
                        "%s %s top-%d: no oos_returns.csv for %s; skipping.",
                        family, measure, universe, ", ".join(absent),
                    )
                    continue
                idx = rj_s.index.intersection(ro_s.index)
                rj, ro = rj_s.loc[idx].to_numpy(), ro_s.loc[idx].to_numpy()

                point = (_risk(ro, measure) - _risk(rj, measure)) / _risk(ro, measure)
                block = politis_white(rj - ro)
                draws = np.array([
                    (_risk(ro[i], measure) - _risk(rj[i], measure)) / _risk(ro[i], measure)
                    for i in circular_index(len(rj), block, n_boot, rng)
                ])
                draws = draws[np.isfinite(draws)]
                lo, hi = np.percentile(draws, [2.5, 97.5])
                rows.append({
                    "family": family, "measure": measure, "universe": universe,
                    "reduction": point, "ci_lo": lo, "ci_hi": hi,
                    "block": block, "n_obs": len(idx),
                    "joint_run": runs["joint"], "overlay_run": runs["overlay"],
                })
                logger.info(
                    "%s %s top-%d: %+.2f%% [%+.2f, %+.2f]",
                    family, measure, universe, 100*point, 100*lo, 100*hi,
                )
    return pd.DataFrame(rows)


def plot(results: pd.DataFrame, stem: Path) -> None:
    """
    One panel per risk measure, realized risk reduction against universe size,
    one line per family. No in-sample series appears anywhere: the point of the
    figure is that the comparison is made entirely out of sample.
    """
    fig, axes = plt.subplots(
        1, len(MEASURES), figsize=(4.33 * len(MEASURES), 4.4), sharex=True, sharey=True
    )
    for ax, measure in zip(axes, MEASURES):
        for family, (color, marker, _) in FAMILY_STYLE.items():
            line = results[
                (results["measure"] == measure) & (results["family"] == family)
            ].sort_values("universe")
            if line.empty:
                continue
            ax.plot(line["universe"], 100 * line["reduction"],
                    marker=marker, ms=5, color=color, zorder=3)
            ax.fill_between(line["universe"], 100 * line["ci_lo"], 100 * line["ci_hi"],
                            color=color, alpha=0.15, lw=0)
        ax.axhline(0.0, color="grey", lw=1.0, zorder=0)
        ax.set_title(MEASURE_LABELS[measure], fontsize=11)
        ax.set_xlabel("Universe size")
        ax.set_xscale("log")
        ax.set_xticks(list(UNIVERSES))
        ax.set_xticklabels([str(u) for u in UNIVERSES])
        ax.set_xticks([], minor=True)
        ax.set_xlim(min(UNIVERSES) * 0.8, max(UNIVERSES) * 1.25)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Realized risk reduction, joint vs overlay (\\%)")
    axes[0].legend(
        handles=[Line2D([], [], color=c, marker=m, label=l)
                 for c, m, l in FAMILY_STYLE.values()],
        fontsize=7, loc="lower left", framealpha=0.9,
    )
    fig.suptitle(
        "Out-of-sample realized risk of the joint program relative to the sequential "
        "overlay, by risk measure and universe size",
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", stem.with_suffix(f".{ext}"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = compare(args.n_boot, args.seed)
    results.to_csv(args.out_dir / "realized_risk_reduction.csv", index=False)
    logger.info("Wrote %s", args.out_dir / "realized_risk_reduction.csv")
    plot(results, args.out_dir / "realized_risk_reduction")

    print("\nRealized risk reduction, joint vs overlay (%), out of sample\n")
    wide = results.pivot_table(
        index=["measure", "universe"], columns="family", values="reduction"
    ) * 100
    print(wide.round(2).to_string())
    sig = results[(results.ci_lo > 0) | (results.ci_hi < 0)]
    print(f"\n{len(sig)} of {len(results)} cells have an interval clear of zero.")


if __name__ == "__main__":
    main()
