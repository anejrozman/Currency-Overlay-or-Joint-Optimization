"""
Do the joint and sequential programs differ in risk-adjusted return?

The gap analysis in in_sample_vs_out_of_sample.py compares the two approaches on
the risk measure they minimize.  This script asks the complementary question a
reader asks first: does joint optimization earn more per unit of risk?  It
compares the realized out-of-sample return series of the joint and overlay
production backtests -- net of transaction costs, with drifted weights -- on the
three ratios reported in the performance tables: Sharpe, Sortino and the
Expected Shortfall ratio.

Why a bespoke test is needed
----------------------------
The two series are strongly dependent: same dates, same asset universe, same
data, differing only in how the weights were chosen.  A two-sample test that
assumes independence is therefore invalid.  Daily equity returns are also
fat-tailed and serially dependent, which rules out the closed-form test of
Jobson and Korkie (1981) with the correction of Memmel (2003): it assumes i.i.d.
normal returns and over-rejects substantially when that fails, as Ledoit and
Wolf (2008) document.

The test used here is the studentized circular block bootstrap of Ledoit and
Wolf (2008), which is designed for exactly this setting -- two dependent return
series, non-normal and serially correlated.

    H0: ratio_joint = ratio_overlay

    1. Point estimate Delta = ratio_joint - ratio_overlay.
    2. HAC standard error by the delta method.  Each ratio is a smooth function
       of a small vector of moments, so Var(Delta) = grad' Psi grad / T with Psi
       a Newey-West (Bartlett kernel) estimate of the long-run covariance of the
       moment contributions.  Ledoit and Wolf use a prewhitened kernel with an
       Andrews-type automatic bandwidth; the simpler Bartlett estimator with the
       4(T/100)^(2/9) rule is used here, which truncates at 10 lags against the
       O(T^(1/3)) ~ 18 the optimal rate would suggest and so understates Psi
       slightly when the autocovariances are positive, as they are here.
    3. Circular block bootstrap in which the SAME resampled indices are applied
       to both series, preserving the dependence between them.  On each
       replication the studentized statistic (Delta* - Delta)/se* is formed,
       with se* from the Gotze-Kunsch block-sum estimator rather than the HAC
       kernel -- see _blocksum_se.  Centring at Delta rather than at 0 does not
       impose the null; it makes t* mimic the pivot (Delta_hat - Delta)/se, and
       not having to impose the null is the point of the procedure.  The
       p-value and the 95% interval are both obtained by SYMMETRIC inversion of
       the |t*| distribution, so they are the same test.

Sharpe and Sortino are handled this way in full.  The Expected Shortfall ratio
is not: see below.

The three ratios
----------------
All three are an annualized excess mean divided by a risk measure, so they share
a numerator.  They are three views of one quantity rather than three independent
tests, and cannot disagree materially.  They differ only in the denominator:

    Sharpe   = mu / sigma,              sigma^2 = E[r^2] - mu^2
    Sortino  = mu / delta,              delta^2 = E[min(r,0)^2]
    ESR      = mu / ES_alpha,           ES_alpha = -E[r | r <= q_alpha]

Sharpe and Sortino are smooth functions of ordinary moments -- (mu, E[r^2]) and
(mu, E[min(r,0)^2]) respectively -- so the delta method applies directly and the
gradients are elementary.  ES is not: it is defined through a quantile.  It can
be brought into the same framework via the Rockafellar-Uryasev representation

    ES_alpha = min_eta { eta + (1-alpha)^{-1} E[(-r-eta)^+] },

whose minimizer is the Value-at-Risk.  Because the derivative in eta vanishes at
the optimum, the estimation error in the fitted quantile does not contribute to
the first-order asymptotics, and ES has a known influence function.  Deriving
and defending that is a heavier piece of work than this comparison warrants, so
the ESR is reported here with the plain (non-studentized) circular block
bootstrap instead.  That is a weaker procedure -- it has poorer finite-sample
coverage than the studentized version -- and the output labels it as such.  Given
that all three ratios share the unestimable numerator, nothing turns on it.

Interpreting the result
-----------------------
A large p-value here is NOT evidence that the two approaches are equivalent.
The numerator of every one of these ratios is an annualized mean return, which
is not estimable to useful precision even over 5935 trading days, while the
denominators are estimable to about one percent.  Report the confidence interval
as a bound on the effect that can be excluded, and note that the same data and
the same bootstrap do detect the volatility and Expected Shortfall differences
between the two programs -- so the null reflects what the Sharpe ratio can
resolve, not a dead instrument.

Usage:
    python code/scripts/sharpe_comparison.py
    python code/scripts/sharpe_comparison.py --table top_runs_w_mean.csv
    python code/scripts/sharpe_comparison.py --n-boot 9999 --block 10

Author: Anej Rozman
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import in_sample_vs_out_of_sample as ivo
from portfolio_backtest import config

logger = logging.getLogger("sharpe_comparison")

MEASURES = ("mv", "mad", "smv", "es", "soes")
FAMILIES = ("unregularized", "regularized")
UNIVERSES = (10, 50, 100)
ES_ALPHA = 0.80
TRADING_DAYS = config.TRADING_DAYS

# Ledoit-Wolf calibrate the block length; the p-values here are flat in it from
# b = 1 (i.i.d.) to b = 63 (a quarter), so a fixed value is used and the
# --block flag is provided to re-check that.
DEFAULT_BLOCK = 5
DEFAULT_N_BOOT = 4999


# Ratios and their moment gradients


def _sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=0)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def _sortino(r: np.ndarray) -> float:
    dd = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    return float(r.mean() / dd) if dd > 0 else float("nan")


def _es(r: np.ndarray, alpha: float = ES_ALPHA) -> float:
    tail = r[r <= np.quantile(r, 1.0 - alpha)]
    return float(-tail.mean()) if tail.size else float("nan")


def _esr(r: np.ndarray, alpha: float = ES_ALPHA) -> float:
    es = _es(r, alpha)
    return float(r.mean() / es) if es > 0 else float("nan")


def _moment_parts(r1: np.ndarray, r2: np.ndarray, ratio: str):
    """
    The moment vector each ratio is a smooth function of, its gradient, and the
    per-observation contributions whose long-run covariance Psi estimates.

    Sharpe:  v = (mu1, mu2, g1, g2),  g = E[r^2],          SR = mu / sqrt(g - mu^2)
    Sortino: v = (mu1, mu2, d1, d2),  d = E[min(r,0)^2],  SO = mu / sqrt(d)
    """
    mu1, mu2 = r1.mean(), r2.mean()
    if ratio == "sharpe":
        a1, a2 = (r1**2).mean(), (r2**2).mean()
        s1, s2 = a1 - mu1**2, a2 - mu2**2
        grad = np.array([
            a1 / s1**1.5, -a2 / s2**1.5,
            -mu1 / (2 * s1**1.5), mu2 / (2 * s2**1.5),
        ])
        contrib = np.column_stack([r1 - mu1, r2 - mu2, r1**2 - a1, r2**2 - a2])
    elif ratio == "sortino":
        a1 = (np.minimum(r1, 0.0) ** 2).mean()
        a2 = (np.minimum(r2, 0.0) ** 2).mean()
        grad = np.array([
            1.0 / np.sqrt(a1), -1.0 / np.sqrt(a2),
            -mu1 / (2 * a1**1.5), mu2 / (2 * a2**1.5),
        ])
        contrib = np.column_stack([
            r1 - mu1, r2 - mu2,
            np.minimum(r1, 0.0) ** 2 - a1, np.minimum(r2, 0.0) ** 2 - a2,
        ])
    else:
        raise ValueError(f"no analytical gradient for '{ratio}'")
    return grad, contrib


def _hac_se(r1: np.ndarray, r2: np.ndarray, ratio: str, bandwidth: int | None = None) -> float:
    """
    Delta-method standard error of the ratio difference with a Newey-West
    (Bartlett kernel) long-run covariance, as in Ledoit and Wolf (2008).
    """
    grad, contrib = _moment_parts(r1, r2, ratio)
    T = len(r1)
    if bandwidth is None:
        # Newey-West rule of thumb; the result is insensitive to it here.
        bandwidth = int(np.ceil(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    psi = contrib.T @ contrib / T
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        cross = contrib[lag:].T @ contrib[:-lag] / T
        psi += weight * (cross + cross.T)
    var = float(grad @ psi @ grad / T)
    return float(np.sqrt(var)) if var > 0 else float("nan")


def _blocksum_se(r1: np.ndarray, r2: np.ndarray, ratio: str, block: int) -> float:
    """
    Delta-method standard error using the Gotze-Kunsch block-sum estimate of the
    long-run covariance,

        Psi* = l^-1 sum_j zeta_j zeta_j',   zeta_j = b^-1/2 sum_{t in block j} c_t.

    This is the estimator to use INSIDE the bootstrap.  A resample is a
    concatenation of independently drawn blocks, so its autocovariance across a
    block junction is zero by construction; running a HAC kernel on it (with a
    bandwidth that here exceeds the block length) would spend most of its weight
    estimating that zero and would break the second-order refinement the
    studentized block bootstrap is meant to deliver.
    """
    grad, contrib = _moment_parts(r1, r2, ratio)
    T = len(r1)
    n_full = T // block                       # whole blocks only
    if n_full < 2:
        return float("nan")
    zeta = contrib[: n_full * block].reshape(n_full, block, -1).sum(axis=1) / np.sqrt(block)
    zeta = zeta - zeta.mean(axis=0)
    psi = zeta.T @ zeta / n_full
    var = float(grad @ psi @ grad / T)
    return float(np.sqrt(var)) if var > 0 else float("nan")


# Bootstrap


def _circular_blocks(n_obs: int, block: int, n_boot: int, rng) -> np.ndarray:
    """
    Resampled position matrix. One row per replication; the SAME row is applied
    to both return series so the dependence between the two strategies survives
    the resampling.
    """
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_obs
    return idx.reshape(n_boot, -1)[:, :n_obs]


def studentized_test(r1, r2, ratio, block, n_boot, seed=0) -> dict:
    """Ledoit-Wolf (2008) studentized circular block bootstrap."""
    fn = {"sharpe": _sharpe, "sortino": _sortino}[ratio]
    delta = fn(r1) - fn(r2)
    se = _hac_se(r1, r2, ratio)
    rng = np.random.default_rng(seed)
    idx = _circular_blocks(len(r1), block, n_boot, rng)

    stats = np.empty(n_boot)
    for i, rows in enumerate(idx):
        a, b = r1[rows], r2[rows]
        se_b = _blocksum_se(a, b, ratio, block)
        stats[i] = (fn(a) - fn(b) - delta) / se_b if se_b > 0 else np.nan
    stats = stats[np.isfinite(stats)]

    if not (se > 0) or stats.size == 0:
        return {"delta": delta, "se": se, "t": float("nan"), "p_value": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "method": "LW studentized"}

    t_obs = delta / se
    # Symmetric inversion for BOTH the p-value and the interval, so the two are
    # the same test: the interval excludes zero exactly when p < alpha.  The
    # (+1)/(B+1) correction keeps the p-value exact under the null.
    n_eff = stats.size
    p_value = (float((np.abs(stats) >= abs(t_obs)).sum()) + 1.0) / (n_eff + 1.0)
    q_sym = float(np.quantile(np.abs(stats), 0.95))
    return {
        "delta": delta,
        "se": se,
        "t": t_obs,
        "p_value": p_value,
        "ci_lo": delta - q_sym * se,
        "ci_hi": delta + q_sym * se,
        "method": "LW studentized",
    }


def percentile_test(r1, r2, ratio, block, n_boot, seed=0) -> dict:
    """
    Plain circular block bootstrap, used for the ESR where no analytical
    gradient is supplied. Weaker finite-sample coverage than the studentized
    version; labelled as such in the output.
    """
    fn = {"esr": _esr}[ratio]
    delta = fn(r1) - fn(r2)
    rng = np.random.default_rng(seed)
    idx = _circular_blocks(len(r1), block, n_boot, rng)
    draws = np.array([fn(r1[rows]) - fn(r2[rows]) for rows in idx])
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # Bootstrap p-value by interval inversion: twice the smaller tail mass,
    # with the same (+1)/(B+1) correction used by the studentized test.
    n_eff = draws.size
    p = 2.0 * min((float((draws <= 0).sum()) + 1.0) / (n_eff + 1.0),
                  (float((draws >= 0).sum()) + 1.0) / (n_eff + 1.0))
    return {
        "delta": delta, "se": float(draws.std(ddof=1)), "t": float("nan"),
        "p_value": float(min(p, 1.0)), "ci_lo": lo, "ci_hi": hi,
        "method": "percentile",
    }


# Data


def _excess_returns(strategy: str, run: int, rf: pd.Series) -> pd.Series:
    path = config.RUNS_DIR / strategy / f"run_{run}" / "oos_returns.csv"
    r = pd.read_csv(path, index_col=0, parse_dates=True)["return"].dropna()
    # method="ffill" looks back into the full risk-free series rather than only
    # within the reindexed frame, so a holiday at the start is not left NaN.
    return r - rf.reindex(r.index, method="ffill")


def _load_table(table: str) -> pd.DataFrame:
    t = pd.read_csv(config.RUNS_DIR / table)
    t.columns = [c.strip() for c in t.columns]
    t["strategy_name"] = t["strategy_name"].astype(str).str.strip()
    return t


def _run_number(table: pd.DataFrame, strategy: str, universe: int) -> int | None:
    row = table[table["strategy_name"] == strategy]
    if row.empty:
        return None
    value = row[f"top_{universe}_run"].iloc[0]
    if isinstance(value, str) and not value.strip():
        return None
    return None if pd.isna(value) else int(value)


def compare(table_name: str, block: int, n_boot: int, seed: int) -> pd.DataFrame:
    rf_raw = pd.read_csv(config.RAW_DIR / config.RF_FILE, index_col=0)
    rf_raw.index = pd.to_datetime(rf_raw.index, format="%Y%m%d")
    rf = rf_raw["RF"] / 100.0
    table = _load_table(table_name)

    rows = []
    for family in FAMILIES:
        for measure in MEASURES:
            try:
                names = ivo._resolve_strategies(measure, family)
            except ValueError:
                logger.warning("No %s %s pair registered; skipping.", family, measure)
                continue
            for universe in UNIVERSES:
                runs = {r: _run_number(table, n, universe) for r, n in names.items()}
                if any(v is None for v in runs.values()):
                    missing = [names[r] for r, v in runs.items() if v is None]
                    logger.warning(
                        "%s %s top-%d: no run recorded for %s; skipping.",
                        family, measure, universe, ", ".join(missing),
                    )
                    continue
                series = {r: _excess_returns(names[r], runs[r], rf) for r in names}
                idx = series["joint"].index.intersection(series["overlay"].index)
                rj = series["joint"].loc[idx].to_numpy()
                ro = series["overlay"].loc[idx].to_numpy()

                for ratio in ("sharpe", "sortino", "esr"):
                    test = studentized_test if ratio != "esr" else percentile_test
                    res = test(rj, ro, ratio, block, n_boot, seed)
                    rows.append({
                        "family": family, "measure": measure, "universe": universe,
                        "ratio": ratio, "n_obs": len(idx),
                        "joint_run": runs["joint"], "overlay_run": runs["overlay"],
                        # Annualized so the numbers match the performance tables.
                        "delta_ann": res["delta"] * np.sqrt(TRADING_DAYS),
                        "se_ann": res["se"] * np.sqrt(TRADING_DAYS),
                        "ci_lo_ann": res["ci_lo"] * np.sqrt(TRADING_DAYS),
                        "ci_hi_ann": res["ci_hi"] * np.sqrt(TRADING_DAYS),
                        "t_stat": res["t"], "p_value": res["p_value"],
                        "method": res["method"],
                    })
                logger.info("%s %s top-%d done (%d obs)", family, measure, universe, len(idx))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--table", default="top_runs.csv",
                        help="Runs table in results/runs/ naming the joint and overlay runs.")
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK,
                        help="Circular block length in trading days.")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path,
                        default=config.RESULTS_DIR / "sharpe_comparison")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = compare(args.table, args.block, args.n_boot, args.seed)
    stem = Path(args.table).stem
    path = args.out_dir / f"ratio_tests_{stem}.csv"
    results.to_csv(path, index=False)
    logger.info("Wrote %s", path)

    for ratio in ("sharpe", "sortino", "esr"):
        part = results[results["ratio"] == ratio]
        if part.empty:
            continue
        print(f"\n=== {ratio.upper()}  (joint minus overlay, annualized) ===")
        print(f"{'family':>14} {'meas':>5} {'uni':>5} {'delta':>9} {'se':>8} "
              f"{'p':>7}  {'95% CI':>20}  method")
        for _, r in part.iterrows():
            print(f"{r.family:>14} {r.measure:>5} {r.universe:>5} {r.delta_ann:+9.4f} "
                  f"{r.se_ann:8.4f} {r.p_value:7.3f}  "
                  f"[{r.ci_lo_ann:+8.4f},{r.ci_hi_ann:+8.4f}]  {r.method}")
    print(f"\nSignificant at 5%: {(results.p_value < 0.05).sum()} of {len(results)} tests.")


if __name__ == "__main__":
    main()
