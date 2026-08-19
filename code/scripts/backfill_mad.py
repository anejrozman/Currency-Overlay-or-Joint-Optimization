"""

Description: One-off maintenance script that backfills the "MAD (ann., %)" metric into
the metrics.json of runs completed before that metric was added to
Backtester.compute_performance_metrics. New runs emit it natively, so this script is
only needed for older run directories.

Only the new key is written. Every other value is left untouched: turnover in
particular is recorded during the walk-forward loop and cannot be recovered from the
saved return series, so nothing else is recomputed.

Before writing, each run's stored annualized volatility is recomputed from its
oos_returns.csv and compared against the stored value. If it does not reproduce, the
return series does not correspond to the metrics file and the run is skipped rather
than silently given a number that belongs to a different series.

The metric matches compute_performance_metrics exactly: the mean absolute deviation
about the realized out-of-sample mean of the full evaluation series, annualized by
sqrt(TRADING_DAYS). Note that sqrt-scaling is the convention shared with the
volatility and ES columns, not an estimate of realized annual dispersion — MAD/sigma
is not invariant across horizons (it rises from ~0.68 daily to ~0.82 annually as
aggregation pulls returns toward Gaussian), so the annualized figure is comparable
across strategies but reads low against directly measured annual MAD.

Usage:
    python code/scripts/backfill_mad.py            # dry run, reports what would change
    python code/scripts/backfill_mad.py --apply    # write the files

Author: Anej Rozman

"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from portfolio_backtest import config

KEY = "MAD (ann., %)"
AFTER = "Volatility (ann., %)"
VOL_TOL = 5e-4  # stored metrics are rounded to 4 decimal places


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill the annualized MAD metric into existing run metrics.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the files. Without this flag the script only reports what would change.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    added = present = skipped = mismatch = 0

    for metrics_path in sorted(config.RUNS_DIR.glob("*/run_*/metrics.json")):
        returns_path = metrics_path.parent / "oos_returns.csv"
        run_name = metrics_path.parent.relative_to(config.RUNS_DIR)

        if not returns_path.exists():
            skipped += 1
            print(f"  skip (no oos_returns.csv)   {run_name}")
            continue

        metrics = json.loads(metrics_path.read_text())
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)["return"].dropna()
        if len(returns) < 2:
            skipped += 1
            print(f"  skip (series too short)     {run_name}")
            continue

        # Guard: the stored volatility must reproduce from this return series.
        vol = round(float(returns.std(ddof=1) * math.sqrt(config.TRADING_DAYS)) * 100, 4)
        stored_vol = metrics.get(AFTER)
        if stored_vol is None or abs(vol - stored_vol) > VOL_TOL:
            mismatch += 1
            print(
                f"  SKIP (volatility mismatch)  {run_name}: "
                f"stored={stored_vol} recomputed={vol}"
            )
            continue

        mad = round(
            float((returns - returns.mean()).abs().mean()) * math.sqrt(config.TRADING_DAYS) * 100,
            4,
        )

        if KEY in metrics:
            present += 1
            continue

        # Insert directly after the volatility, matching where the backtester writes it.
        updated: dict = {}
        for key, value in metrics.items():
            updated[key] = value
            if key == AFTER:
                updated[KEY] = mad
        if KEY not in updated:  # no volatility key: append rather than drop the metric
            updated[KEY] = mad

        if args.apply:
            metrics_path.write_text(json.dumps(updated, indent=2) + "\n")
        added += 1
        print(f"  {'wrote' if args.apply else 'would write'} {KEY}={mad:<9} {run_name}")

    total = added + present + skipped + mismatch
    print(
        f"\n{'APPLIED' if args.apply else 'DRY RUN'}: {added} added, {present} already present, "
        f"{skipped} skipped, {mismatch} volatility mismatch (of {total} runs)"
    )
    if not args.apply and added:
        print("re-run with --apply to write")


if __name__ == "__main__":
    main()
