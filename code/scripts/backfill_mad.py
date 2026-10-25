"""One-off: add the "MAD (ann., %)" metric to the metrics.json of runs that predate it.

A run is skipped unless its stored volatility reproduces from its oos_returns.csv. MAD is computed as in
Backtester.compute_performance_metrics: mean absolute deviation about the mean, annualized by sqrt(252).

Usage:
    python code/scripts/backfill_mad.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from portfolio_backtest import config

KEY = "MAD (ann., %)"
AFTER = "Volatility (ann., %)"
VOL_TOL = 5e-4  # stored metrics are rounded to 4 decimals


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill the annualized MAD into existing metrics.json files.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write the files; without it, only report.")
    apply = parser.parse_args().apply

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
        vol = round(float(returns.std(ddof=1) * math.sqrt(config.TRADING_DAYS)) * 100, 4)
        if metrics.get(AFTER) is None or abs(vol - metrics[AFTER]) > VOL_TOL:
            mismatch += 1
            print(f"  SKIP (volatility mismatch)  {run_name}: stored={metrics.get(AFTER)} recomputed={vol}")
            continue
        if KEY in metrics:
            present += 1
            continue
        mad = round(float((returns - returns.mean()).abs().mean()) * math.sqrt(config.TRADING_DAYS) * 100, 4)
        # placed after the volatility, where the backtester writes it
        updated = {}
        for key, value in metrics.items():
            updated[key] = value
            if key == AFTER:
                updated[KEY] = mad
        if apply:
            metrics_path.write_text(json.dumps(updated, indent=2) + "\n")
        added += 1
        print(f"  {'wrote' if apply else 'would write'} {KEY}={mad:<9} {run_name}")

    total = added + present + skipped + mismatch
    print(f"\n{'APPLIED' if apply else 'DRY RUN'}: {added} added, {present} already present, {skipped} skipped, "
          f"{mismatch} volatility mismatch (of {total} runs)")
    if not apply and added:
        print("re-run with --apply to write")


if __name__ == "__main__":
    main()
