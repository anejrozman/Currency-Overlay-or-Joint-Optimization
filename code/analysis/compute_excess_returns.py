"""Annualized excess return of every top run, for the ER rows of the performance tables."""

from pathlib import Path

import pandas as pd

from portfolio_backtest import config

RESULTS = Path(__file__).resolve().parents[2] / "results"

rf = pd.read_csv(config.RAW_DIR / config.RF_FILE, index_col=0)
rf = rf["RF"].set_axis(pd.to_datetime(rf.index.astype(str), format="%Y%m%d")) / 100.0

rows = []
for file in ("top_runs.csv", "top_runs_w_mean.csv"):
    runs = pd.read_csv(RESULTS / "runs" / file, skipinitialspace=True, index_col=0)
    for optimizer, row in runs.iterrows():
        for column, run in row.items():
            path = RESULTS / "runs" / optimizer / f"run_{run}" / "oos_returns.csv"
            value = float("nan")
            if path.exists():
                r = pd.read_csv(path, index_col=0, parse_dates=True)["return"].dropna()
                cagr = (1 + r).prod() ** (config.TRADING_DAYS / len(r)) - 1
                value = 100 * (cagr - rf.reindex(r.index).ffill().mean() * config.TRADING_DAYS)
            rows.append((optimizer, int(column.split("_")[1]), round(value, 4)))

out = pd.DataFrame(rows, columns=["optimizer", "universe", "annualized_excess_return_pct"]).drop_duplicates()
(RESULTS / "excess_returns").mkdir(exist_ok=True)
out.to_csv(RESULTS / "excess_returns" / "annualized_excess_returns.csv", index=False)
missing = out[out["annualized_excess_return_pct"].isna()]
if not missing.empty:
    print("no oos_returns.csv for:\n" + missing.to_string(index=False))
