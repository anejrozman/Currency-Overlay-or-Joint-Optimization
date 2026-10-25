"""Annualized returns and volatilities of hedged assets and excess currency returns by economy (Table 4.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtest import config
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import compute_daily_forward_premium

ORDER = ["AUD", "CAD", "DKK", "EUR", "HKD", "JPY", "NZD", "NOK", "SGD", "ZAR", "SEK", "CHF", "GBP", "USD"]


def annualized(r: pd.Series | pd.DataFrame, n) -> tuple:
    return ((1 + r).prod() ** (config.TRADING_DAYS / n) - 1) * 100, r.std() * np.sqrt(config.TRADING_DAYS) * 100


if __name__ == "__main__":
    local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    spot = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    premium = compute_daily_forward_premium(load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_1M_FILE),
                                            config.TENOR_DAYS_1M)
    excess_fx = (spot - premium).dropna(how="all").drop(columns=["USD"])
    ret, vol = annualized(excess_fx, excess_fx.count())
    currency_stats = pd.DataFrame({"Currency_Excess_Return": ret, "Currency_Excess_Volatility": vol})
    currency_stats.loc["USD"] = {"Currency_Excess_Return": 0.0, "Currency_Excess_Volatility": 0.0}

    for size, name in config.UNIVERSE_FILES.items():
        atc = build_asset_to_currency(config.RAW_DIR / name)
        print(f"\n Number of assets: {len(atc)}  top-{size} universe.")
        hedged = {}
        for ric in local.columns:
            if ric not in atc:
                continue
            cur = atc[ric]
            # R_fh = R_loc + f_t + R_loc * f_t
            s = (local[ric] if cur == "USD" or cur not in spot.columns
                 else local[ric] + premium[cur] + local[ric] * premium[cur]).dropna()
            hedged[ric] = s[s.abs() <= config.RETURN_FILTER_STATS]

        asset_stats = pd.DataFrame(columns=["Asset_Return", "Asset_Volatility"])
        for cur in set(atc.values()):
            series = [hedged[r] for r, c in atc.items() if c == cur and r in hedged]
            stats = [annualized(s, len(s)) for s in series if len(s) >= config.TRADING_DAYS / 5]
            if stats:
                cagr, vol = zip(*stats)
                asset_stats.loc[cur] = [np.mean(cagr), np.mean(vol)]
        asset_stats.index.name = "Currency"

        print(f"\nTable 1: Annualized Average Returns and Volatilities (%) — Top {size}")
        print("-" * 75)
        print(asset_stats.join(currency_stats).reindex(ORDER).round(2))
