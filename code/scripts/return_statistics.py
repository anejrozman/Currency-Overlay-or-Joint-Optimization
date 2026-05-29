"""

Description: Script for obtaining summary statistics as those reported in 
table 1 in Regularized Multi-Currency Expected Shortfall Portfolios by Ulrych and Lucescu (2026)

Author: Anej Rozman

"""

from __future__ import annotations

import pandas as pd
import numpy as np

from portfolio_backtest import config
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import compute_daily_forward_premium


def main() -> None:

    # File paths
    universe_file = config.RAW_DIR / config.UNIVERSE_FILE
    local_asset_returns_file = config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE
    currency_returns_file = config.PROCESSED_DIR / config.SPOT_RETURNS_FILE
    forward_premium_file = config.PROCESSED_DIR / config.FWD_PREMIUMS_1M_FILE

    # Load data
    local_asset_returns = load_csv(local_asset_returns_file)
    currency_returns = load_csv(currency_returns_file)
    forward_premiums = load_csv(forward_premium_file)

    # Investable universe and asset to currency mapping
    # Use the most recent currency assignment per RIC across all rebalancing dates
    asset_to_currency = build_asset_to_currency(universe_file)

    # Daily Excess Currency Returns
    daily_fwd_premium = compute_daily_forward_premium(forward_premiums, config.TENOR_DAYS_1M)

    # Excess Currency Return  e_{t+1} - f_t
    excess_currency_returns = currency_returns - daily_fwd_premium
    excess_currency_returns = excess_currency_returns.dropna(how='all')
    excess_currency_returns = excess_currency_returns.drop(columns=['USD'])


    # Daily Fully Hedged Asset Returns
    RETURN_FILTER = config.RETURN_FILTER_STATS
    universe_rics = set(asset_to_currency.keys())
    fully_hedged_returns = {}
    for ric in local_asset_returns.columns:
        if ric not in universe_rics:
            continue
        cur = asset_to_currency.get(ric, 'USD')
        if cur == 'USD' or cur not in currency_returns.columns:
            s = local_asset_returns[ric].dropna()
        else:
            # R_fh = (1 + R_loc)(1 + f_t) - 1 = R_loc + f_t + R_loc * f_t
            s = (
                local_asset_returns[ric] +
                daily_fwd_premium[cur] +
                (local_asset_returns[ric] * daily_fwd_premium[cur])
            ).dropna()
        fully_hedged_returns[ric] = s[s.abs() <= RETURN_FILTER]

    # Average the annualized metrics across the stocks in each economy
    economy_asset_stats = pd.DataFrame(columns=['Asset_Return', 'Asset_Volatility'])
    for cur in set(asset_to_currency.values()):
        cur_rics = [r for r, c in asset_to_currency.items() if c == cur and r in fully_hedged_returns]
        if not cur_rics:
            continue
        cagr_list, vol_list = [], []
        for r in cur_rics:
            s = fully_hedged_returns[r]
            n = len(s)
            if n < 10:
                continue
            cagr_list.append(((1 + s).prod() ** (config.TRADING_DAYS / n) - 1) * 100)
            vol_list.append(s.std() * np.sqrt(config.TRADING_DAYS) * 100)
        if cagr_list:
            economy_asset_stats.loc[cur] = [np.mean(cagr_list), np.mean(vol_list)]

    economy_asset_stats.index.name = 'Currency'
    
    # Currency Statistics
    # Annualized return and volatility for the excess currency returns
    currency_count = excess_currency_returns.count()
    currency_ann_ret = (((1 + excess_currency_returns).prod() ** (config.TRADING_DAYS / currency_count)) - 1) * 100
    currency_ann_vol = excess_currency_returns.std() * np.sqrt(config.TRADING_DAYS) * 100
    
    currency_stats = pd.DataFrame({
        'Currency_Excess_Return': currency_ann_ret,
        'Currency_Excess_Volatility': currency_ann_vol
    })
    
    # USD is the base currency so we set its excess return and volatility to 0
    currency_stats.loc['USD'] = {'Currency_Excess_Return': 0.0, 'Currency_Excess_Volatility': 0.0}
    
    # Summary Table Print
    summary_table = economy_asset_stats.join(currency_stats)
    order = ['AUD', 'CAD', 'DKK', 'EUR', 'HKD', 'JPY', 'NZD', 'NOK', 'SGD', 'ZAR', 'SEK', 'CHF', 'GBP', 'USD']
    summary_table = summary_table.reindex(order)
    summary_table = summary_table.round(2)
    
    print("\nTable 1: Annualized Average Returns and Volatilities (%)")
    print("-" * 75)
    print(summary_table)
    

if __name__ == "__main__":
    main()
