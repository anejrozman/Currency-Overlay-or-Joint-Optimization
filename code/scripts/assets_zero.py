"""
Description: Runs the ASSETS-ZERO strategy with RegularizedExpectedShortfallAssetZeroOptimizer.
Uses calendar-based Backtester with quarterly rebalancing. Universe: top-50 stock universe.
Portfolio is unhedged (no currency overlay).
Author: Anej Rozman
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.Backtester import Backtester
from src.backtest.DataHandler import DataHandler
from src.optimizers.RegularizedExpectedShortfallAssetZeroOptimizer import RegularizedExpectedShortfallAssetZeroOptimizer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir      = os.path.dirname(__file__)
data_dir        = os.path.abspath(os.path.join(script_dir, '..', '..', 'data'))
raw_dir         = os.path.join(data_dir, 'raw')
processed_dir   = os.path.join(data_dir, 'processed')
results_dir     = os.path.abspath(os.path.join(script_dir, '..', '..', 'results'))
plots_dir       = os.path.join(results_dir, 'plots')
runs_dir        = os.path.join(results_dir, 'runs')

universe_file       = os.path.join(raw_dir,       'currency_alocation_50_202603081747.csv')
local_returns_file  = os.path.join(processed_dir, 'local_asset_returns.csv')
spot_returns_file   = os.path.join(processed_dir, 'currency_spot_returns.csv')
ru_out_file         = os.path.join(processed_dir, 'unhedgedassetreturns.csv')


# ---------------------------------------------------------------------------
# Run directory setup (auto-incrementing: assets_zero_run_1, run_2, ...)
# ---------------------------------------------------------------------------
os.makedirs(runs_dir, exist_ok=True)
existing_nums = []
for d in os.listdir(runs_dir):
    if d.startswith('assets_zero_run_'):
        try:
            existing_nums.append(int(d.replace('assets_zero_run_', '')))
        except ValueError:
            pass
run_number = max(existing_nums, default=0) + 1
run_dir = os.path.join(runs_dir, f'assets_zero_run_{run_number}')
os.makedirs(run_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load processed data
# ---------------------------------------------------------------------------
def load_df(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # Strip angle brackets from column names if present (Refinitiv Datastream format)
    df.columns = df.columns.str.replace('<', '', regex=False).str.replace('>', '', regex=False)
    return df

R_local = load_df(local_returns_file)   # daily local equity returns
e       = load_df(spot_returns_file)    # e_{t+1}: spot currency returns at t+1

# Filter: replace single-day returns with |r| > 50% with NaN.
RETURN_FILTER = 1.0
R_local = R_local.where(R_local.abs() < RETURN_FILTER)


# ---------------------------------------------------------------------------
# Step 2: Build asset-to-currency mapping from the top-50 universe file
# ---------------------------------------------------------------------------
universe_df = pd.read_csv(universe_file)
ric_col = 'ric_code' if 'ric_code' in universe_df.columns else 'riccode'
universe_df['clean_ric'] = (
    universe_df[ric_col].astype(str)
    .str.replace('<', '', regex=False)
    .str.replace('>', '', regex=False)
)
# Use the most recent entry per RIC to get a stable mapping
sorted_univ = universe_df.sort_values('date', ascending=False)
asset_to_currency_raw = dict(zip(sorted_univ['clean_ric'], sorted_univ['currency']))


# ---------------------------------------------------------------------------
# Step 3: (No forward premiums needed — ASSETS-ZERO is fully unhedged)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 4: Compute unhedged asset returns  R^u = R_local + e + R_local * e
# ---------------------------------------------------------------------------
# Restrict R_local to assets with a known currency mapping
known_assets = [a for a in R_local.columns if a in asset_to_currency_raw]

all_dates  = R_local.index.intersection(e.index)
R_local_a  = R_local.loc[all_dates, known_assets]
e_a        = e.loc[all_dates]

# Build per-asset currency-indexed slices of e
e_asset = pd.DataFrame(index=all_dates, columns=known_assets, dtype=float)

for asset in known_assets:
    ccy = asset_to_currency_raw[asset]
    if ccy in e_a.columns:
        e_asset[asset] = e_a[ccy].values

# Unhedged return: R^u = R_local + e + R_local * e  (no forward premium term)
R_u = R_local_a + e_asset + R_local_a * e_asset
R_u.dropna(how='all', inplace=True)


# ---------------------------------------------------------------------------
# Step 5: Save computed R_u so DataHandler can read it back
# ---------------------------------------------------------------------------
R_u.index.name = 'date'
R_u.to_csv(ru_out_file)


# ---------------------------------------------------------------------------
# Step 6: Load via DataHandler and build point-in-time universe mask
#
# DataHandler is used only for the asset_to_currency mapping and universe_mask.
# We pass ru_out_file as hedged_returns_file so the mask is aligned to R_u dates.
# We also pass ru_out_file as the excess_currency_returns_file placeholder; the
# returned excess-currency DataFrame (3rd return) is discarded — it is not used
# by ASSETS-ZERO.
# ---------------------------------------------------------------------------
handler = DataHandler(
    universe_file=universe_file,
    hedged_returns_file=ru_out_file,
    excess_currency_returns_file=ru_out_file,
)
R_u_filtered, asset_to_currency, _, universe_mask = handler.load_and_process()

# universe_mask is already aligned to R_u_filtered.index by DataHandler;
# reindex defensively to match any date truncation applied during load.
universe_mask = universe_mask.reindex(R_u_filtered.index, method='ffill')


# ---------------------------------------------------------------------------
# Step 7: Run backtest
# ---------------------------------------------------------------------------
data_bundle = {
    'R_u':               R_u_filtered,
    'asset_to_currency': asset_to_currency,
    'universe_mask':     universe_mask,
}

# Sparse parameter grid with 2 options per hyperparameter
param_grid = {
    "alpha": [0.80, 0.95],
    "lambda_l1": [0.0, 0.02],
    "lambda_l2": [0.0, 0.02],
    "leverage_limit": [1, 1.5, 2.0]
}

backtester = Backtester(
    optimizer=RegularizedExpectedShortfallAssetZeroOptimizer,
    param_grid=param_grid,
    window_years=10,
    val_years=1,
    rebalancing_frequency='quarterly',
    asset_return_key='R_u',
    logging=True
)

# Save config before running so it's preserved even if the backtest fails
config = {
    "strategy": "ASSETS-ZERO",
    "run_number": run_number,
    "run_date": datetime.now().isoformat(),
    "optimizer": RegularizedExpectedShortfallAssetZeroOptimizer.__name__,
    "param_grid": param_grid,
    "window_years": 10,
    "val_years": 1,
    "rebalancing_frequency": "quarterly",
    "asset_return_key": "R_u",
    "universe_file": os.path.basename(universe_file),
    "return_filter": RETURN_FILTER,
    "rf_annual": 0.0,
}
with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

oos_returns = backtester.run(data_bundle)


# ---------------------------------------------------------------------------
# Step 8: Save OOS returns
# ---------------------------------------------------------------------------
oos_df = oos_returns.rename('return').to_frame()
oos_df.index.name = 'date'
oos_df.to_csv(os.path.join(run_dir, 'oos_returns.csv'))


# ---------------------------------------------------------------------------
# Step 9: Performance metrics
# ---------------------------------------------------------------------------
RF_ANNUAL = 0.0
metrics = backtester.compute_performance_metrics(oos_returns, label="ASSETS-ZERO (top-50)", rf_annual=RF_ANNUAL)

with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Step 10: Plot cumulative returns
# ---------------------------------------------------------------------------
cumulative_returns = (1 + oos_returns).cumprod()

plt.figure(figsize=(10, 6))
cumulative_returns.plot(label='ASSETS-ZERO (top-50)')
plt.title('Cumulative Returns — ASSETS-ZERO Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(run_dir, 'cumulative_returns_plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
print(f"Run artifacts saved to: {run_dir}")

if __name__ == '__main__':
    pass
