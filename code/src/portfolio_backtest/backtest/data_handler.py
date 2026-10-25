"""Loads the universe, the asset and currency returns and the daily universe mask for the Backtester."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from portfolio_backtest.utils.io import build_asset_to_currency, load_csv


class DataHandler:
    def __init__(
        self,
        universe_file: Path | str,
        hedged_returns_file: Path | str,
        excess_currency_returns_file: Path | str,
    ) -> None:
        self.universe_file = Path(universe_file)
        self.hedged_returns_file = Path(hedged_returns_file)
        self.excess_currency_returns_file = Path(excess_currency_returns_file)
        self.asset_to_currency: dict[str, str] = {}

    def load_and_process(
        self,
    ) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame, pd.DataFrame]:
        """Load and process all data sources."""

        # Investment universe 
        self.asset_to_currency = build_asset_to_currency(self.universe_file)
        universe_df = pd.read_csv(self.universe_file)
        universe_df["clean_ric"] = (
            universe_df["ric_code"]
            .astype(str)
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
        )
        universe_df["date"] = pd.to_datetime(universe_df["date"])

        # Asset returns 
        asset_returns = load_csv(self.hedged_returns_file)

        # Restrict to assets present in the universe
        universe_assets = list(self.asset_to_currency.keys())
        available_assets = [col for col in asset_returns.columns if col in universe_assets]
        asset_returns = asset_returns[available_assets]

        # Investment universe mask 
        # Pivot to (date × RIC) with 1 where the asset is selected, then forward fill 
        universe_df["selected"] = 1
        mask = universe_df.pivot_table(
            index="date", columns="clean_ric", values="selected"
        )
        daily_mask = mask.reindex(asset_returns.index, method="ffill")

        # Excess currency returns 
        excess_currency_returns = load_csv(self.excess_currency_returns_file)

        return asset_returns, self.asset_to_currency, excess_currency_returns, daily_mask
