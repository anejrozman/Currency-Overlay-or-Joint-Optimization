"""Return formulas of Ulrych & Lucescu (2026)."""

from __future__ import annotations

import pandas as pd


def compute_daily_forward_premium(
    f_raw: pd.DataFrame, tenor_days: int
) -> pd.DataFrame:
    """Raw forward premiums scaled to daily and shifted to align at t+1."""
    return (f_raw / tenor_days).shift(1)


def compute_excess_currency_returns(
    e: pd.DataFrame,
    f_daily_shifted: pd.DataFrame,
) -> pd.DataFrame:
    """Excess currency returns R_c = e_{t+1} - f_t, on inputs aligned at t+1."""
    common_dates = e.index.intersection(f_daily_shifted.index)
    common_ccy = e.columns.intersection(f_daily_shifted.columns)
    return (
        e.loc[common_dates, common_ccy]
        - f_daily_shifted.loc[common_dates, common_ccy]
    ).dropna(how="all")


def compute_fully_hedged_returns(
    R_local: pd.DataFrame,
    e: pd.DataFrame,
    f_daily_shifted: pd.DataFrame,
    asset_to_currency: dict[str, str],
) -> pd.DataFrame:
    """Fully hedged asset returns R_fh = (1 + R_local)(1 + f) - 1."""
    known_assets = [a for a in R_local.columns if a in asset_to_currency]
    all_dates = (
        R_local.index
        .intersection(f_daily_shifted.index)
        .intersection(e.index)
    )

    R_local_a = R_local.loc[all_dates, known_assets]
    f_asset = pd.DataFrame(index=all_dates, columns=known_assets, dtype=float)
    for asset in known_assets:
        ccy = asset_to_currency[asset]
        if ccy in f_daily_shifted.columns:
            f_asset[asset] = f_daily_shifted.loc[all_dates, ccy].values

    return (R_local_a + f_asset + R_local_a * f_asset).dropna(how="all")


def compute_unhedged_returns(
    R_local: pd.DataFrame,
    e: pd.DataFrame,
    f_daily_shifted: pd.DataFrame,
    asset_to_currency: dict[str, str],
) -> pd.DataFrame:
    """Unhedged asset returns R_u = (1 + R_local)(1 + e) - 1."""
    known_assets = [a for a in R_local.columns if a in asset_to_currency]
    all_dates = (
        R_local.index
        .intersection(e.index)
        .intersection(f_daily_shifted.index)
    )

    R_local_a = R_local.loc[all_dates, known_assets]
    e_asset = pd.DataFrame(index=all_dates, columns=known_assets, dtype=float)

    for asset in known_assets:
        ccy = asset_to_currency[asset]
        if ccy in e.columns:
            e_asset[asset] = e.loc[all_dates, ccy].values

    return (R_local_a + e_asset + R_local_a * e_asset).dropna(how="all")
