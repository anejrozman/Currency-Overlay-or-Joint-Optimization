"""
Return computation formulas implemented by following the paper Regularized Multi-Currency Expected Shortfall Portfolios by Ulrych & Lucescu (2026).

"""

from __future__ import annotations

import pandas as pd


def compute_daily_forward_premium(
    f_raw: pd.DataFrame, tenor_days: int
) -> pd.DataFrame:
    """
    Scale raw forward points to daily forward premiums and shift to align at time t+1
    """
    return (f_raw / tenor_days).shift(1)


def compute_excess_currency_returns(
    e: pd.DataFrame,
    f_daily_shifted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute excess currency returns R_c = e_{t+1} - f_t
    Both inputs must already share a common time index aligned at time t+1
    """
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
    """
    Compute fully hedged asset returns R_fh = R_local + f_daily + R_local * f_daily. # TODO: inspect formula
    """
    known_assets = [a for a in R_local.columns if a in asset_to_currency]
    all_dates = (
        R_local.index
        .intersection(f_daily_shifted.index)
        .intersection(e.index)
    )

    R_local_a = R_local.loc[all_dates, known_assets]
    f_asset = pd.DataFrame(index=all_dates, columns=known_assets, dtype=float)
    e_asset = pd.DataFrame(index=all_dates, columns=known_assets, dtype=float)

    for asset in known_assets:
        ccy = asset_to_currency[asset]
        if ccy in f_daily_shifted.columns:
            f_asset[asset] = f_daily_shifted.loc[all_dates, ccy].values
        if ccy in e.columns:
            e_asset[asset] = e.loc[all_dates, ccy].values

    return (R_local_a + f_asset + R_local_a * f_asset).dropna(how="all")


def compute_unhedged_returns(
    R_local: pd.DataFrame,
    e: pd.DataFrame,
    asset_to_currency: dict[str, str],
) -> pd.DataFrame:
    """
    Compute unhedged asset returns R_u = R_local + e + R_local * e
    """
    known_assets = [a for a in R_local.columns if a in asset_to_currency]
    all_dates = R_local.index.intersection(e.index)

    R_local_a = R_local.loc[all_dates, known_assets]
    e_asset = pd.DataFrame(index=all_dates, columns=known_assets, dtype=float)

    for asset in known_assets:
        ccy = asset_to_currency[asset]
        if ccy in e.columns:
            e_asset[asset] = e.loc[all_dates, ccy].values

    return (R_local_a + e_asset + R_local_a * e_asset).dropna(how="all")
