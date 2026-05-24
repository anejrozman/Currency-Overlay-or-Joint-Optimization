"""
Return computation utilities.

All formulas follow Ulrych & Lucescu (2026) notation:

    R_fh = R_local + f_daily + R_local * f_daily   (fully hedged)
    R_u  = R_local + e       + R_local * e          (unhedged)
    R_c  = e - f_daily                              (excess currency)

where:
    e_t      = (S_{t+1} / S_t) - 1   spot FX return (observed at t+1)
    f_t      = (F_t - S_t) / S_t     raw forward premium (observed at t)
    f_daily  = f_t / tenor_days       daily-scaled forward premium

The helper functions were previously duplicated across ``assets_full.py``,
``assets_zero.py``, ``base_script_2.py``, and ``return_statistics.py``.
"""

from __future__ import annotations

import pandas as pd


def compute_daily_forward_premium(
    f_raw: pd.DataFrame, tenor_days: int
) -> pd.DataFrame:
    """
    Convert a raw tenor-period forward premium to a daily forward premium,
    shifted by one period so that *f_t* aligns with the return at *t+1*.

    The raw forward premium ``f_t = (F_t - S_t) / S_t`` is observed at date
    *t*.  Dividing by ``tenor_days`` gives the daily equivalent, and shifting
    by one row aligns it with the spot return ``e_{t+1}`` so that the
    excess-currency and fully-hedged formulas apply at the same row index.

    Args:
        f_raw: DataFrame of raw (un-scaled) forward premiums observed at *t*.
        tenor_days: Number of trading days in the forward tenor
                    (63 for 3-month, 21 for 1-month).

    Returns:
        DataFrame of daily forward premiums indexed at *t+1*.
    """
    return (f_raw / tenor_days).shift(1)


def compute_excess_currency_returns(
    e: pd.DataFrame,
    f_daily_shifted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute excess currency returns: *R_c = e_{t+1} - f_t*.

    Both inputs must already share a common time index aligned at *t+1*
    (i.e. ``f_daily_shifted`` has been produced by
    :func:`compute_daily_forward_premium`).

    Args:
        e: Spot currency returns indexed at *t+1*, currencies as columns.
        f_daily_shifted: Daily forward premium already shifted to align
                         at *t+1*, currencies as columns.

    Returns:
        DataFrame of excess currency returns on the intersection of dates
        and currencies, with NaN-only rows dropped.
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
    Compute fully hedged asset returns: *R_fh = R_local + f_daily + R_local * f_daily*.

    Each asset's return is augmented with the forward premium of its home
    currency, mapped via ``asset_to_currency``.  The cross-term
    ``R_local * f_daily`` captures the interaction between the local equity
    return and the hedge.

    Args:
        R_local: Local equity returns, assets as columns.
        e: Spot currency returns indexed at *t+1*, currencies as columns.
           Used only to determine the common date range.
        f_daily_shifted: Daily forward premiums shifted to align at *t+1*,
                         currencies as columns.
        asset_to_currency: Mapping from asset RIC to ISO currency string.

    Returns:
        DataFrame of fully hedged asset returns, NaN-only rows dropped.
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
    Compute unhedged asset returns: *R_u = R_local + e + R_local * e*.

    The investor bears the full currency risk and holds no forward positions.
    Each asset's spot FX return is mapped via ``asset_to_currency``.

    Args:
        R_local: Local equity returns, assets as columns.
        e: Spot currency returns indexed at *t+1*, currencies as columns.
        asset_to_currency: Mapping from asset RIC to ISO currency string.

    Returns:
        DataFrame of unhedged asset returns, NaN-only rows dropped.
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
