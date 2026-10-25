"""Datastream exports in long format to wide interim CSVs.

Usage:
    python -m portfolio_backtest.data_pipeline.format_raw_data
"""

from __future__ import annotations

from datetime import datetime
import polars as pl
from portfolio_backtest.config import INTERIM_DIR, RAW_DIR


def main() -> None:
    # Stock Prices 
    lf_stocks = pl.scan_csv(RAW_DIR / "stock_prices_202603081751.csv")

    df_stocks_wide = (
        lf_stocks
        .select(["date", "ric_code", "price"])
        .collect()
        .pivot(values="price", index="date", on="ric_code", aggregate_function="first")
        .with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False))
        .drop_nulls(subset=["date"])
        .filter(
            (pl.col("date") >= datetime(1990, 6, 1)) &
            (pl.col("date") <= datetime(2023, 6, 30))
        )
        .sort("date")
    )
    df_stocks_wide.write_csv(INTERIM_DIR / "stockprices.csv")

    # Stock Returns
    lf_returns = pl.scan_csv(RAW_DIR / "returns_local_202603081748.csv")

    df_returns_wide = (
        lf_returns
        .select(["date", "ric_code", "returns"])
        .collect()
        .pivot(values="returns", index="date", on="ric_code", aggregate_function="first")
        .with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False))
        .drop_nulls(subset=["date"])
        .filter(
            (pl.col("date") >= datetime(1990, 6, 1)) &
            (pl.col("date") <= datetime(2023, 6, 30))
        )
        .sort("date")
    )
    df_returns_wide.write_csv(INTERIM_DIR / "stockreturns.csv")

    # Currency Spot Rates
    df_spot = (
        pl.scan_csv(RAW_DIR / "spot_202603081751.csv")
        .with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False))
        .drop_nulls(subset=["date"])
        .filter(
            (pl.col("date") >= datetime(1990, 6, 1)) &
            (pl.col("date") <= datetime(2023, 6, 30))
        )
        .sort("date")
        .collect()
    )

    # usd_gbp -> GBP etc., with USD = 1
    rename_mapping = {col: col.split("_")[1].upper() for col in df_spot.columns if col.startswith("usd_")}
    df_spot = df_spot.rename(rename_mapping)
    df_spot = df_spot.with_columns(pl.lit(1.0).alias("USD"))
    df_spot.write_csv(INTERIM_DIR / "exchangerates.csv")

    # Currency Forward Rates
    forward_files = {
        "3m": "fwd_3m_202603081748.csv",
        "1m": "fwd_1m_202603081747.csv",
        "6m": "fwd_6m_202603081748.csv",
    }

    for tenor, filename in forward_files.items():
        df_fwd = (
            pl.scan_csv(RAW_DIR / filename)
            .with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False))
            .drop_nulls(subset=["date"])
            .filter(
                (pl.col("date") >= datetime(1990, 6, 1)) &
                (pl.col("date") <= datetime(2023, 6, 30))
            )
            .sort("date")
            .collect()
        )
        fwd_mapping = {col: col.split("_")[1].upper() for col in df_fwd.columns if col.startswith("usd_")}
        df_fwd = df_fwd.rename(fwd_mapping)
        df_fwd = df_fwd.with_columns(pl.lit(1.0).alias("USD"))
        df_fwd.write_csv(INTERIM_DIR / f"forwardrates_{tenor}.csv")

    # RIC to Currency Mapping
    df_ric = pl.read_csv(RAW_DIR / "ric_currency_202603081751.csv")
    df_ric = df_ric.with_columns(pl.col("currency").str.to_uppercase())
    df_ric.write_csv(INTERIM_DIR / "ric_mapping.csv")


if __name__ == "__main__":
    main()
