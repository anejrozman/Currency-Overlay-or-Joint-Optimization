"""Local asset returns, spot currency returns and raw forward premiums from the interim data.

Usage:
    python -m portfolio_backtest.data_pipeline.preprocess_data
"""

from __future__ import annotations

import polars as pl
from portfolio_backtest.config import INTERIM_DIR, PROCESSED_DIR


def main() -> None:
    df_returns = pl.read_csv(INTERIM_DIR / "stockreturns.csv")
    df_spot = pl.read_csv(INTERIM_DIR / "exchangerates.csv")
    fwd_files = {
        "1m": pl.read_csv(INTERIM_DIR / "forwardrates_1m.csv"),
        "3m": pl.read_csv(INTERIM_DIR / "forwardrates_3m.csv"),
        "6m": pl.read_csv(INTERIM_DIR / "forwardrates_6m.csv"),
    }

    date_col = "date"
    curr_cols = [c for c in df_spot.columns if c != date_col]

    # local asset returns R_{t+1}, unchanged
    df_returns.write_csv(PROCESSED_DIR / "local_asset_returns.csv")

    # spot currency returns e_{t+1} = S_{t+1} / S_t - 1
    df_e = df_spot.with_columns(
        [((pl.col(c) / pl.col(c).shift(1)) - 1).alias(c) for c in curr_cols]
    ).select([date_col] + curr_cols).slice(1)
    df_e.write_csv(PROCESSED_DIR / "currency_spot_returns.csv")

    # raw forward premiums f_t = (F_t - S_t) / S_t
    for tenor, df_fwd in fwd_files.items():
        df_f = df_spot.join(df_fwd, on=date_col, suffix="_fwd")
        df_f = df_f.with_columns(
            [((pl.col(f"{c}_fwd") - pl.col(c)) / pl.col(c)).alias(c) for c in curr_cols]
        ).select([date_col] + curr_cols)
        df_f.write_csv(PROCESSED_DIR / f"raw_forward_premiums_{tenor}.csv")


if __name__ == "__main__":
    main()
