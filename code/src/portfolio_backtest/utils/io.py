"""
I/O utilities: CSV loading that strips angle brackets from column names,
auto incrementing run directory function, and RIC to currency mapping function.

"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(path: Path | str, *, date_col: str = "date") -> pd.DataFrame:
    """
    Load a CSV, parse the date column as the index, and strip
    angle brackets from column names.
    """
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.columns = (
        df.columns
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )
    return df


def next_run_dir(strategy_name: str) -> Path:
    """
    Return a new incremented +1 run directory for *strategy_name* when a backtest is run.
    """
    from portfolio_backtest.config import RUNS_DIR

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{strategy_name}_run_"
    existing = [
        int(d.name.replace(prefix, ""))
        for d in RUNS_DIR.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and d.name[len(prefix):].isdigit()
    ]
    run_number = max(existing, default=0) + 1
    run_dir = RUNS_DIR / f"{prefix}{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)
    return run_dir


def build_asset_to_currency(universe_path: Path | str) -> dict[str, str]:
    """
    Build a RIC to currency mapping from the universe CSV. Angle brackets are stripped from RIC codes.
    """
    universe_df = pd.read_csv(universe_path)
    ric_col = "ric_code" if "ric_code" in universe_df.columns else "riccode"
    universe_df["clean_ric"] = (
        universe_df[ric_col]
        .astype(str)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )
    sorted_univ = universe_df.sort_values("date", ascending=False)
    return dict(zip(sorted_univ["clean_ric"], sorted_univ["currency"]))
