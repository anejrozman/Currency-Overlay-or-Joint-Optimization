"""CSV loading, run directories and the asset-to-currency mapping."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(path: Path | str, *, date_col: str = "date") -> pd.DataFrame:
    """Load a CSV indexed by its date column, with angle brackets stripped from the column names."""
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.columns = (
        df.columns
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )
    return df


def next_run_dir(optimizer_cls: type) -> Path:
    """Claim the next free results/runs/<optimizer file>/run_<N> directory."""
    from portfolio_backtest.config import RUNS_DIR

    optimizer_stem = Path(inspect.getfile(optimizer_cls)).stem
    optimizer_dir = RUNS_DIR / optimizer_stem
    optimizer_dir.mkdir(parents=True, exist_ok=True)

    prefix = "run_"
    existing = [
        int(d.name[len(prefix):])
        for d in optimizer_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and d.name[len(prefix):].isdigit()
    ]
    # mkdir is the lock: concurrent jobs never claim the same number
    run_number = max(existing, default=0) + 1
    while True:
        run_dir = optimizer_dir / f"{prefix}{run_number}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run_number += 1
    logger.info("Run directory: %s", run_dir)
    return run_dir


def build_asset_to_currency(universe_path: Path | str) -> dict[str, str]:
    """RIC -> currency from a universe file, the latest assignment of each RIC winning."""
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
