"""
Constants and path configuration for the portfolio_backtest package.
"""

from __future__ import annotations

from pathlib import Path

# Repository paths
REPO_ROOT: Path = Path(__file__).resolve().parents[3]  # .../master_thesis
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = REPO_ROOT / "results"
RUNS_DIR: Path = RESULTS_DIR / "runs"
LOGS_DIR: Path = REPO_ROOT / "logs"

# Universe filenames — keyed by top-N per economy
UNIVERSE_FILE_10: str = "currency_alocation_10_202603081747.csv"
UNIVERSE_FILE_50: str = "currency_alocation_50_202603081747.csv"
UNIVERSE_FILE_100: str = "currency_alocation_100_202603081747.csv"
UNIVERSE_FILES: dict[int, str] = {10: UNIVERSE_FILE_10, 50: UNIVERSE_FILE_50, 100: UNIVERSE_FILE_100}
RF_FILE: str = "us_risk_free_rates.csv"
LOCAL_RETURNS_FILE: str = "local_asset_returns.csv"
SPOT_RETURNS_FILE: str = "currency_spot_returns.csv"
FWD_PREMIUMS_3M_FILE: str = "raw_forward_premiums_3m.csv"
FWD_PREMIUMS_1M_FILE: str = "raw_forward_premiums_1m.csv"
FH_RETURNS_FILE: str = "fullyhedgedassetreturns_3m.csv"
RC_RETURNS_FILE: str = "excesscurrencyreturns_3m.csv"
UNHEDGED_RETURNS_FILE: str = "unhedgedassetreturns.csv"

# Data processing constants 
RETURN_FILTER_BACKTEST: float = 1.0
RETURN_FILTER_STATS: float = 2.0
TENOR_DAYS_3M: int = 63
TENOR_DAYS_1M: int = 21
TENOR_DAYS_1Y: int = 252
TENOR_DAYS_1W: int = 5
TRADING_DAYS: int = 252 

# Transaction cost assumptions (Appendix C of thesis)
ASSET_COST_BPS: float = 10.0  # basis points per unit traded for equities
FX_COST_BPS: float = 1.0      # basis points per unit traded for currencies
