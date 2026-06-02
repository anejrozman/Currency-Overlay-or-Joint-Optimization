"""
Backtest configuration: per-strategy defaults for window, validation period,
rebalancing frequency, and hyperparameter grids.

Override individual fields at runtime via CLI flags (--window-years, --val-years). #TODO: Consider adding a --rebalancing-frequency flag as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    window_years: int = 10
    val_years: int = 1
    rebalancing_frequency: str = "quarterly"
    param_grid: dict = field(default_factory=dict)


ES_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          [0.80, 0.95],
        "lambda_l1":      [0.0, 0.02],
        "lambda_l2":      [0.0, 0.02],
        "leverage_limit": [1, 1.5, 2.0],
    },
)

EW_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={},
)

OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          [0.80, 0.95],
        "leverage_limit": [1.0, 1.5, 2.0],
    },
)

REG_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":           [0.80, 0.95],
        "lambda_l1_asset": [0.0, 0.02],
        "lambda_l2_asset": [0.0, 0.02],
        "lambda_l1_curr":  [0.0, 0.02],
        "lambda_l2_curr":  [0.0, 0.02],
        "leverage_limit":  [1.0, 1.5, 2.0],
    },
)

JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          [0.80, 0.95],
        "leverage_limit": [1.0, 1.5, 2.0],
    },
)

REG_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          [0.80, 0.95],
        "lambda_l1":      [0.0, 0.02],
        "lambda_l2":      [0.0, 0.02],
        "leverage_limit": [1.0, 1.5, 2.0],
    },
)

REG_MV_ASSET_FULL_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          [1.0, 3.0, 5.0],
        "lambda_l1":      [0.1, 0.3, 0.5],
        "lambda_l2":      [0.1, 0.3, 0.5],
        "leverage_limit": [1.0, 1.5, 2.0],
    },
)

REG_MV_ASSET_ZERO_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          [1.0, 3.0, 5.0],
        "lambda_l1":      [0.1, 0.3, 0.5],
        "lambda_l2":      [0.1, 0.3, 0.5],
        "leverage_limit": [1.0, 1.5, 2.0],
    },
)

MV_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          [1.0, 3.0, 5.0],
        "leverage_limit": [1.0, 1.5, 2.0],
    },
)

REG_MV_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":            [1.0, 3.0, 5.0],
        "lambda_l1_asset":  [0.1, 0.3, 0.5],
        "lambda_l2_asset":  [0.1, 0.3, 0.5],
        "lambda_l1_curr":   [0.1, 0.3, 0.5],
        "lambda_l2_curr":   [0.1, 0.3, 0.5],
        "leverage_limit":   [1.0, 1.5, 2.0],
    },
)
