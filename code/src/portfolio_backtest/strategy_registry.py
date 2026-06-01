"""
Strategy registry for the portfolio backtest framework.

Add an entry here for each new strategy/optimizer. The run_strategy.py script
reads this registry to determine which optimizer to use, which return series
to load, and what the default backtest configuration is.

Each entry must have:
    optimizer       — optimizer class (subclass of BaseOptimizer)
    asset_return_key — "R_fh" (fully hedged) or "R_u" (unhedged)
    label           — short human readable name to use in plots and tables
    use_hedging     — whether to compute and pass R_fh / R_c (vs R_u only)
    config          — BacktestConfig instance with defaults for this strategy
"""

from __future__ import annotations

from portfolio_backtest.backtest_config import (
    ES_CONFIG,
    EW_CONFIG,
    JOINT_CONFIG,
    OVERLAY_CONFIG,
    REG_JOINT_CONFIG,
    REG_MV_ASSET_FULL_CONFIG,
    REG_OVERLAY_CONFIG,
)
from portfolio_backtest.optimizers import (
    EqualWeightOptimizer,
    ExpectedShortfallCurrencyOverlayOptimizer,
    ExpectedShortfallJointOptimizer,
    RegularizedExpectedShortfallAssetFullOptimizer,
    RegularizedExpectedShortfallAssetZeroOptimizer,
    RegularizedExpectedShortfallCurrencyOverlayOptimizer,
    RegularizedExpectedShortfallJointOptimizer,
    RegularizedMeanVarianceAssetFullOptimizer,
)

STRATEGY_REGISTRY: dict[str, dict] = {
    "assets_full": {
        "optimizer": RegularizedExpectedShortfallAssetFullOptimizer,
        "asset_return_key": "R_fh",
        "label": "ASSETS-FULL ES (3m)",
        "use_hedging": True,
        "config": ES_CONFIG,
    },
    "assets_zero": {
        "optimizer": RegularizedExpectedShortfallAssetZeroOptimizer,
        "asset_return_key": "R_u",
        "label": "ASSETS-ZERO ES",
        "use_hedging": False,
        "config": ES_CONFIG,
    },
    "es_curr_overlay": {
        "optimizer": ExpectedShortfallCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label": "ES Currency Overlay",
        "use_hedging": True,
        "config": OVERLAY_CONFIG,
    },
    "reg_es_curr_overlay": {
        "optimizer": RegularizedExpectedShortfallCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label": "Regularized ES Currency Overlay",
        "use_hedging": True,
        "config": REG_OVERLAY_CONFIG,
    },
    "es_joint": {
        "optimizer": ExpectedShortfallJointOptimizer,
        "asset_return_key": "R_fh",
        "label": "ES Joint",
        "use_hedging": True,
        "config": JOINT_CONFIG,
    },
    "reg_es_joint": {
        "optimizer": RegularizedExpectedShortfallJointOptimizer,
        "asset_return_key": "R_fh",
        "label": "Regularized ES Joint",
        "use_hedging": True,
        "config": REG_JOINT_CONFIG,
    },
    "equally_weighted": {
        "optimizer": EqualWeightOptimizer,
        "asset_return_key": "R_fh",
        "label": "1/N Equal Weight (3m)",
        "use_hedging": True,
        "config": EW_CONFIG,
    },
    "reg_mv_asset_full": {
        "optimizer":        RegularizedMeanVarianceAssetFullOptimizer,
        "asset_return_key": "R_fh",
        "label":            "Regularized MV (Assets-Full)",
        "use_hedging":      True,
        "config":           REG_MV_ASSET_FULL_CONFIG,
    },
}
