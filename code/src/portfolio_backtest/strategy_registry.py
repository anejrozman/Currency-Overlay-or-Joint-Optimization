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
    MAD_JOINT_CONFIG,
    MAD_OVERLAY_CONFIG,
    MV_JOINT_CONFIG,
    MV_OVERLAY_CONFIG,
    OVERLAY_CONFIG,
    REG_JOINT_CONFIG,
    REG_MAD_ASSET_FULL_CONFIG,
    REG_MAD_JOINT_CONFIG,
    REG_MAD_OVERLAY_CONFIG,
    REG_MAD_ASSET_ZERO_CONFIG,
    REG_MV_ASSET_FULL_CONFIG,
    REG_MV_ASSET_ZERO_CONFIG,
    REG_MV_JOINT_CONFIG,
    REG_MV_OVERLAY_CONFIG,
    REG_OVERLAY_CONFIG,
    REG_SMV_ASSET_FULL_CONFIG,
    REG_SMV_ASSET_ZERO_CONFIG,
    REG_SMV_JOINT_CONFIG,
    REG_SMV_OVERLAY_CONFIG,
    REG_SOES_ASSET_FULL_CONFIG,
    REG_SOES_ASSET_ZERO_CONFIG,
    REG_SOES_JOINT_CONFIG,
    REG_SOES_OVERLAY_CONFIG,
    SOES_JOINT_CONFIG,
    SOES_OVERLAY_CONFIG,
    SMV_JOINT_CONFIG,
    SMV_OVERLAY_CONFIG,
)
from portfolio_backtest.optimizers import (
    EqualWeightOptimizer,
    ExpectedShortfallCurrencyOverlayOptimizer,
    ExpectedShortfallJointOptimizer,
    MeanAbsoluteDeviationCurrencyOverlayOptimizer,
    MeanAbsoluteDeviationJointOptimizer,
    MeanVarianceCurrencyOverlayOptimizer,
    MeanVarianceJointOptimizer,
    RegularizedExpectedShortfallAssetFullOptimizer,
    RegularizedExpectedShortfallAssetZeroOptimizer,
    RegularizedExpectedShortfallCurrencyOverlayOptimizer,
    RegularizedExpectedShortfallJointOptimizer,
    RegularizedMeanAbsoluteDeviationAssetFullOptimizer,
    RegularizedMeanAbsoluteDeviationAssetZeroOptimizer,
    RegularizedMeanAbsoluteDeviationCurrencyOverlayOptimizer,
    RegularizedMeanAbsoluteDeviationJointOptimizer,
    RegularizedMeanVarianceAssetFullOptimizer,
    RegularizedMeanVarianceAssetZeroOptimizer,
    RegularizedMeanVarianceCurrencyOverlayOptimizer,
    RegularizedMeanVarianceJointOptimizer,
    RegularizedSecondOrderExpectedShortfallAssetFullOptimizer,
    RegularizedSecondOrderExpectedShortfallAssetZeroOptimizer,
    RegularizedSecondOrderExpectedShortfallCurrencyOverlayOptimizer,
    RegularizedSecondOrderExpectedShortfallJointOptimizer,
    SecondOrderExpectedShortfallCurrencyOverlayOptimizer,
    SecondOrderExpectedShortfallJointOptimizer,
    RegularizedSemiVarianceAssetFullOptimizer,
    RegularizedSemiVarianceAssetZeroOptimizer,
    RegularizedSemiVarianceCurrencyOverlayOptimizer,
    SemiVarianceCurrencyOverlayOptimizer,
    RegularizedSemiVarianceJointOptimizer,
    SemiVarianceJointOptimizer,
)

STRATEGY_REGISTRY: dict[str, dict] = {
    "reg_es_asset_full": {
        "optimizer": RegularizedExpectedShortfallAssetFullOptimizer,
        "asset_return_key": "R_fh",
        "label": "ASSETS-FULL ES",
        "use_hedging": True,
        "config": ES_CONFIG,
    },
    "reg_es_asset_zero": {
        "optimizer": RegularizedExpectedShortfallAssetZeroOptimizer,
        "asset_return_key": "R_u",
        # Paper definition (Lucescu & Ulrych 2026, Sec. 4.2): weights are
        # optimized on fully hedged returns (same asset problem as
        # ASSETS-FULL) and realized unhedged.
        "training_return_key": "R_fh",
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
        "label": "1/N Equal Weight",
        "use_hedging": True,
        "config": EW_CONFIG,
    },
    "reg_mad_asset_full": {
        "optimizer":        RegularizedMeanAbsoluteDeviationAssetFullOptimizer,
        "asset_return_key": "R_fh",
        "label":            "ASSETS-FULL MAD",
        "use_hedging":      True,
        "config":           REG_MAD_ASSET_FULL_CONFIG,
    },
    "reg_mad_asset_zero": {
        "optimizer":        RegularizedMeanAbsoluteDeviationAssetZeroOptimizer,
        "asset_return_key": "R_u",
        # Paper definition (Lucescu & Ulrych 2026, Sec. 4.2): weights are
        # optimized on fully hedged returns (same asset problem as
        # ASSETS-FULL) and realized unhedged.
        "training_return_key": "R_fh",
        "label":            "ASSETS-ZERO MAD",
        "use_hedging":      False,
        "config":           REG_MAD_ASSET_ZERO_CONFIG,
    },
    "mad_curr_overlay": {
        "optimizer":        MeanAbsoluteDeviationCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label":            "MAD Currency Overlay",
        "use_hedging":      True,
        "config":           MAD_OVERLAY_CONFIG,
    },
    "reg_mad_curr_overlay": {
        "optimizer":        RegularizedMeanAbsoluteDeviationCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label":            "Regularized MAD Currency Overlay",
        "use_hedging":      True,
        "config":           REG_MAD_OVERLAY_CONFIG,
    },
    "mad_joint": {
        "optimizer":        MeanAbsoluteDeviationJointOptimizer,
        "asset_return_key": "R_fh",
        "label":            "MAD Joint",
        "use_hedging":      True,
        "config":           MAD_JOINT_CONFIG,
    },
    "reg_mad_joint": {
        "optimizer":        RegularizedMeanAbsoluteDeviationJointOptimizer,
        "asset_return_key": "R_fh",
        "label":            "Regularized MAD Joint",
        "use_hedging":      True,
        "config":           REG_MAD_JOINT_CONFIG,
    },
    "reg_mv_asset_full": {
        "optimizer":        RegularizedMeanVarianceAssetFullOptimizer,
        "asset_return_key": "R_fh",
        "label":            "ASSETS-FULL MV",
        "use_hedging":      True,
        "config":           REG_MV_ASSET_FULL_CONFIG,
    },
    "reg_mv_asset_zero": {
        "optimizer":        RegularizedMeanVarianceAssetZeroOptimizer,
        "asset_return_key": "R_u",
        # Paper definition (Lucescu & Ulrych 2026, Sec. 4.2): weights are
        # optimized on fully hedged returns (same asset problem as
        # ASSETS-FULL) and realized unhedged.
        "training_return_key": "R_fh",
        "label":            "ASSETS-ZERO MV",
        "use_hedging":      False,
        "config":           REG_MV_ASSET_ZERO_CONFIG,
    },
    "mv_curr_overlay": {
        "optimizer":        MeanVarianceCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label":            "MV Currency Overlay",
        "use_hedging":      True,
        "config":           MV_OVERLAY_CONFIG,
    },
    "reg_mv_curr_overlay": {
        "optimizer":        RegularizedMeanVarianceCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label":            "Regularized MV Currency Overlay",
        "use_hedging":      True,
        "config":           REG_MV_OVERLAY_CONFIG,
    },
    "mv_joint": {
        "optimizer":        MeanVarianceJointOptimizer,
        "asset_return_key": "R_fh",
        "label":            "MV Joint",
        "use_hedging":      True,
        "config":           MV_JOINT_CONFIG,
    },
    "reg_mv_joint": {
        "optimizer":        RegularizedMeanVarianceJointOptimizer,
        "asset_return_key": "R_fh",
        "label":            "Regularized MV Joint",
        "use_hedging":      True,
        "config":           REG_MV_JOINT_CONFIG,
    },
    "reg_smv_asset_full": {
        "optimizer":        RegularizedSemiVarianceAssetFullOptimizer,
        "asset_return_key": "R_fh",
        "label":            "ASSETS-FULL SMV",
        "use_hedging":      True,
        "config":           REG_SMV_ASSET_FULL_CONFIG,
    },
    "reg_smv_asset_zero": {
        "optimizer":        RegularizedSemiVarianceAssetZeroOptimizer,
        "asset_return_key": "R_u",
        # Paper definition (Lucescu & Ulrych 2026, Sec. 4.2): weights are
        # optimized on fully hedged returns (same asset problem as
        # ASSETS-FULL) and realized unhedged.
        "training_return_key": "R_fh",
        "label":            "ASSETS-ZERO SMV",
        "use_hedging":      False,
        "config":           REG_SMV_ASSET_ZERO_CONFIG,
    },
    "smv_curr_overlay": {
        "optimizer":        SemiVarianceCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label":            "SMV Currency Overlay",
        "use_hedging":      True,
        "config":           SMV_OVERLAY_CONFIG,
    },
    "reg_smv_curr_overlay": {
        "optimizer":        RegularizedSemiVarianceCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label":            "Regularized SMV Currency Overlay",
        "use_hedging":      True,
        "config":           REG_SMV_OVERLAY_CONFIG,
    },
    "smv_joint": {
        "optimizer":        SemiVarianceJointOptimizer,
        "asset_return_key": "R_fh",
        "label":            "SMV Joint",
        "use_hedging":      True,
        "config":           SMV_JOINT_CONFIG,
    },
    "reg_smv_joint": {
        "optimizer":        RegularizedSemiVarianceJointOptimizer,
        "asset_return_key": "R_fh",
        "label":            "Regularized SMV Joint",
        "use_hedging":      True,
        "config":           REG_SMV_JOINT_CONFIG,
    },
    "reg_soes_asset_full": {
        "optimizer":        RegularizedSecondOrderExpectedShortfallAssetFullOptimizer,
        "asset_return_key": "R_fh",
        "label":            "ASSETS-FULL SOES",
        "use_hedging":      True,
        "config":           REG_SOES_ASSET_FULL_CONFIG,
    },
    "reg_soes_asset_zero": {
        "optimizer":        RegularizedSecondOrderExpectedShortfallAssetZeroOptimizer,
        "asset_return_key": "R_u",
        # Paper definition (Lucescu & Ulrych 2026, Sec. 4.2): weights are
        # optimized on fully hedged returns (same asset problem as
        # ASSETS-FULL) and realized unhedged.
        "training_return_key": "R_fh",
        "label":            "ASSETS-ZERO SOES",
        "use_hedging":      False,
        "config":           REG_SOES_ASSET_ZERO_CONFIG,
    },
    "soes_curr_overlay": {
        "optimizer":        SecondOrderExpectedShortfallCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label":            "SOES Currency Overlay",
        "use_hedging":      True,
        "config":           SOES_OVERLAY_CONFIG,
    },
    "reg_soes_curr_overlay": {
        "optimizer":        RegularizedSecondOrderExpectedShortfallCurrencyOverlayOptimizer,
        "asset_return_key": "R_fh",
        "label":            "Regularized SOES Currency Overlay",
        "use_hedging":      True,
        "config":           REG_SOES_OVERLAY_CONFIG,
    },
    "soes_joint": {
        "optimizer":        SecondOrderExpectedShortfallJointOptimizer,
        "asset_return_key": "R_fh",
        "label":            "SOES Joint",
        "use_hedging":      True,
        "config":           SOES_JOINT_CONFIG,
    },
    "reg_soes_joint": {
        "optimizer":        RegularizedSecondOrderExpectedShortfallJointOptimizer,
        "asset_return_key": "R_fh",
        "label":            "Regularized SOES Joint",
        "use_hedging":      True,
        "config":           REG_SOES_JOINT_CONFIG,
    },
}
