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


# Regularization grids
LAMBDA_L1_ASSET_GRID = [0.0, 5e-5, 5e-4, 5e-3]
LAMBDA_L2_ASSET_GRID = [0.0, 0.03, 0.3, 3.0]
LAMBDA_L1_CURR_GRID = [0.0, 5e-6, 5e-5, 5e-4]
LAMBDA_L2_CURR_GRID = [0.0, 5e-7, 5e-6, 5e-5]
# L2 on the overlay shrinks as a power law rather than to a corner, so the bound is
# set where the overlay is economically spent (~2% of unpenalized gross psi) instead
# of where the weights stop moving. The old 5e-5 top was selected by the validation
# search on 26-47% of rebalancing dates while psi was only ~47% shrunk.
# LAMBDA_L2_CURR_GRID = [0.0, 5e-5, 5e-4, 5e-3, 5e-2]


# Asset-level MV penalty grids, for the mean-free (minimum-variance) asset
# problem. The MV risk term is ~3 orders of magnitude smaller than the ES one
# (daily variance ~2e-5 against daily ES_80 ~1.2e-2), so the ES asset grids
# above do not bite here at all. Bounds were calibrated by sweeping each
# penalty on real estimation windows of the top-10 universe: L1 starts moving
# the solution at ~3e-7 and is saturated at a corner (gross = 1) from ~2e-5 on;
# L2 decays smoothly and reaches the equal-weight floor (||x||^2 = 1/N exactly,
# gross = 1) at ~1e-2. Verified stable across five windows spanning 2000-2018.
LAMBDA_L1_MV_ASSET_GRID = [0.0, 1e-6, 5e-6, 5e-5]
LAMBDA_L2_MV_ASSET_GRID = [0.0, 1e-4, 1e-3, 1e-2]


# Fixed
LEVERAGE = [2.0]
ALPHA = [0.80]
GAMMA = [50.0]


ES_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "lambda_l1":      LAMBDA_L1_ASSET_GRID,
        "lambda_l2":      LAMBDA_L2_ASSET_GRID,
        "leverage_limit": LEVERAGE,
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
        "alpha":          ALPHA,
        "leverage_limit": LEVERAGE,
    },
)

REG_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":           ALPHA,
        "lambda_l1_asset": LAMBDA_L1_ASSET_GRID,
        "lambda_l2_asset": LAMBDA_L2_ASSET_GRID,
        "lambda_l1_curr":  LAMBDA_L1_CURR_GRID,
        "lambda_l2_curr":  LAMBDA_L2_CURR_GRID,
        "leverage_limit":  LEVERAGE,
    },
)

JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "leverage_limit": LEVERAGE,
    },
)

REG_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":           ALPHA,
        "lambda_l1_asset": LAMBDA_L1_ASSET_GRID,
        "lambda_l2_asset": LAMBDA_L2_ASSET_GRID,
        "lambda_l1_curr":  LAMBDA_L1_CURR_GRID,
        "lambda_l2_curr":  LAMBDA_L2_CURR_GRID,
        "leverage_limit":  LEVERAGE,
    },
)

# The two asset-level MV configs are deliberately identical: ASSETS-ZERO solves
# the same problem on R_fh and differs only at realization (unhedged, via the
# backtester's training_return_key / asset_return_key split). Neither carries a
# gamma — the mean term is gone, so gamma would only rescale the penalties.
REG_MV_ASSET_FULL_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1":      LAMBDA_L1_MV_ASSET_GRID,
        "lambda_l2":      LAMBDA_L2_MV_ASSET_GRID,
        "leverage_limit": LEVERAGE,
    },
)

REG_MV_ASSET_ZERO_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1":      LAMBDA_L1_MV_ASSET_GRID,
        "lambda_l2":      LAMBDA_L2_MV_ASSET_GRID,
        "leverage_limit": LEVERAGE,
    },
)

MV_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          GAMMA,
        "leverage_limit": LEVERAGE,
    },
)

REG_MV_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":            GAMMA,
        "lambda_l1_asset":  LAMBDA_L1_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_CURR_GRID,
        "leverage_limit":   LEVERAGE,
    },
)

MV_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          GAMMA,
        "leverage_limit": LEVERAGE,
    },
)

REG_MV_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":            GAMMA,
        "lambda_l1_asset":  LAMBDA_L1_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_CURR_GRID,
        "leverage_limit":   LEVERAGE,
    },
)
