"""Backtest window, rebalancing frequency and hyperparameter grid of every strategy."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    window_years: int = 10
    val_years: int = 1
    rebalancing_frequency: str = "quarterly"
    param_grid: dict = field(default_factory=dict)


def config(**param_grid) -> BacktestConfig:
    return BacktestConfig(param_grid=param_grid)


def asset_penalties(l1: list, l2: list) -> dict:
    return {"lambda_l1": l1, "lambda_l2": l2}


def block_penalties(asset_l1: list, asset_l2: list, curr_l1: list, curr_l2: list) -> dict:
    return {"lambda_l1_asset": asset_l1, "lambda_l2_asset": asset_l2,
            "lambda_l1_curr": curr_l1, "lambda_l2_curr": curr_l2}


ALPHA = [0.80]
LEVERAGE = [2.0]
OVERLAY_LIMIT = [2.0]

# Centers the returns of the ES and SOES overlay and joint programs; also their default for --no-grid runs.
# A one-element grid, since raw and de-meaned ES are different measures, not hyperparameters to select.
DEMEAN: bool = False
DEMEAN_GRID = [DEMEAN]

# The risk aversion of each mean-risk family is pinned to one value, which must equal the gamma
# default of its optimizers.
ES_GAMMA_GRID = [2]
MV_GAMMA_GRID = [30]
SMV_GAMMA_GRID = [30]
MAD_GAMMA_GRID = [0.5]
SOES_GAMMA_GRID = [0.5]

EW_CONFIG = config()

# Expected Shortfall
LAMBDA_L1_ASSET_GRID = [0.0, 5e-5, 5e-4, 5e-3]
LAMBDA_L2_ASSET_GRID = [0.0, 0.03, 0.3, 3.0]
LAMBDA_L1_CURR_GRID = [0.0, 5e-6, 5e-5, 5e-4]
LAMBDA_L2_CURR_GRID = [0.0, 5e-7, 5e-6, 5e-5]
ES_PENALTIES = block_penalties(LAMBDA_L1_ASSET_GRID, LAMBDA_L2_ASSET_GRID, LAMBDA_L1_CURR_GRID, LAMBDA_L2_CURR_GRID)

ES_CONFIG = config(alpha=ALPHA, **asset_penalties(LAMBDA_L1_ASSET_GRID, LAMBDA_L2_ASSET_GRID), leverage_limit=LEVERAGE)
OVERLAY_CONFIG = config(alpha=ALPHA, leverage_limit=LEVERAGE, demean=DEMEAN_GRID)
JOINT_CONFIG = config(alpha=ALPHA, leverage_limit=LEVERAGE, demean=DEMEAN_GRID)
REG_OVERLAY_CONFIG = config(alpha=ALPHA, **ES_PENALTIES, leverage_limit=LEVERAGE, demean=DEMEAN_GRID)
REG_JOINT_CONFIG = config(alpha=ALPHA, **ES_PENALTIES, leverage_limit=LEVERAGE, demean=DEMEAN_GRID)
ES_MEAN_OVERLAY_CONFIG = config(alpha=ALPHA, gamma=ES_GAMMA_GRID, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
ES_MEAN_JOINT_CONFIG = config(alpha=ALPHA, gamma=ES_GAMMA_GRID, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
REG_ES_MEAN_OVERLAY_CONFIG = config(alpha=ALPHA, **ES_PENALTIES, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
REG_ES_MEAN_JOINT_CONFIG = config(alpha=ALPHA, **ES_PENALTIES, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)

# Variance
LAMBDA_L1_MV_ASSET_GRID = [0.0, 1e-6, 5e-6, 5e-5]
LAMBDA_L2_MV_ASSET_GRID = [0.0, 1e-4, 1e-3, 1e-2]
LAMBDA_L1_MV_CURR_GRID = [0.0, 5e-7, 5e-6, 5e-5]
LAMBDA_L2_MV_CURR_GRID = [0.0, 5e-6, 1e-4, 2e-3]
MV_PENALTIES = block_penalties(LAMBDA_L1_MV_ASSET_GRID, LAMBDA_L2_MV_ASSET_GRID,
                               LAMBDA_L1_MV_CURR_GRID, LAMBDA_L2_MV_CURR_GRID)

REG_MV_ASSET_FULL_CONFIG = config(**asset_penalties(LAMBDA_L1_MV_ASSET_GRID, LAMBDA_L2_MV_ASSET_GRID),
                                  leverage_limit=LEVERAGE)
REG_MV_ASSET_ZERO_CONFIG = config(**asset_penalties(LAMBDA_L1_MV_ASSET_GRID, LAMBDA_L2_MV_ASSET_GRID),
                                  leverage_limit=LEVERAGE)
MV_OVERLAY_CONFIG = config(leverage_limit=LEVERAGE)
MV_JOINT_CONFIG = config(leverage_limit=LEVERAGE)
REG_MV_OVERLAY_CONFIG = config(**MV_PENALTIES, leverage_limit=LEVERAGE)
REG_MV_JOINT_CONFIG = config(**MV_PENALTIES, leverage_limit=LEVERAGE)
MV_MEAN_OVERLAY_CONFIG = config(gamma=MV_GAMMA_GRID, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
MV_MEAN_JOINT_CONFIG = config(gamma=MV_GAMMA_GRID, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
REG_MV_MEAN_OVERLAY_CONFIG = config(**MV_PENALTIES, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
REG_MV_MEAN_JOINT_CONFIG = config(**MV_PENALTIES, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)

# Semi-variance
LAMBDA_L1_SMV_ASSET_GRID = [0.0, 1e-6, 1e-5, 1e-4]
LAMBDA_L2_SMV_ASSET_GRID = [0.0, 5e-5, 1.5e-3, 5e-2]
LAMBDA_L1_SMV_CURR_GRID = [0.0, 2.5e-7, 2.5e-6, 2.5e-5]
LAMBDA_L2_SMV_CURR_GRID = [0.0, 1e-7, 1e-5, 1e-3]
SMV_PENALTIES = block_penalties(LAMBDA_L1_SMV_ASSET_GRID, LAMBDA_L2_SMV_ASSET_GRID,
                                LAMBDA_L1_SMV_CURR_GRID, LAMBDA_L2_SMV_CURR_GRID)

REG_SMV_ASSET_FULL_CONFIG = config(**asset_penalties(LAMBDA_L1_SMV_ASSET_GRID, LAMBDA_L2_SMV_ASSET_GRID),
                                   leverage_limit=LEVERAGE)
REG_SMV_ASSET_ZERO_CONFIG = REG_SMV_ASSET_FULL_CONFIG
SMV_OVERLAY_CONFIG = config(leverage_limit=LEVERAGE)
SMV_JOINT_CONFIG = config(leverage_limit=LEVERAGE)
REG_SMV_OVERLAY_CONFIG = config(**SMV_PENALTIES, leverage_limit=LEVERAGE)
REG_SMV_JOINT_CONFIG = config(**SMV_PENALTIES, leverage_limit=LEVERAGE)
SMV_MEAN_OVERLAY_CONFIG = config(gamma=SMV_GAMMA_GRID, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
SMV_MEAN_JOINT_CONFIG = config(gamma=SMV_GAMMA_GRID, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
REG_SMV_MEAN_OVERLAY_CONFIG = config(**SMV_PENALTIES, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
REG_SMV_MEAN_JOINT_CONFIG = config(**SMV_PENALTIES, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)

# Mean absolute deviation
LAMBDA_L1_MAD_ASSET_GRID = [0.0, 2.5e-5, 2.5e-4, 2.5e-3]
LAMBDA_L2_MAD_ASSET_GRID = [0.0, 0.01, 0.1, 1.0]
LAMBDA_L1_MAD_CURR_GRID = [0.0, 1.5e-5, 1.5e-4, 1.5e-3]
LAMBDA_L2_MAD_CURR_GRID = [0.0, 2.5e-3, 2.5e-2, 2.5e-1]
MAD_PENALTIES = block_penalties(LAMBDA_L1_MAD_ASSET_GRID, LAMBDA_L2_MAD_ASSET_GRID,
                                LAMBDA_L1_MAD_CURR_GRID, LAMBDA_L2_MAD_CURR_GRID)

REG_MAD_ASSET_FULL_CONFIG = config(**asset_penalties(LAMBDA_L1_MAD_ASSET_GRID, LAMBDA_L2_MAD_ASSET_GRID),
                                   leverage_limit=LEVERAGE)
REG_MAD_ASSET_ZERO_CONFIG = REG_MAD_ASSET_FULL_CONFIG
MAD_OVERLAY_CONFIG = config(leverage_limit=LEVERAGE)
MAD_JOINT_CONFIG = config(leverage_limit=LEVERAGE)
REG_MAD_OVERLAY_CONFIG = config(**MAD_PENALTIES, leverage_limit=LEVERAGE)
REG_MAD_JOINT_CONFIG = config(**MAD_PENALTIES, leverage_limit=LEVERAGE)
MAD_MEAN_OVERLAY_CONFIG = config(gamma=MAD_GAMMA_GRID, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
MAD_MEAN_JOINT_CONFIG = config(gamma=MAD_GAMMA_GRID, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
REG_MAD_MEAN_OVERLAY_CONFIG = config(**MAD_PENALTIES, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)
REG_MAD_MEAN_JOINT_CONFIG = config(**MAD_PENALTIES, leverage_limit=LEVERAGE, overlay_limit=OVERLAY_LIMIT)

# Second-order Expected Shortfall
LAMBDA_L1_SOES_ASSET_GRID = [0.0, 1e-3, 1e-2, 1e-1]
LAMBDA_L2_SOES_ASSET_GRID = [0.0, 5e-2, 1.0, 15.0]
LAMBDA_L1_SOES_CURR_GRID = [0.0, 1e-4, 1e-3, 1e-2]
LAMBDA_L2_SOES_CURR_GRID = [0.0, 1e-4, 1e-2, 1.0]
SOES_PENALTIES = block_penalties(LAMBDA_L1_SOES_ASSET_GRID, LAMBDA_L2_SOES_ASSET_GRID,
                                 LAMBDA_L1_SOES_CURR_GRID, LAMBDA_L2_SOES_CURR_GRID)

REG_SOES_ASSET_FULL_CONFIG = config(alpha=ALPHA, **asset_penalties(LAMBDA_L1_SOES_ASSET_GRID, LAMBDA_L2_SOES_ASSET_GRID),
                                    leverage_limit=LEVERAGE)
REG_SOES_ASSET_ZERO_CONFIG = REG_SOES_ASSET_FULL_CONFIG
SOES_OVERLAY_CONFIG = config(alpha=ALPHA, leverage_limit=LEVERAGE, demean=DEMEAN_GRID)
SOES_JOINT_CONFIG = config(alpha=ALPHA, leverage_limit=LEVERAGE, demean=DEMEAN_GRID)
REG_SOES_OVERLAY_CONFIG = config(alpha=ALPHA, **SOES_PENALTIES, leverage_limit=LEVERAGE, demean=DEMEAN_GRID)
REG_SOES_JOINT_CONFIG = config(alpha=ALPHA, **SOES_PENALTIES, leverage_limit=LEVERAGE, demean=DEMEAN_GRID)
SOES_MEAN_OVERLAY_CONFIG = config(alpha=ALPHA, gamma=SOES_GAMMA_GRID, leverage_limit=LEVERAGE,
                                  overlay_limit=OVERLAY_LIMIT)
SOES_MEAN_JOINT_CONFIG = config(alpha=ALPHA, gamma=SOES_GAMMA_GRID, leverage_limit=LEVERAGE,
                                overlay_limit=OVERLAY_LIMIT)
REG_SOES_MEAN_OVERLAY_CONFIG = config(alpha=ALPHA, **SOES_PENALTIES, leverage_limit=LEVERAGE,
                                      overlay_limit=OVERLAY_LIMIT)
REG_SOES_MEAN_JOINT_CONFIG = config(alpha=ALPHA, **SOES_PENALTIES, leverage_limit=LEVERAGE,
                                    overlay_limit=OVERLAY_LIMIT)
