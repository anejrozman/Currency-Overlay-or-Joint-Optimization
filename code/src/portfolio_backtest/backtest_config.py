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

# Currency-block MV penalty grids, for the mean-free overlay. psi carries no budget
# or leverage constraint, so L1 extinguishes it outright while L2 shrinks it
# smoothly. Calibrated on windows at 2000, 2007 and 2015, where unpenalized gross
# psi is 0.93, 1.28 and 0.96: L1 halves the overlay by ~2e-7, spends 98% of it by
# ~1e-5 and zeroes it by ~5e-5; L2 halves it by ~5e-6 and reaches ~2% by ~2e-3.
LAMBDA_L1_MV_CURR_GRID = [0.0, 5e-7, 5e-6, 5e-5]
LAMBDA_L2_MV_CURR_GRID = [0.0, 5e-6, 1e-4, 2e-3]


# Asset-level MAD penalty grids, for the mean-free (minimum-MAD) asset problem.
# The MAD risk term sits between the other two measures rather than at either's
# scale: ~3.1e-3 at the unpenalized optimum, against ES_80 ~5.7e-3 and daily
# variance ~2.0e-5 on the same windows, so neither neighbouring grid bites here.
# The term is stable across regimes (2.93e-3 to 3.39e-3, a 16% spread), so one
# grid serves the whole sample. Bounds were calibrated by sweeping each penalty
# alone over eight decades on five real estimation windows of the top-10 universe
# (1990-2000 through 2012-2022, T = 2610, N = 101-129).
#
# L1 saturates at a corner. Gross exposure and the active count start moving at
# ~1e-5, gross halves by 1.8e-4 - 3.2e-4, and from 1.3e-3 - 2.4e-3 on the solution
# is pinned at the long-only budget corner: gross = 1.000000 exactly in every
# window, active names down from 128 to 39, and d_inf past that point never above
# 1.0e-4. Note that d_inf is NOT usable to locate the lower end of this range: the
# unpenalized MAD program is a degenerate LP whose optimal face contains many tied
# vertices, so d_inf spikes to ~1e-4 at lambdas as small as 4e-8 while gross moves
# in the fifth decimal and no name enters or leaves. Gross and the active count are
# what identify the onset; d_inf only establishes the upper bound.
#
# L2 decays smoothly and never corners. Gross halves by 1.3e-2 - 4.2e-2 and reaches
# 1.0 by 1.8e-1 - 4.2e-1. The bound is set at the equal-weight floor, which is
# approached asymptotically rather than attained: at lambda = 1.0, ||w||^2 is
# 0.7% - 3.1% above 1/N across the five windows (all names still active, gross = 1).
# If the validation search selects the 1.0 endpoint on a large share of rebalancing
# dates, this bound wants extending, as happened with LAMBDA_L2_CURR_GRID above.
LAMBDA_L1_MAD_ASSET_GRID = [0.0, 2.5e-5, 2.5e-4, 2.5e-3]
LAMBDA_L2_MAD_ASSET_GRID = [0.0, 0.01, 0.1, 1.0]

# Currency-block MAD penalty grids, for the mean-free overlay. psi carries no budget
# or leverage constraint, so L1 extinguishes it outright while L2 shrinks it smoothly
# — the same asymmetry as the ES and MV currency grids, at MAD's own scale. Calibrated
# by sweeping each penalty alone (asset penalties held at zero) on the same five top-10
# estimation windows as the asset grids, where unpenalized gross psi is 0.52 - 1.24
# across 13 non-base currencies.
#
# L1 halves the overlay by 7.5e-6 - 7.5e-5, spends 98% of it by 2.4e-4 - 1.0e-3, and
# zeroes it outright (gross psi = 0, no currency active) by 5.6e-4 - 1.3e-3; the bound
# sits just past the latest of those.
#
# L2 never zeroes the overlay — all 13 currencies stay active at every lambda — so the
# bound is set where it is economically spent rather than where it stops moving, the
# same criterion as LAMBDA_L2_CURR_GRID. It halves the overlay by 1.8e-5 - 1.3e-3 and
# reaches 2% of unpenalized gross in every window by 2.4e-1 (0.5% by 1.0).
LAMBDA_L1_MAD_CURR_GRID = [0.0, 1.5e-5, 1.5e-4, 1.5e-3]
LAMBDA_L2_MAD_CURR_GRID = [0.0, 2.5e-3, 2.5e-2, 2.5e-1]


# Asset-level semi-variance penalty grids, for the mean-free (minimum-semi-variance)
# asset problem. The SMV risk term is the smallest of the four measures: 9.3e-6 - 1.1e-5
# at the unpenalized optimum, against MV variance ~2.0e-5, MAD ~3.1e-3 and ES_80 ~5.7e-3
# on the same windows. It runs at almost exactly half the MV term, as it must for a
# near-symmetric sample, so the MV grids are within a factor of two of biting correctly
# but are not reused: the term is estimated at its own scale and calibrated here. It is
# stable across regimes (9.34e-6 to 1.11e-5, an 18% spread), so one grid serves the whole
# sample. Bounds were calibrated by sweeping each penalty alone over nine decades (L1,
# 1e-9 to 1e-1) and eight decades (L2, 1e-7 to 1e1) at four points per decade, on five
# real estimation windows of the top-10 universe (1990-2000 through 2012-2021,
# T = 2609-2610, N = 101-131).
#
# Note that gross exposure is confined to [1, 2] here: the budget constraint sum(x) = 1
# floors it and the leverage cap tops it. Neither penalty can drive it to zero the way
# the unconstrained overlay grids can, so the progress measure below is the share of the
# excess gross above 1 that the penalty spends, not gross itself.
#
# L1 saturates at a corner. It spends 10% of the excess gross by 3.2e-7 - 1.8e-6, half by
# 1.8e-6 - 3.2e-6 and 98% by 1.0e-5 - 3.2e-5; from 3.2e-5 - 1.0e-4 on the solution is
# pinned at the long-only budget corner, gross = 1.00000000 exactly in every window, with
# the active count down from 101-127 names to 29-49. Past the 1e-4 bound d_inf never
# exceeds 8.1e-5. As with MAD, d_inf alone does not locate the lower end: the unpenalized
# optimum has ties the penalty breaks at lambdas far below where gross moves, so gross and
# the active count identify the onset and d_inf only confirms the upper bound.
#
# L2 decays smoothly and never corners. It kills no names at any lambda — the active count
# never falls below its unpenalized value in any window — and instead pulls the weights
# toward equal weight. Gross reaches the budget floor by 1.8e-3 - 5.6e-3, and ||x||^2 comes
# within 1% of the equal-weight floor 1/N by 1.0e-2 - 3.2e-2 (within 0.1% by 3.2e-2 - 1.0e-1).
# Unlike LAMBDA_L2_MAD_ASSET_GRID, where the floor is only approached, it is attained here
# to five decimals, so the bound is set just past the latest window's 1% crossing. The
# interior point is the log-centre of the two ends.
LAMBDA_L1_SMV_ASSET_GRID = [0.0, 1e-6, 1e-5, 1e-4]
LAMBDA_L2_SMV_ASSET_GRID = [0.0, 5e-5, 1.5e-3, 5e-2]


# Fixed
LEVERAGE = [2.0]
ALPHA = [0.80]

# Every MV program is now mean-free, so no risk-aversion coefficient survives: with
# the mean gone gamma is pure scaling on the risk term, exactly equivalent to dividing
# both penalties by gamma / 2, and the penalty grids carry the tradeoff alone.
#
# The mean-bearing versions were kept just long enough to record why. At gamma = 3:
#   * the asset program collapses onto 19 of 101 names at the leverage bound
#     (results/runs/reg_mv_asset_zero/run_1)
#   * the overlay reaches a gross psi of up to 98x the portfolio notional
#     (results/runs/reg_mv_curr_overlay/run_1)
#   * the joint program reaches up to 74x, with 6-131 active assets
#     (results/runs/reg_mv_joint/run_1)
# The grids those three runs used are recorded in their own config.json.


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

# MAD asset-full. Same mean-free structure, window and rebalancing frequency as the
# MV and ES asset problems so the three risk measures stay directly comparable; only
# the penalty grids differ, since the MAD risk term is on its own scale.
REG_MAD_ASSET_FULL_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1":      LAMBDA_L1_MAD_ASSET_GRID,
        "lambda_l2":      LAMBDA_L2_MAD_ASSET_GRID,
        "leverage_limit": LEVERAGE,
    },
)

# ASSETS-ZERO solves exactly the same asset problem as ASSETS-FULL on R_fh and
# differs only at realization (unhedged, via the backtester's training_return_key /
# asset_return_key split), so it reuses the same config object rather than a copy:
# a single object cannot drift out of sync the way the two MV copies above could.
REG_MAD_ASSET_ZERO_CONFIG = REG_MAD_ASSET_FULL_CONFIG

# SMV asset-full. Same mean-free structure, window and rebalancing frequency as the ES, MV
# and MAD asset problems so the four risk measures stay directly comparable; only the
# penalty grids differ, since the SMV risk term is on its own scale.
REG_SMV_ASSET_FULL_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1":      LAMBDA_L1_SMV_ASSET_GRID,
        "lambda_l2":      LAMBDA_L2_SMV_ASSET_GRID,
        "leverage_limit": LEVERAGE,
    },
)

# ASSETS-ZERO solves exactly the same asset problem as ASSETS-FULL on R_fh and
# differs only at realization (unhedged, via the backtester's training_return_key /
# asset_return_key split), so it reuses the same config object rather than a copy:
# a single object cannot drift out of sync the way the two MV copies above could.
REG_SMV_ASSET_ZERO_CONFIG = REG_SMV_ASSET_FULL_CONFIG

MV_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "leverage_limit": LEVERAGE,
    },
)

REG_MV_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_MV_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_MV_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_MV_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_MV_CURR_GRID,
        "leverage_limit":   LEVERAGE,
    },
)

# The regularized MAD joint program reuses the asset and currency MAD grids, as the ES
# and MV joint configs reuse theirs. Both blocks were calibrated on the asset-level and
# overlay programs rather than on the joint one; reg_mv_joint records a joint-specific
# verification sweep confirming its bands sit inside the same bounds, and no equivalent
# sweep has been run for MAD yet.
REG_MAD_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_MAD_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_MAD_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_MAD_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_MAD_CURR_GRID,
        "leverage_limit":   LEVERAGE,
    },
)

# The unregularized MAD joint program has no penalties, so its grid is the leverage
# cap alone, exactly as MV_JOINT_CONFIG.
MAD_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "leverage_limit": LEVERAGE,
    },
)

# MAD overlays. The unregularized program has no penalties, so its grid is the
# leverage cap alone, exactly as MV_OVERLAY_CONFIG.
MAD_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "leverage_limit": LEVERAGE,
    },
)

# The regularized overlay reuses the calibrated MAD asset grids for the asset block
# and the MAD currency grids for the overlay block, as the ES and MV overlay configs
# reuse theirs.
REG_MAD_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_MAD_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_MAD_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_MAD_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_MAD_CURR_GRID,
        "leverage_limit":   LEVERAGE,
    },
)

MV_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "leverage_limit": LEVERAGE,
    },
)

# The joint program reuses the overlay's MV grids, as the ES configs do. Verified by
# sweeping the joint program itself on windows at 2000/2007/2015: L1 on x drives gross
# to the budget floor at 2.5e-5 (grid top 5e-5), L2 on x reaches equal weight by
# 6.3e-3 - 1.6e-2 (grid top 1e-2), L1 on psi zeroes it by 1.6e-5 - 4e-5 (grid top 5e-5)
# and L2 on psi reaches ~2% by 1.6e-3 (grid top 2e-3). All four bands sit inside the
# existing bounds, so no joint-specific constants are needed.
REG_MV_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_MV_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_MV_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_MV_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_MV_CURR_GRID,
        "leverage_limit":   LEVERAGE,
    },
)
