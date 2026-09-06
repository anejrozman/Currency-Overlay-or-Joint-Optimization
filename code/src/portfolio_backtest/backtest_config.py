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

# Currency-block SMV penalty grids, for the mean-free overlay. psi carries no budget
# or leverage constraint, so L1 extinguishes it outright while L2 shrinks it smoothly
# — the same asymmetry as the ES, MV and MAD currency grids, at SMV's own scale.
# Calibrated by sweeping each penalty alone (asset penalties held at zero) on the same
# five top-10 estimation windows as the asset grids, where unpenalized gross psi is
# 1.09 - 1.87 across 13 non-base currencies.
#
# L1 halves the overlay by 3.2e-8 - 3.2e-7, leaves 2% of it by 3.2e-6 - 1.0e-5, and
# zeroes it outright (gross psi = 0, no currency active) by 5.6e-6 - 1.8e-5; the bound
# sits just past the latest of those. The onset is bracketed: at 1e-9 the overlay still
# retains 97.6% - 99.2% of its unpenalized gross in every window.
#
# L2 never zeroes the overlay, so the bound is set where it is economically spent
# rather than where it stops moving, the same criterion as LAMBDA_L2_MV_CURR_GRID and
# LAMBDA_L2_MAD_CURR_GRID. It reaches 2% of unpenalized gross by 3.2e-4 - 1.0e-3 in
# every window and 0.5% by 1.8e-3 - 5.6e-3. Unlike L1 its onset is NOT bracketed by
# the sweep: at the smallest lambda swept, 1e-7, the overlay is already down to
# 0.47 - 0.75 of unpenalized gross, so it halves at or below 1e-7 and the first grid
# point is placed there rather than at a measured halving point. (Counts of "active"
# currencies fall at large L2 only because individual weights drop below the 1e-6
# tolerance once the whole overlay is spent; L2 produces no true sparsity.)
LAMBDA_L1_SMV_CURR_GRID = [0.0, 2.5e-7, 2.5e-6, 2.5e-5]
LAMBDA_L2_SMV_CURR_GRID = [0.0, 1e-7, 1e-5, 1e-3]


# Asset-level second-order ES (HMCR_2) penalty grids, for the regularized ASSETS-FULL
# HMCR_2 asset problem (Krokhmal 2007; see reg_soes_asset_full.py). HMCR_2 sits on the
# same daily-return scale as ES_80 but runs materially larger, since the tail is
# penalized by its L2 norm rather than its mean and the L2 norm of a set of positive
# numbers dominates their mean by construction — so the ES asset grids above are close
# in shape but not in magnitude, and this measure is calibrated at its own scale rather
# than reusing them. There is no overlay or joint variant of this measure yet, so unlike
# ES/MV/MAD/SMV above there is no companion currency-block grid here.
#
# Bounds were calibrated by sweeping each penalty alone (the other held at zero,
# alpha = 0.80, leverage_limit = 2.0) at three points per decade on the same five real
# estimation windows of the top-10 universe used for the SMV grids above
# (1990-2000 through 2012-2022, T = 2610, N = 101-129), solved with Clarabel (no local
# Gurobi; see the solver fallback in reg_soes_asset_full.py).
#
# Gross exposure is confined to [1, 2] exactly as for MV/MAD/SMV: the budget constraint
# sum(x) = 1 floors it and the leverage cap ||x||_1 <= 2 tops it. Unlike those three
# measures, the unpenalized optimum sits exactly at the leverage cap (gross =
# 2.00000000 in every window) rather than somewhere inside [1, 2] — the unregularized
# HMCR_2 program always wants to lever into short positions to shrink the tail norm
# further. The progress measure below is, as for the other asset grids, the share of
# the excess gross above 1 that the penalty spends.
#
# L1 saturates at a corner. It spends 10% of the excess gross by 1.0e-3 - 2.2e-3, half
# by 2.2e-3 - 4.6e-3 and 98% by 1.0e-2 - 2.2e-2; from 2.2e-2 - 4.6e-2 on the solution is
# pinned at the long-only budget corner, gross = 1.00000000 exactly in every window,
# with the active count down from 74-84 names to 23-30. Past the 4.6e-2 bound d_inf
# never exceeds 1.9e-6. The grid's low end is set at the onset (1e-3) and its high end
# at the point every window is already pinned (1e-1, one full decade past the 4.6e-2
# stabilization point, for margin); the interior point 1e-2 is the exact log-centre of
# the two and itself sits right at the 98%-spent mark.
#
# L2 decays smoothly and never corners: the active count only grows, from 74-84
# unpenalized toward the full universe (101-129, i.e. every name active) as the
# quadratic penalty spreads exposure across all of them, exactly the MV/MAD/SMV
# pattern of pulling weights toward equal weight rather than pruning them. Gross
# starts moving at 4.6e-2 - 1.0e-1, reaches the budget floor (gross = 1.00000000
# exactly) by 1.0 - 2.2, and ||x||^2 comes within 1% of the equal-weight floor 1/N by
# 4.6 - 10.0 (within 0.1% by 21.5 - 46.4). As with SMV, and unlike MAD, the floor is
# attained here to five decimals (from lambda ~ 460 on, in every window) rather than
# only approached, so the bound is set just past the latest window's 1% crossing
# (10.0) rather than pushed out toward full attainment: 15.0. The interior point is
# set at 1.0 — close to, but not exactly, the log-centre of the two ends (0.87) —
# because 1.0 is also where the budget floor is first reached exactly, in the fastest
# window.
LAMBDA_L1_SOES_ASSET_GRID = [0.0, 1e-3, 1e-2, 1e-1]
LAMBDA_L2_SOES_ASSET_GRID = [0.0, 5e-2, 1.0, 15.0]

# Currency-block HMCR_2 penalty grids, for the regularized SOES overlay. psi carries
# no budget or leverage constraint, so L1 extinguishes it outright while L2 shrinks it
# smoothly — the same asymmetry as the ES, MV, MAD and SMV currency grids, at HMCR_2's
# own scale. Calibrated by sweeping each penalty alone (asset penalties held at zero,
# alpha = 0.80, leverage_limit = 2.0) at three points per decade on the same five
# top-10 estimation windows as the SOES asset grids above, where unpenalized gross psi
# is 1.31 - 2.56 across 13 non-base currencies. Exploits the overlay optimizer's own
# step-1 caching (see reg_soes_curr_overlay.py): with the asset penalties fixed at
# zero for the whole sweep, step 1 solves once per window and every swept point reuses
# it, so the full two-grid, five-window sweep completed in 45 seconds on Clarabel.
#
# L1 halves the overlay by 2.2e-5 - 1.0e-4, and zeroes it outright (gross psi = 0, no
# currency active) by 2.2e-3 - 1.0e-2; the bound is set at that upper zeroing point,
# the same criterion as LAMBDA_L1_SMV_CURR_GRID.
#
# L2 never zeroes the overlay — only individual currency weights fall below the 1e-6
# active tolerance at very large lambda, exactly the LAMBDA_L2_SMV_CURR_GRID pattern
# and not true sparsity — so the bound is set where it is economically spent rather
# than where it stops moving. It halves the overlay by 4.6e-5 - 4.6e-4 and reaches 2%
# of unpenalized gross by 2.2e-1 - 1.0 in every window; the bound is set at that upper
# 2%-crossing point.
LAMBDA_L1_SOES_CURR_GRID = [0.0, 1e-4, 1e-3, 1e-2]
LAMBDA_L2_SOES_CURR_GRID = [0.0, 1e-4, 1e-2, 1.0]


# Fixed
LEVERAGE = [2.0]
ALPHA = [0.80]
OVERLAY_LIMIT = [2.0]

# Risk-aversion grid for the mean-bearing ES overlay. Named per risk measure:
# the objective is Risk - mu'w / gamma, so the balancing gamma scales with the
# magnitude of the risk term, and those differ by three orders of magnitude across
# the measures here (variance ~2e-5, SMV ~1e-5, MAD ~3.1e-3, ES_80 ~1.2e-2, HMCR_2
# larger still). An MV or MAD mean-bearing program needs its own grid calibrated at
# its own scale; this one is ES-specific and must not be reused.
# Risk-aversion grid for the mean-bearing ES overlay (es_curr_overlay_w_mean.py),
# the one program here that keeps a mean term. The mean enters its objective with
# coefficient 1 / gamma, so small gamma means an aggressive mean tilt and large
# gamma a nearly mean-free program.
#
# Calibrated by running the backtester at fixed gamma (no selection, so the OOS
# result is attributable to gamma alone) on the top-10 universe, quarterly, 10y
# window, leverage 2.0, overlay cap 2.0, over 2000-07 to 2023-04:
#
#   gamma    ES_80    Vol     MDD      SR       ESR    annRet
#     100   0.6892   8.16   29.29   0.0772   0.0576    2.07%   <- mean term inert
#      10   0.6903   8.17   29.29   0.0829   0.0618    2.11%
#       1   0.6948   8.23   29.61   0.0963   0.0718    2.23%
#     0.5   0.7032   8.31   29.45   0.1041   0.0775    2.30%
#     0.2   0.7395   8.74   29.27   0.1536   0.1144    2.78%   <- optimum
#    0.05   1.1406  13.76   44.94   0.1309   0.0995    3.24%   <- risk running away
#    0.01   2.3278  29.06   76.68  -0.0869  -0.0683   -1.09%   <- blow-up
#
# The optimum is interior at ~0.2 and the curve is single-peaked, so the grid is
# recentered on it rather than left on the original [100 ... 1] range, where every
# point sat in the inert region and the search could only ever run to the bottom
# endpoint. From gamma = 10 up the mean term changes nothing (ES_80 and MDD flat to
# three decimals against the mean-free program), so sweeping 100/50/20 wasted three
# of seven grid points; 10 is retained as the near-mean-free anchor.
#
# The lower bound is set at 0.1: one point past the peak, enough to bracket an
# interior optimum, and a full decade above the 0.01 blow-up. That blow-up is the
# same pathology recorded for the old mean-bearing MV programs above — as gamma
# falls the objective tends to pure mean maximization over sum(x) = 1, ||x||_1 <= 2,
# a linear program whose solution is a levered corner bet on the highest trailing-mean
# economies — but note it takes a gamma two decades below the useful range to
# trigger here, where the raw-mean MV programs blew up at gamma = 3. The economy
# pooling and the James-Stein shrinkage are what buy that margin.
ES_GAMMA_GRID = [10, 5, 2, 1, 0.5, 0.2, 0.1]

# Column-demeaning switch for the eight ES and SOES overlay/joint programs,
# flipped manually. ES and its second-order variant (HMCR_2) are the two risk
# measures here that are not deviation measures: MV, MAD and semi-variance all
# subtract the sample mean before measuring risk, so their objectives are
# location-free, while raw ES and SOES also reward a high mean. Set this to True
# to center the return matrices columnwise before those programs are solved,
# which turns their objectives into pure deviation measures and makes them
# comparable to the other three.
#
# The scalar is what the eight optimizers import as their own default, and
# DEMEAN_GRID puts the same value in their grids, so a grid run and a --no-grid
# run (which falls back to those defaults) demean identically. Kept a one-element
# list rather than a two-point grid: demeaned and raw ES are different risk
# measures, not competing hyperparameters, so the validation search must not
# choose between them.
DEMEAN: bool = True
DEMEAN_GRID = [DEMEAN]

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
        "demean":         DEMEAN_GRID,
    },
)

# Mean-bearing ES overlay. No `demean` entry, deliberately: centering the return
# matrices would zero the sample means this program is built around, so the switch is
# not just unused here but contradictory. Carries the overlay L1 cap and the risk
# aversion grid on top of OVERLAY_CONFIG's alpha and leverage cap.
ES_MEAN_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "gamma":          ES_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
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
        "demean":          DEMEAN_GRID,
    },
)

# Mean-bearing ES joint program. Same grid as ES_MEAN_OVERLAY_CONFIG, and reusing
# ES_GAMMA_GRID: it is the same risk measure at the same scale, so the calibrated
# range transfers, exactly as the other joint configs reuse the asset and currency
# grids calibrated on their standalone programs. The joint program chooses x and psi
# against a single tail rather than in sequence, so the mean pulls somewhat harder at
# any given gamma; if the selected-gamma series clusters at either endpoint, this
# range wants recalibrating on the joint program itself.
ES_MEAN_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "gamma":          ES_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
    },
)

JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "leverage_limit": LEVERAGE,
        "demean":         DEMEAN_GRID,
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
        "demean":          DEMEAN_GRID,
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

# SMV overlay. The unregularized program has no penalties, so its grid is the
# leverage cap alone, exactly as MV_OVERLAY_CONFIG and MAD_OVERLAY_CONFIG.
SMV_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "leverage_limit": LEVERAGE,
    },
)

# The regularized SMV joint program reuses the asset and currency SMV grids, as the ES,
# MV and MAD joint configs reuse theirs. Both blocks were calibrated on the asset-level
# and overlay programs, so the joint program was swept on its own to verify the bands
# sit inside those bounds — the same check REG_MV_JOINT_CONFIG records, run here on the
# five top-10 windows with each penalty swept alone and the other three at zero.
# Unpenalized, the joint program has gross x of 1.72 - 2.00 (101-131 names active) and
# gross psi of 1.42 - 2.10 across 13 non-base currencies.
#
#   L1 on x   pins the solution at the long-only budget corner by 1.0e-5 - 1.8e-5
#             (bound 1e-4)
#   L2 on x   reaches the equal-weight floor, within 1% of 1/N, by 1.0e-2 - 1.8e-2
#             (bound 5e-2)
#   L1 on psi zeroes the overlay outright by 5.6e-6 - 1.8e-5   (bound 2.5e-5)
#   L2 on psi leaves 2% of unpenalized gross psi by 3.2e-4 - 1.0e-3  (bound 1e-3)
#
# The first three sit comfortably inside their bounds. L2 on psi does not: its band tops
# out at 1.0e-3, exactly the grid bound, so the bound is sufficient rather than generous.
# If the validation search selects that endpoint on a large share of rebalancing dates
# the bound wants extending, as happened with LAMBDA_L2_CURR_GRID above.
REG_SMV_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_SMV_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_SMV_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_SMV_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_SMV_CURR_GRID,
        "leverage_limit":   LEVERAGE,
    },
)

# The unregularized SMV joint program has no penalties, so its grid is the leverage
# cap alone, exactly as MV_JOINT_CONFIG and MAD_JOINT_CONFIG.
SMV_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "leverage_limit": LEVERAGE,
    },
)

# The regularized overlay reuses the calibrated SMV asset grids for the asset block
# and the SMV currency grids for the overlay block, as the ES, MV and MAD overlay
# configs reuse theirs.
REG_SMV_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_SMV_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_SMV_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_SMV_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_SMV_CURR_GRID,
        "leverage_limit":   LEVERAGE,
    },
)

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

# Asset-only regularized second-order ES (HMCR_2), ASSETS-FULL strategy. Structured
# like ES_CONFIG (alpha plus the two penalties): unlike MV/MAD/SMV, HMCR_2 keeps a
# confidence level. There is no overlay or joint variant of this measure yet.
REG_SOES_ASSET_FULL_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "lambda_l1":      LAMBDA_L1_SOES_ASSET_GRID,
        "lambda_l2":      LAMBDA_L2_SOES_ASSET_GRID,
        "leverage_limit": LEVERAGE,
    },
)

# ASSETS-ZERO solves exactly the same asset problem as ASSETS-FULL on R_fh and
# differs only at realization (unhedged, via the backtester's training_return_key /
# asset_return_key split), so it reuses the same config object rather than a copy,
# as REG_SMV_ASSET_ZERO_CONFIG does above.
REG_SOES_ASSET_ZERO_CONFIG = REG_SOES_ASSET_FULL_CONFIG

# Unregularized SOES overlay. Keeps alpha (unlike the mean-free MV/MAD/SMV
# overlays), exactly as OVERLAY_CONFIG does for ES; the unregularized program has
# no penalties, so its grid is alpha and the leverage cap alone.
SOES_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "leverage_limit": LEVERAGE,
        "demean":         DEMEAN_GRID,
    },
)

# The regularized overlay reuses the calibrated SOES asset grids for the asset block
# and the SOES currency grids for the overlay block, as the ES, MV, MAD and SMV
# overlay configs reuse theirs.
REG_SOES_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":            ALPHA,
        "lambda_l1_asset":  LAMBDA_L1_SOES_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_SOES_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_SOES_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_SOES_CURR_GRID,
        "leverage_limit":   LEVERAGE,
        "demean":           DEMEAN_GRID,
    },
)

# Unregularized SOES joint program. Keeps alpha, exactly as JOINT_CONFIG does for
# es_joint; the unregularized program has no penalties, so its grid is alpha and
# the leverage cap alone.
SOES_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "leverage_limit": LEVERAGE,
        "demean":         DEMEAN_GRID,
    },
)

# The regularized SOES joint program reuses the asset and currency SOES grids, as the
# ES, MV, MAD and SMV joint configs reuse theirs. Both blocks were calibrated on the
# asset-level and overlay programs, so the joint program was swept on its own to
# verify the bands sit inside those bounds — the same check REG_SMV_JOINT_CONFIG
# records, run here on the five top-10 windows with each penalty swept alone and the
# other three at zero. Unpenalized, the joint program has gross x of 2.00 exactly
# (85-100 of 101-129 names active, pinned at the leverage cap just as the unpenalized
# asset-only program is) and gross psi of 1.52-4.37 across 13 non-base currencies —
# notably larger than the standalone overlay's unpenalized 1.31-2.56, since x and psi
# are chosen together here and psi leans harder on currency risk reduction than a psi
# fit against an already-fixed x.
#
#   L1 on x   pins the solution at the long-only budget corner by 1.0e-2 in every
#             window (bound 1e-1)
#   L2 on x   reaches the budget floor (gross x = 1.00000000 exactly) by 1.0 - 3.2
#             (bound 15.0)
#   L1 on psi zeroes the overlay outright by 3.2e-3 - 1.0e-2   (bound 1e-2)
#   L2 on psi leaves 2% of unpenalized gross psi by 1.0e-1 - 1.0   (bound 1.0)
#
# The first three sit comfortably inside their bounds. L2 on psi does not: its band
# tops out at 1.0, exactly the grid bound, so the bound is sufficient rather than
# generous — the same situation REG_SMV_JOINT_CONFIG flags for its own L2-on-psi
# check. If the validation search selects that endpoint on a large share of
# rebalancing dates, the bound wants extending, as happened with LAMBDA_L2_CURR_GRID
# above.
REG_SOES_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":            ALPHA,
        "lambda_l1_asset":  LAMBDA_L1_SOES_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_SOES_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_SOES_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_SOES_CURR_GRID,
        "leverage_limit":   LEVERAGE,
        "demean":           DEMEAN_GRID,
    },
)
