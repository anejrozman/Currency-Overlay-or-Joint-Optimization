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

# Risk-aversion grid for the mean-bearing MV overlay (mv_curr_overlay_w_mean.py).
# Calibrated on its own program and its own scale; ES_GAMMA_GRID must not be reused
# here, for two independent reasons. The scale differs: the objective is
# Risk - mu'w / gamma, and a daily variance (~2.6e-5) is three orders of magnitude
# below an ES_80 (~1.2e-2), so the balancing gamma is that much larger. And the two
# gammas do different work: variance is a deviation measure, so the mean reaches
# this objective only through the explicit pooled and shrunk mu terms, whereas raw
# undemeaned ES is not a deviation measure -- ES(-r) = ES_deviation(-r) - E[r] --
# so es_curr_overlay_w_mean.py carries a second implicit mean channel at fixed
# weight one, estimated by the raw portfolio sample mean.
#
# Calibrated the same way ES_GAMMA_GRID was: the backtester at fixed gamma (a
# one-point grid, so no selection and the OOS result is attributable to gamma
# alone) on the top-10 universe, quarterly, 10y window, leverage 2.0, overlay cap
# 2.0, over 2000-07 to 2023-04. Solved with Clarabel rather than Gurobi, a ~1e-6
# weight difference.
#
#   gamma   ES_80    Vol     MDD      SR       ESR    annRet   nAct   grossPsi
#     ---  0.6658   7.97   28.23   0.1404   0.1059    2.55%   119.4     1.095   <- mv_curr_overlay
#    3000  0.6658   7.97   28.20   0.1452   0.1096    2.59%   115.8     1.087
#    1000  0.6658   7.98   28.07   0.1552   0.1172    2.67%   115.9     1.072
#     500  0.6660   7.98   27.86   0.1702   0.1285    2.79%   115.9     1.061
#     300  0.6663   7.99   27.58   0.1894   0.1430    2.95%   115.9     1.091
#     200  0.6671   8.00   27.26   0.2109   0.1593    3.12%   115.8     1.160
#     100  0.6712   8.06   26.36   0.2576   0.1948    3.51%   115.7     1.344
#      50  0.6847   8.23   25.45   0.3172   0.2401    4.05%   115.0     1.577   <- MDD minimum
#      30  0.7115   8.57   26.56   0.3718   0.2821    4.62%   113.9     1.771
#      20  0.7568   9.13   28.38   0.4133   0.3140    5.21%   110.5     1.900
#      15  0.8052   9.76   30.53   0.4503   0.3438    5.83%   106.8     1.959
#      10  0.9113  11.15   33.37   0.4808   0.3706    6.80%    98.8     1.991   <- SR optimum
#       7  1.0439  12.91   36.72   0.4708   0.3667    7.51%    88.7     2.000
#       5  1.2021  14.97   41.08   0.4398   0.3450    8.02%    75.7     2.000
#       3  1.5100  19.06   47.70   0.4522   0.3596   10.05%    59.5     2.000
#       2  1.8148  23.64   57.63   0.4364   0.3582   11.75%    50.2     2.000
#       1  2.4961  36.38   77.35   0.2783   0.2554   11.56%    34.5     2.000   <- risk running away
#     0.5  3.1919  51.86   88.97   0.1937   0.1983   11.48%    25.1     2.000   <- blow-up
#
# The Sharpe curve is single-peaked with an interior optimum at gamma = 10, and the
# blow-up is a full decade below that. The grid deliberately does not span the whole
# swept range: it is the [10, 100] band, which is where the mean term earns its
# keep without the program turning into a levered corner bet. Above 100 the mean is
# nearly inert (gamma = 1000 moves the Sharpe by 0.015 over mv_curr_overlay and the
# drawdown by 0.16pp), and below 10 the risk runs away faster than the return: from
# 10 to 5 the Sharpe already falls while volatility rises by half.
#
# Note that gamma = 10 is both the measured Sharpe optimum and the grid's lower
# endpoint, so the selection cannot bracket the optimum from below. In practice it
# does not try to: the gamma-free score leans conservative on this program and picks
# gamma = 100 on 87% of the 91 quarterly dates, never once reaching the lower
# endpoint (SR 0.2702, MDD 26.4%, against 0.2576 / 26.4% at a fixed gamma = 100).
# The binding endpoint is therefore the upper one. Scoring variants were measured
# and move which endpoint is preferred rather than making the search discriminate;
# mv_curr_overlay_w_mean.score records the comparison. Read the selected-gamma
# series accordingly -- as a statement about where the band is cut off, not as a
# fitted interior optimum.
#
# Three caveats worth carrying into the write-up. First, the Sharpe optimum and the
# drawdown optimum disagree: gamma = 50 cuts MDD to 25.4%, below the mean-free
# program's 28.2%, while gamma = 10 more than triples the Sharpe but takes MDD up to
# 33.4%. Selection is on the gamma-free score, not on either directly. Second, at
# the bottom of the band both caps bind (at gamma = 10, gross x 1.99 of 2.0 and gross
# psi 1.99 of 2.0), so the overlay there is a corner of the L1 ball -- its direction
# set by the carry ranking and its size by the cap rather than by the variance
# tradeoff. Over most of the band it is not: gross psi is 1.34 at gamma = 100.
# Third, and for the same reason, the peak is a joint property of gamma and the two
# caps rather than of gamma alone; re-swept caps would move it.
#
# The blow-up is also far milder here than for the raw-mean MV programs recorded
# below, which collapsed onto 19 of 101 names at gamma = 3 in the
# (gamma / 2) * Var - mu'w convention, i.e. gamma = 1.5 in this one. At gamma = 1.5
# here the program still holds ~40 names and a positive Sharpe. The pooling is what
# buys that margin -- and it is the pooling, not the shrinkage on top of it. Measured
# over the 127 estimation windows, the James-Stein weight on the economy panel has a
# median of 0.19 and only reaches 1.0 (mean killed outright) on 1.6% of them, so
# roughly four fifths of the raw economy mean survives into mu_a. Collapsing N ~ 116
# asset means onto at most 14 economy means is the step that does the work.
#
# Ordered descending to match ES_GAMMA_GRID.
MV_GAMMA_GRID = [100, 50, 30, 20, 10]

# Risk-aversion grid for the mean-bearing SMV overlay and joint programs
# (smv_curr_overlay_w_mean.py, smv_joint_w_mean.py). Calibrated on its own programs
# and its own scale; MV_GAMMA_GRID must not be reused. The objective is
# Risk - mu'w / gamma, and a daily semi-variance (~9e-6) sits below a daily variance
# (~2.6e-5), but the balancing gamma does not follow the level ratio -- it follows the
# gradient, and empirically SMV at gamma = 50 produces about the tilt MV shows at
# gamma = 100, so the useful band sits roughly a factor of two below MV's.
#
# Calibrated the same way MV_GAMMA_GRID was: the backtester at fixed gamma (a one-point
# grid, so no selection and the OOS result is attributable to gamma alone) on the top-10
# universe, quarterly, 10y window, leverage 2.0, overlay cap 2.0, over 2000-07 to
# 2023-04, Clarabel rather than Gurobi (~1e-6 weight difference).
#
#   OVERLAY                                                 JOINT
#   gamma   ES_80    Vol     MDD      SR     annRet  nAct  |  ES_80    Vol     MDD      SR     annRet  nAct
#     ---  0.6834   8.15   32.17   0.0827   2.11%  115.7  | 0.6571   7.84   32.63   0.0782   2.05%  119.3  <- mean-free, NO cap
#     1e7  0.6829   8.14   32.16   0.0877   2.15%  113.5  | 0.6562   7.84   32.61   0.0855   2.11%  118.9  <- mean off, cap ON
#     500  0.6836   8.16   31.76   0.1409   2.58%  113.9  | 0.6570   7.85   31.80   0.1441   2.57%  119.0
#     100  0.7012   8.39   29.25   0.2727   3.72%  113.3  | 0.6795   8.13   29.39   0.2822   3.73%  118.6
#      50  0.7447   8.93   28.17   0.3316   4.40%  110.7  | 0.7299   8.77   27.84   0.3387   4.40%  115.5  <- MDD min
#      30  0.8171   9.85   30.52   0.3810   5.19%  104.9  | 0.8108   9.81   30.17   0.3819   5.18%  109.1
#      20  0.9181  11.15   33.35   0.4092   6.00%   96.3  | 0.9189  11.22   32.76   0.3972   5.89%   98.9  <- joint SR peak
#      15  1.0171  12.44   34.89   0.4197   6.66%   87.3  | 1.0227  12.56   34.45   0.3967   6.42%   87.9  <- overlay SR peak
#      10  1.1992  14.81   39.52   0.3977   7.32%   72.5  | 1.2069  14.95   40.39   0.3752   7.04%   69.8
#       5  1.6172  20.45   48.92   0.4152   9.93%   49.8  | 1.6262  20.74   51.33   0.3867   9.46%   46.5
#       2  2.4569  35.99   76.92   0.2479  10.36%   30.6  | 2.4690  36.32   79.82   0.2045   8.86%   28.1  <- blow-up
#
# The Sharpe peaks at 15 (overlay, 0.4197) and 20 (joint, 0.3972), and both curves are
# flat across 15-20, so the grid brackets that plateau with 20 and 15 inside it, 30 and
# 50 above, 100 as the conservative anchor, and 10 one step below. The lower bound of 10
# is five times the gamma = 2 blow-up, where the Sharpe halves and drawdown reaches
# 77-80%. Between 15 and 5 the Sharpe is not monotone (overlay 0.420, 0.398, 0.415);
# that dip and recovery is inside the noise this backtest resolves -- three
# implementation variants of the MV score moved its Sharpe by +/-0.04 -- so it is read
# as a plateau, not as a second optimum, and 5 is excluded because its drawdown (49-51%)
# is not usable regardless.
#
# Two caveats, the first shared with MV_GAMMA_GRID and the second specific to SMV.
# The Sharpe optimum and the drawdown optimum disagree: gamma = 50 cuts MDD to 27.8-28.2%
# against 32.2-32.6% mean-free, while the Sharpe peak at 15-20 takes MDD back up to
# 33-35%. Second, the "---" row is the UNCAPPED mean-free program while every gamma row
# carries the 2.0 overlay cap, which binds on ~20% of windows even with the mean term
# off (see smv_curr_overlay_w_mean.py) -- so the two rows differ by the cap as well as
# by the mean. The 1e7 row isolates that: turning the cap on with the mean term off is
# worth +0.005 Sharpe (overlay) and +0.007 (joint), against +0.053 and +0.059 for the
# mean term alone at gamma = 500. The cap is therefore a negligible part of the effect
# despite binding on a fifth of windows, and the "---" row is a usable baseline after
# all; use the 1e7 row when the decomposition has to be exact.
SMV_GAMMA_GRID = [100, 50, 30, 20, 15, 10]

# Risk-aversion grid for the mean-bearing MAD overlay (mad_curr_overlay_w_mean.py).
# Calibrated on its own program; none of the other grids transfer, and the reason is
# structural rather than incidental. MAD is a DEGREE-ONE risk measure, like ES and
# unlike MV and SMV: the objective is Risk - mu'w / gamma with both terms homogeneous
# of degree one in w, so gamma is a pure information ratio. A daily MAD is ~3.1e-3
# against a daily variance of ~2.6e-5 and semi-variance of ~9e-6, which puts this band
# two to three decades below MV_GAMMA_GRID and SMV_GAMMA_GRID and close to
# ES_GAMMA_GRID's [10 ... 0.1]. The four grids look wildly different because two
# measures are degree one and two are degree two, not because any of them is
# miscalibrated.
#
# Calibrated the same way the others were: the backtester at fixed gamma (a one-point
# grid, so no selection and the OOS result is attributable to gamma alone) on the top-10
# universe, quarterly, 10y window, leverage 2.0, overlay cap 2.0, over 2000-07 to
# 2023-04, Clarabel rather than Gurobi (~1e-6 weight difference).
#
#   gamma   ES_80    Vol     MDD      SR       ESR    annRet   nAct   grossPsi
#     ---  0.6535   7.87   27.67   0.1601   0.1214    2.70%   118.7     0.945   <- mad_curr_overlay
#     100  0.6537   7.87   27.58   0.1619   0.1228    2.71%   118.7     0.947   <- mean term inert
#      10  0.6548   7.89   27.35   0.1712   0.1299    2.79%   118.6     0.924
#       3  0.6571   7.92   27.19   0.2015   0.1531    3.03%   118.7     0.996
#       1  0.6685   8.08   25.84   0.2817   0.2144    3.71%   118.4     1.270   <- MDD minimum
#     0.5  0.6898   8.31   26.52   0.3379   0.2565    4.24%   117.8     1.541
#     0.3  0.7348   8.86   28.82   0.3821   0.2901    4.82%   114.7     1.811
#     0.2  0.8262  10.10   33.22   0.4694   0.3613    6.17%   108.2     1.967   <- SR optimum
#    0.15  0.9629  11.96   38.58   0.4434   0.3469    6.74%    99.3     2.000
#     0.1  1.3066  16.99   48.74   0.4363   0.3574    8.85%    79.3     2.000   <- past gamma_crit
#
# The Sharpe curve is single-peaked at gamma = 0.2 and the grid brackets it: 0.3 above,
# 0.15 below, with 1 and 3 as the conservative end. 10 and 100 are dropped as inert --
# gamma = 100 moves the Sharpe by 0.002 over the mean-free program and the drawdown by
# 0.09pp.
#
# The lower bound of 0.15 is not a judgement call. MAD's degree-one homogeneity makes
# step 2 unbounded below for gamma < gamma_crit = max_d [mu_c'd / MAD(A_c d)], measured
# at 0.062, 0.078 and 0.110 on three windows. 0.15 is the last grid point above every
# measured gamma_crit, so the whole grid stays well posed on its own terms and the 2.0
# overlay cap is a safeguard rather than the only thing holding the program together.
# gamma = 0.1 sits below the largest gamma_crit and is excluded for that reason as well
# as on drawdown (48.7%).
#
# Two caveats, both shared with the MV and SMV grids. The Sharpe optimum and the
# drawdown optimum disagree: gamma = 1 cuts MDD to 25.8% against 27.7% mean-free while
# nearly doubling the Sharpe, whereas the peak at 0.2 takes MDD up to 33.2%. And the cap
# saturates at the aggressive end (gross psi 1.97 at gamma = 0.2, 2.00 at 0.15), so
# there psi is a corner of the L1 ball with its direction set by the carry ranking
# rather than by the risk-return tradeoff.
#
# For reference across measures, the peak Sharpe each mean-bearing overlay reaches on
# this universe: MV 0.481 (gamma 10), MAD 0.469 (gamma 0.2), SMV 0.420 (gamma 15),
# ES 0.154 (gamma 0.2). The three deviation measures land close together; raw ES is the
# outlier, which is consistent with its objective double-counting the mean.
MAD_GAMMA_GRID = [3, 1, 0.5, 0.3, 0.2, 0.15]

# Risk-aversion grid for the mean-bearing SOES overlay (soes_curr_overlay_w_mean.py).
# Calibrated on its own program. HMCR_2 is DEGREE-ONE, like ES and MAD and unlike MV
# and SMV, so gamma is a pure information ratio and the band sits two to three decades
# below MV_GAMMA_GRID and SMV_GAMMA_GRID -- close to ES_GAMMA_GRID and MAD_GAMMA_GRID.
#
# Calibrated the same way the others were: the backtester at fixed gamma (a one-point
# grid, so no selection and the OOS result is attributable to gamma alone) on the top-10
# universe, quarterly, 10y window, leverage 2.0, overlay cap 2.0, over 2000-07 to
# 2023-04, Clarabel rather than Gurobi. The "---" baseline is soes_curr_overlay with
# demean=False, which is the program this one collapses to as gamma -> infinity; note
# that is NOT soes_curr_overlay's own default, since the mean-bearing version does not
# demean (see soes_curr_overlay_w_mean.py).
#
#   gamma   ES_80    Vol     MDD      SR       ESR    annRet   nAct   grossPsi
#     ---  0.7668   9.17   34.43   0.0748   0.0563    2.12%    83.3     2.043   <- mean-free
#       3  0.7650   9.15   34.50   0.0949   0.0715    2.30%    83.4     1.708
#       1  0.7653   9.16   34.18   0.1239   0.0935    2.57%    83.5     1.638
#     0.5  0.7666   9.19   33.77   0.1648   0.1244    2.95%    83.5     1.604
#     0.3  0.7699   9.24   33.41   0.2036   0.1540    3.32%    83.1     1.654
#     0.2  0.7768   9.33   32.92   0.2375   0.1797    3.65%    82.5     1.740
#     0.1  0.8103   9.73   30.73   0.2909   0.2201    4.27%    80.6     1.909   <- SR and MDD optimum
#    0.05  0.9491  11.49   32.73   0.2797   0.2134    4.65%    72.0     1.999
#    0.04  1.0522  12.90   36.49   0.2632   0.2032    4.83%    66.1     2.000
#    0.03  1.2252  15.09   40.76   0.2701   0.2095    5.51%    57.4     2.000
#
# The Sharpe peaks at gamma = 0.1 and the grid brackets it: 0.2 above, 0.05 below, with
# 3 down to 0.3 as the conservative end. Unusually for this family the Sharpe optimum
# and the drawdown optimum COINCIDE at 0.1 -- MDD falls monotonically from 34.4% to
# 30.7% and only deteriorates below the peak. Every other measure has them disagree,
# with the drawdown minimum sitting one to three grid points more conservative than the
# Sharpe peak.
#
# The lower bound of 0.05 is set by the divergence threshold, not by taste. HMCR_2's
# degree-one homogeneity makes step 2 unbounded below for
# gamma < gamma_crit = max_d [mu_c'd / HMCR_2(-(R_c d))], measured at 0.0155, 0.0224 and
# 0.0264 on the raw return matrices this program uses. 0.05 keeps a 1.9x margin over the
# worst of those; 0.04 and 0.03 cut it to 1.5x and 1.14x and are past the peak anyway,
# with drawdowns of 36.5% and 40.8%.
#
# For reference across measures, the peak Sharpe each mean-bearing overlay reaches on
# this universe: MV 0.481 (gamma 10), MAD 0.469 (0.2), SMV 0.420 (15), SOES 0.291 (0.1),
# ES 0.154 (0.2). The three deviation measures cluster at the top; the two undemeaned
# ones -- ES and SOES -- sit well below, which is consistent with their objectives
# double-counting the mean (see soes_curr_overlay_w_mean.py).
SOES_GAMMA_GRID = [3, 1, 0.5, 0.3, 0.2, 0.1, 0.05]



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

# Mean-bearing regularized ES overlay and joint programs. Both reuse the ES penalty
# grids calibrated on their mean-free counterparts, as REG_OVERLAY_CONFIG and
# REG_JOINT_CONFIG do; the risk term is the same measure at the same scale, so the
# bounds transfer. Note the calibration was done without a mean term competing for
# the objective, so the penalty that is optimal here may sit lower than it does
# there -- worth re-reading the selected-lambda series once these have run.
#
# gamma is deliberately NOT in either grid. Both configs already have
# 4 x 4 x 4 x 4 = 256 combinations per rebalancing date, and crossing that with
# ES_GAMMA_GRID's seven points is not affordable at these universe sizes. The
# optimizers default gamma to 2 and still read it from hyperparams, so it can be
# swept deliberately later without touching their code. 2 is a conservative choice
# rather than a calibrated one: the unregularized sweeps favoured 0.2 (overlay) and
# 1.0 (joint), but with no penalties competing against the mean term.
REG_ES_MEAN_OVERLAY_CONFIG = BacktestConfig(
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
        "overlay_limit":   OVERLAY_LIMIT,
    },
)

REG_ES_MEAN_JOINT_CONFIG = BacktestConfig(
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
        "overlay_limit":   OVERLAY_LIMIT,
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

# Mean-bearing SMV overlay and joint programs. Carry the overlay L1 cap and the SMV
# risk-aversion grid on top of the leverage cap. No `demean` entry: the semi-variance
# is already a deviation measure here, since the return matrices are demeaned before
# the program is formed, so the switch the ES overlays carry has no analogue.
SMV_MEAN_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          SMV_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
    },
)

SMV_MEAN_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          SMV_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
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

# Mean-bearing regularized SMV overlay and joint programs. Both reuse the SMV penalty
# grids calibrated on their mean-free counterparts, as REG_SMV_OVERLAY_CONFIG and
# REG_SMV_JOINT_CONFIG do; the risk term is the same measure at the same scale, so the
# bounds transfer. As with the MV pair, the calibration was done without a mean term
# competing for the objective, and the MV production runs showed the mean-bearing
# programs then select STRONGER penalties, increasingly so with universe size -- watch
# the selected-lambda series here for the same, and for endpoint clustering.
#
# gamma is deliberately NOT in either grid, exactly as in REG_MV_MEAN_OVERLAY_CONFIG.
# Both configs already have 4 x 4 x 4 x 4 = 256 combinations per rebalancing date, and
# crossing that with SMV_GAMMA_GRID's six points is not affordable -- still less at SMV's
# solve cost, which is ~50x a mean-variance one. The optimizers default gamma to 30 and
# still read it from hyperparams, so it can be swept deliberately later. 30 is
# calibrated: on the fixed-gamma sweeps of both unregularized SMV programs it is the
# most aggressive grid point that still improves drawdown over the mean-free program
# (MDD 30.5% overlay and 30.2% joint, against 32.2% and 32.6%), while raising the Sharpe
# from 0.083 to 0.381 and from 0.078 to 0.382. Note that 30 sits closer to the SMV
# Sharpe peak (15-20) than the same number does on the MV curve (peak 10), so this pair
# runs a relatively more aggressive tilt than the MV pair at the same nominal gamma.
# gamma is not comparable across risk measures.
#
# Both also carry OVERLAY_LIMIT, which their mean-free counterparts do not. Unlike the
# MV pair this is not free: the mean-free SMV programs already reach gross psi of 2.5-2.9
# on some windows, so the cap binds on ~20% of them even with the mean term off, and
# these are not strict supersets of their mean-free counterparts. Measured on the
# unregularized pair the effect is small -- turning the cap on with the mean term off is
# worth +0.005 (overlay) and +0.007 (joint) Sharpe, against +0.053 and +0.059 for the
# mean term alone.
REG_SMV_MEAN_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_SMV_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_SMV_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_SMV_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_SMV_CURR_GRID,
        "leverage_limit":   LEVERAGE,
        "overlay_limit":    OVERLAY_LIMIT,
    },
)

REG_SMV_MEAN_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_SMV_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_SMV_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_SMV_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_SMV_CURR_GRID,
        "leverage_limit":   LEVERAGE,
        "overlay_limit":    OVERLAY_LIMIT,
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

# Mean-bearing MV overlay. Carries the overlay L1 cap and the MV risk-aversion grid
# on top of MV_OVERLAY_CONFIG's leverage cap. No `demean` entry: variance is already
# location-free, so the switch the ES overlays carry has no analogue here.
MV_MEAN_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          MV_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
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

# Mean-bearing regularized MV overlay and joint programs. Both reuse the MV penalty
# grids calibrated on their mean-free counterparts, as REG_MV_OVERLAY_CONFIG and
# REG_MV_JOINT_CONFIG do; the risk term is the same measure at the same scale, so the
# bounds transfer. The calibration was done without a mean term competing for the
# objective, and the selected-lambda series from the first production runs settles which
# way that cuts: the mean-bearing programs select STRONGER penalties, not weaker, and
# increasingly so with universe size. The mean term pushes the weights out and the
# penalties are what pull them back, so there is more for them to restrain. Median
# selected lambda_l1_asset goes 1e-6 -> 5e-6 at top-50 and top-100; median
# lambda_l1_curr goes 0 -> 5e-7 at both; median lambda_l2_curr goes 0 -> 5e-6 at top-100.
#
# One consequence needs acting on before these grids are reused. At top-100 the
# mean-bearing overlay selects the TOP of LAMBDA_L1_MV_ASSET_GRID (5e-5) on 43 of 91
# rebalancing dates (47%), against 10 of 91 for its mean-free counterpart. That is
# endpoint clustering: the calibrated upper bound was set for a program with no mean
# term to restrain, and the mean-bearing one wants to go past it. LAMBDA_L1_MV_ASSET_GRID
# should be extended upward before the top-100 mean-bearing results are read as a
# selected optimum rather than a truncated one. lambda_l1_curr also reaches its top on
# 18 of 91 dates at top-50 and top-100 (20%), which is worth watching but not yet
# clustering; the two L2 grids are comfortably interior everywhere.
#
# gamma is deliberately NOT in either grid, exactly as in REG_ES_MEAN_OVERLAY_CONFIG.
# Both configs already have 4 x 4 x 4 x 4 = 256 combinations per rebalancing date, and
# crossing that with MV_GAMMA_GRID's five points is not affordable at these universe
# sizes. The optimizers default gamma to 30 and still read it from hyperparams, so it
# can be swept deliberately later without touching their code. Unlike the ES pair's
# conservative choice of 2, 30 is calibrated: it is the centre point of MV_GAMMA_GRID,
# and on the fixed-gamma sweeps of both unregularized MV programs it improves on the
# mean-free program along both axes at once -- Sharpe 0.372 against 0.140 (overlay) and
# 0.373 against 0.162 (joint), at drawdowns of 26.6% and 25.9% against 28.2% and 27.9%.
# It sits one step more aggressive than the drawdown minimum at gamma = 50 and one step
# short of gamma = 20, where both caps saturate. Those sweeps had no penalties competing
# against the mean term, so it is a starting point rather than a jointly optimal one.
#
# Both also carry OVERLAY_LIMIT, which their mean-free counterparts do not. At gamma =
# 30 it binds rarely: over the full 256-point grid on three test windows the overlay
# never reaches it outright (widest solution 1.9999) and the joint sits on it at 3.3% of
# points, all in the lightly-penalized corner where nothing else restrains psi. By
# gamma = 10 it is the only thing between psi and the double-digit gross notional
# recorded above for the discarded mean-bearing MV programs (74x - 98x). Where it binds,
# psi sits on a corner of the L1 ball with its direction set by the carry ranking rather
# than by the variance-mean tradeoff; the per-window table is in
# reg_mv_curr_overlay_w_mean.py.
REG_MV_MEAN_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_MV_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_MV_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_MV_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_MV_CURR_GRID,
        "leverage_limit":   LEVERAGE,
        "overlay_limit":    OVERLAY_LIMIT,
    },
)

REG_MV_MEAN_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_MV_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_MV_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_MV_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_MV_CURR_GRID,
        "leverage_limit":   LEVERAGE,
        "overlay_limit":    OVERLAY_LIMIT,
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

# Mean-bearing MAD joint program. Reuses MAD_GAMMA_GRID, and as with the MV pair that
# reuse is measured rather than assumed -- the joint program was swept on its own:
#
#   gamma   ES_80    Vol     MDD      SR       ESR    annRet   nAct   grossPsi
#     ---  0.6384   7.71   27.31   0.1804   0.1372    2.83%   119.7     1.356   <- mad_joint
#      10  0.6386   7.72   27.57   0.1941   0.1477    2.93%   119.7     1.306   <- nearly inert
#       3  0.6401   7.75   26.71   0.2435   0.1856    3.32%   119.7     1.342
#       1  0.6492   7.88   25.43   0.3225   0.2466    3.98%   119.7     1.506   <- MDD minimum
#     0.5  0.6712   8.13   25.58   0.3749   0.2861    4.48%   119.4     1.713
#     0.3  0.7291   8.84   28.09   0.3876   0.2959    4.86%   117.0     1.894
#     0.2  0.8389  10.33   33.73   0.4327   0.3357    5.91%   110.4     1.990   <- SR optimum
#    0.15  1.0033  12.66   40.37   0.3885   0.3089    6.35%    98.8     2.000
#
# The curve lands on the overlay's: same single peak at gamma = 0.2, same drawdown
# minimum at gamma = 1, same near-inert conservative end at 10. So MAD_GAMMA_GRID
# transfers unchanged, and its lower bound of 0.15 -- the last point above every
# measured gamma_crit (0.062, 0.078, 0.110, identical for the joint) -- applies here
# for the same reason.
#
# The cap is not optional: MAD is degree-one and x is confined to a compact set, so psi
# is the only runaway direction and the program is unbounded below without it for gamma
# under ~0.1. It does more work here than in the overlay -- uncapped, the joint reaches
# gross psi of 4.7 at gamma = 1 and 22.7 at 0.12, well above gamma_crit, where the
# overlay is still below 0.5; mad_joint_w_mean.py records the measurements.
#
# Note the joint does not beat the overlay at either program's optimum (0.433 against
# 0.469 at gamma = 0.2), the same ordering the MV and SMV pairs show.
MAD_MEAN_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          MAD_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
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

# Mean-bearing MAD overlay. Carries the overlay L1 cap and the MAD risk-aversion grid
# on top of MAD_OVERLAY_CONFIG's leverage cap. The cap is not optional here: MAD is
# degree-one, so step 2 is unbounded below without it for gamma under ~0.1. No
# `demean` entry: the MAD is already a deviation measure, formed on demeaned return
# matrices, so the switch the ES overlays carry has no analogue.
MAD_MEAN_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          MAD_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
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

# Mean-bearing regularized MAD overlay and joint programs. Both reuse the MAD penalty
# grids calibrated on their mean-free counterparts, as REG_MAD_OVERLAY_CONFIG and
# REG_MAD_JOINT_CONFIG do; the risk term is the same measure at the same scale, so the
# bounds transfer. As with the MV pair, the calibration was done without a mean term
# competing for the objective, and the MV production runs showed the mean-bearing
# programs then select STRONGER penalties, increasingly so with universe size -- watch
# the selected-lambda series here for the same, and for endpoint clustering.
#
# gamma is deliberately NOT in either grid, exactly as in REG_MV_MEAN_OVERLAY_CONFIG and
# REG_SMV_MEAN_OVERLAY_CONFIG. Both configs already have 4 x 4 x 4 x 4 = 256
# combinations per rebalancing date, and crossing that with MAD_GAMMA_GRID's six points
# is not affordable. The optimizers default gamma to 0.5 and still read it from
# hyperparams. 0.5 is calibrated by the same criterion that picked 30 for the MV and
# SMV pairs: on the fixed-gamma sweeps of both unregularized MAD programs it is the most
# aggressive grid point that still improves drawdown over the mean-free program (MDD
# 26.5% overlay and 25.6% joint, against 27.7% and 27.3%), while more than doubling the
# Sharpe -- 0.160 to 0.338 and 0.180 to 0.375.
#
# gamma is not comparable across risk measures. MAD is degree one, like ES and unlike MV
# and SMV, so its band sits two to three decades below theirs; 0.5 here and 30 there are
# comparable positions on their own curves, not comparable numbers.
#
# Both also carry OVERLAY_LIMIT, and here it is not the safeguard it is for the MV and
# SMV pairs. MAD is degree-one, so the overlay block is unbounded below for gamma under
# gamma_crit = max_d [mu_c'd / MAD(A_c d)], measured at 0.062 - 0.110. The shipped 0.5
# clears that by ~4.5x for the overlay, but the JOINT program is far more fragile: at
# finite scale its asset block partially offsets the carry direction, and uncapped it
# reaches gross psi of 4.7 at gamma = 1 while the solver reports it unbounded at gamma =
# 0.5, the shipped value. For reg_mad_joint_w_mean the cap is what makes the shipped
# configuration solvable at all. The penalties do not substitute for it: the
# lambda_l1_curr = lambda_l2_curr = 0 corner is in the grid.
REG_MAD_MEAN_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_MAD_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_MAD_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_MAD_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_MAD_CURR_GRID,
        "leverage_limit":   LEVERAGE,
        "overlay_limit":    OVERLAY_LIMIT,
    },
)

REG_MAD_MEAN_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "lambda_l1_asset":  LAMBDA_L1_MAD_ASSET_GRID,
        "lambda_l2_asset":  LAMBDA_L2_MAD_ASSET_GRID,
        "lambda_l1_curr":   LAMBDA_L1_MAD_CURR_GRID,
        "lambda_l2_curr":   LAMBDA_L2_MAD_CURR_GRID,
        "leverage_limit":   LEVERAGE,
        "overlay_limit":    OVERLAY_LIMIT,
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

# Mean-bearing MV joint program. Reuses MV_GAMMA_GRID, and unlike ES_MEAN_JOINT_CONFIG
# -- which reuses ES_GAMMA_GRID on the argument that the band should be the right order
# of magnitude and flags it for recalibration -- that reuse is measured here rather
# than assumed. The ES pair gave reason to doubt it would transfer: its unregularized
# sweeps favoured gamma = 0.2 for the overlay and 1.0 for the joint, a factor of five
# in one risk measure, on the argument that choosing x and psi against a single risk
# term rather than in sequence makes the mean pull harder at a given gamma.
#
# It transfers here. Same fixed-gamma protocol as MV_GAMMA_GRID (one-point grid, top-10,
# quarterly, 10y window, leverage 2.0, overlay cap 2.0, 2000-07 to 2023-04, Clarabel):
#
#   gamma   ES_80    Vol     MDD      SR       ESR    annRet   nAct   grossPsi
#     ---  0.6439   7.73   27.92   0.1621   0.1226    2.69%   119.7     1.273   <- mv_joint
#    1000  0.6441   7.73   27.78   0.1758   0.1329    2.79%   119.7     1.221
#     200  0.6462   7.76   27.22   0.2264   0.1713    3.19%   119.7     1.380
#     100  0.6516   7.82   26.56   0.2658   0.2010    3.51%   119.6     1.535
#      50  0.6671   8.02   25.32   0.3189   0.2414    3.99%   119.1     1.715   <- MDD minimum
#      30  0.6970   8.40   25.92   0.3732   0.2833    4.57%   117.9     1.835
#      20  0.7438   9.00   27.95   0.4206   0.3206    5.22%   115.0     1.909
#      10  0.9060  11.14   33.21   0.4732   0.3666    6.71%   101.6     1.994   <- SR optimum
#       5  1.2001  15.01   41.61   0.4378   0.3449    8.01%    74.3     2.000
#       2  1.8197  23.93   60.16   0.4064   0.3366   11.16%    45.3     2.000
#
# The curve is the overlay's to within noise at every point -- Sharpe 0.473 against
# 0.481 at the optimum, 0.319 against 0.317 at gamma = 50 -- with the same interior
# peak at 10, the same drawdown minimum at 50, and the same saturation of both caps
# below 20. So the [10, 100] band needs no adjustment, and the ES pair's five-fold
# shift does not repeat in this measure.
#
# That similarity is itself a result rather than a formality: on the top-10 universe
# the joint program buys almost nothing over the sequential overlay once a mean term
# is present, which is the comparison in_sample_vs_out_of_sample_w_mean.py exists to
# quantify across universe sizes. Note it is not a pure sequencing contrast -- the two
# also differ in how the asset-currency cross-covariance is estimated (raw sample in
# the overlay, shrunk within the stacked Ledoit-Wolf here), inherited from the mean-free
# pair and documented in mv_joint_w_mean.py.
MV_MEAN_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "gamma":          MV_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
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

# Mean-bearing SOES overlay. Carries the overlay L1 cap and the SOES risk-aversion
# grid on top of SOES_OVERLAY_CONFIG's alpha and leverage cap. No `demean` entry,
# deliberately: centering would zero the sample means this program is built around,
# exactly as in ES_MEAN_OVERLAY_CONFIG. The cap is not optional - HMCR_2 is degree-one,
# so step 2 is unbounded below without it for gamma under ~0.026.
SOES_MEAN_OVERLAY_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "gamma":          SOES_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
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

# Mean-bearing regularized SOES overlay and joint programs. Both reuse the SOES penalty
# grids calibrated on their mean-free counterparts; the risk term is the same measure at
# the same scale, so the bounds transfer. As with the MV pair, the calibration was done
# without a mean term competing for the objective, and the MV production runs showed the
# mean-bearing programs then select STRONGER penalties, increasingly so with universe
# size -- watch the selected-lambda series here for the same, and for endpoint clustering.
#
# No `demean` entry in either grid, deliberately: centering would zero the sample means
# these programs are built around, exactly as in the ES mean-bearing configs.
#
# gamma is NOT in either grid. Both already carry 4 x 4 x 4 x 4 = 256 combinations per
# rebalancing date, and crossing that with SOES_GAMMA_GRID's seven points is not
# affordable at SOES's SOCP cost. The optimizers default gamma to 0.5 and still read it
# from hyperparams. 0.5 is a CONSERVATIVE choice rather than the swept optimum, and more
# conservative in relative terms than the values fixed for the other measures: on the
# unregularized sweeps it gives Sharpe 0.165 (overlay) and 0.118 (joint) against peaks of
# 0.291 at gamma = 0.1 and 0.234 at gamma = 0.05. For the overlay the Sharpe peak and the
# drawdown minimum coincide at 0.1, so the criterion that picked 30 for MV/SMV and 0.5
# for MAD -- the most aggressive point still improving drawdown -- would point at 0.1
# here. For the joint no gamma improves drawdown at all, so that criterion does not apply.
#
# Both carry OVERLAY_LIMIT, and it is not the safeguard it is for the MV and SMV pairs.
# HMCR_2 is degree-one, so the overlay block is unbounded below for gamma under
# gamma_crit = max_d [mu_c'd / HMCR_2(-(R_c d))] = 0.0155 - 0.0264. The shipped 0.5
# clears that by ~19x for the overlay. The JOINT is a different matter: uncapped, the
# unregularized version is reported unbounded by the solver at every gamma down from 1,
# including 0.5 itself, because the asset block partially offsets the carry direction at
# finite scale. For reg_soes_joint_w_mean the cap is what makes the shipped configuration
# solvable at all. The penalties do not substitute: the zero-zero currency corner is in
# the grid.
#
# Neither is a strict superset of its mean-free counterpart. Mean-free undemeaned SOES
# runs a gross psi of 0.62 - 2.99 (overlay) and 1.61 - 3.90 (joint) across three windows,
# so the 2.0 cap binds on two of three even with the mean term off -- the largest
# overshoot in the suite.
REG_SOES_MEAN_OVERLAY_CONFIG = BacktestConfig(
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
        "overlay_limit":    OVERLAY_LIMIT,
    },
)

REG_SOES_MEAN_JOINT_CONFIG = BacktestConfig(
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
        "overlay_limit":    OVERLAY_LIMIT,
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

# Mean-bearing SOES joint program. Reuses SOES_GAMMA_GRID, and as with the MAD pair
# that reuse is measured rather than assumed -- the joint was swept on its own, with
# soes_joint(demean=False) as the baseline it collapses to:
#
#   gamma   ES_80    Vol     MDD      SR       ESR    annRet   nAct   grossPsi
#     ---  0.7346   8.72   35.38   0.0216   0.0161    1.62%    92.9     2.986   <- mean-free, UNCAPPED
#     1e9  0.7293   8.66   36.31   0.0556   0.0418    1.92%    93.1     1.868   <- mean OFF, cap ON
#       3  0.7302   8.67   36.31   0.0679   0.0508    2.02%    93.0     1.884
#       1  0.7330   8.70   36.56   0.0882   0.0660    2.20%    92.6     1.893
#     0.5  0.7370   8.75   37.06   0.1181   0.0883    2.47%    92.4     1.889
#     0.3  0.7453   8.83   37.73   0.1495   0.1116    2.76%    91.8     1.947
#     0.2  0.7584   8.98   38.90   0.1716   0.1279    2.98%    90.9     1.964
#     0.1  0.8140   9.69   39.32   0.2290   0.1716    3.65%    87.0     1.995
#    0.05  0.9889  11.94   35.71   0.2335   0.1776    4.22%    74.0     2.000   <- SR optimum
#    0.04  1.1142  13.63   39.70   0.1925   0.1484    4.06%    66.3     2.000
#
# The Sharpe peaks at gamma = 0.05, but 0.1 (0.2290) is within noise of it -- the gap of
# 0.0045 is an order of magnitude below the +/-0.04 sensitivity to implementation
# choices measured on the MV pair -- so the top of this curve is a 0.1-0.05 plateau
# rather than a point. SOES_GAMMA_GRID's [3 ... 0.05] covers it unchanged, and its floor
# of 0.05 keeps the same 1.89x margin over the worst measured gamma_crit (0.0264).
# gamma = 0.04 is excluded on both counts: its Sharpe (0.1925) is clearly off the
# plateau and its margin falls to 1.5x.
#
# Two things distinguish this program from every other joint in the suite, and both
# belong in the write-up. Drawdown gets monotonically WORSE across most of the band
# (35.4% at the mean-free end to 39.3% at gamma = 0.1) rather than improving first --
# so unlike the MV, SMV and MAD joints there is no region that beats the mean-free
# program on both axes. And the "---" row is not a clean baseline: it is the UNCAPPED
# mean-free joint, whose gross psi of 2.986 exceeds the 2.0 cap every mean-bearing row
# carries. The 1e9 row isolates that, and here -- unlike the SMV pair, where the cap was
# worth +0.005 Sharpe -- the cap is most of the story at the conservative end. Imposing
# it with the mean term off is worth +0.034 Sharpe (0.0216 -> 0.0556) and cuts gross psi
# from 2.99 to 1.87. So of the +0.046 jump from "---" to gamma = 3, roughly three
# quarters is the cap and one quarter the mean term. That reverses at the optimum: at
# gamma = 0.05 the mean term is worth +0.178 against the cap's +0.034. USE THE 1e9 ROW,
# not the "---" row, as the baseline for what the mean term buys.
#
# The cap is not optional here. HMCR_2 is degree-one and x is confined to a compact set,
# so psi is the only runaway direction and the asymptotic threshold is
# gamma_crit = max_d [mu_c'd / HMCR_2(-(R_c d))] = 0.0155 - 0.0264, identical to the
# overlay's. But the asymptote badly understates the case: uncapped, this program is
# reported unbounded by the solver at EVERY gamma tested down from 1 -- 38x above
# gamma_crit -- where the uncapped overlay stayed bounded to 0.02. The cap is what makes
# the whole band solvable, not a safeguard near the threshold.
SOES_MEAN_JOINT_CONFIG = BacktestConfig(
    window_years=10,
    val_years=1,
    rebalancing_frequency="quarterly",
    param_grid={
        "alpha":          ALPHA,
        "gamma":          SOES_GAMMA_GRID,
        "leverage_limit": LEVERAGE,
        "overlay_limit":  OVERLAY_LIMIT,
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
