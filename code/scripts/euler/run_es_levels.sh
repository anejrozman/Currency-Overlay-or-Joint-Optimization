#!/bin/bash
# Submit the ES-level sweep to the Euler cluster.
#
# Runs the two UNREGULARIZED Expected Shortfall strategies -- es_curr_overlay
# (two-step: assets, then overlay) and es_joint (assets and overlay in one
# program) -- at six confidence levels on all three universes:
#
#   6 levels (50/60/70/80/90/95) x 2 strategies x 3 universes = 36 jobs
#
# Each job is one sbatch submission via run_backtest.sh, so the scheduler spreads
# them across nodes and they run concurrently; at 4 cores each the sweep needs
# ~144 of the ls_math share's 640 cores, so the wall clock is set by the slowest
# single job (es_joint at top-100).
#
# What is pinned, and why
# -----------------------
# backtest_config.py fixes ALPHA = [0.80] for every strategy, so the level cannot
# be varied from the registry. --alpha pins the 'alpha' axis of the strategy's
# grid to one value per job; --no-demean pins 'demean' to False, overriding the
# global DEMEAN = True. Both land in the run's config.json under "grid_overrides",
# which is how collect_es_level_runs.py later tells a swept run from a default one.
#
# The grids of OVERLAY_CONFIG and JOINT_CONFIG are alpha x leverage x demean, and
# all three axes are singletons once alpha and demean are pinned. So each job's
# "grid search" is a single combination: the backtester still solves the train and
# validation windows before refitting on the full window (3 solves per rebalancing
# date x 91 dates), but nothing is actually being selected. Two consequences:
#
#   * do NOT substitute --no-grid for --alpha. --no-grid drops the grid entirely
#     and the optimizers fall back to their own alpha = 0.80 default, which would
#     silently produce six identical runs per cell.
#   * the val_score column in selected_params.csv is recorded, never used. It is
#     also not comparable across levels: ES at 50% and ES at 95% are different
#     functionals, not two scores of one objective.
#
# Statistical caveat at the top end
# ---------------------------------
# These programs are unregularized; the only thing bounding the asset block is the
# ||x||_1 <= 2 leverage cap. The estimation window is ~2500 days while the top-100
# universe reaches 1161 names, so the level-95 tail is ~125 observations against
# ~1161 weights -- the LP optimum then sits on a vertex pinned by a handful of tail
# days, in-sample ES is driven near zero, and the weights swing hard quarter to
# quarter. Read the top-50 and top-100 cells at levels 90/95 as a diagnostic of
# exactly the estimation-error blowup the regularized variants exist to fix, not as
# a deployable strategy. Top-10 (133 names against 125 tail days) is borderline.
# Expect some "optimal_inaccurate" solver statuses in those cells.
#
# Run directories
# ---------------
# next_run_dir() names directories after the optimizer FILE, not the strategy plus
# level, so all 18 overlay runs land in results/runs/es_curr_overlay/ alongside the
# 15 runs already there, and the 18 joint runs in results/runs/es_joint/. Run
# numbers are claimed by an atomic mkdir when each job starts, so which number goes
# to which level depends on scheduling order and cannot be predicted here. Recover
# the mapping after the fact with:
#
#   python code/scripts/collect_es_level_runs.py            # print the table
#   python code/scripts/collect_es_level_runs.py --write    # update the CSV
#
# Usage:
#   ./run_es_levels.sh              # submit all 36 jobs
#   ./run_es_levels.sh --dry-run    # print the sbatch plan, submit nothing
#   ./run_es_levels.sh --demean     # sweep with demean=True instead of False

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="$HERE/run_backtest.sh"

DRY_RUN=0
DEMEAN_FLAG="--no-demean"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --demean)  DEMEAN_FLAG=""; shift ;;
        -h|--help)
            sed -n '2,60p' "${BASH_SOURCE[0]}"
            exit 1 ;;
        *) echo "Error: unknown argument '$1' (expected --dry-run, --demean or --help)."; exit 1 ;;
    esac
done

LEVELS=(50 60 70 80 90 95)
STRATEGIES=(es_curr_overlay es_joint)
UNIVERSES=(100 50 10)   # heaviest universe first, so the long poles start first

# Same tiers as run_all_backtests.sh gives its light strategies. The LP has q + N + 1
# variables and q constraints regardless of the level -- only the 1/(q(1-alpha))
# coefficient on the objective changes -- so cost scales with the universe, not with
# the level, and one tier per universe is enough.
mem_for() {   # mem_for <universe>
    case "$1" in
        10)  echo "8G"  ;;
        50)  echo "16G" ;;
        100) echo "32G" ;;
    esac
}

time_for() {  # time_for <universe>
    case "$1" in
        100) echo "336:00:00" ;;
        *)   echo "150:00:00" ;;
    esac
}

count=0
for uni in "${UNIVERSES[@]}"; do
    for strat in "${STRATEGIES[@]}"; do
        for level in "${LEVELS[@]}"; do
            alpha="$(awk -v l="$level" 'BEGIN { printf "%.2f", l / 100 }')"
            t="$(time_for "$uni")"
            m="$(mem_for "$uni")"
            if (( DRY_RUN )); then
                printf '  %-16s level=%-3s alpha=%-5s u%-4s time=%-10s mem=%-4s cpus=4 %s\n' \
                    "$strat" "$level" "$alpha" "$uni" "$t" "$m" "$DEMEAN_FLAG"
            else
                "$RUN_ONE" --strategy "$strat" --universe "$uni" --alpha "$alpha" \
                    ${DEMEAN_FLAG:+$DEMEAN_FLAG} --time "$t" --mem-per-cpu "$m" --cpus 4
            fi
            count=$((count + 1))
        done
    done
done

echo ""
if (( DRY_RUN )); then
    echo "${count} jobs planned (dry run, nothing submitted)."
else
    echo "${count} jobs submitted. Track with: squeue --me"
    echo "When they finish: python code/scripts/collect_es_level_runs.py --write"
fi
