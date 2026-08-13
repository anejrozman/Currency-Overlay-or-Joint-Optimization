#!/bin/bash
# Submit ALL backtests to the Euler cluster: every strategy in the registry, on
# every universe (10/50/100), each over its full parameter grid.
#
# 13 strategies x 3 universes = 39 independent jobs. Each is submitted as its own
# SLURM job so the scheduler spreads them across nodes and they run concurrently.
#
# Node allocation
# ---------------
# Every job is a CPU-bound Gurobi QP/LP solve; GPUs give no speed-up, so no GPU is
# requested and all jobs target the high-throughput CPU nodes of the ls_math share
# (5 nodes x 128 cores / 256 GB = 640 cores, 1280 GB). At ~4 cores each the 39 jobs
# need ~156 cores, so they all run at once and the total wall-clock time is set by
# the single slowest job (the large-grid regularized overlay/joint runs at top-100).
#
# Two levers are used to keep the makespan low:
#   * heaviest jobs are submitted first (longest-processing-time first), so the
#     long-pole jobs start immediately and are never stuck behind short ones;
#   * memory is tiered by universe + weight so light jobs pack densely and only the
#     genuinely large runs reserve big memory.
#
# Note: each job is internally sequential over its grid (Gurobi is pinned to 3
# threads), so adding cores to a single job does not speed it up. The only way to
# shorten the long pole further is to parallelise the grid across jobs (code change).
#
# Usage:
#   ./run_all_backtests.sh           # submit everything
#   ./run_all_backtests.sh --dry-run # print the sbatch plan without submitting

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="$HERE/run_backtest.sh"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# All strategies (strategy_registry.py). Heavy ones (large grids / large programs)
# are listed first so they are submitted first.
HEAVY=(reg_mv_curr_overlay reg_mv_joint reg_es_curr_overlay reg_es_joint reg_mv_asset_full reg_mv_asset_zero)
LIGHT=(reg_es_asset_full reg_es_asset_zero es_curr_overlay es_joint mv_curr_overlay mv_joint equally_weighted)

UNIVERSES=(100 50 10)   # heaviest universe first

is_heavy() {
    local s
    for s in "${HEAVY[@]}"; do [[ "$s" == "$1" ]] && return 0; done
    return 1
}

mem_for() {  # mem_for <strategy> <universe>
    local heavy=1; is_heavy "$1" || heavy=0
    case "$2" in
        10)  echo "8G" ;;
        50)  (( heavy )) && echo "32G" || echo "16G" ;;
        100) (( heavy )) && echo "64G" || echo "32G" ;;
    esac
}

time_for() {  # time_for <strategy> <universe>
    if is_heavy "$1" && [[ "$2" == 100 ]]; then echo "336:00:00"
    elif is_heavy "$1" || [[ "$2" == 100 ]];  then echo "336:00:00"
    else echo "150:00:00"; fi
}

submit() {  # submit <strategy> <universe>
    local strat="$1" uni="$2"
    local t m
    t="$(time_for "$strat" "$uni")"
    m="$(mem_for "$strat" "$uni")"
    if (( DRY_RUN )); then
        printf '  %-22s u%-3s  time=%-9s mem=%-4s cpus=4\n' "$strat" "$uni" "$t" "$m"
    else
        "$RUN_ONE" --strategy "$strat" --universe "$uni" --time "$t" --mem-per-cpu "$m" --cpus 4
    fi
}

count=0
# Longest-processing-time-first: all heavy strategies (uni 100 -> 10), then light.
for uni in "${UNIVERSES[@]}"; do
    for strat in "${HEAVY[@]}"; do
        submit "$strat" "$uni"
        count=$((count + 1))
    done
done
for uni in "${UNIVERSES[@]}"; do
    for strat in "${LIGHT[@]}"; do
        submit "$strat" "$uni"
        count=$((count + 1))
    done
done

echo ""
if (( DRY_RUN )); then
    echo "${count} jobs planned (dry run, nothing submitted)."
else
    echo "${count} jobs submitted. Track with: squeue --me"
fi
