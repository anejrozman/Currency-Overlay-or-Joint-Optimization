#!/bin/bash
# Submit every strategy on every universe (10, 50, 100) as separate jobs, heaviest first.
#
# Usage: ./run_all_backtests.sh [--dry-run]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="$HERE/run_backtest.sh"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

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
