#!/bin/bash
# Submit the level sweep: the unregularized <m>_curr_overlay and <m>_joint backtests at every confidence level
# on every universe (14 x 2 x 3 = 84 jobs). Cells already recorded in
# results/runs/top_runs_<m>_at_different_levels.csv are skipped unless --force.
#
# --alpha pins the level of each job; do not use --no-grid instead, which falls back to alpha = 0.80.
# Afterwards, record which run is which cell:
#   python code/scripts/collect_level_runs.py --es --write
#
# Usage:
#   ./run_levels.sh --es                            # every cell not already recorded
#   ./run_levels.sh --soes --levels "72.5 75 97.5"  # only these levels
#   ./run_levels.sh --soes --dry-run                # print the plan, submit nothing
#   ./run_levels.sh --es --force                    # resubmit cells already recorded
#   ./run_levels.sh --es --demean                   # sweep with demean=True instead

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/../../.." && pwd)"
RUN_ONE="$HERE/run_backtest.sh"

MEASURE=""
DRY_RUN=0
FORCE=0
DEMEAN_FLAG="--no-demean"
LEVELS=(50 60 70 72.5 75 77.5 80 82.5 85 87.5 90 92.5 95 97.5)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --es)      MEASURE="es"; shift ;;
        --soes)    MEASURE="soes"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --force)   FORCE=1; shift ;;
        --demean)  DEMEAN_FLAG=""; shift ;;
        --levels)  read -r -a LEVELS <<< "$2"; shift 2 ;;
        -h|--help)
            sed -n '2,15p' "${BASH_SOURCE[0]}"
            exit 1 ;;
        *) echo "Error: unknown argument '$1' (expected --es, --soes, --levels, --dry-run, --force, --demean or --help)."; exit 1 ;;
    esac
done

if [[ -z "$MEASURE" ]]; then
    echo "Error: one of --es or --soes is required."
    exit 1
fi

STRATEGIES=("${MEASURE}_curr_overlay" "${MEASURE}_joint")
RUNS_CSV="$ROOT_DIR/results/runs/top_runs_${MEASURE}_at_different_levels.csv"
UNIVERSES=(100 50 10)   # heaviest universe first, so the long poles start first

# the cost depends on the universe, not on the level
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

# whether the cell's run is recorded; the first field is compared whole, as 72.5 contains a dot
cell_recorded() {  # cell_recorded <strategy> <level> <universe>
    [[ -f "$RUNS_CSV" ]] || return 1
    local col
    case "$3" in 10) col=2 ;; 50) col=3 ;; 100) col=4 ;; esac
    awk -F',' -v name="${1}_level_${2}" -v col="$col" '
        { gsub(/^[ \t]+|[ \t]+$/, "", $1) }
        $1 == name {
            gsub(/[ \t]/, "", $col)
            found = 1
            if ($col != "") filled = 1
            exit
        }
        END { exit (found && filled) ? 0 : 1 }
    ' "$RUNS_CSV"
}

count=0
skipped=0
for uni in "${UNIVERSES[@]}"; do
    for strat in "${STRATEGIES[@]}"; do
        for level in "${LEVELS[@]}"; do
            if (( ! FORCE )) && cell_recorded "$strat" "$level" "$uni"; then
                printf '  %-16s level=%-5s u%-4s  already recorded, skipping\n' \
                    "$strat" "$level" "$uni"
                skipped=$((skipped + 1))
                continue
            fi
            # %g, not %.2f, which would turn level 72.5 into alpha 0.72
            alpha="$(awk -v l="$level" 'BEGIN { printf "%g", l / 100 }')"
            t="$(time_for "$uni")"
            m="$(mem_for "$uni")"
            if (( DRY_RUN )); then
                printf '  %-16s level=%-5s alpha=%-6s u%-4s time=%-10s mem=%-4s cpus=4 %s\n' \
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
(( skipped )) && echo "${skipped} cells skipped (already recorded; --force to resubmit)."
if (( DRY_RUN )); then
    echo "${count} jobs planned (dry run, nothing submitted)."
else
    echo "${count} jobs submitted. Track with: squeue --me"
    echo "When they finish: python code/scripts/collect_level_runs.py --${MEASURE} --write"
fi
