#!/bin/bash
# Submit a level sweep to the Euler cluster.
#
# --es or --soes picks the measure. There is no default: 84 cells is too much
# cluster time to spend on the wrong measure by omission, and the two sweeps are
# recorded in separate tables, so a silent default would fill the wrong one.
#
# Runs the two UNREGULARIZED strategies of the chosen measure -- <m>_curr_overlay
# (two-step: assets, then overlay) and <m>_joint (assets and overlay in one
# program) -- at each confidence level of the sweep, on all three universes:
#
#   14 levels x 2 strategies x 3 universes = 84 cells
#
# The sweep is 50/60/70, then a 2.5-point grid from 72.5 to 97.5. The coarse
# levels were run first; the dense grid was added to resolve the top end, where
# the in-sample gap rises fastest and the out-of-sample tail thins fastest.
#
#   ./run_levels.sh --es --levels "72.5 75 77.5 82.5 85 87.5 92.5 97.5"
#
# is the submission that adds the dense grid to an existing coarse sweep: 48 jobs.
# Cells whose run is already recorded in the measure's
# results/runs/top_runs_<measure>_at_different_levels.csv
# are skipped with a notice unless --force is given, so re-running this script
# after a partial sweep submits only what is missing.
#
# Each job is one sbatch submission via run_backtest.sh, so the scheduler spreads
# them across nodes and they run concurrently; at 4 cores each, 48 jobs need ~192
# of the ls_math share's 640 cores, so the wall clock is set by the slowest single
# job (es_joint at top-100).
#
# What is pinned, and why
# -----------------------
# backtest_config.py fixes ALPHA = [0.80] for every strategy, so the level cannot
# be varied from the registry. --alpha pins the 'alpha' axis of the strategy's
# grid to one value per job; --no-demean pins 'demean' to False, overriding the
# global DEMEAN = True. Both land in the run's config.json under "grid_overrides",
# which is how collect_level_runs.py later tells a swept run from a default one.
#
# The grids of OVERLAY_CONFIG and JOINT_CONFIG are alpha x leverage x demean, and
# all three axes are singletons once alpha and demean are pinned. So each job's
# "grid search" is a single combination: the backtester still solves the train and
# validation windows before refitting on the full window (3 solves per rebalancing
# date x 91 dates), but nothing is actually being selected. Two consequences:
#
#   * do NOT substitute --no-grid for --alpha. --no-grid drops the grid entirely
#     and the optimizers fall back to their own alpha = 0.80 default, which would
#     silently produce identical runs at every level.
#   * the val_score column in selected_params.csv is recorded, never used. It is
#     also not comparable across levels: ES at 50% and ES at 95% are different
#     functionals, not two scores of one objective.
#
# Half levels
# -----------
# Alpha is computed with %g rather than a fixed number of decimals. At two
# decimals level 72.5 becomes alpha 0.72 and 77.5 becomes 0.78, so the job would
# run at a level the sweep does not contain, record that alpha in its config, and
# be dropped later by the collector as unrecognised -- a whole cell of cluster
# time for nothing. The job tag is formed the same way, so 0.725 reads as a72.5
# in squeue rather than colliding with a72 or a73.
#
# Statistical caveat at the top end
# ---------------------------------
# These programs are unregularized; the only thing bounding the asset block is the
# ||x||_1 <= 2 leverage cap. The estimation window is ~2610 days while the top-100
# universe reaches 1161 names, so the level-97.5 tail is ~65 observations against
# ~1161 weights -- the LP optimum then sits on a vertex pinned by a handful of tail
# days and the weights swing hard quarter to quarter. Read the top-50 and top-100
# cells above level 90 as a diagnostic of the estimation-error blowup the
# regularized variants exist to fix, not as a deployable strategy. Note this
# concerns the backtests' own estimation; the realized ES they report still pools
# 5935 days and stays well estimated.
#
# Run directories
# ---------------
# next_run_dir() names directories after the optimizer FILE, not the strategy plus
# level, so every overlay run lands in results/runs/es_curr_overlay/ alongside the
# runs already there, and every joint run in results/runs/es_joint/. Run numbers
# are claimed by an atomic mkdir when each job starts, so which number goes to
# which level depends on scheduling order and cannot be predicted here. Recover
# the mapping after the fact with:
#
#   python code/scripts/collect_level_runs.py --es          # print the table
#   python code/scripts/collect_level_runs.py --es --write  # update the CSV
#
# Usage:
#   ./run_levels.sh --es                          # every cell not already recorded
#   ./run_levels.sh --soes --levels "72.5 75 97.5"  # only these levels
#   ./run_levels.sh --soes --dry-run              # print the plan, submit nothing
#   ./run_levels.sh --es --force                  # resubmit cells already recorded
#   ./run_levels.sh --es --demean                 # sweep with demean=True instead

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
            sed -n '2,84p' "${BASH_SOURCE[0]}"
            exit 1 ;;
        *) echo "Error: unknown argument '$1' (expected --es, --soes, --levels, --dry-run, --force, --demean or --help)."; exit 1 ;;
    esac
done

if [[ -z "$MEASURE" ]]; then
    echo "Error: one of --es or --soes is required."
    exit 1
fi

# The strategy names and the table are both derived from the measure, so the two
# can never disagree -- a mismatch here would submit one measure's backtests and
# record them against the other's table.
STRATEGIES=("${MEASURE}_curr_overlay" "${MEASURE}_joint")
RUNS_CSV="$ROOT_DIR/results/runs/top_runs_${MEASURE}_at_different_levels.csv"
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

# Is this cell's run already recorded? The table is the record of what has landed,
# so a filled cell means that backtest exists and resubmitting it would spend a
# 150-336h reservation to produce a duplicate the collector would then have to
# arbitrate. Matched on the whole first field rather than by regex, because a level
# like 72.5 carries a dot that a regex would read as a wildcard.
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
            # %g, not %.2f: at two decimals level 72.5 becomes alpha 0.72 and 77.5
            # becomes 0.78, so the job would silently run at a level the sweep does
            # not contain and the collector would later drop it as unrecognised.
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
