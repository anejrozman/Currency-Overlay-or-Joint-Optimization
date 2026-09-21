#!/bin/bash
# Submit the level-sweep joint-vs-overlay gap experiment to the Euler cluster.
#
# --es or --soes picks the measure; there is no default, because the two write to
# separate shard directories and a silent default would fill the wrong one.
#
# Runs code/scripts/in_sample_vs_out_of_sample_at_diff_levels.py as one job per
# cell: 14 confidence levels (50/60/70, then 2.5-point steps from 72.5 to 97.5)
# x three universes (10, 50, 100) = 42 independent jobs, each solving the
# unregularized ES joint and overlay programs plus the 1/N control at ~91 quarterly
# rebalancing dates.
#
#   ./run_gap_analysis_levels.sh --es --levels "72.5 75 77.5 82.5 85 87.5 92.5 97.5"
#
# is the submission that adds the dense grid to an existing coarse sweep: 24 jobs.
# Cells whose shard already exists in the output directory are skipped unless
# --force, so re-running after a partial sweep submits only what is missing.
#
# The cells are independent, so the jobs run concurrently and share one output
# directory. Each writes gaps_unregularized_level<L>_top<N>.csv and touches no
# other file, so there is no write race and nothing to merge afterwards. A cell
# that finishes is durable: if a job hits its wall-clock limit the shards already
# written survive, and resubmitting that one cell is enough.
#
# Every cell reuses the settings its production backtest held, read via
# results/runs/top_runs_es_at_different_levels.csv. All 36 cells of that table are
# filled and verified against the run configs, so every job here is submittable.
# The Python script re-checks each run's recorded alpha against the level of the
# cell claiming it, because the table's rows are named <strategy>_level_<L> while
# the run directories are named after the optimizer file alone -- a transposed run
# number is not otherwise detectable.
#
# Node allocation and tiers
# -------------------------
# Time and memory are tiered by universe, reusing the tiers measured for
# run_gap_analysis_demeaned.sh: ES at top-10 took ~5 min under Clarabel, ~35 min
# at top-50 and ~75 min extrapolated at top-100, with cost scaling about N^1.3 in
# the asset count (medians 127 / 584 / 1051), and measured RSS of 1.3 GB at top-50
# and 2.0 GB at top-100. Gurobi should beat those timings. The tiers carry roughly
# 8x headroom.
#
# The level does not change the cost: the ES program is an LP with q + N + 1
# variables and q constraints whatever alpha is, and only the 1/(q(1-alpha))
# coefficient on the objective moves. So one tier per universe covers all six
# levels, and the 18 jobs are 6 identical copies of the demeaned script's 3 ES
# cells.
#
# A note on what the top-level cells mean
# ---------------------------------------
# f_out is scored on the one quarter actually held (65 trading days, measured), so
# the tail it averages thins as the level rises: 32 observations at level 50, 13 at
# level 80, 6 at level 90, 3 at level 95 and ~2 at level 97.5. Those intervals will
# be wide for that reason alone. Each cell therefore also scores a second horizon,
# the following year (~252 days), reported as gap_out_1y alongside gap_out; it
# lifts the level-97.5 tail from ~2 observations to ~6 and is the arm to read when
# the quarterly figure disagrees with the production backtests at the top end. Both
# are in gap_summary.csv, with the median realized tail count of each.
#
# The tables and figures are built separately, once the cells have landed:
#   python code/scripts/in_sample_vs_out_of_sample_at_diff_levels.py --es --plot-only
#
# Usage:
#   ./run_gap_analysis_levels.sh --es|--soes [options]
#
# Options:
#   --level N            Submit one level alone, e.g. 72.5 (default: all 14)
#   --levels "A B C"     Submit this set of levels
#   --force              Submit cells whose shard already exists
#   --universe N         Submit top-10, 50 or 100 alone (default: all three)
#   --time HH:MM:SS      Override the per-universe time limit (max 720:00:00)
#   --cpus N             CPUs per task (default: 4 -> Gurobi runs 3 threads)
#   --mem-per-cpu SIZE   Override the per-universe memory tier
#   --log-level LEVEL    DEBUG|INFO|WARNING (passthrough). DEBUG logs one line
#                        per rebalancing date, which is the only way to see
#                        progress inside a cell.
#   --out-dir DIR        Shared output directory. Relative paths resolve against
#                        the repo root, not the job's working directory.
#   --dry-run            Print what would be submitted and exit.

set -euo pipefail

MEASURE=""
LEVEL=""
UNIVERSE=""
TIME_LIMIT=""
CPUS=4
MEM_PER_CPU=""
OUT_DIR=""
DRY_RUN=0
FORCE=0
PASSTHROUGH=()
MAX_HOURS=720

LEVELS_ALL=(50 60 70 72.5 75 77.5 80 82.5 85 87.5 90 92.5 95 97.5)
UNIVERSES_ALL=(10 50 100)

usage() {
    cat <<'USAGE'
Submit the level-sweep joint-vs-overlay gap experiment to the Euler cluster.

Usage:
  ./run_gap_analysis_levels.sh --es|--soes [options]

  --es | --soes        Which measure to sweep (required, no default)
  --level N            Submit one level alone, e.g. 72.5 (default: all 14)
  --levels "A B C"     Submit this set of levels
  --force              Submit cells whose shard already exists
  --universe N         Submit top-10, 50 or 100 alone (default: all three)
  --time HH:MM:SS      Override the per-universe time limit (max 720:00:00)
  --cpus N             CPUs per task (default: 4 -> Gurobi runs 3 threads)
  --mem-per-cpu SIZE   Override the per-universe memory tier
  --log-level LEVEL    DEBUG|INFO|WARNING (passthrough)
  --out-dir DIR        Shared output directory (passthrough)
  --dry-run            Print what would be submitted and exit
USAGE
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --es)            MEASURE="es"; shift ;;
        --soes)          MEASURE="soes"; shift ;;
        --level)         LEVEL="$2"; shift 2 ;;
        --levels)        read -r -a LEVELS_SET <<< "$2"; shift 2 ;;
        --force)         FORCE=1; shift ;;
        --universe)      UNIVERSE="$2"; shift 2 ;;
        --time)          TIME_LIMIT="$2"; shift 2 ;;
        --cpus)          CPUS="$2"; shift 2 ;;
        --mem-per-cpu)   MEM_PER_CPU="$2"; shift 2 ;;
        --log-level)     PASSTHROUGH+=("--log-level" "$2"); shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               echo "Error: unknown argument '$1'"; usage ;;
    esac
done

# No default measure: an 84-cell sweep is too expensive to start on the wrong one
# by omission, and the two measures write to separate directories, so a silent
# default would quietly fill the other measure's.
if [[ -z "$MEASURE" ]]; then
    echo "Error: one of --es or --soes is required."
    exit 1
fi
PASSTHROUGH+=("--${MEASURE}")

is_sweep_level() {
    local candidate="$1" known
    for known in "${LEVELS_ALL[@]}"; do [[ "$known" == "$candidate" ]] && return 0; done
    return 1
}

if [[ -n "$LEVEL" ]]; then
    LEVELS=("$LEVEL")
elif [[ -n "${LEVELS_SET+x}" ]]; then
    LEVELS=("${LEVELS_SET[@]}")
else
    LEVELS=("${LEVELS_ALL[@]}")
fi
for level in "${LEVELS[@]}"; do
    if ! is_sweep_level "$level"; then
        echo "Error: '$level' is not a sweep level (${LEVELS_ALL[*]})."
        exit 1
    fi
done

if [[ -n "$UNIVERSE" ]]; then
    case "$UNIVERSE" in
        10|50|100) UNIVERSES=("$UNIVERSE") ;;
        *) echo "Error: --universe must be 10, 50 or 100."; exit 1 ;;
    esac
else
    UNIVERSES=("${UNIVERSES_ALL[@]}")
fi

if [[ -n "$TIME_LIMIT" ]]; then
    if [[ ! "$TIME_LIMIT" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]]; then
        echo "Error: --time must be in HH:MM:SS format (e.g. 24:00:00)."
        exit 1
    fi
    if (( ${BASH_REMATCH[1]} > MAX_HOURS )); then
        echo "Error: time limit ${TIME_LIMIT} exceeds the maximum of ${MAX_HOURS}:00:00."
        exit 1
    fi
fi

# Directory paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/code/scripts"
LOGS_DIR="$ROOT_DIR/logs"
mkdir -p "$LOGS_DIR"

# run_in_sample_analysis.sh expects code/.venv, submit_job.sh expects .venv;
# take whichever is actually present so this works in either layout.
if [[ -f "$ROOT_DIR/code/.venv/bin/activate" ]]; then
    VENV="$ROOT_DIR/code/.venv/bin/activate"
else
    VENV="$ROOT_DIR/.venv/bin/activate"
fi

# The job runs with --chdir code/scripts, so a relative --out-dir would land in
# code/scripts/<dir>. Resolve it against the repo root instead.
if [[ -n "$OUT_DIR" ]]; then
    [[ "$OUT_DIR" = /* ]] || OUT_DIR="$ROOT_DIR/$OUT_DIR"
    PASSTHROUGH+=("--out-dir" "$OUT_DIR")
fi

# Per-universe defaults, overridden by --time / --mem-per-cpu when given.
tier_time() {
    case "$1" in
        10)  echo "04:00:00" ;;
        50)  echo "12:00:00" ;;
        100) echo "24:00:00" ;;
    esac
}

tier_mem() {
    case "$1" in
        10)  echo "4G" ;;
        50)  echo "8G" ;;
        100) echo "16G" ;;
    esac
}

# Heaviest universe first, so the long poles start before the short jobs.
IFS=$'\n' read -r -d '' -a UNIVERSES < <(printf '%s\n' "${UNIVERSES[@]}" | sort -rn && printf '\0')

SHARD_DIR="${OUT_DIR:-$ROOT_DIR/results/in_sample_vs_out_of_sample_${MEASURE}_at_diff_levels}"

count=0
skipped=0
for universe in "${UNIVERSES[@]}"; do
    for level in "${LEVELS[@]}"; do
        shard="$SHARD_DIR/gaps_unregularized_level${level}_top${universe}.csv"
        if (( ! FORCE )) && [[ -f "$shard" ]]; then
            printf '  level=%-5s top-%-4s already solved, skipping\n' "$level" "$universe"
            skipped=$((skipped + 1))
            continue
        fi
        job_time="${TIME_LIMIT:-$(tier_time "$universe")}"
        job_mem="${MEM_PER_CPU:-$(tier_mem "$universe")}"
        JOB_NAME="gap_${MEASURE}_l${level}_top${universe}"

        echo "Submitting: ${JOB_NAME}  (time=${job_time}, cpus=${CPUS}, mem-per-cpu=${job_mem})"
        count=$((count + 1))
        if (( DRY_RUN )); then
            echo "  python in_sample_vs_out_of_sample_at_diff_levels.py" \
                 "--levels ${level} --universes ${universe} --sweep-only" \
                 "${PASSTHROUGH[*]:-}"
            continue
        fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${job_time}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem-per-cpu=${job_mem}
#SBATCH --output=${LOGS_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOGS_DIR}/${JOB_NAME}_%j.err
#SBATCH --chdir=${SCRIPTS_DIR}

module load stack/2024-06 gcc/12.2.0 python/3.12.8
module load gurobi/13.0.0
source ${VENV}

echo "Starting ${JOB_NAME} at \$(date) on \$(hostname)"
python in_sample_vs_out_of_sample_at_diff_levels.py \
    --levels ${level} \
    --universes ${universe} \
    --sweep-only ${PASSTHROUGH[*]:-}
status=\$?
echo "Finished ${JOB_NAME} at \$(date) with exit code \${status}"
exit \${status}
EOF
    done
done

echo
(( skipped )) && echo "${skipped} cells skipped (shard already present; --force to resolve)."
if (( DRY_RUN )); then
    echo "${count} jobs planned (dry run, nothing submitted)."
else
    echo "${count} jobs submitted. Check status with: squeue --me"
fi
echo "Once they have finished:"
echo "  python code/scripts/in_sample_vs_out_of_sample_at_diff_levels.py --${MEASURE} --plot-only"
