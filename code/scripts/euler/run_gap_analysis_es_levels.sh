#!/bin/bash
# Submit the ES-level joint-vs-overlay gap experiment to the Euler cluster.
#
# Runs code/scripts/in_sample_vs_out_of_sample_es_at_diff_levels.py as one job per
# cell: six confidence levels (50/60/70/80/90/95) x three universes (10, 50, 100)
# = 18 independent jobs, each solving the unregularized ES joint and overlay
# programs plus the 1/N control at ~91 quarterly rebalancing dates.
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
# f_out is scored on the one quarter actually held (~63 trading days), so the tail
# it averages thins as the level rises: ~31 observations at level 50, ~12 at level
# 80, ~3 at level 95. The level-90 and level-95 out-of-sample intervals will be
# wide for that reason alone. This is a property of the measurement, not of the
# jobs; gap_summary.csv reports the median realized tail count per cell so the
# effect is visible next to the numbers it explains.
#
# The tables and figures are built separately, once the cells have landed:
#   python code/scripts/in_sample_vs_out_of_sample_es_at_diff_levels.py --plot-only
#
# Usage:
#   ./run_gap_analysis_es_levels.sh [options]
#
# Options:
#   --level N            Submit one level alone: 50, 60, 70, 80, 90 or 95
#                        (default: all six)
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

LEVEL=""
UNIVERSE=""
TIME_LIMIT=""
CPUS=4
MEM_PER_CPU=""
OUT_DIR=""
DRY_RUN=0
PASSTHROUGH=()
MAX_HOURS=720

LEVELS_ALL=(50 60 70 80 90 95)
UNIVERSES_ALL=(10 50 100)

usage() {
    cat <<'USAGE'
Submit the ES-level joint-vs-overlay gap experiment to the Euler cluster.

Usage:
  ./run_gap_analysis_es_levels.sh [options]

  --level N            Submit one level alone: 50, 60, 70, 80, 90 or 95 (default: all)
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
        --level)         LEVEL="$2"; shift 2 ;;
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

if [[ -n "$LEVEL" ]]; then
    case "$LEVEL" in
        50|60|70|80|90|95) LEVELS=("$LEVEL") ;;
        *) echo "Error: --level must be 50, 60, 70, 80, 90 or 95."; exit 1 ;;
    esac
else
    LEVELS=("${LEVELS_ALL[@]}")
fi

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

count=0
for universe in "${UNIVERSES[@]}"; do
    for level in "${LEVELS[@]}"; do
        job_time="${TIME_LIMIT:-$(tier_time "$universe")}"
        job_mem="${MEM_PER_CPU:-$(tier_mem "$universe")}"
        JOB_NAME="gap_es_l${level}_top${universe}"

        echo "Submitting: ${JOB_NAME}  (time=${job_time}, cpus=${CPUS}, mem-per-cpu=${job_mem})"
        count=$((count + 1))
        if (( DRY_RUN )); then
            echo "  python in_sample_vs_out_of_sample_es_at_diff_levels.py" \
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
python in_sample_vs_out_of_sample_es_at_diff_levels.py \
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
if (( DRY_RUN )); then
    echo "${count} jobs planned (dry run, nothing submitted)."
else
    echo "${count} jobs submitted. Check status with: squeue --me"
fi
echo "Once they have finished:"
echo "  python code/scripts/in_sample_vs_out_of_sample_es_at_diff_levels.py --plot-only"
