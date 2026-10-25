#!/bin/bash
# Submit code/analysis/in_sample_vs_out_of_sample_demeaned.py to the Euler cluster, one job per cell
# (es, soes x top-10, 50, 100). Each job writes its own shard, so the jobs run concurrently. Afterwards:
#   python code/analysis/in_sample_vs_out_of_sample_demeaned.py --plot-only
#
# Usage: ./run_gap_analysis_demeaned.sh --family <unregularized|regularized> [options]; see --help.

set -euo pipefail

FAMILY=""
MEASURE=""
UNIVERSE=""
TIME_LIMIT=""
CPUS=4
MEM_PER_CPU=""
OUT_DIR=""
DRY_RUN=0
PASSTHROUGH=()
MAX_HOURS=720

MEASURES_ALL=(es soes)
UNIVERSES_ALL=(10 50 100)

usage() {
    cat <<'USAGE'
Submit the de-meaned joint-vs-overlay gap experiment to the Euler cluster.

Usage:
  ./run_gap_analysis_demeaned.sh --family <unregularized|regularized> [options]

  --family NAME        unregularized or regularized (required)
  --measure NAME       Submit es or soes alone (default: both)
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
        --family)        FAMILY="$2"; shift 2 ;;
        --measure)       MEASURE="$2"; shift 2 ;;
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

case "$FAMILY" in
    unregularized|regularized) ;;
    "") echo "Error: --family is required (unregularized or regularized)."; usage ;;
    *)  echo "Error: --family must be unregularized or regularized."; exit 1 ;;
esac

if [[ -n "$MEASURE" ]]; then
    case "$MEASURE" in
        es|soes) MEASURES=("$MEASURE") ;;
        *) echo "Error: --measure must be es or soes."; exit 1 ;;
    esac
else
    MEASURES=("${MEASURES_ALL[@]}")
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ANALYSIS_DIR="$ROOT_DIR/code/analysis"
LOGS_DIR="$ROOT_DIR/logs"
mkdir -p "$LOGS_DIR"

# the virtual environment is code/.venv, or .venv at the repo root
if [[ -f "$ROOT_DIR/code/.venv/bin/activate" ]]; then
    VENV="$ROOT_DIR/code/.venv/bin/activate"
else
    VENV="$ROOT_DIR/.venv/bin/activate"
fi

# the job runs in code/analysis, so a relative --out-dir is resolved against the repo root
if [[ -n "$OUT_DIR" ]]; then
    [[ "$OUT_DIR" = /* ]] || OUT_DIR="$ROOT_DIR/$OUT_DIR"
    PASSTHROUGH+=("--out-dir" "$OUT_DIR")
fi

# per-universe defaults, overridden by --time and --mem-per-cpu
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

for measure in "${MEASURES[@]}"; do
    for universe in "${UNIVERSES[@]}"; do
        job_time="${TIME_LIMIT:-$(tier_time "$universe")}"
        job_mem="${MEM_PER_CPU:-$(tier_mem "$universe")}"
        JOB_NAME="gap_dm_${FAMILY}_${measure}_top${universe}"

        echo "Submitting: ${JOB_NAME}  (time=${job_time}, cpus=${CPUS}, mem-per-cpu=${job_mem})"
        if (( DRY_RUN )); then
            echo "  python in_sample_vs_out_of_sample_demeaned.py --family ${FAMILY}" \
                 "--measures ${measure} --universes ${universe} --sweep-only" \
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
#SBATCH --chdir=${ANALYSIS_DIR}

module load stack/2024-06 gcc/12.2.0 python/3.12.8
module load gurobi/13.0.0
source ${VENV}

echo "Starting ${JOB_NAME} at \$(date) on \$(hostname)"
python in_sample_vs_out_of_sample_demeaned.py \
    --family ${FAMILY} \
    --measures ${measure} \
    --universes ${universe} \
    --sweep-only ${PASSTHROUGH[*]:-}
status=\$?
echo "Finished ${JOB_NAME} at \$(date) with exit code \${status}"
exit \${status}
EOF
    done
done

echo
echo "Check status with: squeue --me"
echo "Once they have finished:"
echo "  python code/analysis/in_sample_vs_out_of_sample_demeaned.py --plot-only"
