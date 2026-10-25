#!/bin/bash
# Submit one arm of code/analysis/in_sample_vs_out_of_sample.py to the Euler cluster.
#
# The currency arm re-solves every cell (~24h); the grid arm reads results/in_sample_analysis/ and needs no job.
# Arms share gaps.csv, so do not run two against the same --out-dir at once. Afterwards:
#   python code/analysis/in_sample_vs_out_of_sample.py --plot-only
#
# Usage: ./run_gap_analysis.sh [--arm <currency|grid>] [options]; see --help.

set -euo pipefail

ARM="currency"
FAMILY="unregularized"
TIME_LIMIT="24:00:00"
CPUS=4
MEM_PER_CPU=""
OUT_DIR=""
PASSTHROUGH=()
MAX_HOURS=720

usage() {
    cat <<'USAGE'
Submit the joint-vs-overlay gap experiment to the Euler cluster.

Usage:
  ./run_gap_analysis.sh [--arm <currency|grid>] [options]

  --arm NAME           Arm to run: currency or grid (default: currency)
  --family NAME        unregularized or regularized (default: unregularized)
  --time HH:MM:SS      SLURM time limit (default: 24:00:00, max 720:00:00)
  --cpus N             CPUs per task (default: 4 -> Gurobi runs 3 threads)
  --mem-per-cpu SIZE   Memory per CPU (default: 8G)
  --log-level LEVEL    DEBUG|INFO|WARNING (passthrough)
  --out-dir DIR        Output directory (passthrough); use distinct dirs to run jobs concurrently
USAGE
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm)           ARM="$2"; shift 2 ;;
        --family)        FAMILY="$2"; shift 2 ;;
        --time)          TIME_LIMIT="$2"; shift 2 ;;
        --cpus)          CPUS="$2"; shift 2 ;;
        --mem-per-cpu)   MEM_PER_CPU="$2"; shift 2 ;;
        --log-level)     PASSTHROUGH+=("--log-level" "$2"); shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *)               echo "Error: unknown argument '$1'"; usage ;;
    esac
done

case "$ARM" in
    currency|grid) ;;
    *) echo "Error: --arm must be currency or grid."; exit 1 ;;
esac

case "$FAMILY" in
    unregularized|regularized) ;;
    *) echo "Error: --family must be unregularized or regularized."; exit 1 ;;
esac

if [[ ! "$TIME_LIMIT" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]]; then
    echo "Error: --time must be in HH:MM:SS format (e.g. 24:00:00)."
    exit 1
fi
if (( ${BASH_REMATCH[1]} > MAX_HOURS )); then
    echo "Error: time limit ${TIME_LIMIT} exceeds the maximum of ${MAX_HOURS}:00:00."
    exit 1
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

MEM_PER_CPU="${MEM_PER_CPU:-8G}"

# the job runs in code/analysis, so a relative --out-dir is resolved against the repo root
if [[ -n "$OUT_DIR" ]]; then
    [[ "$OUT_DIR" = /* ]] || OUT_DIR="$ROOT_DIR/$OUT_DIR"
    PASSTHROUGH+=("--out-dir" "$OUT_DIR")
fi

JOB_NAME="gap_${ARM}_${FAMILY}"
ARM_ARGS="--arms ${ARM} --family ${FAMILY}"

echo "Submitting: ${JOB_NAME}  (time=${TIME_LIMIT}, cpus=${CPUS}, mem-per-cpu=${MEM_PER_CPU})"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH --output=${LOGS_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOGS_DIR}/${JOB_NAME}_%j.err
#SBATCH --chdir=${ANALYSIS_DIR}

module load stack/2024-06 gcc/12.2.0 python/3.12.8
module load gurobi/13.0.0
source ${VENV}

echo "Starting ${JOB_NAME} at \$(date) on \$(hostname)"
python in_sample_vs_out_of_sample.py ${ARM_ARGS} ${PASSTHROUGH[*]:-}
status=\$?
echo "Finished ${JOB_NAME} at \$(date) with exit code \${status}"
exit \${status}
EOF

echo "Check status with: squeue --me"
echo "Once it has finished: python code/analysis/in_sample_vs_out_of_sample.py --plot-only"
