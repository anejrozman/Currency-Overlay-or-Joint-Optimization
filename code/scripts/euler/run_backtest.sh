#!/bin/bash
# Submit one backtest (code/scripts/run_strategy.py) to the Euler cluster.
#
# Usage: ./run_backtest.sh --strategy <name> --universe <10|50|100> [options]; see --help.

set -euo pipefail

STRATEGY=""
UNIVERSE=10
TIME_LIMIT="24:00:00"
CPUS=4
MEM_PER_CPU="4G"
ALPHA=""
PASSTHROUGH=()
MAX_HOURS=720

usage() {
    cat <<'USAGE'
Submit ONE backtest to the Euler cluster.

Usage:
  ./run_backtest.sh --strategy <name> --universe <10|50|100> [options]

  --strategy NAME     Strategy in strategy_registry.py (required)
  --universe N        Top-N universe: 10, 50, or 100 (default: 10)
  --time HH:MM:SS     SLURM time limit (default: 24:00:00, max 720:00:00)
  --cpus N            CPUs per task (default: 4 -> Gurobi runs 3 threads)
  --mem-per-cpu SIZE  Memory per CPU, e.g. 4G (default: 4G)
  --no-grid           Skip grid search (passthrough to run_strategy.py)
  --alpha A           Pin the ES level in the strategy's grid, e.g. 0.95 (passthrough)
  --no-demean         Pin demean=False in the strategy's grid (passthrough)
  --window-years N    Override estimation window (passthrough)
  --val-years N       Override validation window (passthrough)
  --log-level LEVEL   DEBUG|INFO|WARNING (passthrough)
USAGE
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategy)     STRATEGY="$2"; shift 2 ;;
        --universe)     UNIVERSE="$2"; shift 2 ;;
        --time)         TIME_LIMIT="$2"; shift 2 ;;
        --cpus)         CPUS="$2"; shift 2 ;;
        --mem-per-cpu)  MEM_PER_CPU="$2"; shift 2 ;;
        --no-grid)      PASSTHROUGH+=("--no-grid"); shift ;;
        --alpha)        ALPHA="$2"; PASSTHROUGH+=("--alpha" "$2"); shift 2 ;;
        --no-demean)    PASSTHROUGH+=("--no-demean"); shift ;;
        --window-years) PASSTHROUGH+=("--window-years" "$2"); shift 2 ;;
        --val-years)    PASSTHROUGH+=("--val-years" "$2"); shift 2 ;;
        --log-level)    PASSTHROUGH+=("--log-level" "$2"); shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Error: unknown argument '$1'"; usage ;;
    esac
done

[[ -z "$STRATEGY" ]] && { echo "Error: --strategy is required."; usage; }

case "$UNIVERSE" in
    10|50|100) ;;
    *) echo "Error: --universe must be 10, 50, or 100."; exit 1 ;;
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
SCRIPTS_DIR="$ROOT_DIR/code/scripts"
LOGS_DIR="$ROOT_DIR/logs"
VENV="$ROOT_DIR/code/.venv/bin/activate" 
mkdir -p "$LOGS_DIR"

# tag the job with a pinned ES level (0.725 -> a72.5); %g keeps 0.725 and 0.72 apart
if [[ -n "$ALPHA" ]]; then
    if [[ ! "$ALPHA" =~ ^0?\.[0-9]+$ ]]; then
        echo "Error: --alpha must be a decimal strictly between 0 and 1 (e.g. 0.95)."
        exit 1
    fi
    LEVEL="$(awk -v a="$ALPHA" 'BEGIN { printf "%g", a * 100 }')"
    JOB_NAME="${STRATEGY}_a${LEVEL}_u${UNIVERSE}"
else
    JOB_NAME="${STRATEGY}_u${UNIVERSE}"
fi
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
#SBATCH --chdir=${SCRIPTS_DIR}

module load stack/2024-06 gcc/12.2.0 python/3.12.8
module load gurobi/13.0.0
source ${VENV}

# Private, node-local scratch for the intermediate return files so concurrent
# jobs never write the shared data/processed/ files at the same time.
export BACKTEST_SCRATCH_DIR="\${TMPDIR:-/tmp}/bt_\${SLURM_JOB_ID}"
mkdir -p "\${BACKTEST_SCRATCH_DIR}"

echo "Starting ${JOB_NAME} at \$(date) on \$(hostname)"
python run_strategy.py --strategy ${STRATEGY} --universe ${UNIVERSE} ${PASSTHROUGH[*]:-}
status=\$?
echo "Finished ${JOB_NAME} at \$(date) with exit code \${status}"
exit \${status}
EOF
