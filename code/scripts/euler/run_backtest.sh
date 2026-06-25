#!/bin/bash
# Submit ONE backtest to the Euler cluster.
#
# Runs a single strategy on a single universe over its parameter grid (defined in
# code/src/portfolio_backtest/backtest_config.py), mirroring the local
# code/scripts/run_strategy.py interface but wrapped in an sbatch submission.
#
# These are CPU-bound Gurobi QP/LP solves — GPUs do not accelerate them, so no
# GPU is requested and the job lands on the high-throughput CPU nodes.
#
# Usage:
#   ./run_backtest.sh --strategy <name> --universe <10|50|100> [options]
#
# Options:
#   --strategy NAME     Strategy in strategy_registry.py (required)
#   --universe N        Top-N universe: 10, 50, or 100 (default: 10)
#   --time HH:MM:SS     SLURM time limit (default: 24:00:00, max 720:00:00)
#   --cpus N            CPUs per task (default: 4 -> Gurobi runs 3 threads)
#   --mem SIZE          Total memory, e.g. 16G (default: 16G)
#   --no-grid           Skip grid search (passthrough to run_strategy.py)
#   --window-years N    Override estimation window (passthrough)
#   --val-years N       Override validation window (passthrough)
#   --log-level LEVEL   DEBUG|INFO|WARNING (passthrough)

set -euo pipefail

STRATEGY=""
UNIVERSE=10
TIME_LIMIT="24:00:00"
CPUS=4
MEM="16G"
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
  --mem SIZE          Total memory, e.g. 16G (default: 16G)
  --no-grid           Skip grid search (passthrough to run_strategy.py)
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
        --mem)          MEM="$2"; shift 2 ;;
        --no-grid)      PASSTHROUGH+=("--no-grid"); shift ;;
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

# Directory paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/code/scripts"
LOGS_DIR="$ROOT_DIR/logs"
VENV="$ROOT_DIR/.venv/bin/activate"
mkdir -p "$LOGS_DIR"

JOB_NAME="${STRATEGY}_u${UNIVERSE}"
echo "Submitting: ${JOB_NAME}  (time=${TIME_LIMIT}, cpus=${CPUS}, mem=${MEM})"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --output=${LOGS_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOGS_DIR}/${JOB_NAME}_%j.err
#SBATCH --chdir=${SCRIPTS_DIR}

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
