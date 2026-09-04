#!/bin/bash
# Submit the in-sample against realized risk diagnostic to the Euler cluster.
#
# Runs code/scripts/in_sample_analysis.py for one risk measure on one universe,
# or on all three at once as three independent jobs. Each job solves the
# unregularized joint and overlay programs plus the 1/N control at every
# quarterly rebalancing date (~91 dates x 3 solves per universe) and writes its
# own row-level CSV, so the jobs never touch each other's output and can run
# concurrently.
#
# Unlike run_backtest.sh no BACKTEST_SCRATCH_DIR is needed: the diagnostic
# recomputes the hedged and excess-currency returns in memory and writes nothing
# back to data/processed/.
#
# The figure is built separately, once every universe's CSV has landed:
#   python code/scripts/in_sample_analysis.py <measure> --plot-only
#
# Usage:
#   ./run_in_sample_analysis.sh --measure <es|mv|mad|smv|soes> --universe <10|50|100> [options]
#   ./run_in_sample_analysis.sh --measure <es|mv|mad|smv|soes> --all-universes [options]
#
# Options:
#   --measure NAME      Risk measure: es, mv, mad, smv or soes (required)
#   --universe N        Top-N universe: 10, 50, or 100 (default: 10)
#   --all-universes     Submit one job per universe instead of a single one
#   --time HH:MM:SS     SLURM time limit (default: 24:00:00, max 720:00:00)
#   --cpus N            CPUs per task (default: 4 -> Gurobi runs 3 threads)
#   --mem-per-cpu SIZE  Memory per CPU (default: tiered by universe, 4G/8G/16G)
#   --log-level LEVEL   DEBUG|INFO|WARNING (passthrough)

set -euo pipefail

MEASURE=""
UNIVERSE=10
ALL_UNIVERSES=0
TIME_LIMIT="24:00:00"
CPUS=4
MEM_PER_CPU=""
PASSTHROUGH=()
MAX_HOURS=720

usage() {
    cat <<'USAGE'
Submit the in-sample vs realized risk diagnostic to the Euler cluster.

Usage:
  ./run_in_sample_analysis.sh --measure <es|mv|mad|smv|soes> --universe <10|50|100> [options]
  ./run_in_sample_analysis.sh --measure <es|mv|mad|smv|soes> --all-universes [options]

  --measure NAME      Risk measure: es, mv, mad, smv or soes (required)
  --universe N        Top-N universe: 10, 50, or 100 (default: 10)
  --all-universes     Submit one job per universe instead of a single one
  --time HH:MM:SS     SLURM time limit (default: 24:00:00, max 720:00:00)
  --cpus N            CPUs per task (default: 4 -> Gurobi runs 3 threads)
  --mem-per-cpu SIZE  Memory per CPU (default: tiered by universe, 4G/8G/16G)
  --log-level LEVEL   DEBUG|INFO|WARNING (passthrough)
USAGE
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --measure)         MEASURE="$2"; shift 2 ;;
        --universe)        UNIVERSE="$2"; shift 2 ;;
        --all-universes)   ALL_UNIVERSES=1; shift ;;
        --time)            TIME_LIMIT="$2"; shift 2 ;;
        --cpus)            CPUS="$2"; shift 2 ;;
        --mem-per-cpu)     MEM_PER_CPU="$2"; shift 2 ;;
        --log-level)       PASSTHROUGH+=("--log-level" "$2"); shift 2 ;;
        -h|--help)         usage ;;
        *)                 echo "Error: unknown argument '$1'"; usage ;;
    esac
done

[[ -z "$MEASURE" ]] && { echo "Error: --measure is required."; usage; }

case "$MEASURE" in
    es|mv|mad|smv|soes) ;;
    *) echo "Error: --measure must be es, mv, mad, smv or soes."; exit 1 ;;
esac

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
VENV="$ROOT_DIR/code/.venv/bin/activate"
mkdir -p "$LOGS_DIR"

# Estimation windows hold ~2600 daily observations against 101 assets at top-10,
# ~584 at top-50 and ~1051 at top-100, so the programs grow by roughly an order
# of magnitude across the three; memory is tiered to match.
default_mem() {
    case "$1" in
        10)  echo "4G" ;;
        50)  echo "8G" ;;
        100) echo "16G" ;;
    esac
}

submit_one() {
    local universe="$1"
    local mem="${MEM_PER_CPU:-$(default_mem "$universe")}"
    local job_name="insample_${MEASURE}_u${universe}"

    echo "Submitting: ${job_name}  (time=${TIME_LIMIT}, cpus=${CPUS}, mem-per-cpu=${mem})"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem-per-cpu=${mem}
#SBATCH --output=${LOGS_DIR}/${job_name}_%j.out
#SBATCH --error=${LOGS_DIR}/${job_name}_%j.err
#SBATCH --chdir=${SCRIPTS_DIR}

module load stack/2024-06 gcc/12.2.0 python/3.12.8
module load gurobi/13.0.0
source ${VENV}

echo "Starting ${job_name} at \$(date) on \$(hostname)"
python in_sample_analysis.py ${MEASURE} --universes ${universe} --no-plot ${PASSTHROUGH[*]:-}
status=\$?
echo "Finished ${job_name} at \$(date) with exit code \${status}"
exit \${status}
EOF
}

if (( ALL_UNIVERSES )); then
    # Heaviest first so the long pole starts immediately.
    for universe in 100 50 10; do
        submit_one "$universe"
    done
    echo "Three jobs submitted. Check status with: squeue --me"
    echo "Once all have finished: python code/scripts/in_sample_analysis.py ${MEASURE} --plot-only"
else
    submit_one "$UNIVERSE"
    echo "Once it has finished: python code/scripts/in_sample_analysis.py ${MEASURE} --plot-only"
fi
