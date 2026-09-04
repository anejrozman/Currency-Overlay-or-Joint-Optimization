#!/bin/bash
# Submit the joint-vs-overlay gap experiment to the Euler cluster.
#
# Runs code/scripts/in_sample_vs_out_of_sample.py for one arm. The arm that needs
# the cluster is the currency-count arm, which re-solves the joint and overlay
# programs at every rebalancing date for each tradable-currency count:
#
#   currency  top-50 at K = 10, 6 and top-10 at K = 10, 6, 2 (five cells x 4
#             measures x ~91 dates x 2 programs).  The top-50 half dominates the
#             cost; it took ~13h on a laptop under Clarabel for the three original
#             measures.  Adding semi-variance raises the per-date solve cost ~37%
#             (measured: the mv/es/mad joint+overlay pair costs 5.33s per date at
#             N=101, smv adds 1.95s), so budget ~18h under Clarabel, less under
#             Gurobi.  The 24h default still covers it, but not by much.
#   grid      the main cells, one per measure x universe; reads
#             results/in_sample_analysis/ and re-solves nothing, so it takes
#             seconds and does not need a job at all.  A measure with no cached
#             in_sample_<measure>_top<N>.csv is warned about and skipped, so a
#             newly added measure stays absent until in_sample_analysis.py is
#             run for it.
#
# Arms share results/in_sample_vs_out_of_sample/gaps.csv, and each run rewrites
# that file with its own arm's rows replaced.  Do not run two arms concurrently
# against the same --out-dir; submit them in sequence, or give each its own
# --out-dir and merge afterwards.
#
# The tables and figure are built separately, once the arms have landed:
#   python code/scripts/in_sample_vs_out_of_sample.py --plot-only
#
# --family regularized reuses the penalties each production run selected, read via
# results/runs/top_runs.csv.  reg_soes_joint and reg_soes_curr_overlay have finished
# on top-10 alone, so their top-50 and top-100 cells there are blank and the
# regularized arm covers soes only once those land.
#
# Usage:
#   ./run_gap_analysis.sh [--arm <currency|grid>] [options]
#
# Options:
#   --arm NAME           Arm to run: currency or grid (default: currency)
#   --family NAME        unregularized or regularized (default: unregularized)
#   --time HH:MM:SS      SLURM time limit (default: 24:00:00, max 720:00:00)
#   --cpus N             CPUs per task (default: 4 -> Gurobi runs 3 threads)
#   --mem-per-cpu SIZE   Memory per CPU (default: 8G)
#   --log-level LEVEL    DEBUG|INFO|WARNING (passthrough)
#   --out-dir DIR        Output directory.  Relative paths resolve against the repo
#                        root, not the job's working directory.  Give concurrent
#                        jobs distinct directories; the default is shared.

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

# The currency arm touches the top-50 universe (~580 assets against ~2600 daily
# observations) as well as top-10, so size memory for the larger of the two.
MEM_PER_CPU="${MEM_PER_CPU:-8G}"

# The job runs with --chdir code/scripts, so a relative --out-dir would land in
# code/scripts/<dir>.  Resolve it against the repo root instead.
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
#SBATCH --chdir=${SCRIPTS_DIR}

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
echo "Once it has finished: python code/scripts/in_sample_vs_out_of_sample.py --plot-only"
