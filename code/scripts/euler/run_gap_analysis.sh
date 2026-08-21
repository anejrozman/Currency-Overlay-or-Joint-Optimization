#!/bin/bash
# Submit the joint-vs-overlay gap experiment to the Euler cluster.
#
# Runs code/scripts/in_sample_vs_out_of_sample.py for one arm. The arm that
# actually needs the cluster is the shuffle arm: 20 draws x 3 measures x ~91
# rebalancing dates x 2 programs is roughly 11k solves, which took hours on a
# laptop under Clarabel and died twice when the shell went away.
#
#   shuffle   top-10, 20 permutation draws (the long pole, ~3h with Gurobi)
#   currency  top-50 with the overlay block cut to K = 10 and 6 (~13h on Clarabel)
#   grid      the 9 main cells; reads results/in_sample_analysis/ and re-solves
#             nothing, so it takes seconds and does not need a job at all
#
# --shuffle-mode selects what the permutation does:
#
#   rehedge  the security keeps its true local return and true FX exposure but is
#            hedged with the permuted currency (R = R_u + f_perm - e_perm), so it
#            carries residual exposure e_true - e_perm.  One hedge instrument per
#            security and the 14-currency block are preserved; only the alignment
#            with the true returns is destroyed.  This is the informative arm.
#   label    the permutation only selects which forward premium enters R_fh.
#            Because R_fh = R_local + f + R_local*f and R_local is untouched, this
#            moves returns by ~0.6% of their own volatility and leaves the
#            asset-currency correlation block intact to ~1e-04.  Null control only.
#
# Arms share results/in_sample_vs_out_of_sample/gaps.csv, and each run rewrites
# that file with its own arm's rows replaced.  Do not run two arms concurrently
# against the same --out-dir; submit them in sequence, or give each its own
# --out-dir and merge afterwards.  Shuffle rows are stored under the arm name
# "shuffle_<mode>", so the two modes can coexist without overwriting each other.
#
# The tables and figure are built separately, once the arms have landed:
#   python code/scripts/in_sample_vs_out_of_sample.py --plot-only
#
# Usage:
#   ./run_gap_analysis.sh [--arm <shuffle|currency|grid>] [options]
#
# Options:
#   --arm NAME           Arm to run (default: shuffle)
#   --shuffle-mode MODE  rehedge or label (default: rehedge)
#   --time HH:MM:SS      SLURM time limit (default: 24:00:00, max 720:00:00)
#   --cpus N             CPUs per task (default: 4 -> Gurobi runs 3 threads)
#   --mem-per-cpu SIZE   Memory per CPU (default: 4G for shuffle, 8G otherwise)
#   --log-level LEVEL    DEBUG|INFO|WARNING (passthrough)
#   --out-dir DIR        Output directory (passthrough).  Give concurrent jobs
#                        distinct directories; the default is shared.

set -euo pipefail

ARM="shuffle"
SHUFFLE_MODE="rehedge"
TIME_LIMIT="24:00:00"
CPUS=4
MEM_PER_CPU=""
PASSTHROUGH=()
MAX_HOURS=720

usage() {
    cat <<'USAGE'
Submit the joint-vs-overlay gap experiment to the Euler cluster.

Usage:
  ./run_gap_analysis.sh [--arm <shuffle|currency|grid>] [options]

  --arm NAME           Arm to run: shuffle, currency or grid (default: shuffle)
  --shuffle-mode MODE  rehedge (informative) or label (null control), default rehedge
  --time HH:MM:SS      SLURM time limit (default: 24:00:00, max 720:00:00)
  --cpus N             CPUs per task (default: 4 -> Gurobi runs 3 threads)
  --mem-per-cpu SIZE   Memory per CPU (default: 4G for shuffle, 8G otherwise)
  --log-level LEVEL    DEBUG|INFO|WARNING (passthrough)
  --out-dir DIR        Output directory (passthrough); use distinct dirs to run jobs concurrently
USAGE
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm)           ARM="$2"; shift 2 ;;
        --shuffle-mode)  SHUFFLE_MODE="$2"; shift 2 ;;
        --time)          TIME_LIMIT="$2"; shift 2 ;;
        --cpus)          CPUS="$2"; shift 2 ;;
        --mem-per-cpu)   MEM_PER_CPU="$2"; shift 2 ;;
        --log-level)     PASSTHROUGH+=("--log-level" "$2"); shift 2 ;;
        --out-dir)       PASSTHROUGH+=("--out-dir" "$2"); shift 2 ;;
        -h|--help)       usage ;;
        *)               echo "Error: unknown argument '$1'"; usage ;;
    esac
done

case "$ARM" in
    shuffle|currency|grid) ;;
    *) echo "Error: --arm must be shuffle, currency or grid."; exit 1 ;;
esac

case "$SHUFFLE_MODE" in
    rehedge|label) ;;
    *) echo "Error: --shuffle-mode must be rehedge or label."; exit 1 ;;
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

# The shuffle arm runs on top-10 (~130 assets against ~2600 daily observations),
# the currency arm on top-50 (~580 assets); memory is tiered to match.
if [[ -z "$MEM_PER_CPU" ]]; then
    if [[ "$ARM" == "shuffle" ]]; then MEM_PER_CPU="4G"; else MEM_PER_CPU="8G"; fi
fi

if [[ "$ARM" == "shuffle" ]]; then
    JOB_NAME="gap_shuffle_${SHUFFLE_MODE}"
    ARM_ARGS="--arms shuffle --shuffle-mode ${SHUFFLE_MODE}"
else
    JOB_NAME="gap_${ARM}"
    ARM_ARGS="--arms ${ARM}"
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

echo "Starting ${JOB_NAME} at \$(date) on \$(hostname)"
python in_sample_vs_out_of_sample.py ${ARM_ARGS} ${PASSTHROUGH[*]:-}
status=\$?
echo "Finished ${JOB_NAME} at \$(date) with exit code \${status}"
exit \${status}
EOF

echo "Check status with: squeue --me"
echo "Once it has finished: python code/scripts/in_sample_vs_out_of_sample.py --plot-only"
