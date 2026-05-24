#!/bin/bash
# Submit one or more backtest jobs to the Euler cluster, all sharing the same time limit.
#
# Usage:
#   ./run_backtest.sh --time <HH:MM:SS> --script <s1.py> [--script <s2.py> ...]
# Max time limit: 120:00:00

set -euo pipefail

TIME_LIMIT=""
SCRIPTS=()
MAX_HOURS=120

usage() {
    echo "Usage:"
    echo "  $0 --time <HH:MM:SS> --script <s1.py> [--script <s2.py> ...]"
    echo "  $0 --time <HH:MM:SS> <s1.py> [<s2.py> ...]"
    echo ""
    echo "  --time    Time limit for every job (max ${MAX_HOURS}:00:00)"
    echo "  --script  Python script in code/scripts/ to submit"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --time)
            TIME_LIMIT="$2"; shift 2 ;;
        --script)
            SCRIPTS+=("$2"); shift 2 ;;
        -h|--help)
            usage ;;
        --*)
            echo "Error: unknown arguments '$1'"; usage ;;
        *)
            # Bare positional args are treated as script names
            SCRIPTS+=("$1"); shift ;;
    esac
done

[[ -z "$TIME_LIMIT" ]]      && { echo "Error: --time is required.";              usage; }
[[ ${#SCRIPTS[@]} -eq 0 ]]  && { echo "Error: at least one script is required."; usage; }

# Validate time format and limit
if [[ ! "$TIME_LIMIT" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]]; then
    echo "Error: --time must be in HH:MM:SS format (e.g. 90:00:00)."
    exit 1
fi

HOURS="${BASH_REMATCH[1]}"
if (( HOURS > MAX_HOURS )); then
    echo "Error: time limit ${TIME_LIMIT} exceeds the maximum of ${MAX_HOURS}:00:00."
    exit 1
fi

# Directory paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/code/scripts"
LOGS_DIR="$ROOT_DIR/logs"
VENV="$ROOT_DIR/.venv/bin/activate"

# Verify existence of scripts 
for SCRIPT_NAME in "${SCRIPTS[@]}"; do
    if [[ ! -f "$SCRIPTS_DIR/$SCRIPT_NAME" ]]; then
        echo "Error: script not found at $SCRIPTS_DIR/$SCRIPT_NAME"
        exit 1
    fi
done

# Submit one job per script
SUBMITTED=0
for SCRIPT_NAME in "${SCRIPTS[@]}"; do
    JOB_NAME="${SCRIPT_NAME%.py}"
    echo "Submitting: $SCRIPT_NAME  (time limit: $TIME_LIMIT, job name: $JOB_NAME)"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=${LOGS_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOGS_DIR}/${JOB_NAME}_%j.err
#SBATCH --chdir=${SCRIPTS_DIR}

module load gurobi/13.0.0
source ${VENV}

echo "Starting ${SCRIPT_NAME} at \$(date)"
python ${SCRIPT_NAME}
echo "Finished ${SCRIPT_NAME} at \$(date) with exit code \$?"
EOF

    (( ++SUBMITTED ))
done

echo ""
echo "${SUBMITTED} job(s) submitted. Check status with: squeue --me"
