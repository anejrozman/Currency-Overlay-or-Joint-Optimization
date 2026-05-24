#!/bin/bash

# Paths (relative to this script's location, i.e. master_thesis/code/scripts/euler/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/code/scripts"
LOGS_DIR="$ROOT_DIR/logs"
VENV="$ROOT_DIR/.venv/bin/activate"

# Submit assets_zero.py
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=assets_zero
#SBATCH --time=90:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=${LOGS_DIR}/assets_zero_%j.out
#SBATCH --error=${LOGS_DIR}/assets_zero_%j.err
#SBATCH --chdir=${SCRIPTS_DIR}

module load gurobi/13.0.0
source ${VENV}

echo "Starting assets_zero.py at \$(date)"
python assets_zero.py
echo "Finished assets_zero.py at \$(date) with exit code \$?"
EOF

# Submit assets_full.py
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=assets_full
#SBATCH --time=90:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=${LOGS_DIR}/assets_full_%j.out
#SBATCH --error=${LOGS_DIR}/assets_full_%j.err
#SBATCH --chdir=${SCRIPTS_DIR}

module load gurobi/13.0.0
source ${VENV}

echo "Starting assets_full.py at \$(date)"
python assets_full.py
echo "Finished assets_full.py at \$(date) with exit code \$?"
EOF

echo "Both jobs submitted. Check status with: squeue --me"
