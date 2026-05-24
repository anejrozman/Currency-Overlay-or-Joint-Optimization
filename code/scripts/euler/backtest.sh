#!/bin/bash

# -------------------------
# Job 1
# -------------------------
bsub -n 4 -W 24:00 -J "assets_zero" \
-o logs/script1_%J.out \
"cd ~/master_thesis/code/scripts && \
module load stack/2025-06 && \
module load gcc/8.5.0 && \
module load python/3.11.9 && \
module load gurobi/13.0.0 && \
source ~/master_thesis/code/scripts/.venv/bin/activate && \
python assets_zero.py"

# -------------------------
# Job 2
# -------------------------
bsub -n 4 -W 24:00 -J "assets_full" \
-o logs/script2_%J.out \
"cd ~/master_thesis/code/scripts && \
module load stack/2025-06 && \
module load gcc/8.5.0 && \
module load python/3.11.9 && \
module load gurobi/13.0.0 && \
source ~/master_thesis/code/scripts/.venv/bin/activate && \
python assets_full.py"
