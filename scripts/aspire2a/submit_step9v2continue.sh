#!/bin/bash
#PBS -N prism_step9v2b
#PBS -q normal
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o logs/step9/step9v2b.out
#PBS -e logs/step9/step9v2b.err
#PBS -j oe

set -euo pipefail

# ── Navigate to project root ──
cd "${PBS_O_WORKDIR:-$HOME/projects/PRISM}"
mkdir -p logs/step9 results/figures

# ── Activate environment ──
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate prism

# ── Prevent thread over-subscription ──
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Optuna suppress noisy SQLite warnings ──
export PYTHONWARNINGS="ignore"

echo "=========================================================="
echo "PRISM Step 9v2b — Continuation (9.2 + 9.3 + 9.4)"
echo "=========================================================="
echo "Start:     $(date)"
echo "Node:      $(hostname)"
echo "CPUs:      $(nproc)"
echo "Memory:    $(free -h | awk '/Mem:/{print $2}')"
echo "Workdir:   $(pwd)"
echo "Python:    $(which python)"
echo "=========================================================="

# ── Run and log ──
python -u src/models/train_continue.py 2>&1 | tee logs/step9/step9v2b_full.log

echo "=========================================================="
echo "End:       $(date)"
echo "=========================================================="