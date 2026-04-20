#!/bin/bash
#PBS -N prism_step9
#PBS -q normal
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=14:00:00
#PBS -o logs/step9/step9.out
#PBS -e logs/step9/step9.err
#PBS -j oe

set -euo pipefail

# ── Navigate to project root ──
cd "${PBS_O_WORKDIR:-$HOME/projects/PRISM}"
mkdir -p logs/step9 results models

# ── Activate environment ──
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate prism

# ── Prevent thread over-subscription ──
# joblib/sklearn manage parallelism via N_JOBS=-1 (uses all 16 cores).
# Without these, numpy/xgboost may spawn 16 threads PER joblib worker,
# causing 16×16=256 threads competing for 16 cores → massive slowdown.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Optuna suppress noisy SQLite warnings ──
export PYTHONWARNINGS="ignore"

echo "=========================================================="
echo "PRISM Step 9 — Model Training, Ablation, Screening, LC"
echo "=========================================================="
echo "Start:     $(date)"
echo "Node:      $(hostname)"
echo "CPUs:      $(nproc)"
echo "Memory:    $(free -h | awk '/Mem:/{print $2}')"
echo "Workdir:   $(pwd)"
echo "Python:    $(which python)"
echo "=========================================================="

# ── Run ──
python -u src/models/train.py 2>&1 | tee logs/step9/step9_full.log

echo "=========================================================="
echo "End:       $(date)"
echo "=========================================================="