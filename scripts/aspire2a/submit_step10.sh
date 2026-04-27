#!/bin/bash
#PBS -N prism_step10
#PBS -q normal
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=04:00:00
#PBS -o logs/step10/step10.out
#PBS -e logs/step10/step10.err
#PBS -j oe

set -euo pipefail

cd "${PBS_O_WORKDIR:-$HOME/projects/PRISM}"
mkdir -p logs/step10 results

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate prism

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONWARNINGS="ignore"

echo "=========================================================="
echo "PRISM Step 10 — Uncertainty Quantification"
echo "=========================================================="
echo "Start:     $(date)"
echo "Node:      $(hostname)"
echo "CPUs:      $(nproc)"
echo "=========================================================="

python -u src/models/uncertainty.py 2>&1 | tee logs/step10/step10_full.log

echo "=========================================================="
echo "End:       $(date)"
echo "=========================================================="