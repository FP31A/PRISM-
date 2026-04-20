#!/bin/bash
#PBS -N prism_step9b
#PBS -q normal
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o logs/step9/step9b.out
#PBS -e logs/step9/step9b.err
#PBS -j oe

set -euo pipefail

cd "${PBS_O_WORKDIR:-$HOME/projects/PRISM}"
mkdir -p logs/step9 results

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate prism

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONWARNINGS="ignore"

echo "=========================================================="
echo "PRISM Step 9b — Continuation (9.2 + 9.3 + 9.4)"
echo "=========================================================="
echo "Start:     $(date)"
echo "Node:      $(hostname)"
echo "CPUs:      $(nproc)"
echo "Workdir:   $(pwd)"
echo "=========================================================="

python -u src/models/train_continue.py 2>&1 | tee logs/step9/step9b_full.log

echo "=========================================================="
echo "End:       $(date)"
echo "=========================================================="