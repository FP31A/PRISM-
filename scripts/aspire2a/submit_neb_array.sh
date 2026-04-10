#!/bin/bash
#PBS -N prism_neb
#PBS -q normal
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=02:00:00
#PBS -J 1-3000
#PBS -o logs/neb/
#PBS -e logs/neb/

cd ~/projects/PRISM

# Ensure output directories exist before tasks start writing
mkdir -p logs/neb data/transition1x/neb_calibration

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate prism

INDEX_FILE="data/transition1x/processed/neb_calibration/rxn_index.csv"

LINE=$(sed -n "${PBS_ARRAY_INDEX}p" $INDEX_FILE)

RXN_ID=$(echo $LINE | awk -F',' '{print $2}')

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Starting NEB for Reaction ID: $RXN_ID on task $PBS_ARRAY_INDEX"

# 5. Execute the NEB runner 
python src/descriptors/neb_runner.py --rxn_id $RXN_ID