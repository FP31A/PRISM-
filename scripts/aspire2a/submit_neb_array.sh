#!/bin/bash
#PBS -N prism_neb
#PBS -q normal
#PBS -l select=1:ncpus=24:mem=48gb
#PBS -l walltime=02:00:00           
#PBS -J 1-100                        
#PBS -o logs/neb/neb_${PBS_ARRAY_INDEX}.out
#PBS -e logs/neb/neb_${PBS_ARRAY_INDEX}.err

cd ~/projects/PRISM
mkdir -p logs/neb data/transition1x/neb_calibration

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate prism

INDEX_FILE="data/transition1x/processed/neb_calibration/rxn_index.csv"
TOTAL_REACTIONS=$(wc -l < "$INDEX_FILE")

CHUNK_SIZE=30
START=$(( (PBS_ARRAY_INDEX - 1) * CHUNK_SIZE + 1 ))
END=$(( PBS_ARRAY_INDEX * CHUNK_SIZE ))
[ "$END" -gt "$TOTAL_REACTIONS" ] && END=$TOTAL_REACTIONS

echo "Slot $PBS_ARRAY_INDEX: reactions $START to $END ($(( END - START + 1 )) reactions)"

python scripts/aspire2a/launch_chunk.py \
    --start "$START" \
    --end   "$END"   \
    --index "$INDEX_FILE" \
    --workers 24

echo "Slot $PBS_ARRAY_INDEX finished."