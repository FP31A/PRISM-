#!/bin/bash
#PBS -N prism_orca
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=12:00:00
#PBS -J 1-50
#PBS -o logs/orca/orca_${PBS_ARRAY_INDEX}.out
#PBS -e logs/orca/orca_${PBS_ARRAY_INDEX}.err

cd ~/projects/PRISM
mkdir -p logs/orca

# Load OpenMPI — required by ORCA 6 even for nprocs=1
module load openmpi/4.1.6-gcc11

ORCA_BIN="/scratch/users/nus/chmv673/orca/orca"
INDEX_FILE="data/transition1x/tier3_targets/rxn_ids.txt"
DFT_DIR="data/transition1x/dft_singlepoints"
TOTAL=$(wc -l < "$INDEX_FILE")

export OMP_NUM_THREADS=1

# 50 slots × 20 reactions = 1000
CHUNK_SIZE=20
START=$(( (PBS_ARRAY_INDEX - 1) * CHUNK_SIZE + 1 ))
END=$(( PBS_ARRAY_INDEX * CHUNK_SIZE ))
[ "$END" -gt "$TOTAL" ] && END=$TOTAL

echo "Slot $PBS_ARRAY_INDEX: reactions $START to $END"

PASS=0
FAIL=0

for i in $(seq $START $END); do
    RXN_ID=$(sed -n "${i}p" "$INDEX_FILE")
    [ -z "$RXN_ID" ] && continue

    RXN_DIR="${DFT_DIR}/${RXN_ID}"

    # Restart guard — skip if both already converged
    if [ -f "${RXN_DIR}/ts.out" ] && [ -f "${RXN_DIR}/reactant.out" ]; then
        if grep -q "FINAL SINGLE POINT ENERGY" "${RXN_DIR}/ts.out" && \
           grep -q "FINAL SINGLE POINT ENERGY" "${RXN_DIR}/reactant.out"; then
            echo "[$(date +%H:%M:%S)] $RXN_ID already complete, skipping."
            PASS=$(( PASS + 1 ))
            continue
        fi
    fi

    echo "[$(date +%H:%M:%S)] Starting $RXN_ID"

    # TS single-point
    if ! grep -q "FINAL SINGLE POINT ENERGY" "${RXN_DIR}/ts.out" 2>/dev/null; then
        cd "$RXN_DIR"
        timeout 3600 $ORCA_BIN ts.inp > ts.out 2>&1
        cd ~/projects/PRISM
    fi

    # Reactant single-point
    if ! grep -q "FINAL SINGLE POINT ENERGY" "${RXN_DIR}/reactant.out" 2>/dev/null; then
        cd "$RXN_DIR"
        timeout 3600 $ORCA_BIN reactant.inp > reactant.out 2>&1
        cd ~/projects/PRISM
    fi

    # Check results
    TS_OK=$(grep -c "FINAL SINGLE POINT ENERGY" "${RXN_DIR}/ts.out" 2>/dev/null || echo 0)
    R_OK=$(grep -c "FINAL SINGLE POINT ENERGY" "${RXN_DIR}/reactant.out" 2>/dev/null || echo 0)

    if [ "$TS_OK" -gt 0 ] && [ "$R_OK" -gt 0 ]; then
        echo "  $RXN_ID: CONVERGED"
        PASS=$(( PASS + 1 ))
    else
        echo "  $RXN_ID: FAILED (ts=$TS_OK, r=$R_OK)"
        FAIL=$(( FAIL + 1 ))
    fi
done

echo "Slot $PBS_ARRAY_INDEX finished. PASS=$PASS FAIL=$FAIL"