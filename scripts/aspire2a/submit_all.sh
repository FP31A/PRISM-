#!/bin/bash
# Run from login node: bash scripts/aspire2a/submit_all.sh

cd ~/projects/PRISM
mkdir -p data/transition1x/neb_calibration logs/neb

# --- Round 1: all 3000 reactions ---
JOB1=$(qsub scripts/aspire2a/submit_neb_array.sh)
echo "Round 1 submitted: $JOB1"

# --- Round 2: safety pass (restart guard skips already-CONVERGED) ---
# afterany = starts even if some Round 1 slots failed
JOB2=$(qsub -W depend=afterany:${JOB1} scripts/aspire2a/submit_neb_array.sh)
echo "Round 2 (safety) submitted: $JOB2"

echo "$JOB1" >> data/transition1x/neb_calibration/job_ids.txt
echo "$JOB2" >> data/transition1x/neb_calibration/job_ids.txt

echo ""
echo "Expected completion: ~4-6 hours"
echo "Monitor with:  qstat -u $USER"
echo "Check progress: ls data/transition1x/neb_calibration/*/status.txt | wc -l"
echo "Status breakdown: grep -h '' data/transition1x/neb_calibration/*/status.txt | sort | uniq -c"