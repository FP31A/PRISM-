#!/usr/bin/env python3
"""
Parallel NEB launcher for a single PBS array slot.
Runs N reactions concurrently using ProcessPoolExecutor.
OMP_NUM_THREADS=1 per worker to avoid oversubscription.
"""

import os
import sys
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Force single-threaded xTB BEFORE any imports that might touch OpenMP
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def run_reaction(rxn_id: str) -> tuple[str, str]:
    """
    Run neb_runner.py for one reaction. Returns (rxn_id, status).
    Skips if already CONVERGED (restart guard).
    """
    status_file = Path(f"data/transition1x/neb_calibration/{rxn_id}/status.txt")

    # Restart guard
    if status_file.exists():
        status = status_file.read_text().strip()
        if status == "CONVERGED":
            return rxn_id, "SKIPPED_ALREADY_DONE"

    env = os.environ.copy()  # inherits OMP_NUM_THREADS=1 set above

    result = subprocess.run(
        [sys.executable, "src/descriptors/neb_runner.py", "--rxn_id", rxn_id],
        env=env,
        capture_output=False,   # let stdout/stderr flow to PBS log
    )

    # Read status written by neb_runner.py
    try:
        status = status_file.read_text().strip()
    except FileNotFoundError:
        status = "MISSING_STATUS_FILE"

    return rxn_id, status


def load_rxn_ids(index_file: str, start: int, end: int) -> list[str]:
    """Read reaction IDs from a headerless CSV (col 2 = rxn_id)."""
    rxn_ids = []
    with open(index_file) as f:
        lines = f.readlines()
    for i in range(start - 1, end):          # convert to 0-indexed
        if i >= len(lines):
            break
        parts = lines[i].strip().split(",")
        if len(parts) >= 2 and parts[1].strip():
            rxn_ids.append(parts[1].strip())
    return rxn_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   type=int, required=True)
    parser.add_argument("--end",     type=int, required=True)
    parser.add_argument("--index",   type=str, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    rxn_ids = load_rxn_ids(args.index, args.start, args.end)
    print(f"Loaded {len(rxn_ids)} reactions for this slot.", flush=True)

    counts = {"CONVERGED": 0, "SKIPPED_ALREADY_DONE": 0, "FAILED": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_reaction, rxn_id): rxn_id for rxn_id in rxn_ids}
        for future in as_completed(futures):
            rxn_id, status = future.result()
            if status in counts:
                counts[status] += 1
            else:
                counts["FAILED"] += 1
            print(f"  {rxn_id}: {status}", flush=True)

    print("\n--- Slot Summary ---")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()