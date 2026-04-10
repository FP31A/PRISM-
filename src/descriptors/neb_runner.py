# -*- coding: utf-8 -*-
"""
PRISM Step 6B.2 — CI-NEB Runner for ASPIRE2A Array

Runs a 9-image Climbing-Image Nudged Elastic Band calculation using GFN2-xTB.
Executed via PBS Pro array, reading rxn_id from command line.
"""

import os
import sys
import signal
import argparse
import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.mep import NEB
from ase.optimize import FIRE
from xtb.ase.calculator import XTB

# --- Configuration ---
PARQUET_PATH = "data/transition1x/processed/final_curated_reactions.parquet"
OUTPUT_BASE = "data/transition1x/neb_calibration"
N_IMAGES = 9

def setup_signal_handling(out_dir):
    """Catch PBS Pro walltime SIGTERM to save status as PARTIAL."""
    def handle_sigterm(signum, frame):
        with open(os.path.join(out_dir, "status.txt"), "w") as f:
            f.write("PARTIAL")
        print("Caught SIGTERM (Walltime limit). Exiting gracefully.")
        sys.exit(1)
    signal.signal(signal.SIGTERM, handle_sigterm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rxn_id", type=str, required=True, help="Reaction ID to process")
    args = parser.parse_args()
    rxn_id = args.rxn_id

    out_dir = os.path.join(OUTPUT_BASE, rxn_id)
    os.makedirs(out_dir, exist_ok=True)
    setup_signal_handling(out_dir)

    # 1. Load Geometries
    df = pd.read_parquet(PARQUET_PATH)
    row = df[df["rxn_id"] == rxn_id]
    if len(row) == 0:
        print(f"Error: rxn_id {rxn_id} not found.")
        sys.exit(1)
    
    r_path = row.iloc[0]["r_xyz_path"]
    p_path = row.iloc[0]["p_xyz_path"]
    
    reactant = ase_read(r_path)
    product = ase_read(p_path)

    # 2. Build 9-image NEB and Interpolate via IDPP
    images = [reactant.copy()]
    for _ in range(N_IMAGES - 2):
        images.append(reactant.copy())
    images.append(product.copy())

    neb = NEB(images)
    try:
        neb.interpolate("idpp")
    except Exception as e:
        with open(os.path.join(out_dir, "status.txt"), "w") as f:
            f.write("FAILED_IDPP")
        sys.exit(1)

    # 3. Attach xTB Calculators to ALL images
    # ASE requires endpoint forces to compute tangents for adjacent images.
    # ASE will automatically keep the endpoints frozen during optimization.
    for image in images:
        image.calc = XTB(method="GFN2-xTB")

    # 4. Standard NEB Optimization (fmax < 0.1 eV/A)
    # Using a high max_steps limit for the standard NEB phase
    opt = FIRE(neb, trajectory=os.path.join(out_dir, "neb.traj"), logfile=os.path.join(out_dir, "neb.log"))
    try:
        opt.run(fmax=0.1, steps=1000) 
    except Exception as e:
        with open(os.path.join(out_dir, "status.txt"), "w") as f:
            f.write("FAILED_STD_NEB")
        sys.exit(1)

    # 5. Climbing-Image NEB Optimization (fmax < 0.05 eV/A)
    neb.climb = True
    try:
        # Enforce the 50-step limit per the roadmap
        opt.run(fmax=0.05, steps=50)
    except Exception as e:
        with open(os.path.join(out_dir, "status.txt"), "w") as f:
            f.write("FAILED_CI_NEB")
        sys.exit(1)
   

    # 6. Extract TS (Highest Energy Image)
    energies = []
    for img in images[1:-1]:
        energies.append(img.get_potential_energy())
    
    ts_idx = np.argmax(energies) + 1  # +1 because endpoints are excluded
    ts_image = images[ts_idx]
    
    ase_write(os.path.join(out_dir, "ci_neb_ts.xyz"), ts_image)

    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("CONVERGED")

    print(f"Successfully converged CI-NEB for {rxn_id}")

if __name__ == "__main__":
    main()