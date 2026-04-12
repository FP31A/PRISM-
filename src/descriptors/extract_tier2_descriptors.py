# -*- coding: utf-8 -*-
"""
PRISM Step 6B.3 — Extract Tier 2 Descriptors
Parses the output of the CI-NEB array to compute strain, RMSD, and path curvature.
"""

import os
import numpy as np
import pandas as pd
from ase.io import read

# --- Configuration ---
INDEX_FILE = "data/transition1x/processed/neb_calibration/rxn_index.csv"
CALIB_DIR = "data/transition1x/neb_calibration"
OUTPUT_FILE = "data/transition1x/descriptors/stream_b_geometric_tier2_raw.parquet"
N_IMAGES = 9

def compute_rmsd(coords1, coords2):
    """Calculates standard RMSD between two matched coordinate arrays."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def compute_curvature(R_prev, R_ts, R_next):
    """Computes path curvature at the saddle point."""
    v_prev = R_prev.flatten()
    v_ts = R_ts.flatten()
    v_next = R_next.flatten()

    numerator = np.linalg.norm(v_next - 2 * v_ts + v_prev)
    denominator = np.linalg.norm(v_next - v_prev)**2

    if denominator == 0:
        return np.nan
    return numerator / denominator

def main():
    # 1. Load the calibration index
    # Note: Assuming the CSV has no header based on our previous array setup
    index_df = pd.read_csv(INDEX_FILE, header=None, names=["array_idx", "rxn_id", "rmg_family"])
    
    results = []
    
    for _, row in index_df.iterrows():
        rxn_id = row["rxn_id"]
        rxn_dir = os.path.join(CALIB_DIR, str(rxn_id))
        
        status_file = os.path.join(rxn_dir, "status.txt")
        energy_file = os.path.join(rxn_dir, "neb_energies.txt")
        traj_file = os.path.join(rxn_dir, "neb_ci.traj")
        
        # Default empty row
        record = {
            "rxn_id": rxn_id,
            "E_NEB_strain": np.nan,
            "RMSD_IRC": np.nan,
            "curvature_kappa": np.nan,
            "dE_NEB": np.nan,
            "neb_converged": False
        }
        
        # 2. Check convergence status
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = f.read().strip()
                if status == "CONVERGED":
                    record["neb_converged"] = True
        
        if record["neb_converged"]:
            try:
                # 3. Parse Energies
                with open(energy_file, "r") as f:
                    lines = f.readlines()
                    dE_NEB = float(lines[2].split("=")[1].strip())
                    
                record["dE_NEB"] = dE_NEB
                record["E_NEB_strain"] = dE_NEB  # Same baseline reference in this context
                
                # 4. Extract Geometries from the final band in the trajectory
                # ASE NEB trajectories save the whole band at each step.
                # Reading the last N_IMAGES gives the final converged path.
                band = read(traj_file, index=f"-{N_IMAGES}:")
                
                energies = [img.get_potential_energy() for img in band]
                ts_idx = np.argmax(energies)
                
                # Geometries
                R_reactant = band[0].positions
                R_ts = band[ts_idx].positions
                
                # Compute RMSD
                record["RMSD_IRC"] = compute_rmsd(R_reactant, R_ts)
                
                # Compute Curvature (ensure we aren't at the very edge of the band)
                if 0 < ts_idx < (N_IMAGES - 1):
                    R_prev = band[ts_idx - 1].positions
                    R_next = band[ts_idx + 1].positions
                    record["curvature_kappa"] = compute_curvature(R_prev, R_ts, R_next)
                    
            except Exception as e:
                print(f"Warning: Failed to parse data for {rxn_id} despite CONVERGED status. Error: {e}")
                record["neb_converged"] = False
                
        results.append(record)
        
    # 5. Save to Parquet
    df_results = pd.DataFrame(results)
    df_results.to_parquet(OUTPUT_FILE, index=False)
    
    # Print summary
    converged_count = df_results["neb_converged"].sum()
    print(f"Extraction complete. Saved to {OUTPUT_FILE}")
    print(f"Successfully extracted {converged_count} / {len(df_results)} reactions ({(converged_count/len(df_results))*100:.1f}%).")

if __name__ == "__main__":
    main()