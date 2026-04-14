# -*- coding: utf-8 -*-
"""
PRISM Step 6B.3 — Extract Tier 2 Descriptors
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read

INDEX_FILE  = "data/transition1x/processed/neb_calibration/rxn_index.csv"
CALIB_DIR   = "data/transition1x/neb_calibration"               # fixed
OUTPUT_FILE = "data/transition1x/descriptors/stream_b_geometric_tier2_raw.parquet"
N_IMAGES    = 9

def compute_rmsd(coords1, coords2):
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def compute_curvature(R_prev, R_ts, R_next):
    v_prev = R_prev.flatten()
    v_ts   = R_ts.flatten()
    v_next = R_next.flatten()
    numerator   = np.linalg.norm(v_next - 2 * v_ts + v_prev)
    denominator = np.linalg.norm(v_next - v_prev) ** 2
    return np.nan if denominator == 0 else numerator / denominator

def parse_energy_file(energy_file):
    """Returns (E_reactant, E_ts, dE_NEB) from neb_energies.txt."""
    lines = open(energy_file).readlines()
    e_r   = float(lines[0].split("=")[1].strip())
    e_ts  = float(lines[1].split("=")[1].strip())
    dE    = float(lines[2].split("=")[1].strip())
    return e_r, e_ts, dE

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    index_df = pd.read_csv(INDEX_FILE, header=None,
                           names=["array_idx", "rxn_id", "rmg_family"])
    results  = []
    total    = len(index_df)

    for i, row in index_df.iterrows():
        rxn_id   = row["rxn_id"]
        rxn_dir  = os.path.join(CALIB_DIR, str(rxn_id))

        status_file = os.path.join(rxn_dir, "status.txt")
        energy_file = os.path.join(rxn_dir, "neb_energies.txt")
        traj_file   = os.path.join(rxn_dir, "neb_ci.traj")

        record = {
            "rxn_id":          rxn_id,
            "E_NEB_strain":    np.nan,
            "RMSD_IRC":        np.nan,
            "curvature_kappa": np.nan,
            "dE_NEB":          np.nan,
            "neb_converged":   False,
            "neb_quality":     "MISSING",
        }

        # Read quality status
        if os.path.exists(status_file):
            record["neb_quality"] = open(status_file).read().strip()
        record["neb_converged"] = (record["neb_quality"] == "CONVERGED")

        # Process all reactions that produced geometry files, not just CONVERGED
        has_geometry = os.path.exists(energy_file) and os.path.exists(traj_file)
        if not has_geometry:
            results.append(record)
            continue

        try:
            # Energies — E_NEB_strain and dE_NEB are both E_ts - E_reactant
            # (same value, kept as separate columns to match roadmap descriptor schema)
            e_r, e_ts, dE_NEB     = parse_energy_file(energy_file)
            record["dE_NEB"]      = dE_NEB
            record["E_NEB_strain"] = dE_NEB

            # Final converged band — last N_IMAGES frames of trajectory
            band     = read(traj_file, index=f"-{N_IMAGES}:")
            energies = [img.get_potential_energy() for img in band]
            ts_idx   = np.argmax(energies)

            R_reactant = band[0].positions
            R_ts       = band[ts_idx].positions
            record["RMSD_IRC"] = compute_rmsd(R_reactant, R_ts)

            # Curvature — only valid if TS is an interior image
            if 0 < ts_idx < (N_IMAGES - 1):
                record["curvature_kappa"] = compute_curvature(
                    band[ts_idx - 1].positions,
                    R_ts,
                    band[ts_idx + 1].positions
                )

        except Exception as e:
            print(f"  WARNING: {rxn_id} parse failed ({record['neb_quality']}): {e}")

        results.append(record)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total} processed...", flush=True)

    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_FILE, index=False)

    # Summary
    print(f"\nSaved to {OUTPUT_FILE}")
    print(df["neb_quality"].value_counts().to_string())
    print(f"\nDescriptors extracted (has E_NEB_strain): {df['E_NEB_strain'].notna().sum()}")
    print(f"RMSD extracted:      {df['RMSD_IRC'].notna().sum()}")
    print(f"Curvature extracted: {df['curvature_kappa'].notna().sum()}")

if __name__ == "__main__":
    main()