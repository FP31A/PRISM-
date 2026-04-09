# -*- coding: utf-8 -*-
"""
PRISM Step 6 — Stream B: Geometric Descriptors (Tier 1)

For each reaction:
  1. Load reactant and product geometries
  2. Verify atom ordering confidence (fallback to linear if < 0.8)
  3. Run 5-image IDPP interpolation (ASE)
  4. Extract midpoint image (image 2)
  5. Run GFN2-xTB single-point on midpoint and reactant
  6. Compute:
     - E_strain_IDPP = E_xTB(midpoint) - E_xTB(reactant)
     - RMSD between reactant and product
     - Delta PMI (principal moments of inertia)

Run on ASPIRE2A:
    python src/descriptors/geometric_t1x.py
"""

import os
import re
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for compute nodes
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from ase.io import read as ase_read
from ase.mep import NEB

# --- Configuration ---
INPUT_PARQUET = "data/grambow/processed/final_curated_reactions.parquet"
ELECTRONIC_PARQUET = "data/grambow/descriptors/electronic_descriptors.parquet"

# CHANGE (a): Updated output filename to match Task 6.5
OUTPUT_PARQUET = "data/grambow/descriptors/stream_b_geometric_tier1.parquet"
ERROR_LOG = "data/grambow/descriptors/geometric_errors.log"

# EXPLICIT PLOT PATHS
PLOT_PATH_UNFILTERED = "data/grambow/descriptors/estrain_vs_ea_unfiltered.png"
PLOT_PATH_CURATED = "data/grambow/descriptors/estrain_vs_ea_curated.png"

N_IMAGES = 5       # Total images including endpoints
N_JOBS = 24
XTB_CORES = "1"


def run_xtb_energy(atoms, rxn_id, label):
    """
    Run GFN2-xTB on an ASE Atoms object and return energy in eV.
    Writes a temporary xyz, runs xtb, parses energy.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = os.path.join(tmpdir, "input.xyz")

        # Write xyz manually from ASE Atoms
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        with open(xyz_path, "w") as f:
            f.write(f"{len(symbols)}\n")
            f.write(f"PRISM geometric {rxn_id} {label}\n")
            for sym, pos in zip(symbols, positions):
                f.write(f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")

        cmd = ["xtb", xyz_path, "--gfn", "2"]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = XTB_CORES

        try:
            result = subprocess.run(
                cmd, cwd=tmpdir, env=env,
                capture_output=True, text=True, timeout=120
            )
            output = result.stdout
            match = re.search(
                r"\|\s+TOTAL ENERGY\s+([\-\.\d]+)\s+Eh\s+\|", output
            )
            if match:
                return float(match.group(1)) * 27.2114  # Hartree -> eV
            else:
                return None
        except Exception:
            return None


def compute_rmsd(atoms1, atoms2):
    """RMSD between two geometries."""
    pos1 = atoms1.get_positions()
    pos2 = atoms2.get_positions()
    if pos1.shape != pos2.shape:
        return np.nan
    diff = pos1 - pos2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def compute_pmi(atoms):
    """Compute principal moments of inertia (sorted ascending)."""
    moments = atoms.get_moments_of_inertia()
    return np.sort(moments)


# CHANGE (c): Added force_linear to handle mapping confidence logic
def run_interpolation(reactant, product, n_images=5, force_linear=False):
    """
    Attempts IDPP. If force_linear is True, or if IDPP fails, falls back to Cartesian linear.
    Returns (images_list, method_string) or (None, "failed").
    """
    n_intermediate = n_images - 2
    images = [reactant.copy()]
    for _ in range(n_intermediate):
        images.append(reactant.copy())
    images.append(product.copy())

    if not force_linear:
        try:
            neb = NEB(images)
            neb.interpolate("idpp")
            return images, "idpp"
        except Exception:
            pass # Fall through to linear

    # Fallback to Cartesian linear interpolation
    try:
        neb = NEB(images)
        neb.interpolate()  # plain linear
        return images, "linear"
    except Exception:
        return None, "failed"

def process_reaction(row, e_xtb_reactant=None):
    rxn_id = row["rxn_id"]
    r_path = row["r_xyz_path"]
    p_path = row["p_xyz_path"]
    mapping_conf = row.get("mapping_confidence", 1.0)

    result = {
        "rxn_id": rxn_id,
        "E_strain_IDPP": np.nan,
        "RMSD_R_P": np.nan,
        "dPMI_1": np.nan,
        "dPMI_2": np.nan,
        "dPMI_3": np.nan,
        "interpolation_method": "failed", # Track how the geometry was generated
    }

    try:
        reactant = ase_read(r_path)
        product = ase_read(p_path)
    except Exception as e:
        return result

    if len(reactant) != len(product):
        return result

    result["RMSD_R_P"] = compute_rmsd(reactant, product)

    try:
        pmi_r = compute_pmi(reactant)
        pmi_p = compute_pmi(product)
        dpmi = pmi_p - pmi_r
        result["dPMI_1"] = dpmi[0]
        result["dPMI_2"] = dpmi[1]
        result["dPMI_3"] = dpmi[2]
    except Exception:
        pass

    # Force linear if mapping is bad, otherwise try IDPP
    force_linear = False
    if pd.notna(mapping_conf) and mapping_conf < 0.8:
        force_linear = True
        with open(ERROR_LOG, "a") as f:
            f.write(f"{rxn_id} | Warning: Low atom mapping confidence ({mapping_conf:.2f}). Forced linear.\n")

    images, method = run_interpolation(reactant, product, n_images=N_IMAGES, force_linear=force_linear)
    
    # SAVE THE METHOD USED TO THE RESULTS
    result["interpolation_method"] = method

    if method == "failed":
        with open(ERROR_LOG, "a") as f:
            f.write(f"{rxn_id} | Error: Both IDPP and linear interpolation failed\n")
        return result

    mid_idx = len(images) // 2
    midpoint = images[mid_idx]

    e_mid = run_xtb_energy(midpoint, rxn_id, "midpoint")
    if e_mid is None:
        return result

    if e_xtb_reactant is not None and not np.isnan(e_xtb_reactant):
        e_r = e_xtb_reactant
    else:
        e_r = run_xtb_energy(reactant, rxn_id, "reactant")
        if e_r is None:
            return result

    result["E_strain_IDPP"] = e_mid - e_r
    return result


def main():
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

    with open(ERROR_LOG, "w") as f:
        f.write("--- Geometric Descriptor Error Log ---\n")

    print("1. Loading curated reactions...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"   {len(df)} reactions loaded.")

    print("2. Loading Stream A electronic descriptors (for reactant energies)...")
    try:
        elec = pd.read_parquet(ELECTRONIC_PARQUET)
        e_reactant_map = {}
        print("   Will recompute reactant energies (not cached in Stream A).")
    except Exception:
        e_reactant_map = {}
        print("   Stream A not found. Will compute all energies fresh.")

    rows_with_cache = []
    for _, row in df.iterrows():
        rxn_id = row["rxn_id"]
        e_r = e_reactant_map.get(rxn_id, None)
        rows_with_cache.append((row, e_r))

    print(f"\n3. Computing geometric descriptors ({len(df)} reactions, {N_JOBS} workers)...")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_reaction)(row, e_r)
        for row, e_r in rows_with_cache
    )

    # --- 4. Consolidate ---
    print("\n4. Consolidating results...")
    geo_df = pd.DataFrame(results)

    # --- 5. Updated Stats (using the new method column) ---
    # We now calculate success based on the interpolation_method string
    idpp_mask = (geo_df["interpolation_method"] == "idpp")
    linear_mask = (geo_df["interpolation_method"] == "linear")
    
    idpp_count = idpp_mask.sum()
    linear_count = linear_mask.sum()
    idpp_rate = (idpp_count / len(geo_df)) * 100

    strain_valid = geo_df["E_strain_IDPP"].notna().sum()
    # Check physical validity only on the IDPP ones
    strain_negative = (geo_df.loc[idpp_mask, "E_strain_IDPP"] < 0).sum()

    print(f"   IDPP success: {idpp_count}/{len(geo_df)} ({idpp_rate:.1f}%)")
    print(f"   Forced Linear: {linear_count}")
    print(f"   Valid E_strain (total): {strain_valid}")
    print(f"   Negative E_strain in IDPP (unexpected): {strain_negative}")

    # Save the parquet
    geo_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n✓ Saved results to {OUTPUT_PARQUET}")

    # --- 6. Generating Validation Scatter Plots ---
    print("\n6. Generating Validation Scatter Plots...")
    
    # Merge with original DF to get Ea_eV
    plot_df = geo_df.merge(df[['rxn_id', 'Ea_eV']], on='rxn_id', how='inner')
    plot_df = plot_df.dropna(subset=['E_strain_IDPP', 'Ea_eV'])

    if len(plot_df) > 0:
        # --- PLOT 1: THE UNFILTERED DATA (Showing the clashes/spikes) ---
        plt.figure(figsize=(8, 6))
        
        # Plot linear/forced points in RED
        bad_pts = plot_df[plot_df['interpolation_method'] == 'linear']
        plt.scatter(bad_pts['Ea_eV'], bad_pts['E_strain_IDPP'], 
                    alpha=0.6, color='red', edgecolor='k', s=20, label='Linear (Atomic Clashes)')
        
        # Plot IDPP points in BLUE
        good_pts = plot_df[plot_df['interpolation_method'] == 'idpp']
        plt.scatter(good_pts['Ea_eV'], good_pts['E_strain_IDPP'], 
                    alpha=0.6, color='dodgerblue', edgecolor='k', s=20, label='IDPP (Physical)')
        
        plt.xlabel('True Activation Energy $\Delta E^\ddagger$ (eV)')
        plt.ylabel('Strain Energy $E_{strain}$ (eV)')
        plt.title('Unfiltered Strain vs Ea (Demonstrating Data Pathologies)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOT_PATH_UNFILTERED, dpi=150)
        plt.close()
        print(f"   Saved unfiltered plot to {PLOT_PATH_UNFILTERED}")

        # --- PLOT 2: THE CURATED DATA (IDPP only, < 50 eV) ---
        # Only use IDPP results and filter out extreme outliers for the "good" graph
        valid_plot_df = good_pts[good_pts['E_strain_IDPP'] < 50.0]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(valid_plot_df['Ea_eV'], valid_plot_df['E_strain_IDPP'], 
                    alpha=0.5, color='dodgerblue', edgecolor='k', s=20, label='IDPP Only')
        
        # Reference y=x line
        max_val = min(valid_plot_df['Ea_eV'].max(), valid_plot_df['E_strain_IDPP'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='y = x')
        
        plt.xlabel('True Activation Energy $\Delta E^\ddagger$ (eV)')
        plt.ylabel('IDPP Strain Energy $E_{strain}^{IDPP}$ (eV)')
        plt.title('Curated Tier 1 Geometric Strain vs True Activation Energy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOT_PATH_CURATED, dpi=150)
        plt.close()
        print(f"   Saved curated plot to {PLOT_PATH_CURATED}")
    else:
        print("   Not enough valid data points to generate plots.")

if __name__ == "__main__":
    main()