# -*- coding: utf-8 -*-
"""
PRISM Step 6 — Stream B: Geometric Descriptors (Tier 1)

For each reaction:
  1. Build a 5-image IDPP path between reactant and product
  2. Extract the midpoint image
  3. Run GFN2-xTB single-point on midpoint and reactant
  4. Compute E_strain_IDPP = E_xTB(midpoint) - E_xTB(reactant)
  5. Compute RMSD(R, P) and delta-PMI
"""

import os
import re
import subprocess
import tempfile
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.neb import NEB
from joblib import Parallel, delayed

# --- Configuration ---
INPUT_PARQUET = "data/transition1x/processed/final_curated_reactions.parquet"
OUTPUT_PARQUET = "data/transition1x/descriptors/geometric_descriptors.parquet"
ERROR_LOG = "data/transition1x/descriptors/geometric_errors.log"
XTB_CORES = "1"


def run_xtb_energy(xyz_path):
    """Run GFN2-xTB single-point on a file and return energy in eV."""
    if not isinstance(xyz_path, str) or not os.path.exists(xyz_path):
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["xtb", os.path.abspath(xyz_path), "--gfn", "2"]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = XTB_CORES

        try:
            result = subprocess.run(
                cmd, cwd=tmpdir, env=env,
                capture_output=True, text=True, timeout=120
            )
            match = re.search(
                r"\|\s+TOTAL ENERGY\s+([\-\.\d]+)\s+Eh\s+\|", result.stdout
            )
            if match:
                return float(match.group(1)) * 27.2114
            return None
        except Exception:
            return None


def run_xtb_energy_on_atoms(atoms_obj):
    """Run xTB on an ASE Atoms object by writing a temp xyz file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = os.path.join(tmpdir, "input.xyz")
        write(xyz_path, atoms_obj)

        cmd = ["xtb", "input.xyz", "--gfn", "2"]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = XTB_CORES

        try:
            result = subprocess.run(
                cmd, cwd=tmpdir, env=env,
                capture_output=True, text=True, timeout=120
            )
            match = re.search(
                r"\|\s+TOTAL ENERGY\s+([\-\.\d]+)\s+Eh\s+\|", result.stdout
            )
            if match:
                return float(match.group(1)) * 27.2114
            return None
        except Exception:
            return None


def compute_rmsd(atoms_a, atoms_b):
    """RMSD between two structures (same atom count and ordering required)."""
    pos_a = atoms_a.get_positions()
    pos_b = atoms_b.get_positions()
    if pos_a.shape != pos_b.shape:
        return np.nan
    diff = pos_a - pos_b
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def compute_pmi(atoms_obj):
    """Compute principal moments of inertia (sorted ascending)."""
    try:
        moments = atoms_obj.get_moments_of_inertia()
        return np.sort(moments)
    except Exception:
        return np.array([np.nan, np.nan, np.nan])
    
def validate_atom_correspondence(reactant, product):
    """Check that atom ordering is consistent between R and P."""
    r_nums = reactant.get_atomic_numbers()
    p_nums = product.get_atomic_numbers()
    if len(r_nums) != len(p_nums):
        return False, "different atom counts"
    if not np.array_equal(r_nums, p_nums):
        return False, "atomic numbers don't match index-wise"
    # Check no atom needs to travel absurdly far (>5 Angstrom)
    displacements = np.linalg.norm(
        reactant.get_positions() - product.get_positions(), axis=1
    )
    max_disp = displacements.max()
    if max_disp > 8.0:
        return False, f"max displacement {max_disp:.1f} A (possible misorder)"
    return True, "ok"


def run_idpp(reactant, product, n_images=5):
    """
    Run IDPP interpolation between reactant and product.
    Returns the midpoint Atoms object, or None on failure.
    n_images includes endpoints: 5 = 3 intermediates + 2 endpoints.
    """
    try:
        if len(reactant) != len(product):
            return None
        if list(reactant.get_atomic_numbers()) != list(product.get_atomic_numbers()):
            return None

        images = [reactant.copy()]
        for _ in range(n_images - 2):
            images.append(reactant.copy())
        images.append(product.copy())

        neb = NEB(images)
        neb.interpolate("idpp")

        mid_idx = n_images // 2
        return images[mid_idx]

    except Exception:
        return None


def log_error(rxn_id, message):
    """Append error to log file."""
    with open(ERROR_LOG, "a") as f:
        f.write(f"{rxn_id} | {message}\n")


def process_reaction(row):
    """Compute all geometric descriptors for one reaction."""
    rxn_id = row["rxn_id"]

    desc = {
        "rxn_id": rxn_id,
        "E_strain_IDPP": np.nan,
        "rmsd_R_P": np.nan,
        "dPMI_1": np.nan,
        "dPMI_2": np.nan,
        "dPMI_3": np.nan,
        "idpp_success": False,
    }

    try:
        r_path = row["r_xyz_path"]
        p_path = row["p_xyz_path"]

        if not os.path.exists(r_path) or not os.path.exists(p_path):
            log_error(rxn_id, "File not found")
            return desc

        reactant = read(r_path)
        product = read(p_path)

        # --- RMSD(R, P) --- always computable
        desc["rmsd_R_P"] = compute_rmsd(reactant, product)

        # --- Delta PMI --- always computable
        pmi_r = compute_pmi(reactant)
        pmi_p = compute_pmi(product)
        desc["dPMI_1"] = pmi_p[0] - pmi_r[0]
        desc["dPMI_2"] = pmi_p[1] - pmi_r[1]
        desc["dPMI_3"] = pmi_p[2] - pmi_r[2]

        # --- IDPP interpolation ---
        midpoint = run_idpp(reactant, product, n_images=5)

        if midpoint is None:
            log_error(rxn_id, "IDPP interpolation failed")
            return desc

        # --- xTB on midpoint and reactant ---
        E_mid = run_xtb_energy_on_atoms(midpoint)
        E_r = run_xtb_energy(r_path)

        if E_mid is not None and E_r is not None:
            desc["E_strain_IDPP"] = E_mid - E_r
            desc["idpp_success"] = True
        else:
            log_error(rxn_id, "xTB failed on midpoint or reactant")

    except Exception as e:
        log_error(rxn_id, str(e))

    return desc


def main():
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

    with open(ERROR_LOG, "w") as f:
        f.write("--- Geometric Descriptor Error Log ---\n")

    print("1. Loading curated reactions...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"   {len(df)} reactions loaded.")

    est_minutes = len(df) * 5 / 24 / 60
    print(f"\n2. Computing geometric descriptors (IDPP + xTB)...")
    print(f"   ~3-8 s per reaction with 24 workers.")
    print(f"   Estimated: ~{est_minutes:.0f} minutes\n")

    results = Parallel(n_jobs=24, verbose=10)(
        delayed(process_reaction)(row) for _, row in df.iterrows()
    )

    print("\n3. Consolidating descriptors...")
    desc_df = pd.DataFrame(results)
    final_df = pd.merge(df[["rxn_id"]], desc_df, on="rxn_id", how="left")

    # --- Report stats ---
    idpp_ok = final_df["idpp_success"].sum()
    total = len(final_df)
    strain_nans = final_df["E_strain_IDPP"].isna().sum()
    rmsd_nans = final_df["rmsd_R_P"].isna().sum()

    print(f"\n   IDPP success: {idpp_ok}/{total} ({100*idpp_ok/total:.1f}%)")
    print(f"   E_strain NaNs: {strain_nans}")
    print(f"   RMSD NaNs: {rmsd_nans}")

    valid = final_df["E_strain_IDPP"].dropna()
    if len(valid) > 0:
        neg = (valid < 0).sum()
        print(f"   Negative E_strain: {neg}/{len(valid)} ({100*neg/len(valid):.1f}%)")
        print(f"   E_strain: mean={valid.mean():.3f}, "
              f"median={valid.median():.3f}, "
              f"min={valid.min():.3f}, max={valid.max():.3f} eV")

    final_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n   Saved to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()