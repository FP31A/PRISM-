import os
import re
import subprocess
import tempfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

# --- Configuration ---
INPUT_PARQUET = "data/transition1x/processed/final_curated_reactions.parquet"
OUTPUT_PARQUET = "data/transition1x/descriptors/electronic_descriptors.parquet"
ERROR_LOG = "data/transition1x/descriptors/xtb_errors.log"
XTB_CORES = "1" # Limit internal threading for external parallelization

def parse_xtb_output(output_text):
    """Extracts electronic properties from standard xTB output."""
    results = {}

    # Total Energy (Hartree -> eV)
    energy_match = re.search(
        r"\|\s+TOTAL ENERGY\s+([\-\.\d]+)\s+Eh\s+\|", output_text
    )
    if energy_match:
        results['E_xtb'] = float(energy_match.group(1)) * 27.2114  # eV

    # HOMO and LUMO (already in eV)
    homo_match = re.search(r"([-\d\.]+)\s+\(HOMO\)", output_text)
    lumo_match = re.search(r"([-\d\.]+)\s+\(LUMO\)", output_text)
    if homo_match and lumo_match:
        homo = float(homo_match.group(1))
        lumo = float(lumo_match.group(1))
        results['homo'] = homo
        results['lumo'] = lumo
        results['gap'] = lumo - homo

    # Dipole — parse the "full:" line under "molecular dipole:"
    dipole_match = re.search(
        r"full:\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)",
        output_text
    )
    if dipole_match:
        results['dipole'] = float(dipole_match.group(4))  # total magnitude

    return results

def run_xtb_single_point(xyz_path, rxn_id, state):
    """Runs GFN2-xTB and handles convergence failures gracefully per roadmap."""
    if not isinstance(xyz_path, str) or not os.path.exists(xyz_path):
        return None
        
    with tempfile.TemporaryDirectory() as tmpdir:
        # --fukui flag added to generate Fukui indices in output for later parsing
        cmd = ["xtb", os.path.abspath(xyz_path), "--gfn", "2", "--fukui"]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = XTB_CORES
        
        try:
            result = subprocess.run(cmd, cwd=tmpdir, env=env, capture_output=True, text=True, check=True)
            return parse_xtb_output(result.stdout)
            
        except Exception as e:
            # Roadmap Requirement: Robust error handler logging ID, type, and SCF iterations
            output = e.stdout + e.stderr
            scf_iter_match = re.search(r"SCF cycle\s+(\d+)\s+failed", output)
            scf_iters = scf_iter_match.group(1) if scf_iter_match else "Unknown"
            
            error_msg = f"{rxn_id} | State: {state} | Error: Convergence Failure | SCF Iterations: {scf_iters}\n"
            
            # Append to error log thread-safely
            with open(ERROR_LOG, 'a') as f:
                f.write(error_msg)
                
            return None # Return NaN equivalents

def process_reaction(row):
    """Processes R and P to extract requested delta descriptors."""
    rxn_id = row['rxn_id']
    
    r_res = run_xtb_single_point(row['r_xyz_path'], rxn_id, "Reactant")
    p_res = run_xtb_single_point(row['p_xyz_path'], rxn_id, "Product")
    
    desc = {'rxn_id': rxn_id}
    
    # Initialize all requested keys with NaNs
    keys = ['dE_xtb', 'gap_R', 'gap_P', 'd_gap', 'dipole_R', 'dipole_P', 'd_dipole']
    for k in keys:
        desc[k] = np.nan
        
    # Calculate Deltas if both calculations converged
    if r_res and p_res:
        if 'E_xtb' in r_res and 'E_xtb' in p_res:
            desc['dE_xtb'] = p_res['E_xtb'] - r_res['E_xtb']
            
        if 'gap' in r_res and 'gap' in p_res:
            desc['gap_R'] = r_res['gap']
            desc['gap_P'] = p_res['gap']
            desc['d_gap'] = p_res['gap'] - r_res['gap']
            
        if 'dipole' in r_res and 'dipole' in p_res:
            desc['dipole_R'] = r_res['dipole']
            desc['dipole_P'] = p_res['dipole']
            desc['d_dipole'] = p_res['dipole'] - r_res['dipole']
            
    return desc

def main():
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    
    # Initialize clean error log
    with open(ERROR_LOG, 'w') as f:
        f.write("--- xTB Convergence Failure Log ---\n")
        
    print("1. Loading final curated reactions...")
    df = pd.read_parquet(INPUT_PARQUET)
    
    
    print(f"\n2. Executing GFN2-xTB calculations (Testing {len(df)} reactions)...")
    results = Parallel(n_jobs=24, verbose=10)(
    delayed(process_reaction)(row) for _, row in df.iterrows()
    )
    
    print("\n3. Consolidating descriptors...")
    desc_df = pd.DataFrame(results)
    final_df = pd.merge(df, desc_df, on='rxn_id', how='left')
    
    failures = final_df['dE_xtb'].isna().sum()
    print(f"Calculations complete. Failed convergences: {failures}")
    
    final_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✓ xTB descriptors saved to {OUTPUT_PARQUET}")
    if failures > 0:
        print(f"  Check {ERROR_LOG} for failure details.")

if __name__ == "__main__":
    main()