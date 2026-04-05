import os
import glob
import pandas as pd
import numpy as np
import cclib
from tqdm import tqdm
from ase.data import chemical_symbols

# --- Configuration ---
CSV_PATH = "data/grambow/raw/wb97xd3.csv"
LOG_DIR = "data/grambow/raw/qm_logs" # Updated to match your screenshot
PROCESSED_DIR = "data/grambow/processed"
GEOM_DIR = os.path.join(PROCESSED_DIR, "geometries")
PARQUET_PATH = os.path.join(PROCESSED_DIR, "reactions.parquet")

def write_xyz(filepath, atomic_numbers, positions, comment=""):
    """Writes standard .xyz format geometry files."""
    symbols = [chemical_symbols[z] for z in atomic_numbers]
    with open(filepath, 'w') as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{comment}\n")
        for sym, pos in zip(symbols, positions):
            f.write(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

def extract_geometry(log_path):
    """Uses cclib to extract the final atomic numbers and coordinates from a Q-Chem log."""
    try:
        parsed_data = cclib.io.ccread(log_path)
        if parsed_data is None:
            return None, None
        
        atomic_numbers = parsed_data.atomnos
        final_positions = parsed_data.atomcoords[-1] 
        return atomic_numbers, final_positions
    except Exception as e:
        return None, None

def extract_multi_fragment(log_paths):
    """Parses multiple log files and safely stitches them together with a 10A physical separation."""
    all_z = []
    all_pos = []
    offset_x = 0.0 # Shift each subsequent fragment by 10 Angstroms
    
    for path in sorted(log_paths): # Sort ensures _0 comes before _1
        z, pos = extract_geometry(path)
        if z is None:
            return None, None
            
        # Shift geometry to prevent overlapping atoms
        shifted_pos = pos.copy()
        shifted_pos[:, 0] += offset_x
        
        all_z.extend(z)
        all_pos.extend(shifted_pos)
        
        offset_x += 10.0 # Move 10 Angstroms away for the next molecule
        
    return np.array(all_z), np.array(all_pos)

def main():
    os.makedirs(GEOM_DIR, exist_ok=True)
    data_records = []
    
    print("Loading CSV...")
    df_raw = pd.read_csv(CSV_PATH)
    
    # Roadmap requirement: Convert kcal/mol to eV (divide by 23.0605)
    df_raw['Ea_eV'] = df_raw['dE0'] / 23.0605
    df_raw['dE_rxn_eV'] = df_raw['dHrxn298'] / 23.0605
    
    successful_parses = 0
    failures = 0
    
    # Testing the first 20 reactions to guarantee we hit both single and multi-fragment systems
    
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Parsing Q-Chem Logs"):
        idx = int(row['idx'])
        rxn_id = f"rxn_{idx}"
        idx_str = f"{idx:06d}" # Formats '2' as '000002'
        
        # Navigate into the subfolder for this specific reaction
        rxn_dir = os.path.join(LOG_DIR, f"rxn{idx_str}")
        
        # Use glob to find ALL fragments for reactants and products
        r_logs = glob.glob(os.path.join(rxn_dir, f"r{idx_str}*.log"))
        p_logs = glob.glob(os.path.join(rxn_dir, f"p{idx_str}*.log"))
        ts_log = os.path.join(rxn_dir, f"ts{idx_str}.log")
        
        # Ensure we found at least one reactant, one product, and the TS
        if not r_logs or not p_logs or not os.path.exists(ts_log):
            failures += 1
            continue
            
        # Extract and stitch fragments together
        r_z, r_pos = extract_multi_fragment(r_logs)
        p_z, p_pos = extract_multi_fragment(p_logs)
        ts_z, ts_pos = extract_geometry(ts_log) # TS is always a single file
        
        if r_z is not None and p_z is not None and ts_z is not None:
            r_xyz_path = os.path.join(GEOM_DIR, f"{rxn_id}_reactant.xyz")
            p_xyz_path = os.path.join(GEOM_DIR, f"{rxn_id}_product.xyz")
            ts_xyz_path = os.path.join(GEOM_DIR, f"{rxn_id}_ts.xyz")
            
            write_xyz(r_xyz_path, r_z, r_pos, comment=f"{rxn_id} Reactant")
            write_xyz(p_xyz_path, p_z, p_pos, comment=f"{rxn_id} Product")
            write_xyz(ts_xyz_path, ts_z, ts_pos, comment=f"{rxn_id} TS")
            
            data_records.append({
                'rxn_id': rxn_id,
                'formula': "UNKNOWN", 
                'n_atoms': len(r_z),
                'Ea_eV': row['Ea_eV'],
                'dE_rxn_eV': row['dE_rxn_eV'],
                'r_xyz_path': r_xyz_path,
                'p_xyz_path': p_xyz_path,
                'ts_xyz_path': ts_xyz_path
            })
            successful_parses += 1
        else:
            failures += 1
            
    print(f"\nSuccessfully parsed: {successful_parses}")
    print(f"Failed to parse: {failures}")

# Generate Parquet file
    df_out = pd.DataFrame(data_records)
    df_out.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved master table to {PARQUET_PATH} with {len(df_out)} rows.")

if __name__ == "__main__":
    main()