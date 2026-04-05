import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.data import chemical_symbols

# --- Configuration ---
H5_PATH = "data/transition1x/raw/Transition1x.h5"
PROCESSED_DIR = "data/transition1x/processed"
GEOM_DIR = os.path.join(PROCESSED_DIR, "geometries")
PARQUET_PATH = os.path.join(PROCESSED_DIR, "reactions.parquet")

def write_xyz(filepath, symbols, positions, comment=""):
    """Writes standard .xyz format geometry files."""
    with open(filepath, 'w') as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{comment}\n")
        for sym, pos in zip(symbols, positions):
            f.write(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

def main():
    os.makedirs(GEOM_DIR, exist_ok=True)
    data_records = []
    
    with h5py.File(H5_PATH, 'r') as h5:
        data_group = h5['data']
        formulas = list(data_group.keys())
        
        for formula in tqdm(formulas, desc="Extracting Transition1x Data"):
            rxn_group = data_group[formula]
            for rxn_id in rxn_group.keys():
                rxn = rxn_group[rxn_id]
                
                try:
                    # Map atomic numbers to IUPAC chemical symbols (e.g., 6 -> C)
                    atomic_numbers = np.array(rxn['atomic_numbers']).flatten()
                    symbols = [chemical_symbols[z] for z in atomic_numbers]
                    n_atoms = len(symbols)
                    
                    # Extract energies (forcing them to be standard scalar floats)
                    e_key = 'wB97x_6-31G(d).energy'
                    e_r = float(np.squeeze(rxn['reactant'][e_key]))
                    e_p = float(np.squeeze(rxn['product'][e_key]))
                    e_ts = float(np.squeeze(rxn['transition_state'][e_key]))
                    
                    # Calculate thermodynamic and kinetic metrics
                    Ea_eV = e_ts - e_r
                    dE_rxn_eV = e_p - e_r
                    
                    # Define output paths
                    r_xyz_path = os.path.join(GEOM_DIR, f"{rxn_id}_reactant.xyz")
                    p_xyz_path = os.path.join(GEOM_DIR, f"{rxn_id}_product.xyz")
                    ts_xyz_path = os.path.join(GEOM_DIR, f"{rxn_id}_ts.xyz")
                    
                    # Extract positions and force them into an (N_atoms, 3) matrix
                    r_pos = np.array(rxn['reactant']['positions']).reshape(-1, 3)
                    p_pos = np.array(rxn['product']['positions']).reshape(-1, 3)
                    ts_pos = np.array(rxn['transition_state']['positions']).reshape(-1, 3)
                    
                    # Write geometries to disk
                    write_xyz(r_xyz_path, symbols, r_pos, comment=f"{rxn_id} Reactant")
                    write_xyz(p_xyz_path, symbols, p_pos, comment=f"{rxn_id} Product")
                    write_xyz(ts_xyz_path, symbols, ts_pos, comment=f"{rxn_id} TS")
                    
                    # Append to master table
                    data_records.append({
                        'rxn_id': rxn_id,
                        'formula': formula,
                        'n_atoms': n_atoms,
                        'Ea_eV': Ea_eV,
                        'dE_rxn_eV': dE_rxn_eV,
                        'r_xyz_path': r_xyz_path,
                        'p_xyz_path': p_xyz_path,
                        'ts_xyz_path': ts_xyz_path
                    })
                    
                except KeyError as e:
                    print(f"\nSkipping {formula}/{rxn_id} due to missing key: {e}")
                    continue

    # Generate Parquet file
    df = pd.DataFrame(data_records)
    df.to_parquet(PARQUET_PATH, index=False)
    
    # --- Validation Sequences ---
    print("\n========================================")
    print("VALIDATION CHECKPOINT")
    print("========================================")
    
    # 1. Total Reaction Count
    actual_rows = len(df)
    print(f"Total reactions parsed: {actual_rows}")
    if actual_rows == 10073:
        print("✓ Reaction count verified (10,073).")
    else:
        print(f"✗ FAILED: Expected 10,073 rows, got {actual_rows}.")
    
    # 2. Activation Energy Distribution Outliers
    outliers = df[df['Ea_eV'] > 10.0]
    print(f"✓ Found {len(outliers)} reactions with Ea > 10 eV.")
    
    # 3. Spot Check
    print("\nSpot-check of 5 random reactions:")
    print(df[['rxn_id', 'formula', 'Ea_eV', 'dE_rxn_eV']].sample(5).to_string(index=False))
    
    # 4. Plotting Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['Ea_eV'], bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(x=10, color='red', linestyle='--', label='10 eV Checkpoint Threshold')
    plt.title('Distribution of Activation Energies ($\Delta E^\ddagger$)')
    plt.xlabel('$\Delta E^\ddagger$ (eV)')
    plt.ylabel('Frequency')
    plt.legend()
    plot_path = os.path.join(PROCESSED_DIR, "Ea_distribution.png")
    plt.savefig(plot_path)
    print(f"\n✓ Saved distribution plot to {plot_path}")

if __name__ == "__main__":
    main()