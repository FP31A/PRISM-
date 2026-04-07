import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina

# RXNMapper
from rxnmapper import RXNMapper

# --- Configuration ---
INPUT_PARQUET = "data/transition1x/processed/reactions.parquet"
OUTPUT_PARQUET = "data/transition1x/processed/curated_reactions.parquet"
SPLITS_DIR = "data/transition1x/splits"
SPLITS_FILE = os.path.join(SPLITS_DIR, "scaffold_5fold.json")

# Basic SMARTS templates for reaction classification
# (You can expand these later if you need more granular families)
RXN_FAMILIES = {
    "Cycloaddition": "[C:1]=[C:2].[C:3]=[C:4]>>[C:1]1-[C:2]-[C:3]-[C:4]-1",
    "SN2": "[C:1]-[X:2].[Y:3]>>[C:1]-[Y:3].[X:2]",
    # Add more SMARTS templates as needed
}

def xyz_to_smiles(xyz_path):
    """Safely attempts to perceive bonds and generate SMILES from 3D coordinates."""
    try:
        mol = Chem.MolFromXYZFile(xyz_path)
        if mol is None:
            return None
        # Attempt to perceive bonds based on van der Waals radii
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        return None

def cluster_fingerprints(fps, cutoff=0.4):
    """Clusters molecules using the Butina algorithm."""
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    
    # cluster_data returns a tuple of tuples containing the indices of the molecules in each cluster
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs

def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    print("1. Loading raw master table...")
    df = pd.read_parquet(INPUT_PARQUET)
    initial_count = len(df)
    
    print("\n2. Applying Global 3-Sigma Outlier Filter...")
    mean_ea = df['Ea_eV'].mean()
    std_ea = df['Ea_eV'].std()
    upper_bound = mean_ea + (3 * std_ea)
    lower_bound = mean_ea - (3 * std_ea)
    
    outliers = df[(df['Ea_eV'] > upper_bound) | (df['Ea_eV'] < lower_bound)]
    
    # ROADMAP REQUIREMENT: Log every removed reaction with its ID and energy
    if len(outliers) > 0:
        print("   --- GLOBAL OUTLIERS REMOVED ---")
        for _, row in outliers.iterrows():
            print(f"   ID: {row['rxn_id']} | Ea: {row['Ea_eV']:.4f} eV")
            
    # Drop the global outliers
    df = df[(df['Ea_eV'] <= upper_bound) & (df['Ea_eV'] >= lower_bound)].copy()
    print(f"   Remaining reactions after global filter: {len(df)}")
    
    print("\n3. Perceiving SMILES from Geometries...")
    r_smiles_list = []
    p_smiles_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting XYZ to SMILES"):
        r_smi = xyz_to_smiles(row['r_xyz_path'])
        p_smi = xyz_to_smiles(row['p_xyz_path'])
        r_smiles_list.append(r_smi)
        p_smiles_list.append(p_smi)
        
    df['r_smiles'] = r_smiles_list
    df['p_smiles'] = p_smiles_list
    
    # Drop rows where SMILES perception failed
    failed_smiles = df['r_smiles'].isna() | df['p_smiles'].isna()
    print(f"   Failed to perceive SMILES for {failed_smiles.sum()} reactions. Dropping them.")
    df = df[~failed_smiles].reset_index(drop=True)
    
    print("\n4. Running RXNMapper (Atom Mapping)...")
    rxn_mapper = RXNMapper()
    unmapped_rxns = (df['r_smiles'] + ">>" + df['p_smiles']).tolist()
    
    mapped_results = []
    # Process in batches to avoid overwhelming memory
    batch_size = 100
    for i in tqdm(range(0, len(unmapped_rxns), batch_size), desc="Mapping Atoms"):
        batch = unmapped_rxns[i:i+batch_size]
        results = rxn_mapper.get_attention_guided_atom_maps(batch)
        mapped_results.extend(results)
        
    df['mapped_rxn'] = [res['mapped_rxn'] for res in mapped_results]
    df['mapping_confidence'] = [res['confidence'] for res in mapped_results]
    
    flagged_mappings = (df['mapping_confidence'] < 0.8).sum()
    print(f"   Flagged {flagged_mappings} reactions with mapping confidence < 0.8.")
    
    print("\n5. Scaffold-based Splitting (Butina Clustering)...")
    mols = [Chem.MolFromSmiles(smi) for smi in df['r_smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) for x in mols]
    
    clusters = cluster_fingerprints(fps, cutoff=0.4)
    print(f"   Found {len(clusters)} distinct scaffold clusters.")
    
    # Distribute clusters across 5 folds
    folds = {f"fold_{i}": [] for i in range(5)}
    # Sort clusters by size (largest first) to balance folds
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    
    fold_sizes = [0] * 5
    for cluster in sorted_clusters:
        # Find the fold currently with the fewest items
        smallest_fold_idx = np.argmin(fold_sizes)
        
        # Add all rxn_ids from this cluster to that fold
        rxn_ids = df.iloc[list(cluster)]['rxn_id'].tolist()
        folds[f"fold_{smallest_fold_idx}"].extend(rxn_ids)
        fold_sizes[smallest_fold_idx] += len(rxn_ids)
        
    # Save the splits to JSON
    with open(SPLITS_FILE, 'w') as f:
        json.dump(folds, f, indent=4)
        
    print(f"   Saved 5-fold splits to {SPLITS_FILE}.")
    for i, size in enumerate(fold_sizes):
        print(f"      Fold {i}: {size} reactions")

    # Save the updated master table
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n✓ Pipeline complete. Curated table saved to {OUTPUT_PARQUET}.")

if __name__ == "__main__":
    main()