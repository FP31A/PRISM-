import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

df = pd.read_parquet('data/transition1x/processed/final_curated_reactions.parquet')

def get_rot_bonds(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.CalcNumRotatableBonds(mol)
    except Exception:
        return None

df['n_rot_bonds'] = df['r_smiles'].apply(get_rot_bonds)

print(f"Computed n_rot_bonds for {df['n_rot_bonds'].notna().sum()}/{len(df)} reactions")
print(df['n_rot_bonds'].describe())

df.to_parquet('data/transition1x/processed/final_curated_reactions.parquet', index=False)
print('Saved.')