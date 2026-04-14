# -*- coding: utf-8 -*-
"""
PRISM Step 7 — Stream C: Topological Descriptors
src/descriptors/topological.py
"""

import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors
from rdkit.Chem.EState import EState


def get_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None


def compute_topological(r_smiles, p_smiles):
    """
    Compute topological descriptors from reactant and product SMILES.
    Returns dict of descriptors, NaN for any failures.
    """
    result = {
        "MW":                    np.nan,
        "LogP":                  np.nan,
        "TPSA_R":                np.nan,
        "TPSA_P":                np.nan,
        "delta_TPSA":            np.nan,
        "n_rot_bonds_rdkit":     np.nan,
        "n_rings_R":             np.nan,
        "n_rings_P":             np.nan,
        "delta_ring_atoms":      np.nan,
        "balaban_J":             np.nan,
        "estate_sum_R":          np.nan,
        "estate_sum_P":          np.nan,
        "delta_estate_sum":      np.nan,
        "estate_max_R":          np.nan,
        "estate_max_P":          np.nan,
    }

    mol_r = get_mol(r_smiles)
    mol_p = get_mol(p_smiles)

    if mol_r is None or mol_p is None:
        return result

    try:
        # --- Reactant-based descriptors ---
        result["MW"]                = Descriptors.MolWt(mol_r)
        result["LogP"]              = Descriptors.MolLogP(mol_r)
        result["TPSA_R"]            = rdMolDescriptors.CalcTPSA(mol_r)
        result["n_rot_bonds_rdkit"] = rdMolDescriptors.CalcNumRotatableBonds(mol_r)
        result["n_rings_R"]         = rdMolDescriptors.CalcNumRings(mol_r)

        # Balaban J index
        try:
            result["balaban_J"] = GraphDescriptors.BalabanJ(mol_r)
        except Exception:
            result["balaban_J"] = np.nan

        # E-State indices
        try:
            estate_r = EState.EStateIndices(mol_r)
            if estate_r is not None and len(estate_r) > 0:
                result["estate_sum_R"] = float(np.sum(estate_r))
                result["estate_max_R"] = float(np.max(estate_r))
        except Exception:
            pass

    except Exception as e:
        pass

    try:
        # --- Product descriptors ---
        result["TPSA_P"]    = rdMolDescriptors.CalcTPSA(mol_p)
        result["n_rings_P"] = rdMolDescriptors.CalcNumRings(mol_p)

        try:
            estate_p = EState.EStateIndices(mol_p)
            if estate_p is not None and len(estate_p) > 0:
                result["estate_sum_P"] = float(np.sum(estate_p))
                result["estate_max_P"] = float(np.max(estate_p))
        except Exception:
            pass

    except Exception:
        pass

    # --- Delta descriptors ---
    result["delta_TPSA"] = (
        result["TPSA_P"] - result["TPSA_R"]
        if not np.isnan(result["TPSA_R"]) and not np.isnan(result["TPSA_P"])
        else np.nan
    )

    # Ring atoms formed/broken
    if not np.isnan(result["n_rings_R"]) and not np.isnan(result["n_rings_P"]):
        ring_atoms_r = sum(1 for a in mol_r.GetAtoms() if a.IsInRing())
        ring_atoms_p = sum(1 for a in mol_p.GetAtoms() if a.IsInRing())
        result["delta_ring_atoms"] = ring_atoms_p - ring_atoms_r

    if not np.isnan(result["estate_sum_R"]) and not np.isnan(result["estate_sum_P"]):
        result["delta_estate_sum"] = result["estate_sum_P"] - result["estate_sum_R"]

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute topological descriptors.")
    parser.add_argument("--dataset", type=str, required=True, choices=["transition1x", "grambow"],
                        help="Target dataset to process")
    args = parser.parse_args()

    curated_file = f"data/{args.dataset}/processed/final_curated_reactions.parquet"
    output_file = f"data/{args.dataset}/descriptors/stream_c_topological.parquet"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.read_parquet(curated_file)
    print(f"Loaded {len(df)} reactions from {args.dataset}.")

    # 1. Compute topological descriptors
    records = []
    for i, row in df.iterrows():
        desc = compute_topological(row["r_smiles"], row["p_smiles"])
        desc["rxn_id"] = row["rxn_id"]
        records.append(desc)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(df)} processed...", flush=True)

    topo_df = pd.DataFrame(records)

    # Rename the rdkit rotatable bonds to the standard name expected by downstream scripts
    topo_df = topo_df.rename(columns={"n_rot_bonds_rdkit": "n_rot_bonds"})

    # 2. One-hot encode reaction family
    print(f"\nReaction families: {df['rmg_family'].value_counts().to_dict()}")
    family_dummies = pd.get_dummies(
        df[["rxn_id", "rmg_family"]].set_index("rxn_id")["rmg_family"],
        prefix="rxn_class"
    ).reset_index()

    # 3. Merge topological + one-hot
    out_df = topo_df.merge(family_dummies, on="rxn_id", how="left")

    # 4. Save
    out_df.to_parquet(output_file, index=False)

    # --- Summary ---
    print(f"\nSaved to {output_file}")
    print(f"Shape: {out_df.shape}")
    
    print(f"\nNaN counts:")
    nan_counts = out_df.isna().sum()
    print(nan_counts[nan_counts > 0].to_string() if nan_counts.any() else "  None — all descriptors complete.")

    # Checkpoint
    print(f"\nDescriptor completeness:")
    for col in ["MW", "LogP", "TPSA_R", "balaban_J",
                "delta_ring_atoms", "delta_TPSA", "estate_sum_R"]:
        valid = out_df[col].notna().sum()
        print(f"  {col}: {valid}/{len(out_df)} ({valid/len(out_df)*100:.1f}%)")

    # One-hot columns check for Step 8 assembly alignment
    rxn_cols = [c for c in out_df.columns if c.startswith("rxn_class_")]
    print(f"\nOne-hot columns ({len(rxn_cols)}): {rxn_cols}")


if __name__ == "__main__":
    main()