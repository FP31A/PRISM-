# -*- coding: utf-8 -*-
"""
PRISM Step 4.2 — Reaction Classification via Bond-Change Analysis

Instead of SMARTS template matching (which fails on real reaction SMILES),
this classifies reactions by analyzing which bonds broke and formed using
the atom-mapped SMILES from RXNMapper.

Logic:
  1. Parse atom-mapped reaction SMILES
  2. Extract bonds in reactant and product (keyed by atom map numbers)
  3. Compute bonds_broken and bonds_formed
  4. Classify based on bond-change patterns + structural context
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from collections import Counter
from tqdm import tqdm

# --- Configuration ---
INPUT_PARQUET = "data/transition1x/processed/curated_reactions.parquet"
FINAL_PARQUET = "data/transition1x/processed/final_curated_reactions.parquet"
PLOT_PATH = "data/transition1x/processed/mapping_confidence.png"


def get_mapped_bonds(mol):
    """
    Extract bonds as sets of (map_num_1, map_num_2) tuples.
    Only includes bonds where both atoms have atom map numbers.
    Returns a dict of {frozenset(map1, map2): bond_type}.
    """
    bonds = {}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        map1 = a1.GetAtomMapNum()
        map2 = a2.GetAtomMapNum()
        if map1 > 0 and map2 > 0:
            key = frozenset([map1, map2])
            bonds[key] = bond.GetBondTypeAsDouble()
    return bonds


def get_atom_info(mol):
    """
    Build a dict of {atom_map_num: (symbol, num_Hs, is_in_ring, degree)}.
    """
    info = {}
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            info[map_num] = {
                "symbol": atom.GetSymbol(),
                "num_Hs": atom.GetTotalNumHs(),
                "in_ring": atom.IsInRing(),
                "degree": atom.GetDegree(),
            }
    return info


def count_fragments(smiles_side):
    """Count number of molecular fragments (separated by '.')."""
    return len(smiles_side.split("."))


def classify_by_bond_changes(mapped_rxn_smiles):
    """
    Classify a reaction based on bond-change analysis.
    
    Returns (family_name, debug_info_dict).
    """
    try:
        parts = mapped_rxn_smiles.split(">>")
        if len(parts) != 2:
            return "other", {"reason": "no >> separator"}

        r_smi, p_smi = parts
        r_mol = Chem.MolFromSmiles(r_smi)
        p_mol = Chem.MolFromSmiles(p_smi)

        if r_mol is None or p_mol is None:
            return "other", {"reason": "SMILES parse failed"}

        # Get bonds keyed by atom map numbers
        r_bonds = get_mapped_bonds(r_mol)
        p_bonds = get_mapped_bonds(p_mol)

        r_bond_keys = set(r_bonds.keys())
        p_bond_keys = set(p_bonds.keys())

        bonds_broken = r_bond_keys - p_bond_keys
        bonds_formed = p_bond_keys - r_bond_keys

        # Also check for bond ORDER changes (e.g., single -> double)
        common_bonds = r_bond_keys & p_bond_keys
        order_changes = []
        for bk in common_bonds:
            if abs(r_bonds[bk] - p_bonds[bk]) > 0.1:
                order_changes.append((bk, r_bonds[bk], p_bonds[bk]))

        n_broken = len(bonds_broken)
        n_formed = len(bonds_formed)
        n_order_changed = len(order_changes)

        # Get atom info from reactant side
        r_info = get_atom_info(r_mol)
        p_info = get_atom_info(p_mol)

        # Fragment counts
        r_frags = count_fragments(r_smi)
        p_frags = count_fragments(p_smi)

        debug = {
            "n_broken": n_broken,
            "n_formed": n_formed,
            "n_order_changed": n_order_changed,
            "r_frags": r_frags,
            "p_frags": p_frags,
        }

        # --- Classification rules (order matters — most specific first) ---

        # Get elements involved in broken/formed bonds
        def get_bond_elements(bond_set):
            elements = []
            for bk in bond_set:
                atoms = list(bk)
                syms = []
                for a in atoms:
                    if a in r_info:
                        syms.append(r_info[a]["symbol"])
                    elif a in p_info:
                        syms.append(p_info[a]["symbol"])
                elements.append(sorted(syms))
            return elements

        broken_elements = get_bond_elements(bonds_broken)
        formed_elements = get_bond_elements(bonds_formed)

        # Check if any atoms move into or out of rings
        ring_formed = False
        ring_broken = False
        for map_num in p_info:
            if map_num in r_info:
                if p_info[map_num]["in_ring"] and not r_info[map_num]["in_ring"]:
                    ring_formed = True
                if not p_info[map_num]["in_ring"] and r_info[map_num]["in_ring"]:
                    ring_broken = True

        # Check if H is involved in broken/formed bonds
        h_in_broken = any("H" in elems for elems in broken_elements)
        h_in_formed = any("H" in elems for elems in formed_elements)

        # 1. CYCLOADDITION: two fragments merge into one ring
        #    (2 bonds form, ring appears, fragment count decreases)
        if ring_formed and n_formed >= 2 and p_frags < r_frags:
            return "cycloaddition", debug

        # 2. ELECTROCYCLIC RING CLOSURE: single molecule cyclises
        #    (ring forms, no change in fragment count, bond order changes)
        if ring_formed and r_frags == p_frags == 1 and n_order_changed > 0:
            return "electrocyclic ring closure", debug

        # 3. RETRO-[3,3]-SIGMATROPIC / RETROCYCLIC: ring opens
        if ring_broken and n_broken >= 1:
            if p_frags > r_frags:
                return "retro-sigmatropic", debug
            else:
                return "ring opening", debug

        # 4. RADICAL H-ABSTRACTION: X-H breaks, Y-H forms, typically bimolecular
        #    (1 X-H bond breaks, 1 Y-H bond forms)
        if h_in_broken and h_in_formed and n_broken == 1 and n_formed == 1:
            return "radical H-abstraction", debug

        # 5. BETA-ELIMINATION: fragment count increases, H leaves, double bond forms
        if p_frags > r_frags and h_in_broken and n_order_changed > 0:
            return "beta-elimination", debug

        # 6. 1,2-SHIFT: same fragment count, H or group migrates
        #    (1 bond breaks, 1 bond forms, no fragment change, involves adjacent atoms)
        if n_broken == 1 and n_formed == 1 and r_frags == p_frags:
            # Check if the atoms are neighbors (1,2-shift)
            broken_atoms = list(list(bonds_broken)[0])
            formed_atoms = list(list(bonds_formed)[0])
            shared = set(broken_atoms) & set(formed_atoms)
            if len(shared) == 1:
                if h_in_broken or h_in_formed:
                    return "1,2-H shift", debug
                else:
                    return "1,2-shift", debug

        # 7. SN2-LIKE: one bond breaks, one forms, bimolecular
        if n_broken == 1 and n_formed == 1 and r_frags >= 2:
            return "SN2", debug

        # 8. DISSOCIATION: bond breaks, fragment count increases, nothing forms
        if n_broken >= 1 and n_formed == 0 and p_frags > r_frags:
            return "dissociation", debug

        # 9. ASSOCIATION: fragments merge, bond forms, nothing breaks
        if n_formed >= 1 and n_broken == 0 and p_frags < r_frags:
            return "association", debug

        # 10. COMPLEX REARRANGEMENT: multiple bonds change
        if n_broken + n_formed >= 4:
            return "complex rearrangement", debug

        # 11. BOND ORDER CHANGE ONLY (e.g., tautomerism)
        if n_broken == 0 and n_formed == 0 and n_order_changed > 0:
            return "tautomerism", debug

        return "other", debug

    except Exception as e:
        return "other", {"reason": str(e)}


def main():
    print("1. Loading mapped data...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"   Loaded {len(df)} reactions.")

    # --- 4.2 Reaction Classification ---
    print("\n2. Classifying reactions via bond-change analysis...")
    families = []
    debug_infos = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        family, debug = classify_by_bond_changes(row["mapped_rxn"])
        families.append(family)
        debug_infos.append(debug)

    df["rmg_family"] = families

    # Show initial distribution
    family_counts = df["rmg_family"].value_counts()
    print("\nInitial Family Distribution:")
    print(family_counts.to_string())
    print(f"\nTotal families with >= 50 reactions: "
          f"{(family_counts >= 50).sum()}")

    # Merge families with < 50 reactions into "other"
    small_families = family_counts[family_counts < 50].index
    if len(small_families) > 0:
        print(f"\nMerging {len(small_families)} small families into 'other': "
              f"{list(small_families)}")
        df.loc[df["rmg_family"].isin(small_families), "rmg_family"] = "other"

    print("\nFinal Family Distribution:")
    print(df["rmg_family"].value_counts().to_string())

    # --- 4.1 Per-Family 3-Sigma Outlier Filter ---
    print("\n3. Applying Per-Family 3-Sigma Outlier Filter...")
    outlier_indices = []

    for family in df["rmg_family"].unique():
        family_df = df[df["rmg_family"] == family]
        mean_ea = family_df["Ea_eV"].mean()
        std_ea = family_df["Ea_eV"].std()

        if std_ea == 0 or np.isnan(std_ea):
            continue

        upper = mean_ea + 3 * std_ea
        lower = mean_ea - 3 * std_ea

        family_outliers = family_df[
            (family_df["Ea_eV"] > upper) | (family_df["Ea_eV"] < lower)
        ]

        if not family_outliers.empty:
            print(f"\n   --- {family.upper()} ({len(family_outliers)} outliers) ---")
            for _, row in family_outliers.iterrows():
                print(f"   ID: {row['rxn_id']} | Ea: {row['Ea_eV']:.4f} eV "
                      f"| range: [{lower:.2f}, {upper:.2f}]")
            outlier_indices.extend(family_outliers.index)

    df = df.drop(index=outlier_indices).reset_index(drop=True)
    print(f"\nRemoved {len(outlier_indices)} outliers. "
          f"Remaining: {len(df)} reactions.")

    # --- 4.3 Confidence Distribution Plot ---
    print("\n4. Generating Mapping Confidence Distribution Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: confidence histogram
    axes[0].hist(df["mapping_confidence"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0.8, color="red", linestyle="--", label="0.8 threshold")
    axes[0].set_title("RXNMapper Confidence Scores")
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Right: family distribution bar chart
    family_counts_final = df["rmg_family"].value_counts()
    axes[1].barh(family_counts_final.index, family_counts_final.values,
                 edgecolor="black", alpha=0.7)
    axes[1].set_title("Reaction Family Distribution")
    axes[1].set_xlabel("Count")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"   Saved plots to {PLOT_PATH}")

    # Save final table
    df.to_parquet(FINAL_PARQUET, index=False)
    print(f"\nDone. Final curated table: {FINAL_PARQUET}")
    print(f"Final count: {len(df)} reactions across "
          f"{df['rmg_family'].nunique()} families.")


if __name__ == "__main__":
    main()