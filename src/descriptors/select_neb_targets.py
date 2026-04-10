# -*- coding: utf-8 -*-
"""
PRISM Step 6B.1 — Select NEB Calibration Subset
Stratifies and samples ~3,000 reactions from successful IDPP runs for NSCC NEB calculations.
"""

import os
import pandas as pd
import numpy as np

TIER1_PARQUET = "data/transition1x/descriptors/stream_b_geometric_tier1.parquet"
REACTIONS_PARQUET = "data/transition1x/processed/final_curated_reactions.parquet"
OUTPUT_DIR = "data/transition1x/processed/neb_calibration"
INDEX_FILE = os.path.join(OUTPUT_DIR, "rxn_index.csv")
TARGET_SIZE = 3000

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load data
    t1_df = pd.read_parquet(TIER1_PARQUET)
    rxn_df = pd.read_parquet(REACTIONS_PARQUET)
    
    # 2. Filter to only successful IDPP runs
    valid_t1 = t1_df[t1_df["interpolation_method"] == "idpp"]
    
    # Merge to get reaction families
    df = valid_t1.merge(rxn_df[['rxn_id', 'rmg_family']], on='rxn_id', how='inner')
    
    # 3. Stratified sampling
    # Calculate proportions
    family_counts = df['rmg_family'].value_counts()
    proportions = family_counts / len(df)
    
    # Determine sample sizes per family
    sample_sizes = (proportions * TARGET_SIZE).astype(int)
    
    # Adjust for rounding errors to hit exactly TARGET_SIZE
    diff = TARGET_SIZE - sample_sizes.sum()
    if diff > 0:
        sample_sizes.iloc[0] += diff
        
    print(f"Total valid IDPP reactions: {len(df)}")
    print(f"Targeting {TARGET_SIZE} for NEB calibration.\nFamily breakdown:")
    
    sampled_dfs = []
    for family, size in sample_sizes.items():
        family_df = df[df['rmg_family'] == family]
        sampled = family_df.sample(n=size, random_state=42)
        sampled_dfs.append(sampled)
        print(f"  {family}: {size}")
        
    final_sample = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 4. Save index file
    # Shift index to 1-based to strictly align with PBS #PBS -J 1-3000
    final_sample.index = final_sample.index + 1
    
    # Save a single time without headers to ensure line 1 is data, not column names
    final_sample[['rxn_id', 'rmg_family']].to_csv(INDEX_FILE, header=False, index=True)
    
    print(f"\nSaved {len(final_sample)} targets to {INDEX_FILE}")

if __name__ == "__main__":
    main()