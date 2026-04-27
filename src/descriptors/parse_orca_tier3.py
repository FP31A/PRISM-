# -*- coding: utf-8 -*-
"""
PRISM Step 6C.3 — Parse ORCA r²SCAN-3c outputs, extract DFT strain energies.
Computes E_DFT_strain = E_r2SCAN(TS) - E_r2SCAN(R)
and systematic proxy error delta_sys = E_strain_IDPP - E_DFT_strain.
"""

import os
import re
import numpy as np
import pandas as pd

TARGETS_FILE = "data/transition1x/tier3_targets/rxn_ids.txt"
DFT_DIR      = "data/transition1x/dft_singlepoints"
TIER1_FILE   = "data/transition1x/descriptors/stream_b_geometric_tier1.parquet"
CURATED      = "data/transition1x/processed/final_curated_reactions.parquet"
OUTPUT_FILE  = "data/transition1x/descriptors/stream_b_geometric_tier3.parquet"
ERROR_LOG    = "data/transition1x/descriptors/orca_failures.log"

HA_TO_EV = 27.2114


def parse_orca_energy(filepath):
    """Extract FINAL SINGLE POINT ENERGY from ORCA output. Returns eV or None."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath) as f:
            text = f.read()
        match = re.search(r"FINAL SINGLE POINT ENERGY\s+([\-\d\.]+)", text)
        if match:
            return float(match.group(1)) * HA_TO_EV
        return None
    except Exception:
        return None


def check_scf_converged(filepath):
    """Check if ORCA SCF converged."""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath) as f:
            text = f.read()
        return "SCF CONVERGED" in text or "FINAL SINGLE POINT ENERGY" in text
    except Exception:
        return False


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(ERROR_LOG, "w") as f:
        f.write("--- ORCA Tier 3 Failure Log ---\n")

    rxn_ids = [l.strip() for l in open(TARGETS_FILE)]
    print(f"Parsing ORCA outputs for {len(rxn_ids)} reactions...")

    # Load Tier 1 IDPP strain for delta_sys computation
    tier1 = pd.read_parquet(TIER1_FILE).set_index("rxn_id")

    results = []
    converged = 0
    failed    = 0

    for rxn_id in rxn_ids:
        rxn_dir = os.path.join(DFT_DIR, rxn_id)

        record = {
            "rxn_id":         rxn_id,
            "E_DFT_strain":   np.nan,
            "E_DFT_ts":       np.nan,
            "E_DFT_reactant": np.nan,
            "delta_sys":      np.nan,
            "orca_converged": False,
            "ts_source":      "idpp_midpoint", # Standardized geometry source
        }

        # Parse TS and Reactant energies
        e_ts = parse_orca_energy(os.path.join(rxn_dir, "ts.out"))
        e_r  = parse_orca_energy(os.path.join(rxn_dir, "reactant.out"))

        if e_ts is not None and e_r is not None:
            record["E_DFT_ts"]       = e_ts
            record["E_DFT_reactant"] = e_r
            record["E_DFT_strain"]   = e_ts - e_r
            record["orca_converged"] = True
            converged += 1

            # Compute systematic proxy error against IDPP strain
            if rxn_id in tier1.index:
                e_idpp = tier1.loc[rxn_id, "E_strain_IDPP"]
                if not pd.isna(e_idpp):
                    record["delta_sys"] = e_idpp - record["E_DFT_strain"]
        else:
            failed += 1
            ts_ok = "OK" if e_ts is not None else "FAILED"
            r_ok  = "OK" if e_r  is not None else "FAILED"
            with open(ERROR_LOG, "a") as f:
                f.write(f"{rxn_id} | ts={ts_ok} r={r_ok}\n")

        results.append(record)

    # Build dataframe
    df = pd.DataFrame(results)

    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"Converged: {converged}/{len(rxn_ids)} "
          f"({converged/len(rxn_ids)*100:.1f}%)")
    print(f"Failed:    {failed}")

    valid_strain = df["E_DFT_strain"].dropna()
    print(f"\nE_DFT_strain stats:")
    print(f"  count:  {len(valid_strain)}")
    print(f"  mean:   {valid_strain.mean():.4f} eV")
    print(f"  median: {valid_strain.median():.4f} eV")
    print(f"  min:    {valid_strain.min():.4f} eV")
    print(f"  max:    {valid_strain.max():.4f} eV")

    negative = (valid_strain < 0).sum()
    print(f"  negative (unexpected): {negative}")

    valid_delta = df["delta_sys"].dropna()
    if len(valid_delta) > 0:
        print(f"\ndelta_sys (E_strain_IDPP - E_DFT_strain):")
        print(f"  count:  {len(valid_delta)}")
        print(f"  mean:   {valid_delta.mean():+.4f} eV")
        print(f"  std:    {valid_delta.std():.4f} eV")
        print(f"  mean |delta_sys|: {valid_delta.abs().mean():.4f} eV")

    print(f"\nTS source breakdown:")
    print(df["ts_source"].value_counts().to_string())

    # Save
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"Error log: {ERROR_LOG}")


if __name__ == "__main__":
    main()