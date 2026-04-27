# -*- coding: utf-8 -*-
"""
PRISM Step 6C.2 — Generate ORCA r²SCAN-3c input files
Two single-points per reaction: TS geometry + reactant geometry
"""
import os
import pandas as pd
from ase.io import read as ase_read

TARGETS     = "data/transition1x/tier3_targets/rxn_ids.txt"
TS_DIR      = "data/transition1x/tier3_targets/ts_geometries"
CURATED     = "data/transition1x/processed/final_curated_reactions.parquet"
OUTPUT_DIR  = "data/transition1x/dft_singlepoints"

ORCA_TEMPLATE = """! r2SCAN-3c TightSCF
%pal nprocs 1 end
%maxcore 3000
* xyz 0 1
{coords}
*
"""

def atoms_to_xyz_block(atoms):
    lines = []
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f"  {sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}")
    return "\n".join(lines)

def main():
    rxn_ids = [l.strip() for l in open(TARGETS)]
    curated = pd.read_parquet(CURATED).set_index("rxn_id")

    generated = 0
    for rxn_id in rxn_ids:
        rxn_dir = os.path.join(OUTPUT_DIR, rxn_id)
        os.makedirs(rxn_dir, exist_ok=True)

        # TS input
        ts_xyz = os.path.join(TS_DIR, f"{rxn_id}_ts.xyz")
        ts_inp = os.path.join(rxn_dir, "ts.inp")
        if not os.path.exists(ts_inp):
            atoms = ase_read(ts_xyz)
            with open(ts_inp, "w") as f:
                f.write(ORCA_TEMPLATE.format(coords=atoms_to_xyz_block(atoms)))

        # Reactant input
        r_xyz = curated.loc[rxn_id, "r_xyz_path"]
        r_inp = os.path.join(rxn_dir, "reactant.inp")
        if not os.path.exists(r_inp):
            atoms = ase_read(r_xyz)
            with open(r_inp, "w") as f:
                f.write(ORCA_TEMPLATE.format(coords=atoms_to_xyz_block(atoms)))

        generated += 1
        if generated % 200 == 0:
            print(f"  {generated}/1000 generated...", flush=True)

    print(f"\nDone: {generated} reactions, 2 inputs each")
    total_inp = sum(1 for r in os.listdir(OUTPUT_DIR)
                    for f in os.listdir(os.path.join(OUTPUT_DIR, r))
                    if f.endswith('.inp'))
    print(f"Total input files: {total_inp}")

if __name__ == "__main__":
    main()