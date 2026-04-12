# -*- coding: utf-8 -*-
import os, sys, signal, argparse
import numpy as np
import pandas as pd
from ase.io import read as ase_read, write as ase_write
from ase.mep import NEB
from ase.optimize import FIRE
from xtb.ase.calculator import XTB
from ase.optimize import FIRE, LBFGS

PARQUET_PATH = "data/transition1x/processed/final_curated_reactions.parquet"

OUTPUT_BASE  = "data/transition1x/neb_calibration"
N_IMAGES     = 9

def setup_signal_handling(out_dir):
    def handle_sigterm(signum, frame):
        with open(os.path.join(out_dir, "status.txt"), "w") as f:
            f.write("PARTIAL")
        print("Caught SIGTERM. Exiting gracefully.")
        sys.exit(1)
    signal.signal(signal.SIGTERM, handle_sigterm)

def write_status(out_dir, status):
    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write(status)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rxn_id", type=str, required=True)
    args   = parser.parse_args()
    rxn_id = args.rxn_id

    out_dir = os.path.join(OUTPUT_BASE, rxn_id)
    os.makedirs(out_dir, exist_ok=True)
    setup_signal_handling(out_dir)

    # 1. Load geometries
    df  = pd.read_parquet(PARQUET_PATH)
    row = df[df["rxn_id"] == rxn_id]
    if len(row) == 0:
        print(f"rxn_id {rxn_id} not found.")
        sys.exit(1)

    reactant = ase_read(row.iloc[0]["r_xyz_path"])
    product  = ase_read(row.iloc[0]["p_xyz_path"])

    # 2. Build images and IDPP interpolate
    images = [reactant.copy()] + [reactant.copy() for _ in range(N_IMAGES-2)] + [product.copy()]
    neb = NEB(images, method='improvedtangent')
    try:
        neb.interpolate("idpp")
    except Exception:
        write_status(out_dir, "FAILED_IDPP")
        sys.exit(1)

    # 3. Attach calculators to all images including endpoints
    for image in images:
        image.calc = XTB(method="GFN2-xTB")

    # Step 4: Standard NEB — tighter threshold before handing to CI-NEB
    opt_std = FIRE(neb,
                trajectory=os.path.join(out_dir, "neb_std.traj"),
                logfile=os.path.join(out_dir,    "neb_std.log"))
    try:
        opt_std.run(fmax=0.05, steps=2000)  # tighter: 0.05 not 0.08
    except Exception:
        write_status(out_dir, "FAILED_STD_NEB")
        sys.exit(1)

    # Step 5: CI-NEB — try LBFGS, fall back to FIRE with small timestep
    neb_ci = NEB(images, method='improvedtangent', climb=True)

    try:
        opt_ci = LBFGS(neb_ci, memory=10,
                   trajectory=os.path.join(out_dir, "neb_ci.traj"),
                   logfile=os.path.join(out_dir,    "neb_ci.log"))
        converged = opt_ci.run(fmax=0.05, steps=1500)
    except Exception:
    # LBFGS line search failed — fall back to FIRE with conservative timestep
        print(f"LBFGS failed for {rxn_id}, falling back to FIRE")
        neb_ci2 = NEB(images, method='improvedtangent', climb=True)
        try:
            opt_fb = FIRE(neb_ci2, dt=0.05,  # small timestep prevents divergence
                      trajectory=os.path.join(out_dir, "neb_ci_fallback.traj"),
                      logfile=os.path.join(out_dir,    "neb_ci_fallback.log"))
            converged = opt_fb.run(fmax=0.05, steps=2000)
        except Exception:
            write_status(out_dir, "FAILED_CI_NEB")
            sys.exit(1)

    # 6. Extract TS unconditionally — save geometry even if not fully converged
    #    so compute is never wasted
    try:
        energies  = [img.get_potential_energy() for img in images[1:-1]]
        ts_idx    = np.argmax(energies) + 1         # +1 offset for endpoints
        ts_image  = images[ts_idx]

        e_reactant = images[0].get_potential_energy()  # FIX: use images[0], not reactant
        e_ts       = ts_image.get_potential_energy()

        ase_write(os.path.join(out_dir, "ci_neb_ts.xyz"), ts_image)

        with open(os.path.join(out_dir, "neb_energies.txt"), "w") as f:
            f.write(f"E_reactant_eV={e_reactant:.8f}\n")
            f.write(f"E_ts_eV={e_ts:.8f}\n")
            f.write(f"dE_NEB_eV={e_ts - e_reactant:.8f}\n")
            f.write(f"converged={converged}\n")

    except Exception as e:
        write_status(out_dir, "FAILED_ENERGY_EXTRACTION")
        print(f"Energy extraction failed: {e}")
        sys.exit(1)

    # 7. Write final status — trust optimizer's own convergence judgment
    write_status(out_dir, "CONVERGED" if converged else "NOT_CONVERGED")
    print(f"{'CONVERGED' if converged else 'NOT_CONVERGED'}: {rxn_id}")

if __name__ == "__main__":
    main()