# -*- coding: utf-8 -*-
"""
PRISM Step 5 Supplement — Add missing Stream A descriptors:
    - Fukui indices (f+, f- max over all atoms via --vfukui)
    - Wiberg Bond Order changes on reactive bonds (delta_WBO)
    - BEP baseline prediction (global fit, refitted per fold in Step 9)

Merges into existing stream_a parquet files for both datasets.
Run: python src/descriptors/electronic_supplement.py --dataset transition1x
     python src/descriptors/electronic_supplement.py --dataset grambow
"""

import os
import re
import argparse
import subprocess
import tempfile
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from sklearn.linear_model import LinearRegression

DATASET_CONFIG = {
    "transition1x": {
        "parquet":    "data/transition1x/descriptors/stream_a_electronic_trans1x.parquet",
        "error_log":  "data/transition1x/descriptors/xtb_supplement_errors.log",
    },
    "grambow": {
        "parquet":    "data/grambow/descriptors/electronic_descriptors.parquet",
        "error_log":  "data/grambow/descriptors/xtb_supplement_errors.log",
    },
}

XTB_CORES = "1"
N_JOBS    = 24


# ── xTB runner ─────────────────────────────────────────────────────────────────

def run_xtb(xyz_path, flags, tmpdir):
    """Run xTB with given flags, return stdout or None on failure."""
    if not isinstance(xyz_path, str) or not os.path.exists(xyz_path):
        return None
    cmd = ["xtb", os.path.abspath(xyz_path), "--gfn", "2"] + flags
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = XTB_CORES
    try:
        result = subprocess.run(
            cmd, cwd=tmpdir, env=env,
            capture_output=True, text=True, timeout=120
        )
        return result.stdout
    except Exception:
        return None


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_fukui(output):
    """
    Parse xTB --vfukui output block.
    Format:
         #        f(+)     f(-)     f(0)
         1O       0.196    0.151    0.173
         2C       0.072    0.009    0.041
    Returns (f_plus_max, f_minus_max) over all atoms.
    """
    if output is None:
        return np.nan, np.nan

    f_plus, f_minus = [], []
    in_block = False

    for line in output.splitlines():
        if "f(+)" in line and "f(-)" in line:
            in_block = True
            continue
        if in_block:
            parts = line.split()
            # Read from the end to avoid formatting issues with atom symbols
            if len(parts) >= 3:
                try:
                    f_plus.append(float(parts[-3]))
                    f_minus.append(float(parts[-2]))
                except ValueError:
                    in_block = False
            else:
                in_block = False

    if f_plus and f_minus:
        return float(max(f_plus)), float(max(f_minus))
    return np.nan, np.nan


def parse_wbo(output):
    if output is None:
        return {}

    wbo = {}
    in_block = False

    for line in output.splitlines():
        if "Wiberg/Mayer" in line:
            in_block = True
            continue
        if not in_block:
            continue

        # Skip separators, headers, empty lines
        stripped = line.strip()
        if not stripped or "---" in line or stripped.startswith("#"):
            continue

        # Only process lines that contain " -- " (actual data lines)
        # All other lines (descriptive text, section headers) are skipped
        if " -- " not in line:
            continue

        parts = line.split("--")
        try:
            atom_i = int(parts[0].split()[0])
        except (ValueError, IndexError):
            continue

        neighbours = " ".join(parts[1:]).split()
        j = 0
        while j + 2 < len(neighbours):
            try:
                atom_j  = int(neighbours[j])
                wbo_val = float(neighbours[j + 2])
                if wbo_val > 0.1:
                    key = (min(atom_i, atom_j), max(atom_i, atom_j))
                    wbo[key] = wbo_val
                j += 3
            except (ValueError, IndexError):
                j += 1

    return wbo


# ── Reactive bond identification ───────────────────────────────────────────────

def get_reactive_bonds(r_smiles, p_smiles):
    """
    Bonds that changed between reactant and product (formed or broken).
    Returns set of (min_idx, max_idx) tuples, 0-indexed.
    Falls back to empty set on any RDKit failure.
    """
    try:
        mol_r = Chem.MolFromSmiles(r_smiles)
        mol_p = Chem.MolFromSmiles(p_smiles)
        if mol_r is None or mol_p is None:
            return set()
        bonds_r = {(min(b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1),
                    max(b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1))
                   for b in mol_r.GetBonds()}
        bonds_p = {(min(b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1),
                    max(b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1))
                   for b in mol_p.GetBonds()}
        return bonds_r.symmetric_difference(bonds_p)
    except Exception:
        return set()


# ── Per-reaction worker ────────────────────────────────────────────────────────

def process_reaction(row, error_log):
    rxn_id = row["rxn_id"]
    r_path = row["r_xyz_path"]
    p_path = row["p_xyz_path"]

    result = {
        "rxn_id":      rxn_id,
        "fukui_plus":  np.nan,
        "fukui_minus": np.nan,
        "delta_WBO":   np.nan,
    }

    with tempfile.TemporaryDirectory() as tmpdir:

        # 1. Fukui — reactant only
        out_fukui = run_xtb(r_path, ["--vfukui"], tmpdir)
        if out_fukui:
            fp, fm = parse_fukui(out_fukui)
            result["fukui_plus"]  = fp
            result["fukui_minus"] = fm
        else:
            with open(error_log, "a") as f:
                f.write(f"{rxn_id} | Fukui xTB failed\n")

        # 2. WBO — reactant and product
        out_r_wbo = run_xtb(r_path, ["--wbo"], tmpdir)
        out_p_wbo = run_xtb(p_path, ["--wbo"], tmpdir)

        if out_r_wbo and out_p_wbo:
            wbo_r = parse_wbo(out_r_wbo)
            wbo_p = parse_wbo(out_p_wbo)

            # All bonds seen in either R or P geometry (same xTB xyz indexing)
            all_bonds = set(wbo_r.keys()) | set(wbo_p.keys())

            # Reactive bonds: WBO changes by >0.4
            # (full bond breaking: ~1.0 -> ~0.0)
            reactive_deltas = []
            for b in all_bonds:
                w_r = wbo_r.get(b, 0.0)
                w_p = wbo_p.get(b, 0.0)
                diff = abs(w_p - w_r)
                if diff > 0.4:
                    reactive_deltas.append(diff)

            if reactive_deltas:
                result["delta_WBO"] = float(np.mean(reactive_deltas))
            else:
                # Fallback: highly concerted reaction with no single large WBO change
                # Use mean over all bonds with any non-trivial change
                non_zero = [abs(wbo_p.get(b, 0.0) - wbo_r.get(b, 0.0))
                            for b in all_bonds
                            if abs(wbo_p.get(b, 0.0) - wbo_r.get(b, 0.0)) > 0.05]
                if non_zero:
                    result["delta_WBO"] = float(np.mean(non_zero))

        else:
            with open(error_log, "a") as f:
                f.write(f"{rxn_id} | WBO xTB failed\n")

    return result


# ── BEP baseline ───────────────────────────────────────────────────────────────

def compute_bep_baseline(df):
    """
    Global BEP fit: ΔE‡ = α·dE_xtb + β
    Fitted on all reactions with valid dE_xtb and Ea_eV.
    Note: per-fold refitting happens in Step 9 — this global fit
    is included as a feature for the downstream model only.
    """
    valid = df[df["dE_xtb"].notna() & df["Ea_eV"].notna()]
    if len(valid) < 10:
        print("WARNING: Not enough valid reactions for BEP fit.")
        return np.nan, np.nan, pd.Series(np.nan, index=df.index)

    X = valid[["dE_xtb"]].values
    y = valid["Ea_eV"].values

    model  = LinearRegression().fit(X, y)
    alpha  = model.coef_[0]
    beta   = model.intercept_
    r2     = model.score(X, y)

    print(f"  BEP fit: ΔE‡ = {alpha:.4f}·dE_xtb + {beta:.4f}  (R²={r2:.4f})")
    print(f"  Checkpoint: R² should be ~0.3–0.4")

    # Apply to all rows with valid dE_xtb
    bep_pred = pd.Series(np.nan, index=df.index)
    all_valid_mask = df["dE_xtb"].notna()
    bep_pred[all_valid_mask] = (
        alpha * df.loc[all_valid_mask, "dE_xtb"].values + beta
    )
    return alpha, beta, bep_pred


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["transition1x", "grambow"])
    args = parser.parse_args()

    cfg       = DATASET_CONFIG[args.dataset]
    parquet   = cfg["parquet"]
    error_log = cfg["error_log"]

    with open(error_log, "w") as f:
        f.write(f"--- xTB Supplement Error Log ({args.dataset}) ---\n")

    print(f"Loading {parquet}...")
    df = pd.read_parquet(parquet)
    print(f"  {len(df)} reactions loaded.")

    # 1. Fukui + WBO in parallel
    print(f"\nRunning Fukui (--vfukui) + WBO calculations ({N_JOBS} workers)...")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_reaction)(row, error_log)
        for _, row in df.iterrows()
    )

    supp_df = pd.DataFrame(results)

    # 2. BEP baseline
    print("\nFitting BEP baseline...")
    alpha, beta, bep_pred = compute_bep_baseline(df)
    supp_df["bep_prediction"] = bep_pred.values

    # 3. Drop existing columns to avoid _x/_y on re-run
    for col in ["fukui_plus", "fukui_minus", "delta_WBO", "bep_prediction"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    final_df = df.merge(
        supp_df[["rxn_id", "fukui_plus", "fukui_minus",
                 "delta_WBO", "bep_prediction"]],
        on="rxn_id", how="left"
    )

    # 4. Save back to same parquet
    final_df.to_parquet(parquet, index=False)

    # 5. Summary
    print(f"\n--- Summary ({args.dataset}) ---")
    for col in ["fukui_plus", "fukui_minus", "delta_WBO", "bep_prediction"]:
        valid = final_df[col].notna().sum()
        print(f"  {col}: {valid}/{len(final_df)} ({valid/len(final_df)*100:.1f}%)")

    failures = sum(1 for line in open(error_log)
                   if "|" in line and "Log" not in line)
    print(f"\n  xTB failures logged: {failures}")
    print(f"  Error log: {error_log}")
    print(f"\nSaved to {parquet}")
    print(f"Final columns: {final_df.columns.tolist()}")


if __name__ == "__main__":
    main()