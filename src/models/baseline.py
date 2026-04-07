# -*- coding: utf-8 -*-
"""
PRISM Step 5 Checkpoint — BEP Baseline + Descriptor Distributions

Run locally (no HPC needed):
    python src/descriptors/step5_checkpoint.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- Configuration ---
T1X_REACTIONS = "data/transition1x/processed/final_curated_reactions.parquet"
T1X_ELECTRONIC = "data/transition1x/descriptors/electronic_descriptors.parquet"
T1X_SPLITS = "data/transition1x/splits/scaffold_5fold.json"
FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"


def compute_bep_baseline(df, splits):
    """
    Fit BEP: dE_barrier = alpha * dE_rxn_xtb + beta
    per fold, report MAE, MedAE, R2.
    """
    print("=" * 60)
    print("BEP BASELINE (per-fold)")
    print("=" * 60)

    fold_results = []

    for fold_name, test_rxn_ids in splits.items():
        test_mask = df["rxn_id"].isin(test_rxn_ids)
        train = df[~test_mask].dropna(subset=["dE_xtb", "Ea_eV"])
        test = df[test_mask].dropna(subset=["dE_xtb", "Ea_eV"])

        if len(test) == 0:
            continue

        X_train = train[["dE_xtb"]].values
        y_train = train["Ea_eV"].values
        X_test = test[["dE_xtb"]].values
        y_test = test["Ea_eV"].values

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        medae = np.median(np.abs(y_test - preds))
        r2 = r2_score(y_test, preds)

        fold_results.append({
            "fold": fold_name,
            "alpha": model.coef_[0],
            "beta": model.intercept_,
            "mae": mae,
            "medae": medae,
            "r2": r2,
            "n_train": len(train),
            "n_test": len(test),
        })

        print(f"  {fold_name}: alpha={model.coef_[0]:.4f}, beta={model.intercept_:.4f}, "
              f"MAE={mae:.4f} eV, MedAE={medae:.4f} eV, R2={r2:.4f}")

    # Summary
    maes = [r["mae"] for r in fold_results]
    medaes = [r["medae"] for r in fold_results]
    r2s = [r["r2"] for r in fold_results]

    print(f"\n  MEAN +/- STD across folds:")
    print(f"    MAE:   {np.mean(maes):.4f} +/- {np.std(maes):.4f} eV")
    print(f"    MedAE: {np.mean(medaes):.4f} +/- {np.std(medaes):.4f} eV")
    print(f"    R2:    {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")

    # Save results
    results_df = pd.DataFrame(fold_results)
    os.makedirs(TABLES_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(TABLES_DIR, "bep_baseline_results.csv"), index=False)
    print(f"\n  Saved to {TABLES_DIR}/bep_baseline_results.csv")

    return fold_results


def plot_bep_parity(df, splits):
    """BEP parity plot: predicted vs actual barrier."""
    # Use all data with a global fit for the plot
    valid = df.dropna(subset=["dE_xtb", "Ea_eV"])
    X = valid[["dE_xtb"]].values
    y = valid["Ea_eV"].values

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(y, preds, alpha=0.15, s=8, c="steelblue", edgecolors="none")

    lims = [min(y.min(), preds.min()) - 0.2, max(y.max(), preds.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DFT Activation Energy (eV)")
    ax.set_ylabel("BEP Predicted (eV)")
    ax.set_title(f"BEP Baseline: R$^2$ = {r2_score(y, preds):.3f}, "
                 f"MAE = {mean_absolute_error(y, preds):.3f} eV")
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "bep_parity.png")
    plt.savefig(path, dpi=150)
    print(f"  Saved BEP parity plot to {path}")
    plt.close()


def plot_descriptor_distributions(df):
    """Plot distributions of all electronic descriptors."""
    desc_cols = ["dE_xtb", "gap_R", "gap_P", "d_gap", "dipole_R", "dipole_P", "d_dipole"]
    available = [c for c in desc_cols if c in df.columns and df[c].notna().sum() > 0]

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    labels = {
        "dE_xtb": r"$\Delta E_{\mathrm{rxn}}^{\mathrm{xTB}}$ (eV)",
        "gap_R": r"HOMO-LUMO Gap$^{\mathrm{R}}$ (eV)",
        "gap_P": r"HOMO-LUMO Gap$^{\mathrm{P}}$ (eV)",
        "d_gap": r"$\Delta$ HOMO-LUMO Gap (eV)",
        "dipole_R": r"Dipole$^{\mathrm{R}}$ (Debye)",
        "dipole_P": r"Dipole$^{\mathrm{P}}$ (Debye)",
        "d_dipole": r"$\Delta$ Dipole (Debye)",
    }

    for i, col in enumerate(available):
        data = df[col].dropna()
        axes[i].hist(data, bins=60, edgecolor="black", alpha=0.7, color="steelblue")
        axes[i].set_xlabel(labels.get(col, col))
        axes[i].set_ylabel("Count")
        axes[i].set_title(f"n={len(data)}, mean={data.mean():.2f}, std={data.std():.2f}")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Stream A: Electronic Descriptor Distributions (Transition1x)", y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "electronic_descriptor_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved descriptor distributions to {path}")
    plt.close()


def plot_ea_vs_descriptors(df):
    """Scatter plots of each descriptor vs activation energy."""
    desc_cols = ["dE_xtb", "gap_R", "gap_P", "d_gap", "dipole_R", "dipole_P", "d_dipole"]
    available = [c for c in desc_cols if c in df.columns and df[c].notna().sum() > 0]

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(available):
        valid = df[[col, "Ea_eV"]].dropna()
        corr = valid[col].corr(valid["Ea_eV"])
        axes[i].scatter(valid[col], valid["Ea_eV"], alpha=0.1, s=5,
                        c="steelblue", edgecolors="none")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Ea (eV)")
        axes[i].set_title(f"Pearson r = {corr:.3f}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Stream A: Descriptors vs Activation Energy", y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "electronic_vs_ea.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved descriptor vs Ea plots to {path}")
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Load and merge
    print("Loading data...")
    reactions = pd.read_parquet(T1X_REACTIONS)
    electronic = pd.read_parquet(T1X_ELECTRONIC)

    # electronic.parquet may already contain Ea_eV from the merge in electronic.py
    # If not, merge on rxn_id
    if "Ea_eV" not in electronic.columns:
        df = pd.merge(reactions[["rxn_id", "Ea_eV"]], electronic, on="rxn_id", how="inner")
    else:
        df = electronic.copy()

    print(f"Merged dataset: {len(df)} reactions, {df.columns.tolist()}")

    # Load splits
    with open(T1X_SPLITS, "r") as f:
        splits = json.load(f)

    # 1. BEP Baseline
    print()
    compute_bep_baseline(df, splits)

    # 2. BEP Parity Plot
    print()
    plot_bep_parity(df, splits)

    # 3. Descriptor Distributions
    print()
    plot_descriptor_distributions(df)

    # 4. Descriptor vs Ea scatter
    print()
    plot_ea_vs_descriptors(df)

    print("\n" + "=" * 60)
    print("STEP 5 CHECKPOINT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()