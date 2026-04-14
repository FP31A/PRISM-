# -*- coding: utf-8 -*-
"""
PRISM Step 6B.4 — Train NEB Calibration Model h
src/descriptors/neb_calibration.py

Model: E_NEB_strain_hat = h(E_IDPP_strain, dPMI_1,2,3, RMSD_R_P, n_rot_bonds, rxn_class)
Candidates: Random Forest (n=200) and Kernel Ridge Regression only (per roadmap).
Trains on 80% of 3,000-reaction calibration subset, evaluates on held-out 20%.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

TIER1_FILE  = "data/transition1x/descriptors/stream_b_geometric_tier1.parquet"
TIER2_FILE  = "data/transition1x/descriptors/stream_b_geometric_tier2_raw.parquet"
CURATED     = "data/transition1x/processed/final_curated_reactions.parquet"
OUTPUT_DIR  = "models"
OUTPUT_PKL  = "models/neb_calibration_model.pkl"

FEATURES = [
    'E_strain_IDPP',
    'dPMI_1', 'dPMI_2', 'dPMI_3',
    'RMSD_R_P',
    'n_rot_bonds',
    'rxn_class_enc'               # encoded rmg_family
]
TARGET = 'E_NEB_strain'

# Per roadmap: RF or KRR only — no XGBoost on small calibration set
CANDIDATES = {
    "RandomForest_200": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    ),
    "KernelRidge_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  KernelRidge(alpha=0.1, kernel="rbf", gamma=0.1))
    ]),
    "KernelRidge_Matern": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01))
    ]),
    "Ridge_baseline": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ]),
}


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    r2      = r2_score(y_test, y_pred)
    mae     = mean_absolute_error(y_test, y_pred)
    cv_r2   = cross_val_score(model, X_train, y_train,
                               cv=5, scoring='r2', n_jobs=-1)
    print(f"\n  {name}")
    print(f"    Test  R²:  {r2:.4f}   MAE: {mae:.4f} eV")
    print(f"    CV    R²:  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    return {"name": name, "model": model,
            "r2": r2, "mae": mae,
            "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std()}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    tier1   = pd.read_parquet(TIER1_FILE)
    tier2   = pd.read_parquet(TIER2_FILE)
    curated = pd.read_parquet(CURATED)[['rxn_id', 'rmg_family', 'n_rot_bonds']]

    # 2. Build calibration dataframe (3,000-reaction subset)
    df = tier1.merge(tier2,   on='rxn_id', how='inner')
    df = df.merge(curated,    on='rxn_id', how='left')

    # 3. Encode reaction family
    le = LabelEncoder()
    df['rxn_class_enc'] = le.fit_transform(
        df['rmg_family'].fillna('unknown')
    )

    # 4. Training subset: only reactions with valid TARGET
    train_df = df[df[TARGET].notna()].copy()
    print(f"Calibration pool: {len(train_df)} reactions with valid {TARGET}")

    before = len(train_df)
    train_df = train_df.dropna(subset=FEATURES + [TARGET])
    print(f"After dropping NaN in features: {len(train_df)} "
          f"(dropped {before - len(train_df)})")

    X = train_df[FEATURES].values
    y = train_df[TARGET].values

    # 5. 80/20 split — stratify by neb_quality
    # 5. 80/20 split — stratify by neb_quality
    train_df['neb_quality_strat'] = train_df['neb_quality'].replace('MISSING', 'NOT_CONVERGED')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=train_df['neb_quality_strat']
    )
    print(f"Train: {len(X_train)}   Test: {len(X_test)}")

    # 6. Train and compare all candidates
    print("\n--- Model Comparison ---")
    results = []
    for name, model in CANDIDATES.items():
        try:
            results.append(evaluate(name, model,
                                    X_train, X_test,
                                    y_train, y_test))
        except Exception as e:
            print(f"  {name} FAILED: {e}")

    # 7. Rank
    results_df = pd.DataFrame([
        {"Model":       r["name"],
         "Test R²":     round(r["r2"], 4),
         "MAE (eV)":    round(r["mae"], 4),
         "CV R² mean":  round(r["cv_r2_mean"], 4),
         "CV R² std":   round(r["cv_r2_std"], 4)}
        for r in results
    ]).sort_values("Test R²", ascending=False)

    print("\n--- Ranking ---")
    print(results_df.to_string(index=False))

    # 8. Select best by test R²
    best = max(results, key=lambda x: x["r2"])
    print(f"\nBest model: {best['name']}")
    print(f"  Test R²:  {best['r2']:.4f}  (checkpoint: >0.5)")
    print(f"  Test MAE: {best['mae']:.4f} eV")

    if best["r2"] < 0.5:
        print("\nWARNING: R² < 0.5 — per risk mitigation R4:")
        print("  Transfer learning approach abandoned.")
        print("  Use raw NEB subset directly as additional training data in Step 9.")
        print("  Report this outcome honestly in the manuscript.")
    else:
        print("\nCheckpoint PASSED — calibration model viable for 6B.5.")

    # Feature importances for tree models
    underlying = (best["model"].named_steps["model"]
                  if hasattr(best["model"], "named_steps")
                  else best["model"])
    if hasattr(underlying, "feature_importances_"):
        print("\nFeature importances:")
        for feat, imp in sorted(zip(FEATURES, underlying.feature_importances_),
                                 key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.4f}")

    # 9. Serialise — include encoder and feature list for 6B.5
    joblib.dump({
        "model":         best["model"],
        "model_name":    best["name"],
        "label_encoder": le,
        "features":      FEATURES,
        "r2_test":       best["r2"],
        "mae_test":      best["mae"],
    }, OUTPUT_PKL)
    print(f"\nSaved to {OUTPUT_PKL}")

    # Save comparison table
    results_df.to_csv("models/neb_calibration_model_comparison.csv", index=False)
    print("Comparison table saved to models/neb_calibration_model_comparison.csv")

    # Step 6B.5 is a separate script — see src/descriptors/apply_neb_calibration.py


if __name__ == "__main__":
    main()