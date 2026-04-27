# -*- coding: utf-8 -*-
"""
PRISM Step 11 — External Validation on Grambow/RDB7
src/validation/external.py

Applies the trained model (without retraining) to Grambow.
Reports regression, screening, distribution shift, and conformal coverage.
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr, ks_2samp
from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                              r2_score, recall_score, precision_score,
                              f1_score, matthews_corrcoef)

MODEL_PATH     = "models/prism_best_model.pkl"
T1X_MATRIX     = "data/transition1x/features/feature_matrix.parquet"
GRAMBOW_MATRIX = "data/grambow/features/feature_matrix.parquet"
T1X_CONFORMAL  = "results/conformal_predictions.csv"
OUTPUT_DIR     = "results/external_validation"
TARGET         = "Ea_eV"
SCREEN_THRESHOLD = 4.3  # eV — same as internal (Step 9.3)


def compute_metrics(y_true, y_pred):
    mae   = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    r, _  = pearsonr(y_true, y_pred)

    true_pos = (y_true < SCREEN_THRESHOLD).astype(int)
    pred_pos = (y_pred < SCREEN_THRESHOLD).astype(int)

    if len(np.unique(true_pos)) < 2:
        recall = precision = f1 = mcc = np.nan
    else:
        recall    = recall_score(true_pos, pred_pos, zero_division=0)
        precision = precision_score(true_pos, pred_pos, zero_division=0)
        f1        = f1_score(true_pos, pred_pos, zero_division=0)
        mcc       = matthews_corrcoef(true_pos, pred_pos)

    return {
        "MAE": mae, "MedAE": medae, "R2": r2, "Pearson_r": r,
        "Recall": recall, "Precision": precision, "F1": f1, "MCC": mcc,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load model
    print("Loading trained model...")
    artifact     = joblib.load(MODEL_PATH)
    model        = artifact["model"]
    feature_cols = artifact["feature_cols"]
    model_name   = artifact["model_name"]
    print(f"  Model: {model_name}")
    print(f"  Features: {len(feature_cols)}")

    # 2. Load datasets
    print("\nLoading datasets...")
    t1x = pd.read_parquet(T1X_MATRIX)
    grm = pd.read_parquet(GRAMBOW_MATRIX)
    print(f"  Transition1x: {len(t1x)} reactions")
    print(f"  Grambow:      {len(grm)} reactions")

    # Verify columns
    missing = [c for c in feature_cols if c not in grm.columns]
    if missing:
        raise ValueError(f"Grambow missing features: {missing}")

    # ================================================================
    # 11.2 — Apply model (no retraining)
    # ================================================================
    print("\n11.2 APPLYING MODEL TO GRAMBOW")
    print("=" * 60)

    X_grm  = grm[feature_cols].values.astype(float)
    y_grm  = grm[TARGET].values
    y_pred = model.predict(X_grm)

    metrics = compute_metrics(y_grm, y_pred)

    # 11.3 — Regression metrics
    print("\n11.3 REGRESSION METRICS")
    print("-" * 40)
    print(f"  MAE:       {metrics['MAE']:.4f} eV")
    print(f"  MedAE:     {metrics['MedAE']:.4f} eV")
    print(f"  R²:        {metrics['R2']:.4f}")
    print(f"  Pearson r: {metrics['Pearson_r']:.4f}")

    # Internal comparison
    X_t1x  = t1x[feature_cols].values.astype(float)
    y_t1x  = t1x[TARGET].values
    y_t1x_pred = model.predict(X_t1x)
    t1x_metrics = compute_metrics(y_t1x, y_t1x_pred)

    print("\n  Comparison (internal vs external):")
    print(f"  {'Metric':<12} {'Transition1x':>14} {'Grambow':>14}")
    print(f"  {'MAE':<12} {t1x_metrics['MAE']:>14.4f} {metrics['MAE']:>14.4f}")
    print(f"  {'MedAE':<12} {t1x_metrics['MedAE']:>14.4f} {metrics['MedAE']:>14.4f}")
    print(f"  {'R²':<12} {t1x_metrics['R2']:>14.4f} {metrics['R2']:>14.4f}")
    print(f"  {'Pearson r':<12} {t1x_metrics['Pearson_r']:>14.4f} {metrics['Pearson_r']:>14.4f}")

    # Screening
    print(f"\n  SCREENING (threshold: {SCREEN_THRESHOLD} eV)")
    print(f"  {'Metric':<12} {'Transition1x':>14} {'Grambow':>14} {'Target':>10}")
    print(f"  {'Recall':<12} {t1x_metrics['Recall']:>14.4f} {metrics['Recall']:>14.4f} {'≥0.90':>10}")
    print(f"  {'Precision':<12} {t1x_metrics['Precision']:>14.4f} {metrics['Precision']:>14.4f} {'≥0.75':>10}")
    print(f"  {'F1':<12} {t1x_metrics['F1']:>14.4f} {metrics['F1']:>14.4f}")
    print(f"  {'MCC':<12} {t1x_metrics['MCC']:>14.4f} {metrics['MCC']:>14.4f}")

    h3_external = metrics["Recall"] >= 0.90
    print(f"\n  H3 (external): {'PASSED' if h3_external else 'FAILED'} "
          f"(recall={metrics['Recall']:.4f})")

    # ================================================================
    # 11.4 — Distribution shift (KS tests)
    # ================================================================
    print("\n11.4 DISTRIBUTION SHIFT ANALYSIS")
    print("=" * 60)

    ks_rows = []
    for col in feature_cols:
        t1x_vals = t1x[col].dropna().values
        grm_vals = grm[col].dropna().values
        if len(t1x_vals) > 10 and len(grm_vals) > 10:
            stat, pval = ks_2samp(t1x_vals, grm_vals)
            ks_rows.append({
                "feature": col,
                "KS_stat": stat,
                "p_value": pval,
                "significant": pval < 0.01,
            })

    ks_df = pd.DataFrame(ks_rows).sort_values("KS_stat", ascending=False)
    n_sig = ks_df["significant"].sum()
    print(f"  Features with significant shift (p<0.01): {n_sig}/{len(ks_df)}")
    print(f"\n  Top 10 shifted features:")
    print(ks_df.head(10)[["feature", "KS_stat", "p_value"]].to_string(
        index=False, float_format="{:.4f}".format
    ))
    ks_df.to_csv(f"{OUTPUT_DIR}/ks_distribution_shift.csv", index=False)

    # Target distribution comparison
    ks_target, p_target = ks_2samp(y_t1x, y_grm)
    print(f"\n  Target (Ea_eV) shift: KS={ks_target:.4f}, p={p_target:.4e}")
    print(f"  T1x Ea: mean={y_t1x.mean():.3f}, std={y_t1x.std():.3f} eV")
    print(f"  Grm Ea: mean={y_grm.mean():.3f}, std={y_grm.std():.3f} eV")

    # ================================================================
    # 11.5 — Conformal coverage on Grambow
    # ================================================================
    print("\n11.5 CONFORMAL COVERAGE ON GRAMBOW")
    print("=" * 60)

    if os.path.exists(T1X_CONFORMAL):
        conf = pd.read_csv(T1X_CONFORMAL)

        # Get the calibrated q_hat from internal conformal
        if 'q_hat_norm' in conf.columns:
            q_hat = conf['q_hat_norm'].median()
        elif 'interval_width' in conf.columns and 'sigma_ensemble' in conf.columns:
            # Reconstruct: width = 2 * q_hat * sigma
            q_hat = (conf['interval_width'] / (2 * conf['sigma_ensemble'])).median()
        else:
            q_hat = None

        if q_hat is not None:
            print(f"  Using q_hat from internal calibration: {q_hat:.4f}")

            # Need ensemble predictions for Grambow — run simple bootstrap
            print("  Computing ensemble σ for Grambow (10 bootstrap members)...")
            from sklearn.utils import resample

            n_members  = 10
            preds_all  = np.zeros((n_members, len(X_grm)))
            X_t1x_full = t1x[feature_cols].values.astype(float)
            y_t1x_full = t1x[TARGET].values

            for i in range(n_members):
                X_boot, y_boot = resample(X_t1x_full, y_t1x_full,
                                           random_state=i)
                from sklearn.base import clone
                m = clone(model)
                m.fit(X_boot, y_boot)
                preds_all[i] = m.predict(X_grm)
                if (i + 1) % 5 == 0:
                    print(f"    Member {i+1}/10 done", flush=True)

            y_pred_ensemble = preds_all.mean(axis=0)
            sigma_ensemble  = preds_all.std(axis=0)

            # Conformal intervals
            lower = y_pred_ensemble - q_hat * sigma_ensemble
            upper = y_pred_ensemble + q_hat * sigma_ensemble
            covered = ((y_grm >= lower) & (y_grm <= upper))
            coverage = covered.mean()
            width    = (upper - lower).mean()

            print(f"\n  Empirical coverage: {coverage:.4f} (nominal: 0.90)")
            print(f"  Mean interval width: {width:.4f} eV")
            print(f"  Coverage degradation: {0.90 - coverage:+.4f}")

            if coverage < 0.85:
                print("  WARNING: Coverage degraded >5% — distribution shift impact.")
            elif coverage < 0.90:
                print("  NOTE: Mild coverage degradation — expected for cross-dataset.")
            else:
                print("  Coverage maintained — model generalizes well.")
        else:
            print("  Could not extract q_hat from conformal predictions.")
    else:
        print("  Internal conformal predictions not found. Skipping.")

    # ================================================================
    # 11.6 — Save predictions for parity plots
    # ================================================================
    print("\n11.6 SAVING PREDICTIONS")
    print("=" * 60)

    results_df = pd.DataFrame({
        "rxn_id": grm["rxn_id"].values,
        "y_true": y_grm,
        "y_pred": y_pred,
        "abs_error": np.abs(y_grm - y_pred),
    })
    results_df.to_csv(f"{OUTPUT_DIR}/grambow_predictions.csv", index=False)

    # Summary table
    summary = pd.DataFrame([
        {"Dataset": "Transition1x (internal)", **t1x_metrics},
        {"Dataset": "Grambow (external)", **metrics},
    ])
    summary.to_csv(f"{OUTPUT_DIR}/internal_vs_external.csv", index=False)

    print(f"  Predictions saved to {OUTPUT_DIR}/grambow_predictions.csv")
    print(f"  Comparison saved to {OUTPUT_DIR}/internal_vs_external.csv")
    print(f"  KS tests saved to {OUTPUT_DIR}/ks_distribution_shift.csv")

    # ================================================================
    # Final summary
    # ================================================================
    print("\n" + "=" * 60)
    print("=== Step 11 Complete ===")
    print("=" * 60)
    print(f"Model: {model_name} (trained on Transition1x, no retraining)")
    print(f"\n{'Metric':<16} {'Internal':>12} {'External':>12}")
    print(f"{'MAE (eV)':<16} {t1x_metrics['MAE']:>12.4f} {metrics['MAE']:>12.4f}")
    print(f"{'MedAE (eV)':<16} {t1x_metrics['MedAE']:>12.4f} {metrics['MedAE']:>12.4f}")
    print(f"{'R²':<16} {t1x_metrics['R2']:>12.4f} {metrics['R2']:>12.4f}")
    print(f"{'Pearson r':<16} {t1x_metrics['Pearson_r']:>12.4f} {metrics['Pearson_r']:>12.4f}")
    print(f"{'Recall@{SCREEN_THRESHOLD}':<16} {t1x_metrics['Recall']:>12.4f} {metrics['Recall']:>12.4f}")
    print(f"{'Precision@{SCREEN_THRESHOLD}':<16} {t1x_metrics['Precision']:>12.4f} {metrics['Precision']:>12.4f}")
    print(f"\nH3 external: {'PASSED' if h3_external else 'FAILED'}")
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()