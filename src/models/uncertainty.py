# -*- coding: utf-8 -*-
"""
PRISM Step 10 — Uncertainty Quantification
src/models/uncertainty.py

Tasks:
  10.1 Ensemble variance: 10 XGBoost instances with different seeds and
       full bootstrap resamples → σ_ensemble per reaction → flag σ > 0.5 eV
  10.2 Normalized conformal prediction: proper train / calibration split →
       non-conformity scores normalized by σ_ensemble → locally adaptive
       intervals that widen for hard reactions → coverage at 90% nominal
  10.3 Cross-reference: do high-error reactions have high σ_ensemble?

Design decisions:
  - All 10 ensemble members use the same aggregated hyperparameters from
    Step 9.1 (fixed architecture, varied data/seed).
  - Ensemble uses bootstrap (sample with replacement at full training size)
    for proper model diversity; ddof=1 (Bessel's correction) for σ.
  - Ensemble predictions and σ are computed via scaffold 5-fold OOF to
    avoid train-on-predict leakage.
  - Conformal prediction uses NORMALIZED scores (|error| / σ_ensemble),
    producing locally adaptive intervals (Lei et al. 2018, Romano et al.
    2019). Intervals automatically widen for uncertain reactions.
  - Conformal uses 80/20 proper-train/calibration split within each outer
    fold's training set. Coverage is reported both per-fold and pooled.
  - Scaffold splits break the exchangeability assumption; some coverage
    degradation is expected and documented.
"""

from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURE_MATRIX = "data/transition1x/features/feature_matrix.parquet"
SPLITS_FILE    = "data/transition1x/splits/scaffold_5fold.json"
MODEL_PKL      = "models/prism_best_model.pkl"
OUTPUT_DIR     = "results"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET           = "Ea_eV"
N_ENSEMBLE       = 10
SIGMA_THRESHOLD  = 0.5      # eV — flag threshold for high uncertainty
CONFORMAL_ALPHA  = 0.10     # 90% nominal coverage
N_JOBS           = -1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_splits(path, df):
    with open(path) as f:
        raw = json.load(f)
    all_rxn = set(df["rxn_id"].values)
    folds   = []
    for key in sorted(raw.keys()):
        test_ids  = set(raw[key]) & all_rxn
        train_ids = all_rxn - test_ids
        test_idx  = df.index[df["rxn_id"].isin(test_ids)].tolist()
        train_idx = df.index[df["rxn_id"].isin(train_ids)].tolist()
        folds.append((train_idx, test_idx))
    return folds


def build_xgb(params, n_estimators, seed):
    """Build one XGBoost model with a specific random seed."""
    return xgb.XGBRegressor(
        n_estimators     = n_estimators,
        learning_rate    = params["lr"],
        max_depth        = params["max_depth"],
        subsample        = params["subsample"],
        colsample_bytree = params["colsample"],
        reg_lambda       = params["lambda"],
        eval_metric      = "mae",
        n_jobs           = N_JOBS,
        random_state     = seed,
        verbosity        = 0,
    )


# ---------------------------------------------------------------------------
# 10.1 — Ensemble variance (OOF)
# ---------------------------------------------------------------------------
def run_ensemble_variance(df, folds, feature_cols, params, n_estimators):
    """
    For each outer fold:
      - Train N_ENSEMBLE models on 80% subsamples of the training set
        with different random seeds.
      - Predict on the test fold with all 10 models.
      - Compute mean prediction and σ_ensemble per test reaction.

    Returns a DataFrame aligned with df containing:
      y_true, y_pred_ensemble (mean), sigma_ensemble (std),
      abs_error, high_sigma (bool)
    """
    print("\n10.1 ENSEMBLE VARIANCE")
    print("=" * 72)
    print(f"  {N_ENSEMBLE} members, full bootstrap (replace=True), "
          f"σ threshold = {SIGMA_THRESHOLD} eV")

    X_all = df[feature_cols].values.astype(float)
    y_all = df[TARGET].values
    n     = len(df)

    # Preallocate arrays for all reactions (filled via OOF)
    y_pred_mean  = np.full(n, np.nan)
    y_pred_std   = np.full(n, np.nan)
    # Store all ensemble member predictions for later analysis
    ensemble_preds = np.full((n, N_ENSEMBLE), np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n  Fold {fold_idx + 1}/5  "
              f"(train={len(train_idx)}, test={len(test_idx)})")

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test  = X_all[test_idx]
        n_train = len(train_idx)

        fold_preds = np.zeros((len(test_idx), N_ENSEMBLE))

        for m_idx in range(N_ENSEMBLE):
            seed = fold_idx * 1000 + m_idx * 7 + 42
            rng  = np.random.RandomState(seed)

            # Bootstrap: sample WITH replacement at full training size
            # This gives ~63.2% unique points per bag, ensuring enough
            # diversity between members for reliable σ_ensemble.
            boot_idx = rng.choice(n_train, size=n_train, replace=True)
            X_boot   = X_train[boot_idx]
            y_boot   = y_train[boot_idx]

            model = build_xgb(params, n_estimators, seed)
            model.fit(X_boot, y_boot)
            fold_preds[:, m_idx] = model.predict(X_test)

            if m_idx == 0 or (m_idx + 1) % 5 == 0:
                mae = mean_absolute_error(y_all[test_idx],
                                          fold_preds[:, m_idx])
                print(f"    Member {m_idx+1:2d}: MAE={mae:.4f}")

        # Store per-reaction results
        y_pred_mean[test_idx] = fold_preds.mean(axis=1)
        y_pred_std[test_idx]  = fold_preds.std(axis=1, ddof=1)
        ensemble_preds[test_idx, :] = fold_preds

        fold_mae = mean_absolute_error(y_all[test_idx],
                                       y_pred_mean[test_idx])
        print(f"    Ensemble mean MAE: {fold_mae:.4f}")
        print(f"    Mean σ: {y_pred_std[test_idx].mean():.4f} eV")

    # Build results dataframe
    abs_error  = np.abs(y_all - y_pred_mean)
    high_sigma = y_pred_std > SIGMA_THRESHOLD

    results = pd.DataFrame({
        "rxn_id":          df["rxn_id"].values,
        "y_true":          y_all,
        "y_pred_ensemble": y_pred_mean,
        "sigma_ensemble":  y_pred_std,
        "abs_error":       abs_error,
        "high_sigma":      high_sigma,
    })

    # Add family labels if available
    if "rmg_family" in df.columns:
        results["rmg_family"] = df["rmg_family"].values

    # Summary statistics
    pooled_mae = mean_absolute_error(y_all, y_pred_mean)
    n_flagged  = high_sigma.sum()
    pct_flagged = 100 * n_flagged / n

    print(f"\n  --- Ensemble Summary ---")
    print(f"  Pooled ensemble MAE: {pooled_mae:.4f} eV")
    print(f"  Mean σ_ensemble:     {y_pred_std.mean():.4f} eV")
    print(f"  Median σ_ensemble:   {np.median(y_pred_std):.4f} eV")
    print(f"  Max σ_ensemble:      {y_pred_std.max():.4f} eV")
    print(f"  Flagged (σ > {SIGMA_THRESHOLD}): "
          f"{n_flagged}/{n} ({pct_flagged:.1f}%)")

    # σ distribution by decile
    print(f"\n  σ_ensemble percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    {p:3d}th: {np.percentile(y_pred_std, p):.4f} eV")

    results.to_csv(f"{OUTPUT_DIR}/ensemble_predictions.csv", index=False)
    print(f"\n  Saved to {OUTPUT_DIR}/ensemble_predictions.csv")

    # Save raw ensemble predictions for downstream analysis
    ens_df = pd.DataFrame(
        ensemble_preds,
        columns=[f"member_{i}" for i in range(N_ENSEMBLE)]
    )
    ens_df.insert(0, "rxn_id", df["rxn_id"].values)
    ens_df.to_csv(f"{OUTPUT_DIR}/ensemble_raw_predictions.csv", index=False)

    return results


# ---------------------------------------------------------------------------
# 10.2 — Split conformal prediction (OOF)
# ---------------------------------------------------------------------------
def run_conformal_prediction(df, folds, feature_cols, params, n_estimators,
                             ensemble_sigma):
    """
    NORMALIZED split conformal prediction.

    For each outer fold:
      1. Split training set into 80% proper-train + 20% calibration.
      2. Train on proper-train.
      3. Compute NORMALIZED non-conformity scores on calibration:
           s_i = |y_i - ŷ_i| / (σ_ensemble_i + ε)
         where σ_ensemble comes from 10.1 and ε is a small floor to
         prevent division by zero.
      4. Compute q̂_norm = (1-α)(1 + 1/n_cal) quantile of {s_i}.
      5. On test: interval = [ŷ ± q̂_norm · (σ_ensemble + ε)].
         Intervals automatically WIDEN for hard reactions (high σ)
         and NARROW for easy ones (low σ).
      6. Measure empirical coverage.

    This is "locally adaptive" conformal prediction (Lei et al. 2018,
    Romano et al. 2019). The σ_ensemble normalization means the conformal
    guarantee is approximately conditional on difficulty, not just marginal.

    The scaffold split breaks exchangeability, so some coverage degradation
    is expected and physically meaningful.
    """
    print(f"\n10.2 NORMALIZED CONFORMAL PREDICTION")
    print("=" * 72)
    print(f"  Nominal coverage: {100*(1-CONFORMAL_ALPHA):.0f}%")
    print(f"  Normalizer: σ_ensemble from 10.1")
    print(f"  Calibration fraction: 20% of training set per fold")

    X_all = df[feature_cols].values.astype(float)
    y_all = df[TARGET].values
    n     = len(df)

    CAL_FRAC = 0.20
    SIGMA_FLOOR = 0.01   # prevent division by zero

    # Preallocate
    pred_lower = np.full(n, np.nan)
    pred_upper = np.full(n, np.nan)
    pred_point = np.full(n, np.nan)
    pred_width = np.full(n, np.nan)

    fold_coverages = []
    fold_mean_widths = []
    fold_quantiles = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        n_train  = len(train_idx)
        n_cal    = max(10, int(n_train * CAL_FRAC))
        n_proper = n_train - n_cal

        # Deterministic split for reproducibility
        rng       = np.random.RandomState(fold_idx + 100)
        perm      = rng.permutation(n_train)
        proper_i  = [train_idx[i] for i in perm[:n_proper]]
        cal_i     = [train_idx[i] for i in perm[n_proper:]]

        X_proper = X_all[proper_i]
        y_proper = y_all[proper_i]
        X_cal    = X_all[cal_i]
        y_cal    = y_all[cal_i]
        X_test   = X_all[test_idx]
        y_test   = y_all[test_idx]

        # σ_ensemble for calibration and test reactions (from 10.1)
        sigma_cal  = ensemble_sigma[cal_i]
        sigma_test = ensemble_sigma[test_idx]

        # Train on proper set
        model = build_xgb(params, n_estimators, seed=fold_idx + 200)
        model.fit(X_proper, y_proper)

        # Normalized non-conformity scores on calibration set
        y_cal_pred  = model.predict(X_cal)
        cal_scores  = np.abs(y_cal - y_cal_pred) / (sigma_cal + SIGMA_FLOOR)

        # Conformal quantile (on normalized scores)
        level    = min(1.0, (1 - CONFORMAL_ALPHA) * (1 + 1 / len(cal_scores)))
        q_hat    = np.quantile(cal_scores, level)

        # Predict on test — intervals scale with σ_ensemble
        y_test_pred = model.predict(X_test)
        half_width  = q_hat * (sigma_test + SIGMA_FLOOR)

        pred_point[test_idx] = y_test_pred
        pred_lower[test_idx] = y_test_pred - half_width
        pred_upper[test_idx] = y_test_pred + half_width
        pred_width[test_idx] = 2 * half_width

        # Empirical coverage
        covered   = ((y_test >= y_test_pred - half_width) &
                     (y_test <= y_test_pred + half_width))
        coverage  = covered.mean()
        mean_w    = (2 * half_width).mean()

        fold_coverages.append(coverage)
        fold_mean_widths.append(mean_w)
        fold_quantiles.append(q_hat)

        print(f"\n  Fold {fold_idx + 1}/5:")
        print(f"    Proper train: {n_proper}, Calibration: {n_cal}, "
              f"Test: {len(test_idx)}")
        print(f"    Calibration MAE:  "
              f"{mean_absolute_error(y_cal, y_cal_pred):.4f} eV")
        print(f"    Normalized q̂:    {q_hat:.4f}")
        print(f"    Mean interval width: {mean_w:.4f} eV")
        print(f"    Width range:     [{2*half_width.min():.4f}, "
              f"{2*half_width.max():.4f}] eV")
        print(f"    Empirical coverage: {coverage:.4f} "
              f"(nominal: {1-CONFORMAL_ALPHA:.2f})")

    # Summary
    mean_cov   = np.mean(fold_coverages)
    std_cov    = np.std(fold_coverages)
    mean_width = np.mean(fold_mean_widths)

    print(f"\n  --- Conformal Summary ---")
    print(f"  Mean coverage:  {mean_cov:.4f} ± {std_cov:.4f}  "
          f"(nominal: {1-CONFORMAL_ALPHA:.2f})")
    print(f"  Mean width:     {mean_width:.4f} eV")
    print(f"  Mean q̂_norm:   {np.mean(fold_quantiles):.4f}")

    # Check coverage
    if mean_cov >= (1 - CONFORMAL_ALPHA) - 0.02:
        print(f"  Coverage check: PASSED (within 2% of nominal)")
    else:
        print(f"  Coverage check: DEGRADED "
              f"(expected — scaffold splits break exchangeability)")

    # Pooled coverage
    pooled_covered = ((y_all >= pred_lower) & (y_all <= pred_upper))
    pooled_cov     = pooled_covered.mean()
    print(f"  Pooled coverage: {pooled_cov:.4f}")

    # Width distribution
    valid_widths = pred_width[~np.isnan(pred_width)]
    print(f"\n  Interval width percentiles:")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"    {p:3d}th: {np.percentile(valid_widths, p):.4f} eV")

    # Save
    conf_df = pd.DataFrame({
        "rxn_id":     df["rxn_id"].values,
        "y_true":     y_all,
        "y_pred":     pred_point,
        "lower":      pred_lower,
        "upper":      pred_upper,
        "width":      pred_width,
        "covered":    pooled_covered,
        "sigma_ensemble": ensemble_sigma,
    })
    conf_df.to_csv(f"{OUTPUT_DIR}/conformal_predictions.csv", index=False)
    print(f"\n  Saved to {OUTPUT_DIR}/conformal_predictions.csv")

    fold_summary = pd.DataFrame({
        "fold":       range(1, 6),
        "coverage":   fold_coverages,
        "mean_width": fold_mean_widths,
        "q_hat_norm": fold_quantiles,
    })
    fold_summary.to_csv(f"{OUTPUT_DIR}/conformal_fold_summary.csv",
                        index=False)

    return {
        "mean_coverage":   mean_cov,
        "std_coverage":    std_cov,
        "pooled_coverage": pooled_cov,
        "mean_width":      mean_width,
        "mean_q_hat":      np.mean(fold_quantiles),
        "fold_coverages":  fold_coverages,
    }


# ---------------------------------------------------------------------------
# 10.3 — Cross-reference: error vs uncertainty
# ---------------------------------------------------------------------------
def run_cross_reference(ensemble_results):
    """
    Test whether high-σ_ensemble reactions are also high-error.

    Analyses:
      1. Pearson & Spearman correlation between |error| and σ_ensemble
      2. Fraction of worst-5% errors that are also high-σ
      3. Mean error inside vs outside the high-σ flag
      4. Calibration: observed error quantiles vs σ_ensemble quantiles
    """
    print(f"\n10.3 CROSS-REFERENCE: ERROR vs UNCERTAINTY")
    print("=" * 72)

    df = ensemble_results.copy()
    ae = df["abs_error"].values
    sigma = df["sigma_ensemble"].values

    # Correlations
    r_pearson, p_pearson   = pearsonr(ae, sigma)
    r_spearman, p_spearman = spearmanr(ae, sigma)

    print(f"  Pearson  r(|error|, σ): {r_pearson:.4f}  "
          f"(p={p_pearson:.2e})")
    print(f"  Spearman ρ(|error|, σ): {r_spearman:.4f}  "
          f"(p={p_spearman:.2e})")

    # Worst 5% errors
    error_95  = np.percentile(ae, 95)
    worst_5   = ae >= error_95
    n_worst   = worst_5.sum()
    worst_high_sigma = (worst_5 & df["high_sigma"].values).sum()
    pct_worst_flagged = 100 * worst_high_sigma / n_worst if n_worst > 0 else 0

    print(f"\n  Worst 5% errors (|error| >= {error_95:.3f} eV):")
    print(f"    Count: {n_worst}")
    print(f"    Flagged by σ > {SIGMA_THRESHOLD}: "
          f"{worst_high_sigma}/{n_worst} ({pct_worst_flagged:.1f}%)")

    # Mean error: flagged vs unflagged
    flagged = df["high_sigma"].values
    mae_flagged   = ae[flagged].mean()   if flagged.any()  else np.nan
    mae_unflagged = ae[~flagged].mean()  if (~flagged).any() else np.nan

    print(f"\n  Mean |error| for σ > {SIGMA_THRESHOLD}: "
          f"{mae_flagged:.4f} eV  "
          f"(n={flagged.sum()})")
    print(f"  Mean |error| for σ ≤ {SIGMA_THRESHOLD}: "
          f"{mae_unflagged:.4f} eV  "
          f"(n={(~flagged).sum()})")
    if not np.isnan(mae_flagged) and not np.isnan(mae_unflagged):
        ratio = mae_flagged / mae_unflagged
        print(f"  Ratio: {ratio:.2f}x")

    # Calibration: bin by σ quintile, report mean |error| per bin
    print(f"\n  Calibration by σ_ensemble quintile:")
    print(f"  {'Quintile':>10} {'σ range':>18} {'Mean |error|':>14} {'n':>6}")
    print(f"  {'-'*52}")
    quintile_labels = pd.qcut(sigma, 5, labels=False, duplicates="drop")
    for q in sorted(np.unique(quintile_labels)):
        mask     = quintile_labels == q
        sig_lo   = sigma[mask].min()
        sig_hi   = sigma[mask].max()
        mean_err = ae[mask].mean()
        print(f"  {q+1:>10} [{sig_lo:.3f}, {sig_hi:.3f}]"
              f"       {mean_err:.4f}      {mask.sum():>5}")

    # Per-family analysis if available
    if "rmg_family" in df.columns:
        print(f"\n  Per-family mean σ_ensemble:")
        fam_stats = (df.groupby("rmg_family")
                     .agg(n=("sigma_ensemble", "size"),
                          mean_sigma=("sigma_ensemble", "mean"),
                          mean_error=("abs_error", "mean"))
                     .sort_values("mean_sigma", ascending=False))
        print(fam_stats.to_string())

    # Save cross-reference summary
    xref = {
        "pearson_r":          r_pearson,
        "pearson_p":          p_pearson,
        "spearman_rho":       r_spearman,
        "spearman_p":         p_spearman,
        "n_worst_5pct":       int(n_worst),
        "worst_5pct_flagged": int(worst_high_sigma),
        "pct_worst_flagged":  pct_worst_flagged,
        "mae_flagged":        mae_flagged,
        "mae_unflagged":      mae_unflagged,
        "ratio":              ratio if not np.isnan(mae_flagged) else np.nan,
    }
    pd.DataFrame([xref]).to_csv(
        f"{OUTPUT_DIR}/uncertainty_cross_reference.csv", index=False
    )
    print(f"\n  Saved to {OUTPUT_DIR}/uncertainty_cross_reference.csv")

    return xref


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model artifact
    print("Loading model artifact from Step 9...")
    artifact     = joblib.load(MODEL_PKL)
    best_name    = artifact["model_name"]
    params       = artifact["params"]
    n_estimators = artifact.get("n_estimators", 2000)
    feature_cols = artifact["feature_cols"]

    print(f"  Model: {best_name}")
    print(f"  Params: {params}")
    print(f"  n_estimators: {n_estimators}")

    if best_name != "XGBoost":
        print(f"  WARNING: Script optimised for XGBoost but got {best_name}. "
              f"Proceeding with XGBoost architecture using saved params.")

    # Load data
    print("\nLoading feature matrix...")
    df = pd.read_parquet(FEATURE_MATRIX)
    print(f"  {len(df)} reactions, {df.shape[1]} columns")

    print("\nLoading scaffold splits...")
    folds = load_splits(SPLITS_FILE, df)
    for i, (tr, te) in enumerate(folds):
        print(f"  fold_{i}: train={len(tr)}, test={len(te)}")

    # 10.1 Ensemble variance
    ensemble_results = run_ensemble_variance(
        df, folds, feature_cols, params, n_estimators
    )

    # 10.2 Conformal prediction (uses σ_ensemble from 10.1)
    conformal_results = run_conformal_prediction(
        df, folds, feature_cols, params, n_estimators,
        ensemble_sigma=ensemble_results["sigma_ensemble"].values
    )

    # 10.3 Cross-reference
    xref = run_cross_reference(ensemble_results)

    # Final summary
    print("\n" + "=" * 72)
    print("=== Step 10 Complete ===")
    print("=" * 72)
    print(f"Ensemble:")
    print(f"  Mean σ:    {ensemble_results['sigma_ensemble'].mean():.4f} eV")
    print(f"  Flagged:   {ensemble_results['high_sigma'].sum()} reactions "
          f"(σ > {SIGMA_THRESHOLD} eV)")
    print(f"  Ensemble MAE: "
          f"{mean_absolute_error(ensemble_results['y_true'], ensemble_results['y_pred_ensemble']):.4f} eV")
    print(f"Conformal:")
    print(f"  Coverage:  {conformal_results['mean_coverage']:.4f} ± "
          f"{conformal_results['std_coverage']:.4f}  "
          f"(nominal: {1-CONFORMAL_ALPHA:.2f})")
    print(f"  Width:     {conformal_results['mean_width']:.4f} eV")
    print(f"Cross-reference:")
    print(f"  ρ(|error|, σ): {xref['spearman_rho']:.4f}")
    print(f"  Worst 5% flagged: {xref['pct_worst_flagged']:.1f}%")
    print(f"  Error ratio (flagged/unflagged): {xref.get('ratio', 'N/A')}")

    print(f"\nOutputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()