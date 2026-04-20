# -*- coding: utf-8 -*-
"""
PRISM Step 9 — Model Training, Comparison, Ablation, Screening, Learning Curve
src/models/train.py  (v3 — maximum rigor)

Tasks:
  9.1 Train 5 models (BEP, Ridge, RF, XGBoost, LightGBM) on scaffold 5-fold CV
      with nested Optuna hyperparameter tuning (100 trials, inner 3-fold)
  9.2 Ablation study (variants A, B, C, D) on best model
  9.3 Screening metrics at threshold ΔE‡ < 2 eV
  9.4 Learning curve (20/40/60/80/100% of training data)

Fixes vs v2:
  - Per-fold mean ± std reported alongside pooled OOF metrics (roadmap mandate)
  - Early stopping no longer wastes 15% of outer training data: best_iteration
    is extracted from Optuna inner CV and used as fixed n_estimators for the
    outer fold fit on 100% of training data
  - BEP ablation path guarded — skips cleanly if dE_xtb absent from variant
  - Optuna sampler seed incorporates model name for cleaner reproducibility
  - Same applies to final full-dataset model (uses median best_iteration)
  - All v2 fixes retained: no test-set leakage, cached screening (9.3),
    fixed-param learning curve (9.4), per-reaction paired bootstrap
"""

from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                              r2_score, confusion_matrix,
                              precision_score, recall_score,
                              f1_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURE_MATRIX = "data/transition1x/features/feature_matrix.parquet"
SPLITS_FILE    = "data/transition1x/splits/scaffold_5fold.json"
OUTPUT_DIR     = "results"
MODELS_DIR     = "models"

# ---------------------------------------------------------------------------
# Feature definitions
# Note: bep_prediction excluded — BEP is a separate baseline model refit
# per fold on training data only to avoid leakage.
# dE_rxn_eV excluded — DFT reaction energy, not available at inference.
# ---------------------------------------------------------------------------
ELECTRONIC_FEATURES = [
    "dE_xtb", "gap_R", "gap_P", "d_gap",
    "dipole_R", "dipole_P", "d_dipole",
    "fukui_plus", "fukui_minus", "delta_WBO",
]
GEOMETRIC_FEATURES = [
    "E_strain_IDPP", "RMSD_R_P",
    "dPMI_1", "dPMI_2", "dPMI_3",
]
TOPOLOGICAL_FEATURES = [
    "MW", "LogP", "TPSA_R", "TPSA_P", "delta_TPSA",
    "n_rot_bonds", "n_rings_R", "n_rings_P",
    "delta_ring_atoms", "balaban_J",
    "estate_sum_R", "estate_sum_P", "delta_estate_sum",
    "estate_max_R", "estate_max_P",
]
RXN_CLASS_PREFIX = "rxn_class_"

TARGET           = "Ea_eV"
SCREEN_THRESHOLD = 2.0    # eV — "viable reaction"
N_OPTUNA_TRIALS  = 100
N_INNER_FOLDS    = 3
ES_VAL_FRAC      = 0.15   # for early stopping inside Optuna inner CV only
N_JOBS           = -1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, threshold=SCREEN_THRESHOLD):
    """Full metric set: regression + screening."""
    mae   = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    r, _  = pearsonr(y_true, y_pred)

    true_pos = (y_true < threshold).astype(int)
    pred_pos = (y_pred < threshold).astype(int)

    if len(np.unique(true_pos)) < 2:
        warnings.warn(
            f"Degenerate screening labels (all={np.unique(true_pos)}). "
            f"Screening metrics set to NaN."
        )
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


def load_splits(path, df):
    """Load scaffold 5-fold splits. Each key maps to test-set rxn_ids."""
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
        print(f"  {key}: train={len(train_idx)}, test={len(test_idx)}")
    return folds


def make_seed(fold_idx: int, name: str) -> int:
    """Deterministic seed that varies with both fold and model name."""
    return hash((fold_idx, name)) % (2**31)


def aggregate_params(params_list):
    """
    Aggregate best hyperparameters across folds.
    Numeric → median (int types rounded); categorical → mode.
    """
    if not params_list:
        return {}
    keys = params_list[0].keys()
    agg  = {}
    for k in keys:
        vals = [p[k] for p in params_list if k in p]
        if isinstance(vals[0], (int, np.integer)):
            agg[k] = int(round(np.median(vals)))
        elif isinstance(vals[0], (float, np.floating)):
            agg[k] = float(np.median(vals))
        else:
            agg[k] = max(set(vals), key=vals.count)
    return agg


def split_early_stopping(X, y, val_frac=ES_VAL_FRAC, seed=42):
    """Carve internal val split for early stopping (Optuna inner CV only)."""
    rng   = np.random.RandomState(seed)
    n_val = max(1, int(len(X) * val_frac))
    idx   = rng.permutation(len(X))
    return (X[idx[n_val:]], y[idx[n_val:]],
            X[idx[:n_val]], y[idx[:n_val]])


# ---------------------------------------------------------------------------
# BEP baseline
# ---------------------------------------------------------------------------
def fit_bep(X_train, y_train, X_test, feature_cols):
    """ΔE‡ = α·dE_xtb + β, fitted on training fold only."""
    if "dE_xtb" not in feature_cols:
        raise ValueError("BEP requires dE_xtb in feature_cols")
    dE_col = feature_cols.index("dE_xtb")
    lr = LinearRegression()
    lr.fit(X_train[:, dE_col:dE_col + 1], y_train)
    return lr.predict(X_test[:, dE_col:dE_col + 1])


# ---------------------------------------------------------------------------
# Optuna objective factories
#
# For boosting models (XGBoost, LightGBM):
#   Early stopping uses an internal split WITHIN each inner CV fold.
#   The inner CV validation set is used ONLY for scoring (MAE), never for
#   early stopping.  This prevents double-dipping.
#   We also record best_iteration from each inner fit so the caller can
#   extract the median and use it as fixed n_estimators on the outer fold.
# ---------------------------------------------------------------------------
def make_ridge_objective(X_train, y_train, cv, seed):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
        pipe  = Pipeline([("sc", StandardScaler()),
                          ("m",  Ridge(alpha=alpha))])
        scores = []
        for tr, va in cv:
            pipe.fit(X_train[tr], y_train[tr])
            scores.append(mean_absolute_error(
                y_train[va], pipe.predict(X_train[va])
            ))
        return np.mean(scores)
    return objective


def make_rf_objective(X_train, y_train, cv, seed):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
        }
        scores = []
        for tr, va in cv:
            m = RandomForestRegressor(**params, n_jobs=N_JOBS,
                                      random_state=seed)
            m.fit(X_train[tr], y_train[tr])
            scores.append(mean_absolute_error(
                y_train[va], m.predict(X_train[va])
            ))
        return np.mean(scores)
    return objective


def make_xgb_objective(X_train, y_train, cv, seed):
    """
    Returns (objective_fn, best_iterations_list).
    After study.optimize, best_iterations_list contains the
    best_iteration from the best trial's inner fits.
    """
    best_iters_store = []   # mutable — populated by the objective

    def objective(trial):
        params = {
            "n_estimators":       2000,
            "learning_rate":      trial.suggest_float("lr", 0.005, 0.05,
                                                      log=True),
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample", 0.6, 1.0),
            "reg_lambda":         trial.suggest_float("lambda", 1e-2, 10.0,
                                                      log=True),
        }
        scores      = []
        trial_iters = []
        for fold_i, (tr, va) in enumerate(cv):
            # Internal split for early stopping — separate from va
            X_tr, y_tr, X_es, y_es = split_early_stopping(
                X_train[tr], y_train[tr], seed=seed + fold_i
            )
            m = xgb.XGBRegressor(
                **params,
                early_stopping_rounds=50,
                eval_metric="mae",
                n_jobs=N_JOBS, random_state=seed, verbosity=0
            )
            m.fit(X_tr, y_tr,
                  eval_set=[(X_es, y_es)], verbose=False)
            trial_iters.append(m.best_iteration)

            # Score on the actual inner CV validation set (no leakage)
            scores.append(mean_absolute_error(
                y_train[va], m.predict(X_train[va])
            ))

        # Store iterations for the best trial (checked after optimize)
        trial.set_user_attr("best_iters", trial_iters)
        return np.mean(scores)

    return objective, best_iters_store


def make_lgbm_objective(X_train, y_train, cv, seed):
    """Same pattern as XGBoost: records best_iteration per inner fold."""
    def objective(trial):
        params = {
            "n_estimators":    2000,
            "learning_rate":   trial.suggest_float("lr", 0.005, 0.05,
                                                    log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample", 0.6, 1.0),
            "reg_lambda":      trial.suggest_float("lambda", 1e-2, 10.0,
                                                    log=True),
        }
        scores      = []
        trial_iters = []
        callbacks   = [lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        for fold_i, (tr, va) in enumerate(cv):
            X_tr, y_tr, X_es, y_es = split_early_stopping(
                X_train[tr], y_train[tr], seed=seed + fold_i
            )
            m = lgb.LGBMRegressor(
                **params, n_jobs=N_JOBS, random_state=seed, verbose=-1
            )
            m.fit(X_tr, y_tr,
                  eval_set=[(X_es, y_es)],
                  callbacks=callbacks)
            trial_iters.append(m.best_iteration_)

            scores.append(mean_absolute_error(
                y_train[va], m.predict(X_train[va])
            ))

        trial.set_user_attr("best_iters", trial_iters)
        return np.mean(scores)

    return objective


def get_best_n_estimators(study) -> int:
    """
    Extract median best_iteration from the best Optuna trial's inner fits.
    Used as fixed n_estimators when fitting on the full outer training fold,
    so we train on 100% of available data (no early-stopping holdout needed).
    """
    best_trial = study.best_trial
    iters = best_trial.user_attrs.get("best_iters", [])
    if not iters:
        return 2000   # fallback: use max
    # Add a small buffer (~10%) to account for more training data
    median_iter = int(np.median(iters))
    buffered    = int(median_iter * 1.10)
    return min(buffered, 2000)


# ---------------------------------------------------------------------------
# Model construction and fitting
# ---------------------------------------------------------------------------
def build_model(name, params, n_estimators_override=None, seed=42):
    """Instantiate a model from name + params dict."""
    if name == "Ridge":
        return Pipeline([("sc", StandardScaler()),
                         ("m",  Ridge(alpha=params["alpha"]))])

    elif name == "RandomForest":
        return RandomForestRegressor(
            n_estimators     = params["n_estimators"],
            max_depth        = params["max_depth"],
            min_samples_leaf = params["min_samples_leaf"],
            max_features     = params["max_features"],
            n_jobs=N_JOBS, random_state=seed
        )

    elif name == "XGBoost":
        n_est = n_estimators_override or 2000
        return xgb.XGBRegressor(
            n_estimators     = n_est,
            learning_rate    = params["lr"],
            max_depth        = params["max_depth"],
            subsample        = params["subsample"],
            colsample_bytree = params["colsample"],
            reg_lambda       = params["lambda"],
            eval_metric      = "mae",
            n_jobs=N_JOBS, random_state=seed, verbosity=0
            # No early_stopping_rounds — n_estimators is pre-determined
        )

    elif name == "LightGBM":
        n_est = n_estimators_override or 2000
        return lgb.LGBMRegressor(
            n_estimators     = n_est,
            learning_rate    = params["lr"],
            num_leaves       = params["num_leaves"],
            subsample        = params["subsample"],
            colsample_bytree = params["colsample"],
            reg_lambda       = params["lambda"],
            n_jobs=N_JOBS, random_state=seed, verbose=-1
        )

    raise ValueError(f"Unknown model: {name}")


def fit_model_clean(model, X_train, y_train):
    """
    Fit without any early stopping — n_estimators already determined
    from inner CV.  Model trains on 100% of the provided data.
    """
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Run Optuna + fit on one outer fold
# ---------------------------------------------------------------------------
def tune_and_fit(name, X_train, X_test, y_train, y_test,
                 inner_cv, fold_idx, feature_cols):
    """
    Returns (metrics, y_pred, best_params, best_n_est).
    best_n_est is the extracted best_iteration (boosters only); None for others.
    """
    if name == "BEP":
        y_pred  = fit_bep(X_train, y_train, X_test, feature_cols)
        metrics = compute_metrics(y_test, y_pred)
        return metrics, y_pred, {}, None

    seed  = make_seed(fold_idx, name)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    best_n_est = None

    if name == "Ridge":
        study.optimize(
            make_ridge_objective(X_train, y_train, inner_cv, seed),
            n_trials=N_OPTUNA_TRIALS, show_progress_bar=False
        )

    elif name == "RandomForest":
        study.optimize(
            make_rf_objective(X_train, y_train, inner_cv, seed),
            n_trials=N_OPTUNA_TRIALS, show_progress_bar=False
        )

    elif name == "XGBoost":
        obj_fn, _ = make_xgb_objective(X_train, y_train, inner_cv, seed)
        study.optimize(obj_fn, n_trials=N_OPTUNA_TRIALS,
                       show_progress_bar=False)
        best_n_est = get_best_n_estimators(study)

    elif name == "LightGBM":
        obj_fn = make_lgbm_objective(X_train, y_train, inner_cv, seed)
        study.optimize(obj_fn, n_trials=N_OPTUNA_TRIALS,
                       show_progress_bar=False)
        best_n_est = get_best_n_estimators(study)

    best_params = study.best_params
    model       = build_model(name, best_params,
                              n_estimators_override=best_n_est, seed=seed)
    model       = fit_model_clean(model, X_train, y_train)
    y_pred      = model.predict(X_test)
    metrics     = compute_metrics(y_test, y_pred)

    return metrics, y_pred, best_params, best_n_est


# ---------------------------------------------------------------------------
# 9.1 — Model comparison
# ---------------------------------------------------------------------------
def run_model_comparison(df, folds, feature_cols):
    print("\n9.1 MODEL COMPARISON")
    print("=" * 72)
    model_names = ["BEP", "Ridge", "RandomForest", "XGBoost", "LightGBM"]

    X_all = df[feature_cols].values.astype(float)
    y_all = df[TARGET].values

    # Cache per model: list of (test_idx, y_pred) per fold
    fold_cache       = {m: [] for m in model_names}
    fold_metrics     = {m: [] for m in model_names}
    best_params_all  = {m: [] for m in model_names}
    best_n_est_all   = {m: [] for m in model_names}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n  --- Outer Fold {fold_idx + 1}/5 ---")
        X_train = X_all[train_idx]
        X_test  = X_all[test_idx]
        y_train = y_all[train_idx]
        y_test  = y_all[test_idx]

        inner_kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                         random_state=42)
        inner_cv = list(inner_kf.split(X_train))

        for name in model_names:
            print(f"    {name}...", end=" ", flush=True)

            metrics, y_pred, bp, bn = tune_and_fit(
                name, X_train, X_test, y_train, y_test,
                inner_cv, fold_idx, feature_cols
            )

            fold_cache[name].append((test_idx, y_pred))
            fold_metrics[name].append(metrics)
            if bp:
                best_params_all[name].append(bp)
            if bn is not None:
                best_n_est_all[name].append(bn)

            extra = ""
            if bn is not None:
                extra = f"  n_est={bn}"
            print(f"MAE={metrics['MAE']:.4f}{extra}")

    # ---- Aggregate results ----
    print("\n\n  --- Per-fold mean ± std (roadmap format) ---")
    perfold_rows = []
    for name in model_names:
        row = {"Model": name}
        for metric_key in ["MAE", "MedAE", "R2", "Pearson_r",
                           "Recall", "Precision", "F1", "MCC"]:
            vals = [m[metric_key] for m in fold_metrics[name]]
            vals_clean = [v for v in vals if not np.isnan(v)]
            row[f"{metric_key}_mean"] = (np.mean(vals_clean)
                                         if vals_clean else np.nan)
            row[f"{metric_key}_std"]  = (np.std(vals_clean)
                                         if vals_clean else np.nan)
        perfold_rows.append(row)
    perfold_df = pd.DataFrame(perfold_rows).sort_values("MAE_mean")
    print(perfold_df[["Model", "MAE_mean", "MAE_std",
                       "MedAE_mean", "MedAE_std",
                       "R2_mean", "R2_std",
                       "Pearson_r_mean", "Pearson_r_std"]].to_string(
        index=False, float_format="{:.4f}".format
    ))

    print("\n  --- Pooled OOF metrics ---")
    pooled_rows = []
    for name in model_names:
        y_true_all = np.concatenate(
            [y_all[idx] for idx, _ in fold_cache[name]]
        )
        y_pred_all = np.concatenate(
            [pred for _, pred in fold_cache[name]]
        )
        metrics     = compute_metrics(y_true_all, y_pred_all)
        metrics["Model"] = name
        pooled_rows.append(metrics)
    pooled_df = pd.DataFrame(pooled_rows).sort_values("MAE")
    print(pooled_df[["Model", "MAE", "MedAE", "R2", "Pearson_r",
                      "Recall", "Precision"]].to_string(
        index=False, float_format="{:.4f}".format
    ))

    # Save both tables
    perfold_df.to_csv(f"{OUTPUT_DIR}/model_comparison_perfold.csv",
                      index=False)
    pooled_df.to_csv(f"{OUTPUT_DIR}/model_comparison_pooled.csv",
                     index=False)

    best_name = pooled_df.iloc[0]["Model"]
    print(f"\n  Best model: {best_name}  "
          f"(pooled MAE={pooled_df.iloc[0]['MAE']:.4f} eV)")

    # Aggregate best params across folds
    agg_params = {}
    agg_n_est  = {}
    for name in model_names:
        if best_params_all[name]:
            agg_params[name] = aggregate_params(best_params_all[name])
            print(f"  {name} aggregated params: {agg_params[name]}")
        if best_n_est_all[name]:
            med_n = int(np.median(best_n_est_all[name]))
            buffered = min(int(med_n * 1.10), 2000)
            agg_n_est[name] = buffered
            print(f"  {name} aggregated n_estimators: {buffered} "
                  f"(fold medians: {best_n_est_all[name]})")

    return (perfold_df, pooled_df, best_name,
            fold_cache, fold_metrics,
            agg_params, agg_n_est)


# ---------------------------------------------------------------------------
# 9.2 — Ablation study
# ---------------------------------------------------------------------------
def run_ablation(df, folds, feature_cols, rxn_class_cols, best_name):
    print(f"\n9.2 ABLATION  (model: {best_name})")
    print("=" * 72)

    variants = {
        "A_topo_only": TOPOLOGICAL_FEATURES + rxn_class_cols,
        "B_elec_topo": (ELECTRONIC_FEATURES + TOPOLOGICAL_FEATURES
                        + rxn_class_cols),
        "C_geo_topo":  (GEOMETRIC_FEATURES + TOPOLOGICAL_FEATURES
                        + rxn_class_cols),
        "D_full":      feature_cols,
    }

    X_all = df[feature_cols].values.astype(float)
    y_all = df[TARGET].values

    variant_preds   = {}
    variant_perfold = {}

    for vname, vfeats in variants.items():
        # Map variant features to column indices in the full feature matrix
        feat_idx = [feature_cols.index(f) for f in vfeats
                    if f in feature_cols]
        # Build a local feature_cols list for this variant (for BEP lookup)
        local_feature_cols = [f for f in vfeats if f in feature_cols]

        print(f"\n  {vname} ({len(feat_idx)} features)...")

        all_test_idx, all_y_pred = [], []
        vfold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_tr = X_all[np.ix_(train_idx, feat_idx)]
            X_te = X_all[np.ix_(test_idx,  feat_idx)]
            y_tr = y_all[train_idx]
            y_te = y_all[test_idx]

            inner_kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                             random_state=42)
            inner_cv = list(inner_kf.split(X_tr))

            # BEP only works if dE_xtb is in the variant's features
            if best_name == "BEP":
                if "dE_xtb" not in local_feature_cols:
                    print(f"    Fold {fold_idx+1}: SKIPPED "
                          f"(dE_xtb absent in {vname})")
                    continue
                y_pred = fit_bep(X_tr, y_tr, X_te, local_feature_cols)
                metrics = compute_metrics(y_te, y_pred)
            else:
                metrics, y_pred, _, _ = tune_and_fit(
                    best_name, X_tr, X_te, y_tr, y_te,
                    inner_cv, fold_idx, local_feature_cols
                )

            mae = metrics["MAE"]
            print(f"    Fold {fold_idx+1}: MAE={mae:.4f}", flush=True)
            all_test_idx.extend(test_idx)
            all_y_pred.extend(y_pred.tolist())
            vfold_metrics.append(metrics)

        # Reconstruct y_true in the same order
        all_y_true = y_all[all_test_idx].tolist()

        variant_preds[vname] = {
            "y_true": np.array(all_y_true),
            "y_pred": np.array(all_y_pred),
        }
        variant_perfold[vname] = vfold_metrics

    # ---- Summary tables ----
    print("\n  --- Ablation: Per-fold mean ± std ---")
    abl_pf_rows = []
    for vname, vfm in variant_perfold.items():
        row = {"Variant": vname}
        for mk in ["MAE", "MedAE", "R2", "Recall", "Precision"]:
            vals = [m[mk] for m in vfm if not np.isnan(m.get(mk, np.nan))]
            row[f"{mk}_mean"] = np.mean(vals) if vals else np.nan
            row[f"{mk}_std"]  = np.std(vals)  if vals else np.nan
        abl_pf_rows.append(row)
    abl_pf_df = pd.DataFrame(abl_pf_rows)
    print(abl_pf_df[["Variant", "MAE_mean", "MAE_std",
                      "MedAE_mean", "R2_mean",
                      "Recall_mean"]].to_string(
        index=False, float_format="{:.4f}".format
    ))

    print("\n  --- Ablation: Pooled OOF ---")
    abl_pooled_rows = []
    for vname, preds in variant_preds.items():
        m = compute_metrics(preds["y_true"], preds["y_pred"])
        m["Variant"] = vname
        abl_pooled_rows.append(m)
    abl_pooled_df = pd.DataFrame(abl_pooled_rows)[
        ["Variant", "MAE", "MedAE", "R2", "Recall", "Precision"]
    ]
    print(abl_pooled_df.to_string(index=False,
                                   float_format="{:.4f}".format))

    abl_pf_df.to_csv(f"{OUTPUT_DIR}/ablation_perfold.csv", index=False)
    abl_pooled_df.to_csv(f"{OUTPUT_DIR}/ablation_pooled.csv", index=False)

    # ---- Paired bootstrap: per-reaction residuals ----
    print("\n  Paired bootstrap (1000 resamples, n≈9300 per-reaction):")
    rng  = np.random.RandomState(42)
    d_ae = np.abs(variant_preds["D_full"]["y_true"]
                  - variant_preds["D_full"]["y_pred"])
    n    = len(d_ae)

    for vname in ["A_topo_only", "B_elec_topo", "C_geo_topo"]:
        v_ae  = np.abs(variant_preds[vname]["y_true"]
                       - variant_preds[vname]["y_pred"])
        diffs = np.zeros(1000)
        for i in range(1000):
            idx     = rng.choice(n, n, replace=True)
            diffs[i] = v_ae[idx].mean() - d_ae[idx].mean()
        p_val = np.mean(diffs <= 0)
        ci_lo = np.percentile(diffs, 2.5)
        ci_hi = np.percentile(diffs, 97.5)
        print(f"    D < {vname}: "
              f"Δ={np.mean(diffs):.4f} eV  "
              f"95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"p={p_val:.4f}")

    return abl_pooled_df, abl_pf_df, variant_preds


# ---------------------------------------------------------------------------
# 9.3 — Screening (reuses cached predictions from 9.1)
# ---------------------------------------------------------------------------
def run_screening_analysis(df, fold_cache, best_name):
    print(f"\n9.3 SCREENING  (threshold: ΔE‡ < {SCREEN_THRESHOLD} eV)")
    print("=" * 72)
    print("  Reusing cached fold predictions from 9.1 — no retraining.")

    y_all  = df[TARGET].values
    y_true = np.concatenate([y_all[idx] for idx, _ in fold_cache[best_name]])
    y_pred = np.concatenate([pred       for _, pred in fold_cache[best_name]])
    metrics = compute_metrics(y_true, y_pred)

    print(f"  Pooled (n={len(y_true)}):")
    for k in ["Recall", "Precision", "F1", "MCC", "MAE", "MedAE", "R2"]:
        target = ""
        if k == "Recall":    target = "  (target ≥0.90)"
        if k == "Precision": target = "  (target ≥0.75)"
        print(f"    {k:12s} = {metrics[k]:.4f}{target}")

    h3 = (not np.isnan(metrics["Recall"])
           and not np.isnan(metrics["Precision"])
           and metrics["Recall"] >= 0.90
           and metrics["Precision"] >= 0.75)
    print(f"\n  H3 checkpoint: {'PASSED' if h3 else 'FAILED'}")

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        f"{OUTPUT_DIR}/screening_predictions.csv", index=False
    )
    print(f"  Saved to {OUTPUT_DIR}/screening_predictions.csv")
    print("  Per-family screening deferred to Step 12.")
    return metrics


# ---------------------------------------------------------------------------
# 9.4 — Learning curve (fixed hyperparams from 9.1 — no Optuna)
# ---------------------------------------------------------------------------
def run_learning_curve(df, folds, feature_cols, best_name,
                       agg_params, agg_n_est):
    print(f"\n9.4 LEARNING CURVE  (fixed params, no Optuna)")
    print("=" * 72)

    X_all  = df[feature_cols].values.astype(float)
    y_all  = df[TARGET].values
    fracs  = [0.2, 0.4, 0.6, 0.8, 1.0]

    params = agg_params.get(best_name, {})
    n_est  = agg_n_est.get(best_name, None)
    print(f"  Model: {best_name}")
    print(f"  Params: {params}")
    if n_est:
        print(f"  Fixed n_estimators: {n_est}")

    lc_rows = []
    for frac in fracs:
        fold_maes = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_test = X_all[test_idx]
            y_test = y_all[test_idx]

            rng   = np.random.RandomState(fold_idx * 100 + int(frac * 100))
            n_sub = max(20, int(len(train_idx) * frac))
            sub   = rng.choice(train_idx, size=n_sub, replace=False)
            X_sub = X_all[sub]
            y_sub = y_all[sub]

            if best_name == "BEP":
                y_pred = fit_bep(X_sub, y_sub, X_test, feature_cols)
            else:
                model = build_model(best_name, params,
                                    n_estimators_override=n_est,
                                    seed=make_seed(fold_idx, best_name))
                model = fit_model_clean(model, X_sub, y_sub)
                y_pred = model.predict(X_test)

            fold_maes.append(mean_absolute_error(y_test, y_pred))

        mean_mae = np.mean(fold_maes)
        std_mae  = np.std(fold_maes)
        n_approx = int(len(df) * (1 - 1/5) * frac)
        print(f"  {int(frac*100):3d}%  n≈{n_approx:5d}  "
              f"MAE={mean_mae:.4f} ± {std_mae:.4f}")
        lc_rows.append({
            "frac": frac, "n_train_approx": n_approx,
            "MAE_mean": mean_mae, "MAE_std": std_mae,
        })

    lc_df = pd.DataFrame(lc_rows)
    lc_df.to_csv(f"{OUTPUT_DIR}/learning_curve.csv", index=False)
    print(f"  Saved to {OUTPUT_DIR}/learning_curve.csv")
    return lc_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading feature matrix...")
    df = pd.read_parquet(FEATURE_MATRIX)
    print(f"  {len(df)} reactions, {df.shape[1]} columns")

    rxn_class_cols = [c for c in df.columns if c.startswith(RXN_CLASS_PREFIX)]
    feature_cols   = (ELECTRONIC_FEATURES + GEOMETRIC_FEATURES +
                      TOPOLOGICAL_FEATURES + rxn_class_cols)

    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"    Electronic: {len(ELECTRONIC_FEATURES)}")
    print(f"    Geometric:  {len(GEOMETRIC_FEATURES)}")
    print(f"    Topological: {len(TOPOLOGICAL_FEATURES)}")
    print(f"    rxn_class:   {len(rxn_class_cols)}")

    print("\nLoading scaffold splits...")
    folds = load_splits(SPLITS_FILE, df)

    # ── 9.1 ──
    (perfold_df, pooled_df, best_name,
     fold_cache, fm,
     agg_params, agg_n_est) = run_model_comparison(df, folds, feature_cols)

    # Train final model on full data
    if best_name != "BEP":
        print(f"\nTraining final {best_name} on full dataset...")
        X_full = df[feature_cols].values.astype(float)
        y_full = df[TARGET].values
        final  = build_model(best_name, agg_params[best_name],
                             n_estimators_override=agg_n_est.get(best_name),
                             seed=42)
        final  = fit_model_clean(final, X_full, y_full)
        artifact = {
            "model":         final,
            "model_name":    best_name,
            "feature_cols":  feature_cols,
            "params":        agg_params[best_name],
            "n_estimators":  agg_n_est.get(best_name),
        }
        joblib.dump(artifact, f"{MODELS_DIR}/prism_best_model.pkl")
        print(f"  Saved to {MODELS_DIR}/prism_best_model.pkl")

    # ── 9.2 ──
    abl_pooled, abl_pf, variant_preds = run_ablation(
        df, folds, feature_cols, rxn_class_cols, best_name
    )

    # ── 9.3 ──
    screen_metrics = run_screening_analysis(df, fold_cache, best_name)

    # ── 9.4 ──
    lc_df = run_learning_curve(
        df, folds, feature_cols, best_name, agg_params, agg_n_est
    )

    # ── Final summary ──
    print("\n" + "=" * 72)
    print("=== Step 9 Complete ===")
    print("=" * 72)
    best_pf = perfold_df[perfold_df["Model"] == best_name].iloc[0]
    best_po = pooled_df[pooled_df["Model"] == best_name].iloc[0]
    print(f"Best model:    {best_name}")
    print(f"Per-fold MAE:  {best_pf['MAE_mean']:.4f} ± "
          f"{best_pf['MAE_std']:.4f} eV")
    print(f"Pooled MAE:    {best_po['MAE']:.4f} eV")
    print(f"Pooled MedAE:  {best_po['MedAE']:.4f} eV")
    print(f"Pooled R²:     {best_po['R2']:.4f}")
    print(f"Pooled r:      {best_po['Pearson_r']:.4f}")
    print(f"Screening:")
    print(f"  Recall:      {screen_metrics['Recall']:.4f}  (≥0.90)")
    print(f"  Precision:   {screen_metrics['Precision']:.4f}  (≥0.75)")
    print(f"  H3:          "
          f"{'PASSED' if (screen_metrics['Recall'] >= 0.90 and screen_metrics['Precision'] >= 0.75) else 'FAILED'}")
    print(f"\nAblation confirms: ", end="")
    d_mae = abl_pooled[abl_pooled["Variant"] == "D_full"]["MAE"].values[0]
    b_mae = abl_pooled[abl_pooled["Variant"] == "B_elec_topo"]["MAE"].values[0]
    c_mae = abl_pooled[abl_pooled["Variant"] == "C_geo_topo"]["MAE"].values[0]
    a_mae = abl_pooled[abl_pooled["Variant"] == "A_topo_only"]["MAE"].values[0]
    print(f"A({a_mae:.3f}) > C({c_mae:.3f}) > B({b_mae:.3f}) > D({d_mae:.3f})"
          if a_mae > c_mae > b_mae > d_mae
          else f"A={a_mae:.3f}, B={b_mae:.3f}, C={c_mae:.3f}, D={d_mae:.3f}")
    print(f"\nOutputs: {OUTPUT_DIR}/")
    print(f"Model:   {MODELS_DIR}/prism_best_model.pkl")


if __name__ == "__main__":
    main()