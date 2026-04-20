# -*- coding: utf-8 -*-
"""
PRISM Step 9 — CONTINUATION (9.2, 9.3, 9.4)
src/models/train_continue.py

9.1 already completed and outputs are on disk:
  - results/model_comparison_perfold.csv
  - results/model_comparison_pooled.csv
  - models/prism_best_model.pkl

This script:
  1. Loads 9.1 results from disk
  2. Regenerates fold-level predictions (5 fast fits, no Optuna) for screening
  3. Runs full ablation with per-variant Optuna tuning (rigorous)
  4. Runs screening from regenerated fold cache
  5. Runs learning curve with fixed params
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
MODEL_PKL      = f"{MODELS_DIR}/prism_best_model.pkl"

# ---------------------------------------------------------------------------
# Feature definitions
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
SCREEN_THRESHOLD = 2.0
N_OPTUNA_TRIALS  = 100
N_INNER_FOLDS    = 3
ES_VAL_FRAC      = 0.15
N_JOBS           = -1


# ---------------------------------------------------------------------------
# Utilities (same as train.py v3)
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, threshold=SCREEN_THRESHOLD):
    mae   = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    r, _  = pearsonr(y_true, y_pred)

    true_pos = (y_true < threshold).astype(int)
    pred_pos = (y_pred < threshold).astype(int)

    if len(np.unique(true_pos)) < 2:
        warnings.warn("Degenerate screening labels. Metrics set to NaN.")
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
    return hash((fold_idx, name)) % (2**31)


def split_early_stopping(X, y, val_frac=ES_VAL_FRAC, seed=42):
    rng   = np.random.RandomState(seed)
    n_val = max(1, int(len(X) * val_frac))
    idx   = rng.permutation(len(X))
    return (X[idx[n_val:]], y[idx[n_val:]],
            X[idx[:n_val]], y[idx[:n_val]])


def aggregate_params(params_list):
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


def fit_bep(X_train, y_train, X_test, feature_cols):
    if "dE_xtb" not in feature_cols:
        raise ValueError("BEP requires dE_xtb in feature_cols")
    dE_col = feature_cols.index("dE_xtb")
    lr = LinearRegression()
    lr.fit(X_train[:, dE_col:dE_col + 1], y_train)
    return lr.predict(X_test[:, dE_col:dE_col + 1])


# ---------------------------------------------------------------------------
# Optuna objectives (identical to train.py v3)
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
        for fold_i, (tr, va) in enumerate(cv):
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
            trial.set_user_attr("best_iters",
                                trial.user_attrs.get("best_iters", [])
                                + [m.best_iteration])
            scores.append(mean_absolute_error(
                y_train[va], m.predict(X_train[va])
            ))
        return np.mean(scores)
    return objective


def make_lgbm_objective(X_train, y_train, cv, seed):
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
                  eval_set=[(X_es, y_es)], callbacks=callbacks)
            trial.set_user_attr("best_iters",
                                trial.user_attrs.get("best_iters", [])
                                + [m.best_iteration_])
            scores.append(mean_absolute_error(
                y_train[va], m.predict(X_train[va])
            ))
        return np.mean(scores)
    return objective


def get_best_n_estimators(study) -> int:
    best_trial = study.best_trial
    iters = best_trial.user_attrs.get("best_iters", [])
    if not iters:
        return 2000
    median_iter = int(np.median(iters))
    return min(int(median_iter * 1.10), 2000)


# ---------------------------------------------------------------------------
# Model construction (identical to train.py v3)
# ---------------------------------------------------------------------------
def build_model(name, params, n_estimators_override=None, seed=42):
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
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Tune + fit one fold (for ablation)
# ---------------------------------------------------------------------------
def tune_and_fit(name, X_train, X_test, y_train, y_test,
                 inner_cv, fold_idx, feature_cols):
    if name == "BEP":
        if "dE_xtb" not in feature_cols:
            return None, None, {}, None
        y_pred  = fit_bep(X_train, y_train, X_test, feature_cols)
        metrics = compute_metrics(y_test, y_pred)
        return metrics, y_pred, {}, None

    seed  = make_seed(fold_idx, name)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    best_n_est = None
    objectives = {
        "Ridge":        make_ridge_objective,
        "RandomForest": make_rf_objective,
        "XGBoost":      make_xgb_objective,
        "LightGBM":     make_lgbm_objective,
    }
    study.optimize(
        objectives[name](X_train, y_train, inner_cv, seed),
        n_trials=N_OPTUNA_TRIALS, show_progress_bar=False
    )

    if name in ("XGBoost", "LightGBM"):
        best_n_est = get_best_n_estimators(study)

    best_params = study.best_params
    model       = build_model(name, best_params,
                              n_estimators_override=best_n_est, seed=seed)
    model       = fit_model_clean(model, X_train, y_train)
    y_pred      = model.predict(X_test)
    metrics     = compute_metrics(y_test, y_pred)

    return metrics, y_pred, best_params, best_n_est


# ---------------------------------------------------------------------------
# Regenerate fold cache (quick — no Optuna, uses saved params)
# ---------------------------------------------------------------------------
def regenerate_fold_cache(df, folds, feature_cols, best_name, params,
                          n_est_override):
    """
    Retrain the best model on each fold using the aggregated params
    from 9.1 (already on disk in the model pkl).  No Optuna.
    Returns fold_cache for screening in 9.3.
    """
    print("\n  Regenerating fold-level predictions for screening...")
    X_all = df[feature_cols].values.astype(float)
    y_all = df[TARGET].values

    fold_cache = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train = X_all[train_idx]
        X_test  = X_all[test_idx]
        y_train = y_all[train_idx]
        y_test  = y_all[test_idx]

        if best_name == "BEP":
            y_pred = fit_bep(X_train, y_train, X_test, feature_cols)
        else:
            model = build_model(best_name, params,
                                n_estimators_override=n_est_override,
                                seed=make_seed(fold_idx, best_name))
            model = fit_model_clean(model, X_train, y_train)
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        print(f"    Fold {fold_idx+1}: MAE={mae:.4f}")
        fold_cache.append((test_idx, y_pred))

    return fold_cache


# ---------------------------------------------------------------------------
# 9.2 — Ablation (full Optuna per variant — rigorous)
# ---------------------------------------------------------------------------
def run_ablation(df, folds, feature_cols, rxn_class_cols, best_name):
    print(f"\n9.2 ABLATION  (model: {best_name}, full Optuna per variant)")
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
        feat_idx = [feature_cols.index(f) for f in vfeats
                    if f in feature_cols]
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

            if best_name == "BEP" and "dE_xtb" not in local_feature_cols:
                print(f"    Fold {fold_idx+1}: SKIPPED (dE_xtb absent)")
                continue

            metrics, y_pred, bp, bn = tune_and_fit(
                best_name, X_tr, X_te, y_tr, y_te,
                inner_cv, fold_idx, local_feature_cols
            )

            if metrics is None:
                continue

            mae = metrics["MAE"]
            extra = f"  n_est={bn}" if bn else ""
            print(f"    Fold {fold_idx+1}: MAE={mae:.4f}{extra}", flush=True)
            all_test_idx.extend(test_idx)
            all_y_pred.extend(y_pred.tolist())
            vfold_metrics.append(metrics)

        all_y_true = y_all[all_test_idx].tolist()
        variant_preds[vname] = {
            "y_true": np.array(all_y_true),
            "y_pred": np.array(all_y_pred),
        }
        variant_perfold[vname] = vfold_metrics

    # ---- Summary tables ----
    print("\n  --- Ablation: Per-fold mean +/- std ---")
    abl_pf_rows = []
    for vname, vfm in variant_perfold.items():
        row = {"Variant": vname}
        for mk in ["MAE", "MedAE", "R2", "Recall", "Precision"]:
            vals = [m[mk] for m in vfm
                    if not np.isnan(m.get(mk, np.nan))]
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

    # ---- Paired bootstrap ----
    print("\n  Paired bootstrap (1000 resamples, per-reaction residuals):")
    rng  = np.random.RandomState(42)
    d_ae = np.abs(variant_preds["D_full"]["y_true"]
                  - variant_preds["D_full"]["y_pred"])
    n    = len(d_ae)

    for vname in ["A_topo_only", "B_elec_topo", "C_geo_topo"]:
        v_ae  = np.abs(variant_preds[vname]["y_true"]
                       - variant_preds[vname]["y_pred"])
        diffs = np.zeros(1000)
        for i in range(1000):
            idx      = rng.choice(n, n, replace=True)
            diffs[i] = v_ae[idx].mean() - d_ae[idx].mean()
        p_val = np.mean(diffs <= 0)
        ci_lo = np.percentile(diffs, 2.5)
        ci_hi = np.percentile(diffs, 97.5)
        print(f"    D < {vname}: "
              f"Delta={np.mean(diffs):.4f} eV  "
              f"95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"p={p_val:.4f}")

    return abl_pooled_df, abl_pf_df, variant_preds


# ---------------------------------------------------------------------------
# 9.3 — Screening (reuses regenerated fold cache)
# ---------------------------------------------------------------------------
def run_screening_analysis(df, fold_cache, best_name):
    print(f"\n9.3 SCREENING  (threshold < {SCREEN_THRESHOLD} eV)")
    print("=" * 72)

    y_all  = df[TARGET].values
    y_true = np.concatenate([y_all[idx] for idx, _ in fold_cache])
    y_pred = np.concatenate([pred       for _, pred in fold_cache])
    metrics = compute_metrics(y_true, y_pred)

    print(f"  Pooled (n={len(y_true)}):")
    for k in ["Recall", "Precision", "F1", "MCC", "MAE", "MedAE", "R2"]:
        target = ""
        if k == "Recall":    target = "  (target >= 0.90)"
        if k == "Precision": target = "  (target >= 0.75)"
        print(f"    {k:12s} = {metrics[k]:.4f}{target}")

    h3 = (not np.isnan(metrics.get("Recall", np.nan))
           and not np.isnan(metrics.get("Precision", np.nan))
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
# 9.4 — Learning curve (fixed params, no Optuna)
# ---------------------------------------------------------------------------
def run_learning_curve(df, folds, feature_cols, best_name, params,
                       n_est_override):
    print(f"\n9.4 LEARNING CURVE  (fixed params)")
    print("=" * 72)

    X_all  = df[feature_cols].values.astype(float)
    y_all  = df[TARGET].values
    fracs  = [0.2, 0.4, 0.6, 0.8, 1.0]

    print(f"  Model: {best_name}")
    print(f"  Params: {params}")
    if n_est_override:
        print(f"  Fixed n_estimators: {n_est_override}")

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
                                    n_estimators_override=n_est_override,
                                    seed=make_seed(fold_idx, best_name))
                model = fit_model_clean(model, X_sub, y_sub)
                y_pred = model.predict(X_test)

            fold_maes.append(mean_absolute_error(y_test, y_pred))

        mean_mae = np.mean(fold_maes)
        std_mae  = np.std(fold_maes)
        n_approx = int(len(df) * (1 - 1/5) * frac)
        print(f"  {int(frac*100):3d}%  n~{n_approx:5d}  "
              f"MAE={mean_mae:.4f} +/- {std_mae:.4f}")
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

    # ── Load 9.1 outputs ──
    print("Loading 9.1 outputs from disk...")
    perfold_df = pd.read_csv(f"{OUTPUT_DIR}/model_comparison_perfold.csv")
    pooled_df  = pd.read_csv(f"{OUTPUT_DIR}/model_comparison_pooled.csv")
    artifact   = joblib.load(MODEL_PKL)

    best_name    = artifact["model_name"]
    saved_params = artifact["params"]
    saved_n_est  = artifact.get("n_estimators", None)
    feature_cols = artifact["feature_cols"]

    print(f"  Best model from 9.1: {best_name}")
    print(f"  Params: {saved_params}")
    print(f"  n_estimators: {saved_n_est}")

    # ── Load data ──
    print("\nLoading feature matrix...")
    df = pd.read_parquet(FEATURE_MATRIX)
    print(f"  {len(df)} reactions, {df.shape[1]} columns")

    rxn_class_cols = [c for c in df.columns if c.startswith(RXN_CLASS_PREFIX)]

    print("\nLoading scaffold splits...")
    folds = load_splits(SPLITS_FILE, df)

    # ── Regenerate fold cache for 9.3 ──
    fold_cache = regenerate_fold_cache(
        df, folds, feature_cols, best_name,
        saved_params, saved_n_est
    )

    # ── 9.2 Ablation (full Optuna — rigorous) ──
    abl_pooled, abl_pf, variant_preds = run_ablation(
        df, folds, feature_cols, rxn_class_cols, best_name
    )

    # ── 9.3 Screening ──
    screen_metrics = run_screening_analysis(df, fold_cache, best_name)

    # ── 9.4 Learning curve ──
    lc_df = run_learning_curve(
        df, folds, feature_cols, best_name,
        saved_params, saved_n_est
    )

    # ── Final summary ──
    print("\n" + "=" * 72)
    print("=== Step 9 Complete ===")
    print("=" * 72)

    best_po = pooled_df[pooled_df["Model"] == best_name].iloc[0]
    print(f"Best model:    {best_name}")
    print(f"Pooled MAE:    {best_po['MAE']:.4f} eV  (from 9.1)")
    print(f"Pooled R2:     {best_po['R2']:.4f}")
    print(f"Screening:")
    print(f"  Recall:      {screen_metrics['Recall']:.4f}  (>= 0.90)")
    print(f"  Precision:   {screen_metrics['Precision']:.4f}  (>= 0.75)")
    h3 = (screen_metrics["Recall"] >= 0.90
           and screen_metrics["Precision"] >= 0.75)
    print(f"  H3:          {'PASSED' if h3 else 'FAILED'}")

    print(f"\nAblation:")
    for _, row in abl_pooled.iterrows():
        print(f"  {row['Variant']:15s}  MAE={row['MAE']:.4f}  "
              f"R2={row['R2']:.4f}  Recall={row['Recall']:.4f}")

    d_mae = abl_pooled[abl_pooled["Variant"] == "D_full"]["MAE"].values[0]
    b_mae = abl_pooled[abl_pooled["Variant"] == "B_elec_topo"]["MAE"].values[0]
    c_mae = abl_pooled[abl_pooled["Variant"] == "C_geo_topo"]["MAE"].values[0]
    a_mae = abl_pooled[abl_pooled["Variant"] == "A_topo_only"]["MAE"].values[0]
    if a_mae > c_mae > b_mae > d_mae:
        print(f"\n  Hierarchy: A > C > B > D (expected)")
    elif a_mae > b_mae > d_mae:
        print(f"\n  Hierarchy: D best, consistent with H1 (CMI > 0)")
    else:
        print(f"\n  Hierarchy: {a_mae=:.3f} {b_mae=:.3f} "
              f"{c_mae=:.3f} {d_mae=:.3f}")

    print(f"\nOutputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()