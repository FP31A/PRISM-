# -*- coding: utf-8 -*-
"""
PRISM Step 8 — Descriptor Assembly and Information-Theoretic Analysis
src/models/train.py  (v3 — rigorous CMI pipeline)

Tasks:
  8.1 Merge Stream A + B Tier1 + C on rxn_id
  8.2 Drop NaN rows, record count
  8.3 Mutual Information ranking (bootstrap x100)
  8.4 Conditional MI: CMI(x_geo; Ea | x_elec) — tests H1

WHY THIS VERSION EXISTS
-----------------------
v2 of this script produced a NEGATIVE observed CMI (−0.075 nats), which is
mathematically impossible (CMI >= 0).  Diagnosis: the chain-rule subtraction
  CMI = I(X_elec, X_geo ; Y)  −  I(X_elec ; Y)
compounded the downward bias of the KSG estimator in high dimensions.
The 11D joint-space KSG under-estimated true MI more than the 7D electronic-only
KSG, so the difference went negative.

FIXES IN v3
-----------
1. Primary estimator switched to GAUSSIAN COPULA CMI (GCMI-CMI):
     - Each variable is rank-transformed to standard Gaussian (empirical copula).
     - CMI is then computed in closed form from covariance determinants.
     - Gives a provable LOWER BOUND on true CMI (Ince et al. 2017, HBM).
     - Degrades gracefully in high dimensions (no k-NN curse).
2. Verification estimator: FRENZEL-POMPE direct k-NN CMI.
     - Single joint-space k-NN pass, no chain-rule subtraction.
     - Bias does not compound.
3. Statistical test: LOCAL PERMUTATION (Runge 2018 "CMIknn" style).
     - Shuffles X_geo among nearest-neighbours in X_elec space.
     - Preserves X_elec-X_geo coupling, breaks X_geo-Y link.
     - Gives a correctly-calibrated null for the conditional independence test.
4. Synthetic validation FIRST: we verify the estimators reproduce known
   ground-truth CMI on a toy Gaussian dataset before trusting real results.
5. Sensitivity sweep across PCA dimensionality so the H1 conclusion is not
   an artefact of one arbitrary choice.
6. All bootstrap / permutation work is parallelised via joblib.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import norm
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STREAM_A   = "data/transition1x/descriptors/stream_a_electronic_trans1x.parquet"
STREAM_B   = "data/transition1x/descriptors/stream_b_geometric_tier1.parquet"
STREAM_C   = "data/transition1x/descriptors/stream_c_topological.parquet"
CURATED    = "data/transition1x/processed/final_curated_reactions.parquet"
OUTPUT_DIR = "data/transition1x/features"
OUTPUT_MATRIX  = f"{OUTPUT_DIR}/feature_matrix.parquet"
OUTPUT_MI      = f"{OUTPUT_DIR}/mi_ranking.csv"
OUTPUT_CMI     = f"{OUTPUT_DIR}/cmi_result.txt"
OUTPUT_CMI_CSV = f"{OUTPUT_DIR}/cmi_sensitivity.csv"

# ---------------------------------------------------------------------------
# Feature definitions per stream
# ---------------------------------------------------------------------------
ELECTRONIC_FEATURES = [
    "dE_xtb", "gap_R", "gap_P", "d_gap",
    "dipole_R", "dipole_P", "d_dipole",
    "fukui_plus", "fukui_minus", "delta_WBO",
    "bep_prediction",
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

TARGET = "Ea_eV"


# ============================================================================
# TASK 8.3 — univariate MI (unchanged; validated in v2 review)
# ============================================================================
def bootstrap_mi(X, y, feature_names, n_bootstrap=100, k=5, random_state=42):
    """Univariate MI(Xi; y) with bootstrap CIs.  One MI call per feature is
    what sklearn.mutual_info_regression does internally — the CORRECT use of
    that function, unlike the Task 8.4 misuse in v1."""
    rng     = np.random.RandomState(random_state)
    n       = len(X)
    mi_boot = np.zeros((n_bootstrap, X.shape[1]))

    for b in range(n_bootstrap):
        idx        = rng.choice(n, size=n, replace=True)
        mi_boot[b] = mutual_info_regression(
            X[idx], y[idx], n_neighbors=k, random_state=b
        )

    mi_mean = mi_boot.mean(axis=0)
    mi_std  = mi_boot.std(axis=0)

    return pd.DataFrame({
        "feature":  feature_names,
        "MI_mean":  mi_mean,
        "MI_std":   mi_std,
        "MI_lower": np.percentile(mi_boot, 2.5, axis=0),
        "MI_upper": np.percentile(mi_boot, 97.5, axis=0),
        "stream":   [
            "electronic" if f in ELECTRONIC_FEATURES else
            "geometric"  if f in GEOMETRIC_FEATURES  else
            "topological"
            for f in feature_names
        ],
    }).sort_values("MI_mean", ascending=False).reset_index(drop=True)


# ============================================================================
# TASK 8.4 — rigorous multivariate CMI infrastructure
# ============================================================================

# ---- 1. Gaussian Copula transforms -----------------------------------------
def _copula_transform(X):
    """
    Map each column of X to standard Gaussian via rank-based empirical CDF.
    This is the 'Gaussian copula' preprocessing: it preserves rank-order
    dependencies while removing all marginal distributional quirks.

    After this transform, any linear Gaussian analysis captures the true
    non-linear dependence structure between variables (as long as the
    dependence has a Gaussian copula).  Ince et al. 2017 showed this gives
    a provable LOWER BOUND on the true MI.
    """
    X = np.atleast_2d(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    ranks    = np.apply_along_axis(
        lambda col: np.argsort(np.argsort(col)), 0, X
    ).astype(float) + 1.0
    uniforms = ranks / (n + 1.0)
    return norm.ppf(uniforms)


def _logdet_psd(M, eps=1e-8):
    """
    Log-determinant of a (near-)PSD matrix via Cholesky with jitter.

    The jitter is essential when features are strongly collinear
    (e.g. dE_xtb and bep_prediction in this dataset, which are
    linearly related via the BEP relation).  After the copula
    transform all diagonal entries are ≈ 1, so a fixed 1e-8 is
    effectively a relative jitter of 10⁻⁸.
    """
    M = np.atleast_2d(M)
    d = M.shape[0]
    try:
        L = np.linalg.cholesky(M + eps * np.eye(d))
        return 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        w = np.linalg.eigvalsh(M + eps * np.eye(d))
        w = np.clip(w, eps, None)
        return np.sum(np.log(w))


# ---- 2. GCMI primary estimator ---------------------------------------------
def gcmi_cmi(X, Y, Z):
    """
    Gaussian Copula Conditional Mutual Information.

        CMI(X; Y | Z) = 0.5 * [ logdet(Σ_XZ) + logdet(Σ_YZ)
                              − logdet(Σ_XYZ) − logdet(Σ_Z) ]

    computed on copula-transformed variables.  Lower bound on true CMI;
    exact when the joint distribution has a Gaussian copula.  Output in
    nats.  Clipped at 0 for tiny negative numerical noise.
    """
    X_g = _copula_transform(X)
    Y_g = _copula_transform(Y)
    Z_g = _copula_transform(Z)

    XZ   = np.hstack([X_g, Z_g])
    YZ   = np.hstack([Y_g, Z_g])
    XYZ  = np.hstack([X_g, Y_g, Z_g])

    C_XZ  = np.cov(XZ,  rowvar=False)
    C_YZ  = np.cov(YZ,  rowvar=False)
    C_XYZ = np.cov(XYZ, rowvar=False)
    C_Z   = np.cov(Z_g, rowvar=False)

    cmi = 0.5 * (
        _logdet_psd(C_XZ)
        + _logdet_psd(C_YZ)
        - _logdet_psd(C_XYZ)
        - _logdet_psd(C_Z)
    )
    return max(cmi, 0.0)


def gcmi_mi(X, Y):
    """Gaussian Copula MI (for sanity checks)."""
    X_g = _copula_transform(X)
    Y_g = _copula_transform(Y)
    XY  = np.hstack([X_g, Y_g])
    C_X  = np.cov(X_g, rowvar=False)
    C_Y  = np.cov(Y_g, rowvar=False)
    C_XY = np.cov(XY,  rowvar=False)
    mi = 0.5 * (_logdet_psd(C_X) + _logdet_psd(C_Y) - _logdet_psd(C_XY))
    return max(mi, 0.0)


# ---- 3. Frenzel-Pompe direct k-NN CMI (verification) -----------------------
def fp_cmi(X, Y, Z, k=5):
    """
    Frenzel & Pompe (2007) direct k-NN estimator of CMI.

        CMI = ψ(k) + < ψ(n_z + 1) − ψ(n_xz + 1) − ψ(n_yz + 1) >

    with ε_i = Chebyshev distance to the k-th NN in the JOINT (X,Y,Z) space
    and n_xz, n_yz, n_z = counts of points within ε_i in the marginal
    subspaces.  DIRECT estimator — no chain-rule subtraction, so biases do
    not compound.  Returns nats.  Can be slightly negative at finite N
    for small true CMI (artefact).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if X.ndim == 1: X = X.reshape(-1, 1)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    if Z.ndim == 1: Z = Z.reshape(-1, 1)

    W   = np.hstack([X, Y, Z])
    XZ  = np.hstack([X, Z])
    YZ  = np.hstack([Y, Z])
    n   = W.shape[0]

    tree_w = cKDTree(W)
    dists, _ = tree_w.query(W, k=k + 1, p=np.inf)
    eps = dists[:, -1]

    tree_xz = cKDTree(XZ)
    tree_yz = cKDTree(YZ)
    tree_z  = cKDTree(Z)

    # emulate strict inequality for continuous data
    eps_strict = np.nextafter(eps, -np.inf)

    n_xz = np.array(tree_xz.query_ball_point(
        XZ, r=eps_strict, p=np.inf, return_length=True
    )) - 1
    n_yz = np.array(tree_yz.query_ball_point(
        YZ, r=eps_strict, p=np.inf, return_length=True
    )) - 1
    n_z  = np.array(tree_z.query_ball_point(
        Z,  r=eps_strict, p=np.inf, return_length=True
    )) - 1

    n_xz = np.clip(n_xz, 1, None)
    n_yz = np.clip(n_yz, 1, None)
    n_z  = np.clip(n_z,  1, None)

    cmi = (
        digamma(k)
        + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))
    )
    return cmi   # caller decides whether to clip


# ---- 4. Local permutation (Runge 2018) -------------------------------------
def local_permutation(X_geo, Z, rng, k_perm=10):
    """
    Generate a permutation of X_geo that preserves its joint distribution
    with Z but destroys its conditional link to anything else.

    For each i: find its k_perm NN in Z-space, pick one at random, set
    X_geo_perm[i] = X_geo[neighbour].  Under H0 (X_geo ⊥ Y | Z) this
    preserves the null's sampling distribution.
    """
    n = len(Z)
    tree = cKDTree(Z)
    _, nn_idx = tree.query(Z, k=k_perm + 1, p=np.inf)
    nn_idx    = nn_idx[:, 1:]                   # drop self
    choices   = rng.randint(0, k_perm, size=n)
    swap_idx  = nn_idx[np.arange(n), choices]
    return X_geo[swap_idx]


# ---- 5. Synthetic validation -----------------------------------------------
def validate_estimators_on_synthetic(n=5000, seed=42):
    """
    Sanity-check estimators against ground-truth CMI on synthetic Gaussians.

    Z      ~ N(0, I_3)                        (analogue of X_elec)
    X      = 0.5·Z[:,0] + √(1-0.25)·η         (partially coupled with Z,
                                               analogue of X_geo)
    Y_null = Z[:,0] + 0.5·ε                   → CMI(X; Y_null | Z) = 0
    Y_alt  = Z[:,0] + 0.8·X + 0.5·ε           → CMI(X; Y_alt  | Z) > 0

    For Y_alt and jointly-Gaussian vars, CMI is computable exactly from the
    5×5 covariance matrix.
    """
    rng = np.random.RandomState(seed)
    Z        = rng.randn(n, 3)
    X_unique = rng.randn(n, 1)
    X        = 0.5 * Z[:, :1] + np.sqrt(1 - 0.25) * X_unique

    eps_null = rng.randn(n, 1) * 0.5
    eps_alt  = rng.randn(n, 1) * 0.5
    Y_null   = Z[:, :1] + eps_null
    Y_alt    = Z[:, :1] + 0.8 * X + eps_alt

    # Analytic CMI for Y_alt via sample covariance
    data = np.hstack([X, Y_alt, Z])
    C    = np.cov(data, rowvar=False)
    iX, iY, iZ = [0], [1], [2, 3, 4]
    sub = lambda i: C[np.ix_(i, i)]
    cmi_true = 0.5 * (
        np.log(np.linalg.det(sub(iX + iZ)))
        + np.log(np.linalg.det(sub(iY + iZ)))
        - np.log(np.linalg.det(sub(iX + iY + iZ)))
        - np.log(np.linalg.det(sub(iZ)))
    )

    rows = []
    for name, Y, truth in [
        ("NULL (H0: CMI=0)", Y_null, 0.0),
        ("ALT  (H1: CMI>0)", Y_alt,  cmi_true),
    ]:
        rows.append({
            "case":       name,
            "truth_nats": truth,
            "GCMI_nats":  gcmi_cmi(X, Y, Z),
            "FP_nats":    fp_cmi(X, Y, Z, k=5),
        })
    df = pd.DataFrame(rows)
    print("\n   Synthetic validation (n=%d):" % n)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    return df


# ---- 6. CMI + permutation test ---------------------------------------------
def cmi_with_permutation(
    X_geo, X_elec, y,
    estimator="gcmi",
    n_permutations=1000,
    k=5,
    k_perm=10,
    n_jobs=-1,
    random_state=42,
):
    """
    Observed CMI + one-sided permutation p-value via local permutation.
    """
    rng = np.random.RandomState(random_state)

    if estimator == "gcmi":
        est = lambda Xg, Xe, Y: gcmi_cmi(Xg, Y, Xe)
    elif estimator == "fp":
        est = lambda Xg, Xe, Y: fp_cmi(Xg, Y, Xe, k=k)
    else:
        raise ValueError(f"Unknown estimator {estimator}")

    cmi_obs    = est(X_geo, X_elec, y)
    perm_seeds = rng.randint(0, 2**31 - 1, size=n_permutations)

    def _one_perm(seed):
        local_rng  = np.random.RandomState(seed)
        X_geo_perm = local_permutation(X_geo, X_elec, local_rng, k_perm=k_perm)
        return est(X_geo_perm, X_elec, y)

    null_dist = np.array(Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_one_perm)(s) for s in perm_seeds
    ))

    p_value = (np.sum(null_dist >= cmi_obs) + 1) / (n_permutations + 1)

    return {
        "cmi":       cmi_obs,
        "p_value":   p_value,
        "null_mean": null_dist.mean(),
        "null_std":  null_dist.std(),
        "null_dist": null_dist,
    }


def sensitivity_sweep(df, geo_features, elec_features, target,
                      pca_variance_list=(None, 0.95, 0.90, 0.80),
                      n_permutations=500,
                      k=5,
                      n_jobs=-1,
                      random_state=42):
    """
    Run the CMI + permutation test under multiple preprocessing settings
    and both estimators.  `None` = no PCA, raw standardised features.
    """
    y          = df[target].values.reshape(-1, 1)
    X_elec_raw = StandardScaler().fit_transform(df[elec_features].values)
    X_geo_raw  = StandardScaler().fit_transform(df[geo_features].values)

    rows = []
    for pv in pca_variance_list:
        if pv is None:
            X_e, X_g, tag = X_elec_raw, X_geo_raw, "raw"
        else:
            X_e = PCA(n_components=pv).fit_transform(X_elec_raw)
            X_g = PCA(n_components=pv).fit_transform(X_geo_raw)
            tag = f"PCA{int(pv*100)}"

        for estimator in ("gcmi", "fp"):
            print(f"\n   [{tag} | {estimator}] dims: "
                  f"elec={X_e.shape[1]}, geo={X_g.shape[1]}")
            res = cmi_with_permutation(
                X_g, X_e, y,
                estimator=estimator,
                n_permutations=n_permutations,
                k=k,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            print(f"       CMI_obs = {res['cmi']:+.4f}   "
                  f"null_mean = {res['null_mean']:+.4f}   "
                  f"p = {res['p_value']:.4f}")
            rows.append({
                "preprocessing": tag,
                "elec_dim":      X_e.shape[1],
                "geo_dim":       X_g.shape[1],
                "estimator":     estimator,
                "cmi":           res["cmi"],
                "null_mean":     res["null_mean"],
                "null_std":      res["null_std"],
                "p_value":       res["p_value"],
                "h1_accepted":   bool((res["cmi"] > 0)
                                     and (res["p_value"] < 0.01)),
            })

    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 8.1 Merge streams ─────────────────────────────────────────────────
    print("8.1 Merging descriptor streams...")
    stream_a = pd.read_parquet(STREAM_A)
    stream_b = pd.read_parquet(STREAM_B)
    stream_c = pd.read_parquet(STREAM_C)
    curated  = pd.read_parquet(CURATED)[["rxn_id", TARGET,
                                          "rmg_family", "dE_rxn_eV"]]
    rxn_class_cols = [c for c in stream_c.columns
                      if c.startswith("rxn_class_")]

    df = (curated
          .merge(stream_a[["rxn_id"] + ELECTRONIC_FEATURES],
                 on="rxn_id", how="left")
          .merge(stream_b[["rxn_id"] + GEOMETRIC_FEATURES],
                 on="rxn_id", how="left")
          .merge(stream_c[["rxn_id"] + TOPOLOGICAL_FEATURES + rxn_class_cols],
                 on="rxn_id", how="left"))
    print(f"   After merge: {len(df)} reactions, {df.shape[1]} columns")

    # ── 8.2 Drop NaN rows ────────────────────────────────────────────────
    print("\n8.2 Dropping NaN rows...")
    all_features = ELECTRONIC_FEATURES + GEOMETRIC_FEATURES + TOPOLOGICAL_FEATURES
    before       = len(df)
    df_clean     = df.dropna(subset=all_features + [TARGET]).copy()
    dropped      = before - len(df_clean)
    print(f"   Dropped:   {dropped} reactions")
    print(f"   Remaining: {len(df_clean)} reactions")
    nan_counts = df[all_features].isna().sum()
    print(f"   NaN breakdown per feature:")
    print(nan_counts[nan_counts > 0].to_string())

    df_clean.to_parquet(OUTPUT_MATRIX, index=False)
    print(f"\n   Feature matrix saved to {OUTPUT_MATRIX}")
    print(f"   Shape: {df_clean.shape}")

    # ── 8.3 Univariate MI ranking ────────────────────────────────────────
    print("\n8.3 Computing MI rankings (100 bootstrap resamples)...")
    feature_names = all_features + rxn_class_cols
    df_mi    = df_clean.dropna(subset=feature_names + [TARGET])
    X        = df_mi[feature_names].values.astype(float)
    y        = df_mi[TARGET].values

    mi_df = bootstrap_mi(X, y, feature_names, n_bootstrap=100)
    print(f"\n   Top 10 features by MI:")
    print(mi_df.head(10)[["feature", "stream",
                           "MI_mean", "MI_std"]].to_string(index=False))
    mi_df.to_csv(OUTPUT_MI, index=False)
    print(f"\n   Full MI ranking saved to {OUTPUT_MI}")

    # ── 8.4 Conditional MI — proper test of H1 ───────────────────────────
    print("\n" + "=" * 72)
    print("8.4 CONDITIONAL MUTUAL INFORMATION — H1 (DECOUPLING PRINCIPLE)")
    print("=" * 72)

    # (a) Synthetic validation
    print("\n(a) Validating estimators on synthetic ground truth...")
    synth_df = validate_estimators_on_synthetic(n=5000)

    # (b) Sensitivity sweep on real data
    print("\n(b) Running sensitivity sweep on real data...")
    print("    (4 preprocessing settings × 2 estimators × 500 permutations)")
    sweep = sensitivity_sweep(
        df_mi,
        geo_features      = GEOMETRIC_FEATURES,
        elec_features     = ELECTRONIC_FEATURES,
        target            = TARGET,
        pca_variance_list = (None, 0.95, 0.90, 0.80),
        n_permutations    = 500,
        k                 = 5,
        n_jobs            = -1,
    )

    print("\n   Sensitivity table:")
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    sweep.to_csv(OUTPUT_CMI_CSV, index=False)

    # Headline: GCMI on raw (standardised) features.  GCMI handles high D,
    # so no PCA needed for the primary answer.  Agreement with FP and with
    # PCA-reduced variants provides robustness.
    headline = sweep[(sweep.preprocessing == "raw")
                     & (sweep.estimator == "gcmi")].iloc[0]

    n_total      = len(sweep)
    n_h1_pass    = sweep.h1_accepted.sum()
    robust_h1    = n_h1_pass >= (n_total // 2 + 1)
    primary_pass = bool(headline.h1_accepted)

    cmi_result = (
        "=== CMI Result (H1 Test — v3 Rigorous Pipeline) ===\n"
        "\n"
        "PRIMARY ESTIMATOR: Gaussian Copula CMI (GCMI-CMI)\n"
        "   - Rank-copula transform + closed-form Gaussian CMI\n"
        "   - Provable lower bound on true CMI (Ince et al. 2017)\n"
        "   - Unlike k-NN estimators, bias does not blow up in high D\n"
        "\n"
        "VERIFICATION ESTIMATOR: Frenzel-Pompe direct k-NN CMI\n"
        "   - Single joint-space k-NN pass (no chain-rule subtraction)\n"
        "\n"
        "NULL DISTRIBUTION: Local permutation (Runge 2018)\n"
        "   - Shuffle X_geo within k-NN neighbourhoods of X_elec\n"
        "   - Preserves X_elec-X_geo coupling, breaks X_geo-Y link\n"
        "\n"
        "--- SYNTHETIC VALIDATION (sanity check) ---\n"
        f"{synth_df.to_string(index=False)}\n"
        "\n"
        "--- SENSITIVITY SWEEP ---\n"
        f"{sweep.to_string(index=False)}\n"
        "\n"
        "--- HEADLINE (GCMI, raw standardised features) ---\n"
        f"CMI(X_geo; Ea | X_elec) = {headline.cmi:+.4f} nats\n"
        f"Permutation p-value     = {headline.p_value:.4f}\n"
        f"Null mean ± std         = {headline.null_mean:+.4f} ± "
        f"{headline.null_std:.4f}\n"
        f"Primary H1 verdict      = "
        f"{'ACCEPTED' if primary_pass else 'REJECTED'}\n"
        "\n"
        f"--- ROBUSTNESS ACROSS SETTINGS ---\n"
        f"Settings accepting H1   = {n_h1_pass} / {n_total}\n"
        f"Robust H1 verdict       = "
        f"{'ACCEPTED' if robust_h1 else 'REJECTED'}\n"
        "\n"
        "--- INTERPRETATION ---\n"
        f"Geometric descriptors "
        f"{'ARE' if (primary_pass and robust_h1) else 'ARE NOT'} "
        f"non-redundant given electronic descriptors "
        f"(CMI > 0 with p < 0.01).\n"
    )

    print("\n" + cmi_result)
    with open(OUTPUT_CMI, "w") as f:
        f.write(cmi_result)
    print(f"   CMI result saved to {OUTPUT_CMI}")
    print(f"   Sensitivity table saved to {OUTPUT_CMI_CSV}")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("=== Step 8 Complete ===")
    print("=" * 72)
    print(f"Feature matrix: {df_clean.shape[0]} reactions × "
          f"{len(all_features + rxn_class_cols)} features")
    print(f"Target range:   {y.min():.3f} – {y.max():.3f} eV")
    print(f"Top feature:    {mi_df.iloc[0]['feature']} "
          f"(MI={mi_df.iloc[0]['MI_mean']:.4f})")
    print(f"H1 primary:     {'ACCEPTED' if primary_pass else 'REJECTED'} "
          f"(CMI={headline.cmi:+.4f}, p={headline.p_value:.4f})")
    print(f"H1 robust:      {'ACCEPTED' if robust_h1 else 'REJECTED'} "
          f"({n_h1_pass}/{n_total} settings)")


if __name__ == "__main__":
    main()