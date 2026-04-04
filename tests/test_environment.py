# -*- coding: utf-8 -*-

"""
PRISM Environment Smoke Test
Run: python tests/test_environment.py
"""
import sys
import traceback

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
results = []


def run_test(name, func):
    """Run a test function and record the result."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        func()
        print(f"\n  {PASS}: {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"\n  {FAIL}: {name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))


# ─────────────────────────────────────────────────────────────
# TEST 1: xTB can compute an energy
# ─────────────────────────────────────────────────────────────
def test_xtb():
    """Run GFN2-xTB water test."""
    from ase import Atoms
    from ase.calculators.calculator import CalculationFailed

    # Build a water molecule (positions in Angstrom)
    water = Atoms(
        "OH2",
        positions=[
            [0.000, 0.000, 0.117],
            [0.000, 0.757, -0.469],
            [0.000, -0.757, -0.469],
        ],
    )

    # Try the Python API first, fall back to command-line
    try:
        from xtb.ase.calculator import XTB
        water.calc = XTB(method="GFN2-xTB")
        print("  Using: xtb-python API")
    except ImportError:
        # Fallback: use command-line xTB through ASE
        from ase.calculators.xtb import XTB
        water.calc = XTB(method="GFN2-xTB")
        print("  Using: ASE command-line xTB interface")

    energy = water.get_potential_energy()  # in eV
    print(f"  Water energy: {energy:.6f} eV")

    # GFN2-xTB water energy should be roughly -137.9 eV
    # We just check it's negative and in a reasonable range
    assert energy < 0, f"Energy should be negative, got {energy}"
    assert -200 < energy < -100, f"Energy outside expected range: {energy}"
    print(f"  Energy is in expected range (-200 to -100 eV)")


# ─────────────────────────────────────────────────────────────
# TEST 2: RDKit can parse SMILES and do basic operations
# ─────────────────────────────────────────────────────────────
def test_rdkit():
    """smiles, fingerprint test"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem

    smiles = "CC(=O)O"  # acetic acid
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    print(f"  Parsed SMILES: {smiles}")
    print(f"  Num atoms: {mol.GetNumAtoms()}")

    # Compute molecular weight
    mw = Descriptors.MolWt(mol)
    print(f"  Molecular weight: {mw:.2f}")
    assert 59 < mw < 61, f"Unexpected MW for acetic acid: {mw}"

    # Compute ECFP4 fingerprint (needed for Butina scaffold splitting)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    assert fp.GetNumBits() == 2048, "Fingerprint wrong size"
    print(f"  ECFP4 fingerprint: {fp.GetNumBits()} bits, {fp.GetNumOnBits()} on-bits")

    # Test SMILES round-trip
    canonical = Chem.MolToSmiles(mol)
    print(f"  Canonical SMILES: {canonical}")


# ─────────────────────────────────────────────────────────────
# TEST 3: ASE can build molecules and run IDPP interpolation
# ─────────────────────────────────────────────────────────────
def test_ase_idpp():
    """
    Build two simple geometries (reactant and product) and run IDPP
    interpolation between them.
    """
    from ase import Atoms
    from ase.neb import NEB

    # Simple test: H2 molecule with two different bond lengths
    reactant = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    product = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.50]])

    # Create 5 images for NEB (including endpoints)
    images = [reactant.copy()]
    for _ in range(3):
        images.append(reactant.copy())
    images.append(product.copy())

    # Run IDPP interpolation
    neb = NEB(images)
    neb.interpolate("idpp")

    # Check that intermediate images are between reactant and product
    midpoint = images[2]  # middle image
    mid_dist = midpoint.get_distance(0, 1)
    print(f"  Reactant H-H distance: {reactant.get_distance(0, 1):.3f} A")
    print(f"  Midpoint H-H distance: {mid_dist:.3f} A")
    print(f"  Product  H-H distance: {product.get_distance(0, 1):.3f} A")

    assert 0.74 < mid_dist < 1.50, f"Midpoint distance should be between R and P, got {mid_dist}"
    print(f"  IDPP interpolation produced sensible intermediate geometry")


# ─────────────────────────────────────────────────────────────
# TEST 4: h5py can open Transition1x.h5
# ─────────────────────────────────────────────────────────────
def test_h5py():
    """
    Open the Transition1x HDF5 file and inspect its structure.
    Adjust the path below to match your setup.
    """
    import h5py
    import os

    # Try common paths — adjust for your setup
    possible_paths = [
        "data/transition1x/raw/Transition1x.h5",
        "../data/transition1x/raw/Transition1x.h5",
        os.path.expanduser("/Users/fidelepoh/PRISM-/data/transition1x/raw/Transition1x.h5"),
    ]

    h5_path = None
    for p in possible_paths:
        if os.path.exists(p):
            h5_path = p
            break

    if h5_path is None:
        print("  WARNING: Transition1x.h5 not found at expected paths.")
        print("  Searched:", possible_paths)
        print("  Skipping HDF5 content check — testing h5py itself only.")
        # At least verify h5py works
        import h5py
        print(f"  h5py version: {h5py.__version__}")
        print(f"  h5py is importable and functional")
        return

    print(f"  Found: {h5_path}")
    print(f"  Size: {os.path.getsize(h5_path) / 1e9:.2f} GB")

    with h5py.File(h5_path, "r") as f:
        top_keys = list(f.keys())
        print(f"  Top-level keys: {top_keys}")
        assert "data" in top_keys, f"Expected 'data' key, got {top_keys}"

        # Count reactions
        n_reactions = 0
        for formula in f["data"]:
            n_reactions += len(f["data"][formula])
        print(f"  Total reactions: {n_reactions}")
        assert n_reactions > 10000, f"Expected >10k reactions, got {n_reactions}"

        # Inspect one reaction
        first_formula = list(f["data"].keys())[0]
        first_rxn = list(f["data"][first_formula].keys())[0]
        rxn = f["data"][first_formula][first_rxn]
        print(f"  Sample reaction: {first_formula}/{first_rxn}")
        print(f"    Datasets: {list(rxn.keys())}")

        if "reactant" in rxn:
            r_pos = rxn["reactant"]["positions"][:]
            r_eng = rxn["reactant"]["energies"][:]
            print(f"    Reactant: {r_pos.shape[1]} atoms, energy = {r_eng[0]:.4f} eV")


# ─────────────────────────────────────────────────────────────
# TEST 5: ML stack works
# ─────────────────────────────────────────────────────────────
def test_ml_stack():
    """Quick check that XGBoost, SHAP, and Optuna all work together."""
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    # Generate dummy data
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Train a small XGBoost model
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    print(f"  XGBoost trained: MAE = {mae:.2f} on dummy data")

    # Test SHAP
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:5])
    print(f"  SHAP values computed: shape = {shap_values.shape}")

    # Test Optuna (just create a study, don't run trials)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    print(f"  Optuna study created: {study.study_name}")

    # Test pyarrow / parquet round-trip
    import pandas as pd
    df = pd.DataFrame(X_train[:5], columns=[f"feat_{i}" for i in range(10)])
    df.to_parquet("/tmp/prism_test.parquet")
    df_read = pd.read_parquet("/tmp/prism_test.parquet")
    assert df_read.shape == df.shape, "Parquet round-trip failed"
    print(f"  Parquet round-trip OK: {df_read.shape}")


# ─────────────────────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PRISM ENVIRONMENT SMOKE TEST")
    print("=" * 60)

    run_test("1. GFN2-xTB energy calculation", test_xtb)
    run_test("2. RDKit SMILES parsing + fingerprints", test_rdkit)
    run_test("3. ASE IDPP interpolation", test_ase_idpp)
    run_test("4. HDF5 / Transition1x access", test_h5py)
    run_test("5. ML stack (XGBoost + SHAP + Optuna + Parquet)", test_ml_stack)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, err in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if err:
            print(f"         → {err}")

    print(f"\n  {passed}/{total} tests passed")

    if passed < total:
        print("\n  Fix the failing tests before proceeding to Step 2.")
        sys.exit(1)
    else:
        print("\n  Environment is ready. Proceed to Step 2.")
        sys.exit(0)
