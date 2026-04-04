# Grambow / RDB7 — Raw Data

## Purpose in PRISM
External validation dataset. The trained PRISM model is applied to this dataset
WITHOUT retraining to test cross-dataset generalization. The deliberate level-of-theory
mismatch (different functional, dispersion correction, and basis set vs. Transition1x)
tests two axes of transferability: dispersion robustness and basis set convergence.

## Source
This is Spiekermann's cleaned and upgraded version of the original Grambow dataset,
referred to as **RDB7** (Reaction Database, up to 7 heavy atoms).

- **Paper:** Spiekermann, K. A.; Pattanaik, L.; Green, W. H. (2022).
  "High accuracy barrier heights, enthalpies, and rate coefficients for chemical
  reactions." *Sci. Data* 9, 417.
- **DOI:** https://doi.org/10.1038/s41597-022-01529-6
- **Download:** https://zenodo.org/records/6618262
- **Original Grambow dataset:** https://zenodo.org/records/3715478
- **License:** CC BY 4.0

### Why RDB7 instead of the original Grambow?
Spiekermann re-optimized products and cleaned SMILES (fixed incorrect bond orders
and formal charges from Open Babel perception). Reaction indices match the original
Grambow numbering for literature comparability.

## Citations
```bibtex
@article{spiekermann2022rdb7,
  title     = {High accuracy barrier heights, enthalpies, and rate coefficients
               for chemical reactions},
  author    = {Spiekermann, Kevin A. and Pattanaik, Lagnajit and Green, William H.},
  journal   = {Scientific Data},
  volume    = {9},
  pages     = {417},
  year      = {2022},
  doi       = {10.1038/s41597-022-01529-6}
}

@article{grambow2020original,
  title     = {Reactants, products, and transition states of elementary chemical
               reactions based on quantum chemistry},
  author    = {Grambow, Colin A. and Pattanaik, Lagnajit and Green, William H.},
  journal   = {Scientific Data},
  volume    = {7},
  pages     = {137},
  year      = {2020},
  doi       = {10.1038/s41597-020-0460-4}
}
```

## Dataset Summary
- **Reactions (ωB97X-D3/def2-TZVP):** 11,926
- **Elements:** H, C, N, O only
- **Max heavy atoms:** 7
- **Level of theory:** ωB97X-D3/def2-TZVP (geometry opt + freq, Q-Chem)
- **TS method:** Growing String Method (not NEB)
- **Reaction families:** Pre-classified using RMG templates

### Level-of-theory mismatch vs. Transition1x (deliberate)
| Property | Transition1x | Grambow/RDB7 |
|----------|-------------|--------------|
| Functional | ωB97x | ωB97X-D3 |
| Dispersion | None | D3 correction |
| Basis set | 6-31G(d) (Pople DZ) | def2-TZVP (Ahlrichs TZ) |
| TS method | NEB | Growing String Method |
| Software | ORCA | Q-Chem |

## Files in this directory

### `wb97xd3.csv` (3.8 MB) — PRIMARY
CSV with one row per reaction. Columns (from Spiekermann Table 1):
| Column | Description |
|--------|-------------|
| `idx` | Reaction index (matches original Grambow numbering) |
| `rsmi` | Atom-mapped SMILES of reactant(s) |
| `psmi` | Atom-mapped SMILES of product(s) |
| `ea` | Forward activation energy (kcal/mol) |
| `dh` | Reaction enthalpy (kcal/mol) |
| `rmg_family` | RMG reaction family classification |
| `rinchi` | InChI of reactant (added in v1.0.1) |
| `pinchi` | InChI of product (added in v1.0.1) |

**Unit warning:** Energies are in kcal/mol, NOT eV. Convert: 1 eV = 23.0605 kcal/mol.

### `wb97xd3.tar.gz` (1.4 GB) — GEOMETRIES
Archive of Q-Chem log files. Structure after extraction:
```
wb97xd3/
├── rxn000001/
│   ├── r000001.log    # Reactant: geom opt + freq
│   ├── ts000001.log   # Transition state: geom opt + freq
│   └── p000001.log    # Product: geom opt + freq (re-optimized by Spiekermann)
├── rxn000002/
│   └── ...
```
Each `.log` file contains optimized Cartesian coordinates, total energies, and
harmonic frequencies. Parse with cclib, Q-Chem ASE reader, or custom parser.

### `ccsdtf12_tz.csv` (5 KB) — BONUS HIGH-LEVEL REFERENCE
15 validation reactions at CCSD(T)-F12a/cc-pVTZ-F12 — near benchmark quality.
Same column format as wb97xd3.csv. Useful as a high-accuracy reference check
but too small for statistical validation.

## What PRISM extracts from this dataset
1. **Activation energies** from `wb97xd3.csv` → column `ea` (convert to eV)
2. **Reaction enthalpies** from `wb97xd3.csv` → column `dh` (convert to eV)
3. **Atom-mapped SMILES** from `wb97xd3.csv` → columns `rsmi`, `psmi`
4. **RMG family labels** from `wb97xd3.csv` → column `rmg_family`
5. **3D geometries** from `wb97xd3.tar.gz` → parsed from Q-Chem log files
   (needed to compute xTB descriptors on R, P structures)

## Notes
- Do NOT retrain PRISM on this data — it is held-out external validation only.
- Conformal prediction coverage will degrade on this set due to distribution shift;
  report both nominal and empirical coverage (see proposal §4.4.3).
- Downloaded: [DATE]
