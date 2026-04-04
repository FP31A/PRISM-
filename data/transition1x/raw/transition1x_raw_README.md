# Transition1x — Raw Data

## Purpose in PRISM
Primary training dataset. Contains organic reaction geometries and DFT energies
used to train the barrier screening model and compute all three descriptor streams.

## Source
- **Paper:** Schreiner, M. et al. (2022). "Transition1x — a dataset for building
  generalizable reactive machine learning potentials." *Sci. Data* 9, 779.
- **DOI:** https://doi.org/10.1038/s41597-022-01870-w
- **Download:** https://doi.org/10.6084/m9.figshare.19614657
- **Code/Dataloaders:** https://gitlab.com/matschreiner/Transition1x
- **License:** CC BY 4.0

## Citation
```bibtex
@article{schreiner2022transition1x,
  title     = {Transition1x -- a dataset for building generalizable reactive
               machine learning potentials},
  author    = {Schreiner, Mathias and Bhowmik, Arghya and Vegge, Tejs and
               Busk, Jonas and Winther, Ole},
  journal   = {Scientific Data},
  volume    = {9},
  pages     = {779},
  year      = {2022},
  doi       = {10.1038/s41597-022-01870-w}
}
```

## Dataset Summary
- **Reactions:** 10,073 organic elementary reactions
- **Configs:** ~9.6 million DFT calculations (intermediate NEB images + endpoints)
- **Elements:** H, C, N, O only
- **Level of theory:** ωB97x/6-31G(d) (ORCA 5.0.2)
- **Generation method:** NEB with Climbing-Image refinement
- **Max heavy atoms:** 7 (derived from GDB7 reactants)

## File: `Transition1x.h5`
Single HDF5 file (~7 GB). Structure:

```
Transition1x.h5
├── data/                          # All 10,073 reactions
│   ├── {chemical_formula}/        # e.g., "C2H4N2O"
│   │   ├── {reaction_id}/         # e.g., "reaction_0001"
│   │   │   ├── atomic_numbers     # shape (m,) — element Z for each atom
│   │   │   ├── positions          # shape (n, m, 3) — Cartesian coords in Å
│   │   │   ├── energies           # shape (n,) — DFT total energies in eV
│   │   │   ├── forces             # shape (n, m, 3) — DFT forces in eV/Å
│   │   │   ├── reactant/          # endpoint geometry (n=1)
│   │   │   ├── transition_state/  # highest-energy image (n=1)
│   │   │   └── product/           # endpoint geometry (n=1)
├── train/                         # Symlinks to data/ (author's split)
├── val/                           # Symlinks to data/ (author's split)
└── test/                          # Symlinks to data/ (author's split)
```

- `n` = number of saved NEB images for that reaction (varies per reaction)
- `m` = number of atoms in the system
- Reactions are grouped by chemical formula, then by reaction ID
- The `reactant/`, `transition_state/`, `product/` subgroups each have the
  same dataset structure but with `n=1`
- Products from one reaction can be reactants for another (linkable via hash)

## What PRISM extracts from this file
For each reaction, we need:
1. **Reactant geometry** — `data/{formula}/{rxn}/reactant/positions` (for xTB calculations)
2. **Product geometry** — `data/{formula}/{rxn}/product/positions` (for xTB calculations)
3. **TS geometry** — `data/{formula}/{rxn}/transition_state/positions` (reference only)
4. **Activation energy** — `transition_state/energies - reactant/energies` (DFT label, in eV)
5. **Reaction energy** — `product/energies - reactant/energies` (DFT label, for comparison)
6. **Atomic numbers** — `data/{formula}/{rxn}/atomic_numbers` (to build ASE Atoms objects)

We do NOT need the millions of intermediate NEB images for PRISM — only the three
endpoint/TS geometries per reaction.

## Notes
- ~6 reactions have Ea > 10 eV (unphysical NEB artifacts). Filtered in curation.
- The author's train/val/test split is random; PRISM uses scaffold-based splitting instead.
- Downloaded: [DATE]
