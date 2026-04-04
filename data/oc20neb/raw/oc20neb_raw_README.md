# OC20NEB — Raw Data

## Purpose in PRISM
Heterogeneous catalysis extension (exploratory, Phase 3). Task shifts from barrier
prediction to **anomaly detection**: flagging DFT NEB calculations where cheap
xTB-derived descriptors are wildly inconsistent with reported DFT values.

**Status: NOT YET DOWNLOADED.** Gated on feasibility experiment (xTB on ~500
structures; if failure rate > 30%, this extension is deferred).

## Source
- **Paper:** Wander, B.; Shuaibi, M.; Kitchin, J. R.; Ulissi, Z. W.; Zitnick, C. L.
  (2025). "CatTSunami: Accelerating Transition State Energy Calculations with
  Pre-trained Graph Neural Networks." *ACS Catal.* 15(7), 5283–5294.
- **DOI:** https://doi.org/10.1021/acscatal.4c04272
- **Download:** https://fair-chem.github.io/core/datasets/oc20neb.html
- **License:** CC BY 4.0

## Citation
```bibtex
@article{wander2025cattsunami,
  title     = {CatTSunami: Accelerating Transition State Energy Calculations
               with Pretrained Graph Neural Networks},
  author    = {Wander, Brook and Shuaibi, Muhammed and Kitchin, John R. and
               Ulissi, Zachary W. and Zitnick, C. Lawrence},
  journal   = {ACS Catalysis},
  volume    = {15},
  number    = {7},
  pages     = {5283--5294},
  year      = {2025},
  doi       = {10.1021/acscatal.4c04272}
}
```

## Dataset Summary
- **Converged NEB trajectories:** 932
- **Reaction types:** Dissociations, desorptions, transfers
- **Level of theory:** RPBE (GGA functional, periodic DFT)
- **Format:** ASE `.traj` files, organized by reaction type

## Expected file structure after download
```
oc20neb/raw/
├── dissociations/
│   └── dissociation_id_XX_XXXX_X_XXX-X_nebX.X.traj
├── desorptions/
│   └── desorption_id_XX_XXXX_X_XXX-X_nebX.X.traj
└── transfers/
    └── transfer_id_XX_XXXX_X_XXX-X_nebX.X.traj
```

Each `.traj` contains the full NEB band (typically 10 images). No pre-built LMDB
is provided; use ASE directly to read trajectories.

## What PRISM will extract
1. **Adsorbate + coordination shell** as a finite cluster (fragment extraction)
2. **xTB descriptors** on the extracted cluster
3. **Binary labels:** converged vs. unconverged (anomaly detection target)

## Open question
The manuscript references 2,827 total trajectories (932 converged + 1,895
unconverged). The public OC20NEB release contains only the 932 converged ones.
Clarify whether unconverged trajectories are separately available before committing.

## Notes
- This is periodic slab data — PRISM's molecular descriptors require adsorbate
  fragment extraction (see proposal §8).
- xTB was parameterized for molecular systems; performance on surface fragments
  is unknown and must be validated.
- Downloaded: [NOT YET]
