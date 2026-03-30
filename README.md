# PRISM: Physics-Informed Reactivity Interpretable Screening Model

Interpretable, physics-informed Tier-1 screening framework for organic 
reaction activation barriers. Decomposes barriers into thermodynamic 
driving force and geometric distortion using GFN2-xTB and NEB-calibrated 
geometric proxies.

## Setup
```
conda env create -f environment.yml
conda activate prism
```

## Project Structure
```
PRISM/
├── data/
│   ├── raw/              # Original Transition1x, Grambow downloads
│   ├── processed/        # Curated, filtered, split datasets
│   └── external/         # Organometallic, OC20NEB data
├── src/
│   ├── descriptors/      # Stream A, B, C descriptor generation
│   ├── models/           # Training, ablation, calibration
│   ├── validation/       # Internal, external, cross-domain
│   └── utils/            # I/O, chemistry helpers, error handling
├── scripts/
│   └── nscc/             # PBS Pro job scripts for ASPIRE2A
├── notebooks/            # Exploration and analysis
└── results/
    ├── figures/
    └── tables/
```

## Inference
```
python prism.py --input reactions.xyz --output predictions.csv
```
