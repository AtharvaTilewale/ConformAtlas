# ConformAtlas

```text
 ▄████▄   ▒█████   ███▄    █   █████▒▒█████   ██▀███   ███▄ ▄███▓ ▄▄▄     ▄▄▄█████▓ ██▓    ▄▄▄        ██████ 
▒██▀ ▀█  ▒██▒  ██▒ ██ ▀█   █ ▓██   ▒▒██▒  ██▒▓██ ▒ ██▒▓██▒▀█▀ ██▒▒████▄   ▓  ██▒ ▓▒▓██▒   ▒████▄    ▒██    ▒ 
▒▓█    ▄ ▒██░  ██▒▓██  ▀█ ██▒▒████ ░▒██░  ██▒▓██ ░▄█ ▒▓██    ▓██░▒██  ▀█▄ ▒ ▓██░ ▒░▒██░   ▒██  ▀█▄  ░ ▓██▄   
▒▓▓▄ ▄██▒▒██   ██░▓██▒  ▐▌██▒░▓█▒  ░▒██   ██░▒██▀▀█▄  ▒██    ▒██ ░██▄▄▄▄██░ ▓██▓ ░ ▒██░   ░██▄▄▄▄██   ▒   ██▒
▒ ▓███▀ ░░ ████▓▒░▒██░   ▓██░░▒█░   ░ ████▓▒░░██▓ ▒██▒▒██▒   ░██▒ ▓█   ▓██▒ ▒██▒ ░ ░██████▒▓█   ▓██▒▒██████▒▒
░ ░▒ ▒  ░░ ▒░▒░▒░ ░ ▒░   ▒ ▒  ▒ ░   ░ ▒░▒░▒░ ░ ▒▓ ░▒▓░░ ▒░   ░  ░ ▒▒   ▓▒█░ ▒ ░░   ░ ▒░▓  ░▒▒   ▓▒█░▒ ▒▓▒ ▒ ░
  ░  ▒     ░ ▒ ▒░ ░ ░░   ░ ▒░ ░       ░ ▒ ▒░   ░▒ ░ ▒░░  ░      ░  ▒   ▒▒ ░   ░    ░ ░ ▒  ░ ▒   ▒▒ ░░ ░▒  ░ ░
░        ░ ░ ░ ▒     ░   ░ ░  ░ ░   ░ ░ ░ ▒    ░░   ░ ░      ░     ░   ▒    ░        ░ ░    ░   ▒   ░  ░  ░  
░ ░          ░ ░           ░            ░ ░     ░            ░         ░  ░            ░  ░     ░  ░      ░  
░                                                                                                            
```

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**ConformAtlas** is a rigorous Python package and CLI tool for molecular dynamics (MD) ensemble, Principal Component Analysis (PCA), and Free Energy Landscape (FEL) characterization using GROMACS or direct trajectory streaming.

---

## Key Features

* **Temperature-Aware Free Energy Surfaces**:
  $$\Delta G(x, y) = -R \cdot T \cdot \ln\left[ \frac{P(x, y)}{P_{\max}} \right]$$
  Evaluated in $\text{kJ/mol}$ with explicit temperature validation ($T > 0$ K). Zero-occupancy bins are safely masked as $\text{NaN}$ rather than erroneously assigned zero free energy.
* **Topological Watershed Basin Segmentation**:
  Automatically identifies discrete conformational energy basins and assigns every trajectory frame to its underlying basin.
* **Verified Representative Structure Extraction**:
  Extracts the real trajectory frame closest in PC space to each basin minimum as a full-atom PDB structure with complete provenance metadata.
* **Multi-Replicate Analysis**:
  Maintains replicate identity throughout, avoids continuous-trajectory statistical inflation, and supports balanced replicate weighting (`equal-replicate`).
* **Condition & Mutant Comparisons in Shared PCA Space**:
  Compares Wild-Type and mutants within an identical PCA coordinate system with invariant atom selection validation.
* **Rigorous Convergence & Sampling Diagnostics**:
  Evaluates first-half vs second-half subspace overlap (RMSIP), progressive overlap across sampling fractions, cosine content (diffusion test), and state population stability curves.
* **Replicate Uncertainty Estimation**:
  Reports Mean, SD, SEM, and 95% Confidence Intervals for state populations.
* **Automated Publication Figures & HTML Reports**:
  Generates publication-quality 2D/3D figures without artificial interpolation distortion and compiles a self-contained HTML analysis report.

---

## Installation

### Requirements
* Python $\ge 3.10$
* (Optional) GROMACS (`gmx` or `gmx_mpi`)

### Standard Install
```bash
git clone https://github.com/AtharvaTilewale/ConformAtlas.git
cd ConformAtlas
pip install .
```

For development and automated test running:
```bash
pip install -e ".[dev]"
```

Verify your environment:
```bash
conformatlas doctor
```

---

## Quick Start

### 1. Single or Multi-Replicate Analysis
```bash
conformatlas analyze \
    -s topology.tpr \
    -f rep1.xtc \
    -f rep2.xtc \
    -f rep3.xtc \
    -T 310 \
    --bins 64 \
    --output results/WT
```

### 2. WT vs Mutant Comparison
```bash
conformatlas compare --config comparison.yaml -o results/comparison
```

### 3. Generate Demonstration Project
```bash
conformatlas example --output demo_project
```

---

## Output Organization

Every analysis run produces a structured, reproducible results directory:

```text
results/
├── report.html                  # Standalone interactive HTML report
├── run_metadata.json            # Machine-readable execution provenance
├── summary.json                 # State occupancies and convergence summary
├── pca/                         # Eigenvalues, explained variance, projections
├── fel/                         # 2D free energy grid, probability density (.npy, .csv)
├── states/                      # Watershed basin map, frame assignments, representative PDBs
├── convergence/                 # RMSIP, cosine content, progressive overlap tables
├── statistics/                  # State populations with replicate SD, SEM, and 95% CI
└── figures/                     # Publication-ready 2D/3D FEL and PCA figures
```

---

## Important Thermodynamic Limitations

1. **Equilibrium Sampling Assumption**:
   Boltzmann inversion assumes the trajectory represents an **unbiased, equilibrium thermodynamic ensemble**. For biased or enhanced sampling simulations (Metadynamics, Umbrella Sampling, Accelerated MD), raw histogram occupancy must not be interpreted as an equilibrium free energy surface without statistical reweighting.
2. **PCA Projection Limits**:
   While the first two principal components often capture dominant collective domain motions, higher-dimensional transitions may project into overlapping regions in 2D. Always check cumulative explained variance and convergence diagnostics.

---

## Documentation

* [Installation Guide](docs/installation.md)
* [Quickstart Walkthrough](docs/quickstart.md)
* [Mathematical & Algorithmic Methods](docs/methods.md)
* [Scientific Interpretation Guide](docs/interpretation.md)
* [Multi-Replicate Analysis](docs/multiple_replicates.md)
* [Condition & Mutant Comparison](docs/comparison.md)
* [Sampling & Convergence Diagnostics](docs/convergence.md)
* [Basin Analysis & Representatives](docs/basin_analysis.md)
* [Output Manifest](docs/outputs.md)

---

## Citation

If you use ConformAtlas in your research, please cite:

```bibtex
@software{tilewale2026conformatlas,
  author = {Tilewale, Atharva},
  title = {ConformAtlas: MD Ensemble, PCA, and Free Energy Landscape Analysis},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/AtharvaTilewale/ConformAtlas}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
