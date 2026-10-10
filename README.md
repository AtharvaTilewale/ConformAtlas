<p align="center">
  <img src="https://raw.githubusercontent.com/AtharvaTilewale/ConformAtlas/main/docs/assets/conformatlas-logo.png" alt="ConformAtlas" width="520" />
</p>

<h3 align="center">Conformational landscapes from molecular dynamics</h3>

<p align="center">
  PCA, TICA and dihedral decompositions, free-energy landscapes with bootstrap uncertainty, watershed states, barriers and kinetics, convergence diagnostics and shared-space comparisons of MD ensembles — from the command line, with a self-contained report.
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/AtharvaTilewale/ConformAtlas/actions/workflows/ci.yml/badge.svg)](https://github.com/AtharvaTilewale/ConformAtlas/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/conformatlas/badge/?version=latest)](https://conformatlas.readthedocs.io/en/latest/?badge=latest)

**ConformAtlas** is a rigorous Python package and CLI tool for molecular dynamics (MD) ensemble, Principal Component Analysis (PCA), and Free Energy Landscape (FEL) characterization using GROMACS or direct trajectory streaming.

---

## Key Features

* **Superposition before PCA**:
  Every frame is least-squares fitted (Kabsch) on the `--lsq-group` atoms to the structure file before the covariance matrix is built, so the principal components describe internal motion rather than tumbling and drift in the box. Equivalent to `gmx covar -fit`; disable with `--no-fit` only for pre-fitted trajectories.
* **Choice of collective variables**:
  Cartesian PCA (default), backbone dihedral PCA (`--pca-type dihedral`, sin/cos of φ/ψ), or time-lagged independent component analysis (`--pca-type tica --tica-lag N`, with implied timescales). The landscape can be built on any pair of components (`--fel-pcs 1,3`), on backbone RMSD vs radius of gyration (`--cv rmsd-rg`), or on two columns of an external file such as a PLUMED `COLVAR` or `gmx` `.xvg` (`--cv file --cv-file ... --cv-cols 1,2`). Per-frame statistical weights (`--weights-file`) reweight biased simulations.
* **Temperature-aware free energy surfaces**:
  $$\Delta G(x, y) = -R \cdot T \cdot \ln\left[ \frac{P(x, y)}{P_{\max}} \right]$$
  Evaluated in $\text{kJ/mol}$ with explicit temperature validation ($T > 0$ K). Zero-occupancy bins are masked as $\text{NaN}$ rather than assigned zero free energy. The grid resolution is chosen automatically from the Freedman–Diaconis rule (`--bins auto`, default) or set explicitly.
* **Statistical uncertainty of the landscape**:
  A circular block bootstrap (block length from the statistical inefficiency of the CVs, blocks drawn within each replicate) gives the standard deviation and 95 % confidence interval of ΔG in every bin, of every state population and state free energy, and of every barrier (`--bootstrap N`, default 200).
* **Topological watershed basin segmentation, barriers and kinetics**:
  Discrete conformational basins are identified on the landscape and every frame is assigned. Sub-basins separated by less than `--min-barrier` (default 1 kT) are merged. Saddle-point barriers between adjacent states, transition counts, transition-probability matrix, dwell times and Markov mean first-passage times are reported.
* **Equilibration detection and effective sample size**:
  The statistical inefficiency g of PC1/PC2 and the effective number of independent frames N_eff are reported per replicate; the equilibration point that maximises N_eff (Chodera 2016) is detected on the fit RMSD and either reported or applied (`--equilibration auto`). Time windows are set with `-b/-e` (ps). Fewer than 50 effective samples force the sampling verdict to "Insufficient evidence".
* **Representative structures and state ensembles**:
  One or several structures per state (`--n-representatives`, `--representative minimum|medoid`) as full-atom PDBs with provenance, plus per-state average structures and RMSF profiles.
* **Mode interpretation**:
  Per-atom and per-residue contributions to the leading modes, a VMD/NMWiz `modes.nmd` file, an average structure and extreme-projection structures for each component.
* **Standard descriptors**:
  Backbone RMSD, radius of gyration, per-atom/per-residue RMSF and, with `--lig`, ligand RMSD after protein superposition.
* **Multi-replicate analysis**:
  Replicate identity is maintained throughout; balanced replicate weighting (`equal-replicate`) avoids one long trajectory dominating the covariance.
* **Condition and mutant comparisons in a shared PCA space, with statistics**:
  Any number of systems are projected onto a common basis. For every system versus the reference: a ΔΔG map with bootstrap uncertainty, population differences with block-bootstrap confidence intervals and p-values (plus Welch's t-test across replicates when each system has at least two), state ΔΔG with confidence intervals, landscape overlap metrics (Bhattacharyya coefficient, Jensen–Shannon divergence) and the RMSIP between each system's own essential subspace.
* **Convergence and sampling diagnostics**:
  First-half vs second-half subspace overlap (RMSIP) pooled over all replicates and per replicate, progressive overlap, cosine content, population stability on cumulative time per replicate, N_eff and equilibration tables.
* **Input validation**:
  Atom-count consistency between structure and trajectory, broken Cα chains (periodic-boundary artefacts) and memory requirements are checked before the run; GROMACS `.ndx` groups are accepted everywhere a group name is (`-n index.ndx`).
* **Figures and reports**:
  Publication-quality 2D/3D figures, 1-D projections with error bands, uncertainty maps, descriptor and kinetics panels, a self-contained HTML report, optional PDF (`--pdf`), and a YAML configuration file for the full `analyze` run (`conformatlas init-config --type analyze`, `conformatlas analyze --config run.yaml`).

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

All options can be stored in a YAML file (`conformatlas init-config --type analyze -o run.yaml`, then `conformatlas analyze --config run.yaml`); command-line flags override the file. A fuller protein–ligand run:

```bash
conformatlas analyze -s md.gro -f md_noPBC.xtc -n index.ndx \
    --cov-group C-alpha --lsq-group C-alpha --lig ZMR \
    -T 300 -b 5000 --equilibration auto \
    --bootstrap 200 --n-representatives 3 --representative medoid \
    --pdf -o results/complex
```

Alternative landscapes on the same trajectory:

```bash
conformatlas analyze ... --pca-type dihedral                 # backbone φ/ψ PCA
conformatlas analyze ... --pca-type tica --tica-lag 50       # slow modes (lag in frames)
conformatlas analyze ... --cv rmsd-rg                        # RMSD vs Rg
conformatlas analyze ... --cv file --cv-file COLVAR --cv-cols 1,2 --cv-labels "d1 (nm),d2 (nm)"
```

### 2. WT vs Mutant Comparison
```bash
conformatlas compare --config comparison.yaml -o results/comparison
```

### 3. Generate Demonstration Project
```bash
conformatlas example --output demo_project
```
The demonstration peptide switches between a bent and a twisted conformation and tumbles freely in the box, so it only resolves into two states once frames have been superposed. Run it with `--bins 32 --smooth-sigma 1.0` (200 frames do not populate a 64 × 64 grid).

### Input requirements
* Trajectories must be whole across periodic boundaries (`gmx trjconv -pbc mol`); fitting a broken molecule is meaningless.
* `-s` should carry coordinates (`.gro`, `.pdb`) to serve as the superposition reference. With a `.tpr` the first trajectory frame is used instead.
* For enhanced-sampling or biased trajectories supply per-frame weights (`--weights-file`, e.g. exp(βV) from a reweighting scheme); without weights see the thermodynamic limitations below.
* Group names follow GROMACS conventions: `Backbone` = N, CA, C; `MainChain` = N, CA, C, O; `C-alpha`; `Protein`; or any GROMACS index group (`-n`) / MDAnalysis selection string.
* `--lig RES` adds the ligand's atoms to the analysis group (the landscape then describes the complex) and reports its RMSD after protein superposition.

---

## Output Organization

Every analysis run produces a structured, reproducible results directory:

```text
results/
├── report.html                  # Standalone HTML report (report.pdf with --pdf)
├── run_metadata.json            # Machine-readable execution provenance and warnings
├── summary.json                 # State occupancies, convergence and uncertainty summary
├── pca/                         # Eigenvalues, variance, projections, eigenvectors.npy,
│                                # modes.nmd, average.pdb, PC*_extremes.pdb, mode_contributions_*.csv
├── fel/                         # Free energy grid, probability, bootstrap SD and CI grids
├── states/                      # Basins, frame assignments, barriers, state free energies,
│                                # State_X/ (representatives, average.pdb, RMSF), ensemble summary
├── kinetics/                    # Transition counts/matrix, dwell times, Markov MFPT
├── descriptors/                 # RMSD, Rg, ligand RMSD time series; RMSF per atom and residue
├── convergence/                 # RMSIP, cosine content, progressive overlap, N_eff, equilibration
├── statistics/                  # State populations with block SD/SEM, 95% CI and bootstrap CI
└── figures/                     # 2D/3D FEL, 1-D projections, uncertainty, basins, kinetics, …
```

---

## Validation

`validation/validate.py` checks the numerical core against analytical results and independent implementations (MDAnalysis, GROMACS, pymbar, deeptime, SciPy, ProDy): landscape inversion, PCA/TICA, superposition, statistical inefficiency and equilibration, basin populations and barriers on an analytic double well, bootstrap confidence-interval coverage, kinetics on a known Markov chain, descriptors, overlap metrics and the calibration of the comparison test. The current report is in [`validation/VALIDATION.md`](validation/VALIDATION.md) (51/51 checks). Convergence verdicts, the Freedman–Diaconis grid and the minor-basin threshold are heuristics by construction and are documented as such in `docs/methods.md`.

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
  version = {0.3.0},
  url = {https://github.com/AtharvaTilewale/ConformAtlas}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
