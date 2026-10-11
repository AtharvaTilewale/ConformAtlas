<p align="center">
  <img src="https://raw.githubusercontent.com/AtharvaTilewale/ConformAtlas/main/docs/assets/conformatlas-logo-on-white.png" alt="ConformAtlas" width="560" />
</p>

<h3 align="center">Mapping conformational landscapes from molecular simulations</h3>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10 | 3.11 | 3.12"></a>
  <a href="https://github.com/AtharvaTilewale/ConformAtlas/actions/workflows/ci.yml"><img src="https://github.com/AtharvaTilewale/ConformAtlas/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://conformatlas.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/conformatlas/badge/?version=latest" alt="Documentation"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-you-get">Outputs</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#citation">Citation</a>
</p>

---

**ConformAtlas** turns molecular dynamics trajectories into a quantitative map of conformational space: principal components, a free-energy landscape with error bars, the metastable states on it, the barriers and rates between them, and a verdict on whether the sampling supports those numbers. It runs as a single command (`conformatlas analyze`) and as a Python library, reads anything [MDAnalysis](https://www.mdanalysis.org/) can open, and ships with a [validation suite](docs/benchmarks.md) that checks every numerical component against analytical results or an independent implementation.

<p align="center">
  <img src="https://raw.githubusercontent.com/AtharvaTilewale/ConformAtlas/main/docs/assets/readme-landscape.png" alt="Free-energy landscape of influenza neuraminidase bound to zanamivir" width="860" />
  <br>
  <sub>Influenza A/H1N1 neuraminidase bound to zanamivir, 2001 frames at 300 K: free-energy landscape on the first two Cα principal components with three watershed states (48 / 41 / 11 %). Unmodified <code>figures/fel_2d_3d.png</code> from a single <code>conformatlas analyze</code> run; see the <a href="docs/tutorial_neuraminidase.md">tutorial</a>.</sub>
</p>

## Features

| | |
|---|---|
| **Decompositions** | Cartesian PCA after Kabsch superposition, backbone / side-chain / nucleic dihedral PCA, TICA with implied timescales; landscapes on any component pair, on RMSD vs R<sub>g</sub>, or on user-supplied collective variables |
| **Free-energy landscape** | Boltzmann inversion at the simulation temperature (kJ/mol), automatic grid resolution, unsampled bins left empty; agrees with `gmx sham` on the same input |
| **Uncertainty** | Circular block bootstrap within replicates: standard deviation and 95 % CI of every bin, state population, state free energy and barrier |
| **States and kinetics** | Watershed segmentation with barrier-based merging, saddle-point barriers, transition matrix, dwell times, mean first-passage times; optional Markov state model (microstates, reversible estimator, implied-timescale scan, Chapman–Kolmogorov test, PCCA+) |
| **Convergence** | Statistical inefficiency and effective sample size, equilibration detection, first/second-half subspace overlap, cosine content, population stability per replicate |
| **Structures** | Representative and average structures per state, RMSF profiles, per-residue mode contributions, NMWiz mode files, dynamic cross-correlation maps, ligand–residue contacts |
| **Comparisons** | Any number of systems projected into a shared PCA space: ΔΔG maps with uncertainty, population differences with bootstrap confidence intervals and p-values, landscape overlap metrics |
| **Enhanced sampling** | `conformatlas reweight` turns PLUMED metadynamics or umbrella-sampling output into per-frame weights (final bias, Tiwary–Parrinello, WHAM, MBAR) for a reweighted landscape |
| **Scale** | Streaming two-pass PCA engine with memory independent of trajectory length, multi-replicate aware, HDF5 results container, re-analysis of saved results without re-reading trajectories |
| **Reporting** | Self-contained HTML report with interactive landscape and 3-D structure viewer, publication-ready PNG/SVG/PDF figures, optional PDF report |

## Installation

```bash
pip install conformatlas            # core (includes MDAnalysis)
pip install "conformatlas[all]"     # + interactive figures, HDF5, progress bars, PDF report
```

From source:

```bash
git clone https://github.com/AtharvaTilewale/ConformAtlas.git
cd ConformAtlas
pip install -e ".[dev]"
conformatlas doctor                 # checks Python dependencies and GROMACS (optional)
```

Requires Python ≥ 3.10. GROMACS is optional and only needed for `--pca-backend gromacs`.

## Quick start

**Analyse one system** (any number of replicates):

```bash
conformatlas analyze -s topology.gro -f rep1.xtc -f rep2.xtc -f rep3.xtc -T 300 -o results/WT
```

**Protein–ligand complex**, discarding the equilibration phase and adding a Markov state model:

```bash
conformatlas analyze -s md.gro -f md_noPBC.xtc -n index.ndx \
    --cov-group C-alpha --lig ZMR -T 300 \
    --equilibration auto --bootstrap 200 --msm \
    -o results/complex
```

**Compare conditions** (wild type vs mutant, apo vs holo, …) in a shared PCA space:

```bash
conformatlas compare --config comparison.yaml -o results/comparison
```

**Try it** on the bundled synthetic two-state peptide:

```bash
conformatlas example -o demo
conformatlas analyze -s demo/WT/topology.pdb -f demo/WT/rep1.xtc -f demo/WT/rep2.xtc -T 300 --bins 32 --smooth-sigma 1.0 -o demo/results
```

Every option can live in a YAML file (`conformatlas init-config --type analyze -o run.yaml`). Trajectories must be made whole across periodic boundaries first (`gmx trjconv -pbc mol`). The full option reference is in [docs/cli.md](docs/cli.md).

From Python:

```python
from conformatlas import api

ens  = api.load_trajectories("topology.gro", ["rep1.xtc", "rep2.xtc"], group="C-alpha")
pcs  = api.pca(ens)
land = api.landscape(pcs, temperature=300)
st   = api.states(land)
fig, ax, _ = api.plot_landscape(land, states=st.states)
```

## What you get

```text
results/
├── report.html         # interactive landscape, structure viewer, interpretation
├── summary.json        # state populations and minima, convergence assessment
├── pca/                # eigenvalues, projections, eigenvectors, modes.nmd, mode contributions
├── fel/                # free-energy grid, probability, bootstrap SD and CI
├── states/             # basins, frame assignments, barriers, state ΔG, representative structures
├── statistics/         # populations with block SD/SEM and bootstrap 95 % CI
├── kinetics/  msm/     # transition matrices, dwell times, MFPT; MSM diagnostics (--msm)
├── convergence/        # RMSIP, N_eff, equilibration, cosine content, population stability
├── residues/           # DCCM, state differences, ligand contacts (--residue-analysis)
└── figures/            # 2-D/3-D landscapes, uncertainty maps, kinetics, MSM, DCCM panels
```

Units: Å, ps, kJ/mol. The complete file manifest is in [docs/outputs.md](docs/outputs.md).

## Validation

`validation/validate.py` compares every numerical component with an analytical result or an independent implementation — MDAnalysis, GROMACS (`covar`, `anaeig`, `sham`), pymbar, deeptime, SciPy and ProDy — and writes the report that is published as [docs/benchmarks.md](docs/benchmarks.md) (147/147 checks). Components that are conventions rather than derivations (convergence thresholds, grid heuristics, MSM lag selection) are labelled as such in [docs/methods.md](docs/methods.md).

Boltzmann inversion assumes an unbiased equilibrium ensemble; for biased simulations use `conformatlas reweight` and `--weights-file`. See the [FAQ](docs/faq.md) for this and other pitfalls.

## Documentation

Full documentation: **[conformatlas.readthedocs.io](https://conformatlas.readthedocs.io)**

| Getting started | Reference | Background |
|---|---|---|
| [Installation](docs/installation.md) | [Command line](docs/cli.md) | [Methods](docs/methods.md) |
| [Quickstart](docs/quickstart.md) | [Python API](docs/python_api.md) | [Interpreting results](docs/interpretation.md) |
| [Tutorial: neuraminidase–zanamivir](docs/tutorial_neuraminidase.md) | [Output files](docs/outputs.md) | [Convergence diagnostics](docs/convergence.md) |
| [FAQ](docs/faq.md) | [Comparing systems](docs/comparison.md) | [Validation and benchmarks](docs/benchmarks.md) |

## Citation

```bibtex
@software{conformatlas2026,
  author  = {Tilewale, Atharva and Patel, Dhaval},
  title   = {ConformAtlas: mapping conformational landscapes from molecular simulations},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/AtharvaTilewale/ConformAtlas}
}
```

A machine-readable citation is in [CITATION.cff](CITATION.cff).

## Contributing

Bug reports, feature requests and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Please run `ruff check`, the test suite and `validation/validate.py` before opening a pull request.

## Authors and license

Developed by **Atharva Tilewale** and **Dhaval Patel**. Released under the [MIT License](LICENSE).
