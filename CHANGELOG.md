# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-10

### Changed
- **Frames are superposed before PCA.** The MDAnalysis streaming backend now least-squares fits every frame (Kabsch, proper rotations only) on the `--lsq-group` atoms to the structure file and applies the transform to the `--cov-group` atoms before the covariance matrix is built, matching `gmx covar -fit`. Previously `--lsq-group` was only honoured by the GROMACS backend and the Python backend ran Cartesian PCA on unfitted coordinates, so PC1/PC2 contained rigid-body translation and rotation. In `compare`, all systems are superposed on one shared reference (the reference system's structure restricted to the common fit atoms). New options: `--fit/--no-fit`, `--mass-weighted-fit`; YAML keys `pca.fit`, `pca.mass_weighted_fit`. `run_metadata.json` records the superposition settings.
- **Basin segmentation.** Density maxima on the grid border are now valid watershed markers (`exclude_border=False`; a basin at the extreme of PC1/PC2 sits on the border by construction). Every connected region of sampled space receives at least one marker. Watershed runs on the full finite-energy support and isolated occupied bins inherit the nearest basin, so every frame is assigned; previously bins not connected to a marker were reported as "Unassigned" (43.5 % in the shipped demo). Sub-basins separated by a barrier below `--min-barrier` (default 1 kT) are merged, removing the splitting of one broad well into several states by counting noise. Frames lying exactly on the right-most histogram edge are binned instead of dropped.
- **Convergence diagnostics.** Half-vs-half RMSIP pools the first halves of all replicates against the second halves (previously only the first replicate was used) and reports each replicate separately in `rmsip.csv`. Progressive overlap is computed for every fraction instead of hard-coding 1.0 at 100 %. Population-stability fractions are taken per replicate (first f % of every replicate, pooled) instead of slicing the concatenated frame table, where the 25 % point consisted of replicate 1 only. Interpretative notes use qualitative wording that matches the value (a cosine content of 0.02 is "low", not "moderate").
- **Demonstration dataset.** The generated WT/MUT trajectories now encode genuine internal motion (hinge bending vs. twisting of the C-terminal half, switched by a two-state Markov chain) plus random rigid-body rotation and translation per frame. The previous generator shifted the whole peptide along x or y, i.e. its "states" were pure translations that disappear under superposition. The bundled `examples/demo_dataset` files were regenerated.
- A warning is printed when the FEL grid holds fewer than two frames per occupied bin.
- Figures are written once as `<name>.png` and `<name>_unannotated.png`; the duplicate `<name>_non_annotated.png` copies are no longer produced.
- The four-panel convergence figure wraps its interpretative notes so the panels keep their size.

### Added
- `conformatlas.superpose` module: `kabsch_rotation`, `superpose_frame`, `weighted_centroid`, `rmsd`.
- `conformatlas.basins.merge_shallow_basins`; `ConvergenceResults.rmsip_halves_per_replicate`.
- 12 new unit tests (superposition, edge basins, barrier merging, edge binning, per-replicate convergence, note wording); 39 tests in total.

### Removed
- Second author entry in `pyproject.toml`; machine-specific paths in `TESTING_REPORT.md`.

---

## [0.1.1] - 2026-09-19

### Added
- Integrated comprehensive **ReadTheDocs** (`readthedocs.io`) documentation infrastructure:
  - Added `.readthedocs.yaml` (v2 build configuration).
  - Configured Sphinx documentation suite in `docs/conf.py` with MyST Markdown parser, MathJax, autodoc, and `sphinx_rtd_theme`.
  - Created master documentation hub `docs/index.rst` and Python API reference `docs/api.rst`.
  - Added `docs` optional dependencies (`sphinx`, `sphinx-rtd-theme`, `myst-parser`) in `pyproject.toml` and `docs/requirements.txt`.
  - Added ReadTheDocs documentation status badge in `README.md`.
- Hardened GitHub Actions CI/CD workflows:
  - Structured pipeline into dedicated stages: `lint`, multi-OS matrix `test` (Python 3.10–3.12 on Ubuntu and macOS), and `build`.
  - Added defensive integration testing safeguards (`@pytest.mark.skipif`) for headless CI environments lacking binary trajectory drivers.
  - Added `MDAnalysis` and `twine` to developer test dependencies.

### Changed
- Standardized and formatted entire codebase using Ruff.
- Incremented package version to `0.1.1`.

---

## [0.1.0] - 2026-09-19

### Added
- Initial public release of **ConformAtlas** (`conformatlas`).
- Standard PEP 517/621 Python packaging layout (`pyproject.toml`, `src/conformatlas/`).
- Subcommand-based command-line interface (`conformatlas`):
  - `conformatlas analyze`: Single and multi-replicate trajectory PCA and FEL evaluation.
  - `conformatlas compare`: WT vs mutant comparison in shared PCA coordinate space.
  - `conformatlas doctor`: Environment, dependency, and GROMACS diagnostics.
  - `conformatlas init-config`: Interactive and starter YAML template generation.
  - `conformatlas example`: Miniature demonstration dataset generation.
- Python-native 2D Free Energy Landscape (FEL) evaluation with Boltzmann inversion, explicit temperature validation ($T > 0$ K), and zero-occupancy masking.
- Topological watershed segmentation on free energy grids for discrete conformational basin identification.
- Automated extraction of full-atom representative PDB structures closest to basin energy minima with provenance metadata.
- Multi-replicate analysis preserving replicate identity with balanced weighting (`equal-replicate` vs `frames`).
- Shared-PCA space WT vs Mutant comparison with atom selection validation.
- Rigorous sampling and convergence diagnostics:
  - First-half vs second-half subspace overlap (RMSIP).
  - Progressive subspace overlap (25%, 50%, 75%, 100%).
  - Principal component cosine content (diffusion check).
  - State population stability trajectories.
- Statistical uncertainty reporting (replicate Mean, SD, SEM, and 95% Confidence Intervals).
- Standalone 300-DPI publication figures with boundary-aware labels, dynamic headroom, and unannotated variants.
- Self-contained interactive HTML report with embedded figures and parameter manifests.
- Machine-readable execution logs and metadata (`run_metadata.json`, `summary.json`).
- Automated test suite covering unit and end-to-end integration workflows.
