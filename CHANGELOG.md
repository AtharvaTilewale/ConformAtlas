# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-10-12

### Fixed
- **Streaming PCA engine on macOS.** With `--n-workers` greater than 1 the worker processes were forked on every platform. On macOS a forked worker could crash with a segmentation fault (the garbage collector closed an MDAnalysis XDR reader inherited from the parent), so `--engine streaming --n-workers N` aborted intermittently. Workers are now forked on Linux only and started with `spawn` on macOS and Windows, and unreachable objects are collected before the pool starts. The default `--n-workers 1` was not affected.

### Added
- The Zenodo concept DOI 10.5281/zenodo.22847857 (covers every release, resolves to the latest) in `CITATION.cff`, the README (badge and BibTeX) and the documentation landing page, and as a project link on PyPI.
- The conda-forge recipe in the v1 `recipe.yaml` format under `conda-recipe/`, as submitted to conda-forge.

## [1.0.1] - 2026-10-12

### Fixed
- **Console output no longer depends on the platform code page.** With stdout or stderr redirected (CI logs, `> file`, pipes) Python encodes with the locale code page, which on Windows (cp1252) cannot represent `ΔG`, `Å`, `✔` or `✘`: `conformatlas doctor`, `example`, `analyze --help` and the progress output of `analyze` aborted with `UnicodeEncodeError`. Redirected streams are now switched to UTF-8 before anything is printed, and a terminal keeps its encoding but replaces unencodable characters instead of failing. Remaining text-mode file reads and writes name their encoding explicitly. Found by the Windows build of the conda-forge recipe.

### Added
- Windows (Python 3.12) in the continuous-integration matrix.

## [1.0.0] - 2026-10-11

First release after 0.1.1; all changes are described relative to 0.1.1.

### Changed
- **Frames are superposed before PCA.** The Python (MDAnalysis) backend least-squares fits every frame (Kabsch, proper rotations only) on the `--lsq-group` atoms to the reference structure and applies the transform to the `--cov-group` atoms before the covariance matrix is built, matching `gmx covar -fit`. In 0.1.1 `--lsq-group` was honoured only by the GROMACS backend and the Python backend ran Cartesian PCA on unfitted coordinates, so PC1/PC2 contained rigid-body translation and rotation. In `compare` all systems are superposed on one shared reference (the reference system's structure restricted to the common fit atoms). New options `--fit/--no-fit`, `--mass-weighted-fit`; YAML keys `pca.fit`, `pca.mass_weighted_fit`. With a `.tpr` topology (no reference coordinates) the first trajectory frame is the reference.
- **'Backbone' follows the GROMACS definition (N, CA, C)** in both backends; the Python backend used the MDAnalysis `backbone` keyword (N, CA, C, O), so the two backends analysed different atoms and their eigenvalues differed by ~50 %. New group name `MainChain` = N, CA, C, O. Results obtained with the default group differ slightly from 0.1.1.
- **Basin segmentation.** Density maxima on the grid border are valid watershed markers; every connected region of sampled space receives at least one marker; watershed runs on the full finite-energy support and isolated occupied bins inherit the nearest basin, so every frame is assigned (in 0.1.1, 43.5 % of the demo frames were "Unassigned"). Sub-basins separated by a barrier below `--min-barrier` (default 1 kT) are merged. Frames lying exactly on the right-most histogram edge are binned instead of dropped. Frames in sub-threshold minor basins are reported as "Unassigned / Minor".
- **Convergence diagnostics.** Half-vs-half RMSIP pools the first halves of all replicates against the second halves (0.1.1 used the first replicate only) and reports each replicate in `rmsip.csv`; progressive overlap is computed for every fraction (0.1.1 hard-coded 1.0 at 100 %); population stability is computed on the first f % of every replicate (0.1.1 sliced the concatenated table, so the 25 % point contained replicate 1 only); cosine content is evaluated on the mean-centred series as in Hess's definition; interpretative notes use wording that matches the value. Fewer than 50 effective samples force the sampling verdict to "Insufficient evidence".
- **Demonstration dataset.** `conformatlas example` and the bundled `examples/demo_dataset` encode genuine internal motion (hinge bending vs. twisting of the C-terminal half, switched by a two-state Markov chain) plus random rigid-body motion per frame; the 0.1.1 generator translated the whole peptide, i.e. its "states" disappeared under superposition.
- `--bins` accepts `auto` (Freedman–Diaconis rule, clipped to 16–128 bins; now the default) or an integer; a warning is printed when the grid holds fewer than two frames per occupied bin.
- `--lig` is functional: the ligand's atoms join the analysis group and the ligand RMSD after protein superposition is reported.
- Figure axis labels follow the collective variables in use (PC, dPC, IC, RMSD/Rg, user labels; nm for the GROMACS backend, Å for the Python backend); CSV columns keep `PC1..PCn` for every decomposition.
- Figures are written once as `<name>.png` and `<name>_unannotated.png` (the duplicate `<name>_non_annotated.png` copies are gone); the convergence figure wraps its notes so the panels keep their size; population error bars are clipped to [0, 100] %.
- `cli.analyze` is a thin wrapper over `conformatlas.pipeline.run_analysis`, the full single-system workflow as a library function.
- CLI: the banner is printed only on a terminal and never for `--help`/`--version`; `--quiet`; friendly messages instead of tracebacks for malformed YAML (with line/column), missing files, unknown group names (available groups are listed) and missing optional dependencies (with the `pip install` command); exit codes 0 (success), 1 (analysis error), 2 (usage error).
- Common-atom mapping for comparisons is chain-aware (`chain/residue/atom` keys with automatic fallback when chain labels differ), so repeated residue numbers in multi-chain systems are no longer collapsed.
- Project logo and brand assets (`docs/assets/`, generated by `docs/assets/make_logo.py`); README header uses the logo; Sphinx documentation carries the logo and favicon.

### Added

#### Inputs, time window and configuration
- Time window `-b/--begin`, `-e/--end` (ps); GROMACS index files `-n/--index` for every group option and for comparisons (same atom numbering required across systems); group names may also be MDAnalysis selection strings.
- Input validation before the run: structure/trajectory atom-count mismatch, broken Cα chains (periodic-boundary artefacts), memory estimate with a warning above 8 GB.
- YAML configuration for `analyze` (`conformatlas init-config --type analyze`, `conformatlas analyze --config run.yaml`; command-line flags override the file). Comparison YAML accepts `begin_ps`, `end_ps`, `index_file`, `bins: auto`, `n_bootstrap`, `seed`, `bootstrap_mode` and the figure options.
- `run_metadata.json` records every option, the superposition settings, the time window, equilibration, decomposition, CVs, weights, bootstrap and representative settings, all warnings, and the software environment (Python, NumPy, SciPy, pandas, matplotlib, scikit-image, click, Jinja2, PyYAML, MDAnalysis, WeasyPrint, GROMACS, platform; module `conformatlas.provenance`). `summary.json` carries uncertainties and N_eff.

#### Decompositions and collective variables
- `--pca-type dihedral`: dihedral PCA on sin/cos of backbone φ/ψ, optionally with side-chain χ₁/χ₂ or nucleic-acid backbone/glycosidic torsions (`--dihedral-set backbone|backbone+chi|chi|nucleic`).
- `--pca-type tica --tica-lag N`: time-lagged independent component analysis with implied timescales (lags computed within replicates; PCA pre-reduction when frames do not greatly outnumber coordinates).
- `--fel-pcs i,j` (any two retained components), `--n-components`, `--cv rmsd-rg`, and external CVs `--cv file --cv-file … --cv-cols … --cv-labels …` (PLUMED COLVAR, `.xvg`, CSV; one file per trajectory).
- Per-frame statistical weights `--weights-file` for histogram, populations and bootstrap (kinetics are skipped for weighted frames).
- SVD (dual) route for the covariance eigenproblem when 3N exceeds the number of frames.

#### Uncertainty and sampling quality
- **Block bootstrap** (`--bootstrap N`, default 200; `--seed`): circular blocks within each replicate, block length from the statistical inefficiency; standard deviation and 95 % interval of ΔG per bin (`fel/free_energy_bootstrap_*.npy`, `figures/fel_uncertainty.png`), of state populations (merged into `statistics/state_populations.csv`), state free energies (`states/state_free_energies.csv`) and barriers; 1-D projections of the landscape with bootstrap bands (`figures/fel_1d.png`).
- `--bootstrap-mode resegment`: the watershed segmentation is re-run on every resample; reports the distribution of the number of states and per-state population intervals that include segmentation uncertainty.
- Statistical inefficiency g and effective sample size N_eff per replicate (exact FFT autocorrelation estimator; `convergence/effective_samples.csv`).
- Equilibration detection by the maximum-N_eff criterion (Chodera 2016) on the fit RMSD or on the landscape CVs (`--equilibration-on rmsd|cv`), reported in `convergence/equilibration.csv` and applied with `--equilibration auto`.
- Per-replicate landscapes on the shared grid as a convergence view.

#### States, barriers, kinetics and Markov state models
- Saddle-point barriers between adjacent basins (`states/barriers.csv`).
- Lag-1 kinetics within replicates: transition counts and probabilities, dwell times per visit with censoring flags, Markov mean first-passage times (`kinetics/`, `figures/state_timeseries.png`).
- **Markov state models** (`--msm`; `--msm-lag`, `--msm-lags`, `--msm-clusters`, `--msm-clustering kmeans|regular-space`, `--msm-dmin`, `--msm-components`, …): microstates by k-means or regular-space clustering in the retained components, sliding-window counts within replicates, reversible maximum-likelihood transition matrix, implied-timescale scan with a reported (never auto-applied) plateau lag, Chapman–Kolmogorov test, PCCA+ metastable sets, mean first-passage times between sets, and an overlap table between metastable sets and watershed basins (`msm/`, with figures).

#### Enhanced sampling
- `conformatlas reweight metad`: metadynamics reweighting from PLUMED HILLS + COLVAR (final bias, or the time-dependent c(t) correction of Tiwary & Parrinello), writing weight files for `--weights-file`.
- `conformatlas reweight umbrella`: umbrella-sampling weights and PMF by MBAR or WHAM (`pmf.csv`). Module `conformatlas.reweighting` with PLUMED COLVAR/HILLS readers.

#### Structures, residues and descriptors
- Representative structures: `--n-representatives N`, `--representative minimum|medoid`; per-state `average.pdb`, `rmsf_atoms.csv`, `rmsf_residues.csv`; `states/state_ensemble_summary.csv`.
- Mode interpretation: per-atom and per-residue mode contributions (`pca/mode_contributions_*.csv`, `figures/mode_contributions.png`), `pca/modes.nmd` (VMD NMWiz / ProDy), `pca/average.pdb`, `pca/PC*_extremes.pdb`, `pca/eigenvectors.npy`.
- Descriptors: backbone RMSD, radius of gyration, RMSF per atom and residue, and ligand RMSD with `--lig` (`descriptors/`, `figures/descriptors.png`).
- `--residue-analysis`: dynamic cross-correlation maps (overall and per state), per-residue differences between states (ΔRMSF, displacement of state means), ligand–residue contact persistence per state and binding-site RMSF (`residues/`, with figures).

#### Comparisons
- `compare` with any number of systems: ΔΔG maps with bootstrap standard deviation against the reference system (`comparison/fel_diff_<SYS>_vs_<REF>.csv/.npy`, `figures/ddg_map_*.png`, hatched where |ΔΔG| < 2 SD); population differences with block-bootstrap interval and p-value and Welch's t-test across replicates (when each system has at least two); state ΔΔG with intervals; landscape overlap (Bhattacharyya coefficient, Hellinger distance, Jensen–Shannon divergence, sampled-bin Jaccard; `fel_overlap.csv`); RMSIP between each system's own essential subspace (`subspace_overlap.csv`); `comparison_summary.json`; bootstrap error bars on the population figures; side-by-side interactive comparison landscape.

#### Performance and programmatic use
- Streaming PCA engine (`--engine auto|memory|streaming`): chunked two-pass covariance accumulation with per-frame superposition that never holds the whole trajectory in memory, numerically identical to the in-memory path; `--n-workers` (one process per replicate), `--precision float32`, memory-mapped projection storage; `auto` switches to streaming above a memory threshold.
- `--from <results_dir>`: re-bin, re-segment, re-bootstrap or build an MSM from an existing results directory without re-reading trajectories or recomputing PCA.
- `--save-hdf5`: single-file results container `results.h5` (`conformatlas.results_io`: `save_results`, `load_results`, `load_results_dir`, `rebin`).
- TTY-aware progress bars (`rich` when installed; silent in pipes and logs).
- Python API `conformatlas.api` (`load_trajectories`, `pca`, `landscape`, `states`, `kinetics`, `convergence`, `analyze`, `plot_landscape`) returning in-memory objects; `conformatlas.mda_analysis.ConformAtlasPCA`, an MDAnalysis `AnalysisBase` implementation; notebook `examples/notebooks/neuraminidase_walkthrough.ipynb`; streaming benchmark in `benchmarks/`.

#### Report and figures
- Interactive HTML report: Plotly landscape with hover ΔG/probability/state, basin outlines, time-coloured per-replicate paths and a time slider, 1-D profile panel; embedded NGL viewer of the state representatives and the average structure coloured by per-residue RMSF; all JavaScript and structures are inlined (no external resources; `--plotly-js cdn` for a smaller file; `--interactive/--no-interactive`).
- Interpretive sentences next to every table (N_eff and its consequence, RMSIP wording, interval widths, barriers in kT), a "Limits of this analysis" section and a warnings box; `report_interpretation.json`. Optional `report.pdf` (`--pdf`; static figures, stated in the PDF).
- Figure control: `--fig-format png|svg|pdf|all` (SVG/PDF are vector), `--colormap`, `--font-scale`, `--fig-dpi`, `--path-overlay` (trajectory path on the static landscape).

#### Validation, tests and documentation
- Scientific validation suite `validation/validate.py` (report `validation/VALIDATION.md`): checks against analytical results and independent implementations — GROMACS `covar`, `anaeig`, `sham` and `gyrate`; MDAnalysis PCA, QCP superposition, RMSD/Rg/RMSF, Ramachandran and Janin dihedrals; pymbar (statistical inefficiency, equilibration, MBAR); deeptime (TICA, reversible MSM estimator, PCCA+); `plumed sum_hills`; SciPy; ProDy; analytical double wells, Gaussians and Markov chains; coverage of the bootstrap intervals and type-I error of the comparison test.
- Unit and integration tests, including a real-protein fixture (influenza neuraminidase Cα + zanamivir heavy atoms, 81 frames, `tests/data/h1n1_ca/`).
- Documentation: real-data tutorial (`docs/tutorial_neuraminidase.md`), FAQ of common pitfalls, validation/benchmark page generated from the validation results, Python API guide, methods sections for every new component with a "validated vs heuristic" summary; Sphinx builds with `-W`.
- Project: `CONTRIBUTING.md` (including how to add a validation check), Contributor Covenant 2.1, issue forms and pull-request template, Codecov upload and a documentation job in CI, conda-forge recipe draft (`conda-recipe/`), JOSS paper draft (`paper/`), `.zenodo.json`, `requirements-lock.txt`.
- Optional dependency groups: `pdf` (WeasyPrint), `interactive` (Plotly), `hdf5` (h5py), `progress` (rich), `all`.

### Fixed
- **GROMACS PCA backend**: `gmx anaeig -proj` writes one `&`-separated data set per eigenvector; the parser read them as a single column, so every component after the first was lost. Projections are now parsed per set.
- **GROMACS PCA backend** built the covariance from the first trajectory only and projected the other replicates onto it; all trajectories are now concatenated (`gmx trjcat -cat`) before `gmx covar`, and the GROMACS and Python backends give identical explained variance.
- XPM parser: `gmx sham` writes bin *edges* (width + 1 values) on the axis lines, which were rejected with a size error; edges are converted to centres, so `--fel-backend gromacs` works with current GROMACS.
- Representative-structure extraction indexed the raw trajectory with the streamed frame index, which is wrong under `--stride`; frames are now located by time.

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
