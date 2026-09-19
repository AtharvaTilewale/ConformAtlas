# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
