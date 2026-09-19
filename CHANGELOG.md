# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
