# Installation Guide

## 1. Prerequisites
* **Python**: Version 3.10, 3.11, or 3.12.
* **MDAnalysis**: 2.2 or newer, installed automatically by `pip` together with numpy, scipy, pandas, matplotlib, scikit-image, click, jinja2 and PyYAML.
* **GROMACS**: (Optional) GROMACS 2020 or newer (`gmx` or `gmx_mpi` accessible in PATH) for the `--pca-backend gromacs` route.
* **Package Manager**: `pip` or `conda` (conda-forge).

---

## 2. Standard Installation

### conda (conda-forge)

```bash
conda create -n conformatlas -c conda-forge conformatlas
conda activate conformatlas
```

The conda package contains the core dependencies. The optional features
listed below are separate conda-forge packages:

```bash
conda install -c conda-forge plotly h5py rich weasyprint
```

| Optional feature | conda-forge package |
|---|---|
| `--pdf` | `weasyprint` |
| interactive report figures | `plotly` |
| `--save-hdf5` | `h5py` |
| terminal progress bars | `rich` |

### pip

Install directly into your active Python environment:

```bash
cd /path/to/ConformAtlas
pip install .
```

Optional features are installed as extras:

| Extra | Installs | Enables |
|---|---|---|
| `pdf` | WeasyPrint | `--pdf` (static-figure PDF of the report) |
| `interactive` | plotly ≥ 5.18 | Plotly landscape and NGL viewer in `report.html`, `comparison_interactive.html` |
| `hdf5` | h5py ≥ 3.6 | `--save-hdf5` (`results.h5`), `results_io.save_results/load_results` |
| `progress` | rich ≥ 12 | progress bars on a terminal (a plain bar is used without it) |
| `all` | all of the above | |
| `dev` | pytest, pytest-cov, ruff, build, twine, plotly, h5py, rich, deeptime, pymbar | tests and the validation suite's cross-checks |
| `docs` | sphinx, sphinx-rtd-theme, myst-parser | building this documentation |

```bash
pip install ".[all]"          # every optional feature
pip install -e ".[dev,docs]"  # development
```

The core numerics (PCA, landscapes, bootstrap, MSM, reweighting, residue
analyses) need none of the extras.

---

## 3. Verify Installation

Run the system diagnostics tool:

```bash
conformatlas doctor
```

Example output:
```text
ConformAtlas System Diagnostics
========================================
ConformAtlas Version: 1.0.2
Python Version:     3.11.16 (/home/user/projects/ConformAtlas/.venv/bin/python3.11)

Checking GROMACS installation:
  ✔ Found GROMACS: /usr/bin/gmx (2023.3-Ubuntu_2023.3_1ubuntu3)

Checking Python libraries:
  ✔ numpy        [Required]: version 2.4.6
  ✔ scipy        [Required]: version 1.17.1
  ✔ pandas       [Required]: version 3.0.6
  ✔ matplotlib   [Required]: version 3.11.2
  ✔ click        [Required]: version 8.5.0
  ✔ jinja2       [Required]: version 3.1.6
  ✔ yaml         [Required]: version 6.0.3
  ✔ skimage      [Required]: version 0.26.0
  ✔ MDAnalysis   [Required]: version 2.10.0 — trajectory reading, PCA backend, descriptors, residue analyses
  ✔ weasyprint   [Optional]: version 70.0 — report.pdf (--pdf)
  ✔ plotly       [Optional]: version 7.1.0 — interactive report figures (--interactive)
  ✔ h5py         [Optional]: version 3.16.0 — HDF5 results container (--save-hdf5)
  ✔ rich         [Optional]: version 15.0.0 — progress bars on a terminal
```

Run test suite:
```bash
pytest -v
```
