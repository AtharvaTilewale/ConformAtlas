# Installation Guide

## 1. Prerequisites
* **Python**: Version 3.10, 3.11, or 3.12.
* **GROMACS**: (Optional but recommended) GROMACS 2020 or newer (`gmx` or `gmx_mpi` accessible in PATH).
* **Package Manager**: Standard `pip`.

---

## 2. Standard Installation

Install directly into your active Python environment:

```bash
cd /path/to/ConformAtlas
pip install .
```

For development (editable mode with testing dependencies):

```bash
pip install -e ".[dev]"
```

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
ConformAtlas Version: 0.1.1
Python Version:     3.11.7 (/home/.../python)

Checking GROMACS installation:
  ✔ Found GROMACS: /usr/local/gromacs-2024.4/bin/gmx (GROMACS 2024.4)

Checking Python libraries:
  ✔ numpy        [Required]: version 1.26.4
  ✔ scipy        [Required]: version 1.13.1
  ✔ pandas       [Required]: version 3.0.5
  ✔ matplotlib   [Required]: version 3.10.7
  ✔ click        [Required]: version 8.4.2
  ✔ jinja2       [Required]: version 3.1.3
  ✔ yaml         [Required]: version 6.0.2
  ✔ skimage      [Required]: version 0.22.0
  ✔ MDAnalysis   [Optional]: version 2.7.0
  ○ weasyprint   [Optional]
```

Run test suite:
```bash
pytest -v
```
