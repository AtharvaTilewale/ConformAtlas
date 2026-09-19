# Existing Architecture & Audit of ConformAtlas

## 1. Overview of the Existing System

Prior to modernization, ConformAtlas consisted of a set of flat Python scripts and a shell installer:

```text
fel_pca_cli.py        # Top-level CLI (Click-based)
gromacs_pipeline.py   # GROMACS subprocess pipeline
plot_fel.py           # Matplotlib plotting with SciPy cubic interpolation
xpm2dat.py            # GROMACS XPM to 3-column DAT converter (duplicated code)
requirements.txt      # Dependencies: numpy, scipy, matplotlib, pandas, click, pyarmor
install.sh            # Sudo-based installation script copying to /usr/local/lib
```

### Execution Flow in the Original Version

```text
Topology + MD Trajectory
         │
         ▼
[gmx covar] ───► eigenvalue.xvg, eigenvector.trr, average.pdb, covapic.xpm
         │
         ▼
[gmx anaeig] ──► pc1.xvg (first 1, last 1)
         │
         ▼
[gmx anaeig] ──► pc2.xvg (first 2, last 2)
         │
         ▼
[paste | awk] ─► PC1PC2.xvg (time, PC1, PC2)
         │
         ▼
[gmx sham] ────► FES.xpm (Free Energy Surface)
         │
         ▼
[xpm2dat.py] ──► free_energy.dat
         │
         ▼
[plot_fel.py] ─► 2D contour & 3D surface plot (FEL.png) via SciPy griddata (cubic)
```

---

## 2. Identified Deficiencies and Scientific Issues

### 2.1 Scientific & Thermodynamic Limitations
1. **Missing Temperature Parameterization**:
   - GROMACS `gmx sham` was invoked without the `-tsham <temperature>` argument, silently defaulting to GROMACS's internal default (298.15 K).
   - In physical reality:
     $$\Delta G(x, y) = -R T \ln\left[\frac{P(x,y)}{P_{\max}}\right]$$
     Free energy scales linearly with absolute temperature $T$. A calculation at 330 K or 300 K must reflect $T$ explicitly.
2. **Cubic Interpolation Artifacts**:
   - `plot_fel.py` applied `scipy.interpolate.griddata(..., method="cubic")` to an already gridded energy landscape. Cubic interpolation on discrete bin counts can introduce spurious local minima, artificial oscillations, and negative energy overshoots.
3. **Fragile XPM Parsing & Silent Zero-Energy Assignment**:
   - `xpm2dat.py` mapped missing/unknown color symbols with `letter_to_value.get(ch, 0.0)`. Unobserved regions (with zero probability) therefore silently became **0.0 kJ/mol** (the global free energy minimum)!
   - Characters-per-pixel (`cpp`) was not parsed from the XPM header, breaking files with multi-byte colormaps.
4. **Lack of Multi-Replicate Statistics**:
   - Multiple trajectories could not be combined with proper replicate weighting or replicate-level uncertainty (SD, SEM, 95% CI). Treating concatenation as a single trajectory falsely inflates statistical confidence.
5. **No Shared PCA Space for Condition Comparisons**:
   - Comparing WT vs Mutant was impossible without building separate PCA coordinates, which cannot be compared directly along PC1 and PC2.

### 2.2 Software Architecture & Engineering Deficiencies
1. **Duplicated Code in `xpm2dat.py`**:
   - The file literally contained the exact functions duplicated twice in sequence.
2. **Hardcoded Subprocess Executable**:
   - `fel_pca_cli.py` checked for `gmx_mpi` or `gmx`, but `gromacs_pipeline.py` hardcoded `["gmx", ...]`. If only `gmx_mpi` was installed, the pipeline failed.
3. **Plot-Only Mode Inefficiencies**:
   - Plot-only mode (`-p`) checked for GROMACS installation before plotting, even though GROMACS is not needed to plot a `.dat` file.
4. **Shell Pipelines for Data Processing**:
   - Merging `pc1.xvg` and `pc2.xvg` via `paste pc1.xvg pc2.xvg | awk '{print $1, $2, $4}' > PC1PC2.xvg` relied on Unix shell binaries rather than cross-platform Python manipulation.
5. **Unstructured Output & Fixed Directory**:
   - All output was hardcoded into `FEL_output/` without machine-readable run metadata (`run_metadata.json`).
6. **Logging Handler Leaks**:
   - `setup_logging` attached handlers to the root logger without clearing previous handlers, causing duplicated log lines on repeated calls.
7. **Packaging & Security Issues**:
   - No `pyproject.toml` or setuptools packaging.
   - Installation script required `sudo` and installed files directly to `/usr/local/lib/conformatlas`.
   - `pyarmor` was listed as a mandatory dependency in `requirements.txt`.
8. **No Automated Testing Suite**:
   - Zero unit or integration tests existed.

---

## 3. Modernization Strategy

1. **New Package Structure (`src/conformatlas/`)**:
   - Proper PEP 517/621 packaging with `pyproject.toml` and entrypoint `conformatlas`.
   - Separation into modular components: `cli`, `config`, `models`, `trajectory`, `gromacs`, `pca`, `fel`, `basins`, `representatives`, `convergence`, `uncertainty`, `comparison`, `plotting`, `report`, `xpm`, `utils`.
2. **Python-Native FEL Calculation**:
   - Histogram-based probability distribution with zero-count masking ($P = 0 \implies \Delta G = \text{NaN}$).
   - Explicit temperature validation and Boltzmann weighting:
     $$\Delta G = -R T \ln(P / P_{\max}), \quad R = 0.008314462618\ \text{kJ}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$$
3. **Automated Basin Segmentation & Representation**:
   - Topological watershed segmentation on the FEL grid to identify discrete energy basins.
   - Extraction of real trajectory frames corresponding to basin minima as PDB files.
4. **Multi-Replicate & Comparison Support**:
   - Common PCA basis construction for multi-replicate and WT vs Mutant trajectories.
   - Replicate-level uncertainty (SD, SEM, 95% CI) and PCA convergence metrics (RMSIP, cosine content, progressive overlap).
5. **Traceable Outputs & Automated Reports**:
   - HTML report with embedded figures and parameter summaries.
   - Structured JSON outputs (`run_metadata.json`, `summary.json`).
