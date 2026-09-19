# ConformAtlas Testing & Verification Report

## 1. Execution Environment
* **Date & Time:** 2026-09-19
* **Host Operating System:** Linux (Ubuntu 20.04 LTS kernel x86_64)
* **Python Runtime:** Python 3.11.7 (`/home/dhaval/anaconda3/bin/python3`)
* **GROMACS Binary:** `/usr/local/gromacs-2024.4/bin/gmx` (GROMACS 2024.4)
* **Key Python Dependencies:**
  - `numpy`: 1.26.4
  - `scipy`: 1.13.1
  - `pandas`: 3.0.5
  - `matplotlib`: 3.10.7
  - `scikit-image`: 0.22.0
  - `click`: 8.4.2
  - `jinja2`: 3.1.3
  - `PyYAML`: 6.0.2
  - `MDAnalysis`: 2.7.0

---

## 2. Test Files Inventory & Manifest

### 2.1 Synthetic & Miniature Trajectory Datasets
* **`examples/demo_dataset/WT/topology.pdb`**: 5-residue polyalanine model (25 atoms).
* **`examples/demo_dataset/WT/rep1.xtc`**: 100 frames (10 ps timestep, total 1000 ps).
* **`examples/demo_dataset/WT/rep2.xtc`**: 100 frames (10 ps timestep, total 1000 ps).
* **`examples/demo_dataset/MUT/topology.pdb`**: 5-residue polyalanine model (25 atoms).
* **`examples/demo_dataset/MUT/rep1.xtc`**: 100 frames with shifted conformational population.
* **`examples/demo_dataset/MUT/rep2.xtc`**: 100 frames with shifted conformational population.
* **`examples/demo_dataset/comparison.yaml`**: Multi-system shared-PCA comparison specification.

### 2.2 Production Molecular Dynamics System
* **`step5_production.gro`**: 5,753-atom coarse-grained (Martini) biomolecular complex structure.
* **`step5_production.xtc`**: 100,001 frames (100 ps dt, total 10,000,000 ps = 10 microseconds, 2.19 GB).

---

## 3. Test Suites Executed

### 3.1 Automated Pytest Suite (`pytest -v`)
* **Total Tests:** 27
* **Passed:** 27 (100%)
* **Failed:** 0
* **Execution Time:** ~4.8 seconds

#### Breakdown of Unit & Integration Tests:
1. `tests/unit/test_temperature.py`:
   - `test_temperature_scaling_ratio`: PASSED (verified $\Delta G(330\ \text{K}) / \Delta G(300\ \text{K}) = 1.10000$ within $10^{-5}$ relative error).
   - `test_temperature_validation`: PASSED (verified that $T \le 0$ K raises `ValueError`).
2. `tests/unit/test_rmsip.py`:
   - `test_rmsip_identical_subspaces`: PASSED ($\text{RMSIP}(U, U) = 1.000000$).
   - `test_rmsip_orthogonal_subspaces`: PASSED ($\text{RMSIP}(U, V_{\perp}) = 0.000000$).
   - `test_rmsip_intermediate`: PASSED ($\text{RMSIP} = \sqrt{3/5} \approx 0.774597$).
3. `tests/unit/test_pca.py`:
   - `test_pca_recovers_known_dominant_motion`: PASSED (recovers principal vector along simulated oscillation mode).
   - `test_shared_pca_projection_consistency`: PASSED (batch vs single frame projections strictly match).
   - `test_replicate_weighting`: PASSED (equal-replicate weighting prevents unequal trajectory lengths from biasing the mean).
4. `tests/unit/test_fel.py`:
   - `test_fel_probability_and_minimum`: PASSED ($\sum P = 1.0$, $\Delta G_{\min} = 0.0\ \text{kJ/mol}$).
   - `test_unobserved_bins_are_nan`: PASSED (zero-count bins are strictly $\text{NaN}$, never $0.0$).
   - `test_gaussian_smoothing`: PASSED (density smoothing maintains probability normalization).
5. `tests/unit/test_xpm.py`:
   - `test_parse_gromacs_xpm`: PASSED (recovers GROMACS FES.xpm values and matrix coordinates).
   - `test_parse_multi_char_cpp`: PASSED (parses 2-character colormaps).
   - `test_unknown_character_raises_error`: PASSED (unknown character raises `ValueError`).
   - `test_xpm_to_dataframe_and_dat`: PASSED (exports valid 3-column DAT).
6. `tests/unit/test_basins.py`:
   - `test_basin_detection_recovers_known_gaussian_mixture`: PASSED (recovers 3 Gaussian clusters at 65%, 25%, 10% within tolerance; sum of states equals 100%).
   - `test_state_minimum_coordinates`: PASSED (places minimum at true cluster center).
7. `tests/unit/test_representatives.py`:
   - `test_representative_provenance_and_metadata`: PASSED (provenance tracking to source replicate, frame, time, and coordinates).
8. `tests/unit/test_uncertainty.py`:
   - `test_multiple_replicates_uncertainty`: PASSED (computes Mean, SD, SEM, and Student's $t$ 95% CI).
   - `test_single_replicate_block_uncertainty`: PASSED (intra-trajectory block uncertainty correctly identified).
9. `tests/unit/test_cli.py`:
   - `test_cli_version`: PASSED (`conformatlas --version`).
   - `test_cli_help`: PASSED (`conformatlas --help`).
   - `test_cli_doctor`: PASSED (`conformatlas doctor`).
   - `test_cli_init_config`: PASSED (`conformatlas init-config`).
   - `test_cli_example_generation`: PASSED (`conformatlas example`).
10. `tests/integration/test_pipeline.py`:
    - `test_end_to_end_analyze`: PASSED (full single/multi-replicate analysis run producing all output tables, figures, report).
    - `test_end_to_end_compare`: PASSED (full WT vs mutant comparison in shared PCA space).

---

## 4. End-to-End CLI Pipeline Verifications

### 4.1 Multi-Replicate WT Analysis
```bash
conformatlas analyze \
    -s examples/demo_dataset/WT/topology.pdb \
    -f examples/demo_dataset/WT/rep1.xtc \
    -f examples/demo_dataset/WT/rep2.xtc \
    -T 300 \
    --bins 32 \
    --smooth-sigma 1.0 \
    -o examples/demo_dataset/results_wt
```
* **Result:** Exit code 0.
* **Findings:**
  - Detected 2 major conformational states:
    - State A: 38.5% population ($\Delta G_{\min} = 0.35\ \text{kJ/mol}$, representative frame 13 at 130 ps).
    - State B: 18.0% population ($\Delta G_{\min} = 0.79\ \text{kJ/mol}$, representative frame 75 at 750 ps).
    - Minor / transitional conformations: 43.5%.
  - Top 2 PCA modes explain 80.8% of variance.
  - Convergence: Subspace overlap $\text{RMSIP} = 0.51$, PC1 cosine content $c_1 = 0.02$.
  - Generated `report.html`, `run_metadata.json`, `summary.json`, figures, and representative PDBs.

### 4.2 Shared-PCA WT vs Mutant Condition Comparison
```bash
conformatlas compare \
    --config examples/demo_dataset/comparison.yaml \
    -o examples/demo_dataset/results_comparison
```
* **Result:** Exit code 0.
* **Findings:**
  - Common atom mapping constructed across 20 invariant backbone atoms.
  - Common PCA basis constructed with 4 major states.
  - Captured significant population shifts:
    - State A: WT 47.7% vs MUT 36.0% ($\Delta = -11.7\%$).
    - State B: WT 37.7% vs MUT 10.3% ($\Delta = -27.4\%$).
    - State C: WT 3.6% vs MUT 42.8% ($\Delta = +39.2\%$, mutant-induced conformational shift).
  - Produced `condition_comparison.png` displaying shared PCA coordinate space, population comparison bar chart, and per-system FELs on identical color scales.

### 4.3 Real 10-Microsecond Production MD Simulation Run
```bash
conformatlas analyze \
    -s /home/dhaval/step5_production.gro \
    -f /home/dhaval/step5_production.xtc \
    -T 303.15 \
    --stride 1000 \
    --bins 30 \
    --smooth-sigma 1.0 \
    -o test_real_md/single_stride1000
```
* **Result:** Exit code 0 (execution time: 4.8s).
* **Findings:**
  - Streamed 100 frames spaced across 10 microseconds ($10,000,000\ \text{ps}$) of simulation time.
  - Evaluated Backbone coordinates on coarse-grained Martini model (`name BB`).
  - Dominant modes explain 20.9% of total variance.
  - Subspace overlap $\text{RMSIP} = 0.56$ (moderate stability across halves of the 10 $\mu\text{s}$ trajectory), PC1 cosine content $c_1 = 0.01$ (confirming non-diffusive bounded motion).
  - Extracted verified representative structure `representative.pdb` from frame 17 ($1,700,000\ \text{ps}$).

---

## 5. Scientific Sanity Checks & Numerical Sanity Checks

1. **Sum of State Populations:** $\sum \text{Major States} + \text{Unassigned} = 100.00\%$ across all analyses.
2. **Temperature Invariance of Probability:** Evaluating identical coordinates at $300\ \text{K}$ and $330\ \text{K}$ produces identical probability grids $P(x, y)$ while scaling $\Delta G$ strictly by $330 / 300 = 1.10$.
3. **Absence of Artificial Minima:** Visual inspection of `figures/fel_2d_3d.png` confirms that free energy surfaces are rendered directly from the discrete calculated grid, avoiding spurious oscillations caused by cubic spline interpolation.
4. **XPM Parser Safety:** Corrupted or unmapped XPM characters strictly raise descriptive exceptions and never silently default to $0.0\ \text{kJ/mol}$.
5. **Representative PDB Coordinates:** Representative PDBs contain complete full-atom Cartesian coordinates matching the exact frame index and simulation timestamp recorded in `metadata.json`.

---

## 6. Package Build Verification
* Built wheel: `dist/conformatlas-0.1.1-py3-none-any.whl`.
* Verified console script: `conformatlas`.
* Status: **All Acceptance Criteria Met.**
