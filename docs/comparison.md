# Condition & Mutant Comparison in Shared PCA Space

Comparing conformational spaces between Wild-Type (WT) and mutants requires projecting all systems onto a common coordinate system.

---

## 1. Why Shared PCA Space is Critical

Calculating separate PCAs for WT and Mutant produces distinct, non-equivalent eigenvectors ($\mathbf{v}_{1, \text{WT}} \ne \mathbf{v}_{1, \text{MUT}}$). Plotting them side-by-side as if the axes were equivalent is mathematically invalid and biologically misleading.

ConformAtlas constructs a **shared PCA space** using either:
1. **`combined` mode** (Recommended default for unbiased comparisons): Pools WT and mutant frames with balanced weighting to construct a shared basis.
2. **`reference-system` mode**: Computes eigenvectors strictly from the reference condition (e.g. WT) and projects mutants onto that basis to detect deviations from the WT baseline.

---

## 2. Common Atom Mapping Validation

Mutants often introduce or delete side-chain atoms. ConformAtlas strictly matches invariant atoms across all systems (typically `Backbone` or `C-alpha`):
* Verified by `(chain, residue_number, atom_name)`.
* Saved to `pca/common_atom_mapping.csv`.
* Guarantees that Cartesian displacements map to identical physical positions.

---

## 3. Comparison Outputs

When running `conformatlas compare --config compare.yaml`, the following outputs are generated:
* `pca/projections.csv`: All frames projected onto the shared $PC_1 / PC_2$ axes.
* `comparison/state_population_comparison.csv`: Direct shifts in major state occupancies.
* `comparison/fel_diff.npy`: $\Delta\Delta G = \Delta G_{\text{mutant}} - \Delta G_{\text{WT}}$ free energy difference map.
* `figures/condition_comparison.png`: 4-panel comparison figure.


---

## 4. Statistics Reported for Every System vs the Reference

* **Population differences** (`state_population_comparison.csv`): block-bootstrap 95 % interval and two-sided p-value for the difference, and Welch's t-test across per-replicate populations when every system has at least two replicates. With two or three replicates per system the t-test has very little power; the bootstrap, which resamples correlated blocks of frames, is the primary estimate and the t-test a complement.
* **ΔΔG maps** (`fel_diff_<SYS>_vs_<REF>.csv/.npy`, `figures/ddg_map_*.png`): $\Delta G_S - \Delta G_R$ on the shared grid, each landscape referenced to its own minimum, with a bootstrap standard deviation per bin. Bins where $|\Delta\Delta G| < 2\,\text{SD}$ are hatched in the figure; regions sampled by one system only are blank.
* **State ΔΔG** (`state_free_energy_comparison.csv`): difference of the state minima with bootstrap interval.
* **Landscape overlap** (`fel_overlap.csv`): Bhattacharyya coefficient (1 = identical densities), Hellinger distance, Jensen–Shannon divergence in bits and the Jaccard index of sampled bins.
* **Essential-subspace similarity** (`subspace_overlap.csv`): RMSIP between the leading eigenvectors of each system's own PCA. Values ≳ 0.7 indicate that the systems move along similar collective directions even where their populations differ.

Comparison configurations accept `begin_ps`, `end_ps`, `index_file`, `bins: auto`, `n_bootstrap` and `seed`; any number of systems may be listed and `reference_system` chooses the baseline for all differences.
