# Condition & Mutant Comparison in Shared PCA Space

Comparing conformational spaces between Wild-Type (WT) and mutants requires projecting all systems onto a common coordinate system.

---

## 1. Why Shared PCA Space is Critical

Calculating separate PCAs for WT and Mutant produces distinct, non-equivalent eigenvectors ($\mathbf{v}_{1, \text{WT}} \ne \mathbf{v}_{1, \text{MUT}}$). Plotting them side-by-side as if the axes were equivalent is mathematically invalid and biologically misleading.

ConformAtlas constructs a **shared PCA space** using either:
1. **`combined` mode** (default): pools all replicates of all systems into one covariance. With equal-replicate weighting every replicate weighs the same, so the systems are balanced when they have the same number of replicates (`pca_weighting: frames` weights every frame equally instead).
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
* `comparison/fel_diff_<SYS>_vs_<REF>.csv/.npy`: $\Delta\Delta G = \Delta G_{\text{system}} - \Delta G_{\text{reference}}$ map on the shared grid (with exactly two systems also saved as `comparison/fel_diff.npy`).
* `figures/condition_comparison.png`: 4-panel comparison figure.


---

## 4. Statistics Reported for Every System vs the Reference

* **Population differences** (`state_population_comparison.csv`):
  * **Welch's t-test with its 95 % interval** on per-replicate populations (`*_Welch_p`, `*_Welch_CI95_Low/High`, clipped to the possible range ±100 percentage points; Welch–Satterthwaite `*_Welch_dof`), reported when every system has at least two replicates. Independent replicates are the units of evidence for a difference between conditions, so this is the primary test. With two or three replicates per system it has little power: a non-significant result is not evidence of no difference.
  * **Bootstrap 95 % interval and two-sided p-value.** By default (`bootstrap_level: replicate`, `--bootstrap-level replicate`) each resample draws the system's replicates with replacement and then resamples blocks of frames inside every drawn replicate, so the interval contains the spread between independent simulations. With few replicates it is still somewhat too narrow, since a replicate-level bootstrap underestimates the between-replicate variance by $(k-1)/k$. `bootstrap_level: frame` restores the 1.0.x within-replicate block bootstrap. That interval describes the uncertainty given these particular trajectories and is far too narrow for a between-condition claim whenever replicates disagree: in a published-data null test (T4 lysozyme, `benchmarks/RESULTS_T4L_HIV.md`) it excluded zero in 20 of 20 comparisons of identical conditions.
  * **Replicate agreement.** Per-replicate minimum and maximum populations, and a heterogeneity ratio: the between-replicate SD of a state population divided by the RMS within-replicate block-bootstrap SE. The ratio is about 1 when the replicates sample one distribution. Above 2, `compare` warns that the replicates disagree (log, and `warnings` in `comparison_summary.json`): the simulations have not converged to a common distribution, and differences involving that system can change with more or longer runs. A system with a single replicate also triggers a warning, because its intervals reflect within-trajectory sampling only.
* **ΔΔG maps** (`fel_diff_<SYS>_vs_<REF>.csv/.npy`, `figures/ddg_map_*.png`): $\Delta G_S - \Delta G_R$ on the shared grid, each landscape referenced to its own minimum, with a bootstrap standard deviation per bin. Bins where $|\Delta\Delta G| < 2\,\text{SD}$ are hatched in the figure; regions sampled by one system only are blank.
* **State ΔΔG** (`state_free_energy_comparison.csv`): difference of the state minima with bootstrap interval.
* **Landscape overlap** (`fel_overlap.csv`): Bhattacharyya coefficient (1 = identical densities), Hellinger distance, Jensen–Shannon divergence in bits and the Jaccard index of sampled bins.
* **Essential-subspace similarity** (`subspace_overlap.csv`): RMSIP between the leading eigenvectors of each system's own PCA. Values ≳ 0.7 indicate that the systems move along similar collective directions even where their populations differ.

Comparison configurations accept `begin_ps`, `end_ps`, `index_file`, `bins: auto`, `n_bootstrap`, `bootstrap_level` (`replicate` or `frame`, also `--bootstrap-level`), `bootstrap_mode` (`fixed` or `resegment`, also `--bootstrap-mode`; re-segmentation of the pooled landscape on every resample, writing `comparison/state_count_distribution.csv` and `state_populations_resegmented.csv`) and `seed`; any number of systems may be listed and `reference_system` (under `pca:`, or at the top level) chooses the baseline for all differences. `pca_weighting` (or `pca: weighting`) is `equal-replicate` (default) or `frames`. Unknown or misspelled keys, and invalid values of `pca: reference`, `pca_weighting`, `bootstrap_level` and `bootstrap_mode`, stop with an error instead of being ignored.

When plotly is installed (or with `--interactive`), `compare` also writes `comparison_interactive.html`: the per-system landscapes side by side on the shared grid with linked axes and a shared colour scale. The common-atom mapping is chain-aware (`chain/residue/atom` keys) and falls back to `residue/atom` keys with a warning when the systems carry different chain labels, so repeated residue numbers across chains are no longer collapsed.
