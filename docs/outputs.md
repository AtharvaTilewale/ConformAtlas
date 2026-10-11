# Output Directory Structure & File Manifest

ConformAtlas creates an organized, reproducible, and machine-readable output directory:

The tree below was listed from a run of the bundled demonstration data set
(`conformatlas analyze ... --msm --residue-analysis --bootstrap-mode resegment
--save-hdf5 --path-overlay`); entries marked *(option)* appear only with the
option named. Units: Å for Python-backend coordinates and projections, nm for
the GROMACS backend, ps for time, kJ/mol for free energies.

```text
results/
├── report.html                  # Self-contained HTML report (Plotly landscape, NGL viewer, interpretive text); report.pdf with --pdf
├── report_interpretation.json   # The report's interpretive sentences and every number they quote (`values`)
├── run_metadata.json            # Command line, options, engine, superposition, bootstrap, MSM summary, warnings,
│                                # and `environment`: Python, NumPy, SciPy, pandas, matplotlib, scikit-image, click, Jinja2,
│                                # PyYAML, MDAnalysis, WeasyPrint, GROMACS versions and the platform
├── summary.json                 # States, populations (+ bootstrap CI), convergence, N_eff
├── results.h5                   # (--save-hdf5) HDF5 container of every table and array; see below
├── conformatlas_<date>_<time>.log  # Full log including tracebacks
│
├── pca/
│   ├── eigenvalues.csv          # Eigenvalues (TICA: autocorrelations and implied timescales)
│   ├── explained_variance.csv   # Individual and cumulative variance (%)
│   ├── projections.csv          # system, replicate, frame, time_ps, PC1..PCn (dPC/IC for other types)
│   ├── projections.npy          # (--engine streaming) memory-mapped (frames, components) array, same row order as projections.csv;
│   │                            # float32 with --precision float32
│   ├── eigenvectors.npy         # (3N or n_features) x k eigenvector matrix
│   ├── mean_coordinates.npy     # Mean structure of the analysis atoms (Å)
│   ├── average.pdb              # Average structure of the analysis atoms
│   ├── modes.nmd                # Leading modes for VMD NMWiz / ProDy
│   ├── PC1_extremes.pdb …       # Structures at the extreme projections of each mode (two models)
│   ├── mode_contributions_atoms.csv     # Per-atom weight of each leading mode
│   ├── mode_contributions_residues.csv  # Per-residue weight (sqrt of summed squares)
│   └── common_atom_mapping.csv  # Atom selection mapping (comparison only)
│
├── fel/
│   ├── free_energy.csv          # Tabular FEL: CV1, CV2, FreeEnergy_kJ_mol, Probability
│   ├── free_energy.npy          # 2D free energy (NaN where unobserved)
│   ├── probability.npy          # 2D normalised probability density
│   ├── x_edges.npy, y_edges.npy # Bin edges
│   ├── free_energy_bootstrap_sd.npy        # Bootstrap SD of ΔG per bin
│   └── free_energy_bootstrap_ci_low/high.npy  # 95 % interval of ΔG per bin
│
├── states/
│   ├── basins.csv               # Detected states: population, minimum, representative frame
│   ├── frame_assignments.csv    # Every frame with its state label and basin id
│   ├── basin_map.npy            # 2D integer basin labels
│   ├── barriers.csv             # Saddle and barrier heights between adjacent states (+ CI)
│   ├── state_free_energies.csv  # ΔG of each state relative to the most populated (+ CI)
│   ├── state_ensemble_summary.csv  # Per-state frame count, mean RMSD to average, mean RMSF
│   ├── state_count_distribution.csv       # (--bootstrap-mode resegment) n_states, count, fraction over resamples
│   ├── state_populations_resegmented.csv  # (--bootstrap-mode resegment) state, population, match_rate,
│   │                                      # population_mean/sd/ci95_low/ci95_high_resegmented (fractions)
│   └── State_A/
│       ├── representative.pdb   # Frame closest to the minimum (or medoid); representative_2.pdb …
│       ├── metadata.json        # Provenance (replicate, frame, time, distance, method)
│       ├── average.pdb          # Average structure of the state's frames
│       ├── rmsf_atoms.csv       # Per-atom RMSF within the state
│       └── rmsf_residues.csv    # Per-residue RMSF within the state
│
├── kinetics/
│   ├── transition_counts.csv    # Pooled lag-1 transition counts between major states
│   ├── transition_counts_per_replicate.csv
│   ├── transition_matrix.csv    # Row-normalised transition probabilities
│   ├── dwell_times.csv          # One row per visit (replicate, state, start, dwell_ps, censored)
│   ├── dwell_summary.csv        # Visits, mean/median/max dwell per state
│   ├── mfpt_markov_ps.csv       # Markov mean first-passage times (ps)
│   └── replicate_summary.csv    # Frames used / skipped per replicate
│
├── descriptors/
│   ├── rmsd_rg_timeseries.csv   # replicate, time_ps, RMSD, Rg (+ Ligand_RMSD with --lig)
│   ├── rmsf_atoms.csv           # Per-atom RMSF over all frames
│   ├── rmsf_residues.csv        # Per-residue RMSF
│   └── summary.csv              # Mean / SD of the descriptors per replicate
│
├── convergence/
│   ├── rmsip.csv                # Half-vs-half RMSIP: pooled, then one row per replicate
│   ├── cosine_content.csv       # Cosine content of PC1–PC5
│   ├── progressive_overlap.csv  # Overlap of the first 25/50/75/100 % with the full data
│   ├── population_stability.csv # State populations on the first f % of every replicate
│   ├── pairwise_rmsip.csv       # Inter-replicate subspace overlap matrix
│   ├── effective_samples.csv    # Statistical inefficiency g (PC1, PC2) and N_eff per replicate
│   └── equilibration.csv        # Replicate, observable, t0_index, t0_ps, frames_discarded, g_production, N_eff_production,
│                                # N_eff_full, applied, deciding_observable. One row per replicate for the fit-group RMSD;
│                                # with --equilibration-on cv a second row per replicate ("PC1+PC2 (latest t0)")
│
├── statistics/
│   ├── state_populations.csv    # Mean, block SD/SEM, 95 % CI, bootstrap 95 % CI per state (percent);
│   │                            # + Resegment_CI_95_Lower/Upper and Resegment_match_rate with --bootstrap-mode resegment
│   └── replicate_statistics.csv # Per-replicate state populations
│
├── msm/                         # (--msm) see "Markov state model outputs" below
├── residues/                    # (--residue-analysis) see "Residue-level outputs" below
│
└── figures/                     # One file per --fig-format (png default; svg/pdf are vector)
    ├── pca.png, pca_scatter.png, pca_density.png, explained_variance.png
    ├── fel_2d.png, fel_2d_3d.png, fel_3d.png, fel_1d.png   # Landscape; 1-D projections with bootstrap bands
    ├── fel_2d_path.png                           # (--path-overlay) time-coloured path of every replicate on the landscape
    ├── fel_uncertainty.png                       # ΔG next to its bootstrap SD
    ├── basin_map.png, state_populations.png
    ├── state_timeseries.png                      # State sequence per replicate
    ├── per_replicate_landscapes.png              # (> 1 replicate) each replicate's landscape on the shared grid, own minimum at 0
    ├── descriptors.png                           # RMSD, Rg, RMSF, RMSD distribution
    ├── mode_contributions.png                    # Per-residue mode weights
    ├── convergence.png, convergence_progressive.png, convergence_cosine.png, convergence_stability.png
    ├── msm_implied_timescales.png, msm_ck_test.png, msm_pcca_landscape.png, msm_metastable_timeseries.png   # (--msm)
    └── dccm.png, dccm_state_A.png …, state_residue_differences.png, ligand_contact_persistence.png          # (--residue-analysis)
```

Annotated figures are also written without labels as `<name>_unannotated.<ext>` for manual annotation in a figure editor.

With `--engine streaming` the steps that need every coordinate in memory are
skipped (listed from a streaming run on the demonstration data): `descriptors/`,
`pca/eigenvectors.npy`, `pca/mean_coordinates.npy`, `pca/average.pdb`,
`pca/PC*_extremes.pdb`, `pca/modes.nmd`, `pca/mode_contributions_*.csv`,
`states/state_ensemble_summary.csv`, the per-state `average.pdb`/`rmsf_*.csv`
and `convergence/pairwise_rmsip.csv` are not written; `pca/projections.npy`
is added and representative structures are still extracted. With `--from DIR`
the `pca/` tables and `descriptors/` are copied from the source directory,
representatives are re-extracted when the recorded trajectories still exist,
and `states/state_ensemble_summary.csv`, the per-state average structures and
RMSF profiles and `residues/` are not regenerated.

## Markov state model outputs (`--msm`)

```text
msm/
├── count_matrix.csv              # Microstate transition counts at the model lag (sliding window, within replicates); all microstates
├── transition_matrix.csv         # Reversible (or row-normalised) T on the largest connected set; index column `microstate`
├── stationary.csv                # microstate, stationary_probability (active set)
├── eigenvalues.csv               # index, eigenvalue, timescale_ps (inf for the stationary eigenvalue)
├── cluster_centers.csv           # microstate, one column per clustering component (PC1, PC2, PC3 …), component units
├── implied_timescales.csv        # lag_frames, lag_ps, n_active, active_count_fraction, ts_1 … ts_k (ps)
├── implied_timescale_changes.csv # lag_frames, next_lag_frames, max_rel_change, n_resolved — the plateau report
├── ck_test.csv                   # k, lag_frames, lag_ps, from_set, to_set, predicted, estimated, estimated_se, abs_deviation, within_tolerance
├── pcca_memberships.csv          # microstate, crisp_set, MS1 … MSm (PCCA+ memberships of each active microstate)
├── metastable_assignments.csv    # Per frame: replicate, frame, time_ps, microstate, metastable_set, membership_MS1 …
├── mfpt.csv                      # from_set, to_MS1 … (mean first-passage times between metastable sets, ps)
├── coarse_transition_matrix.csv  # from_set, MS1 … (PCCA+ coarse-grained T; descriptive)
├── coarse_stationary.csv         # metastable_set, stationary_probability
├── basin_overlap.csv             # metastable_set, one column per watershed state (+ Unassigned), best_basin, row_fraction_max
└── msm_summary.json              # lag_frames, dt_ps, reversible, clustering, n_microstates, n_active, active_count_fraction,
                                  # n_metastable, suggested_lag_frames, plateau_found, ck_test {lag_frames, k_max, abs_tol,
                                  # max_abs_deviation, max_abs_deviation_diagonal, n_violations, passed}, estimator {iterations,
                                  # converged}, notes
```

`run_metadata.json` repeats the headline numbers under `msm` (`lag_frames`,
`lag_ps`, `components`, `clustering`, `n_microstates`, `n_active`,
`reversible`, `n_metastable`, `suggested_lag_frames`, `plateau_found`,
`ck_passed`, `ck_max_abs_deviation`, `implied_timescales_ps`). Microstates
outside the largest connected set are absent from `transition_matrix.csv` and
`stationary.csv` and are reported in `notes`.

## Residue-level outputs (`--residue-analysis`)

```text
residues/
├── dccm.csv, dccm.npy             # Atom-level dynamic cross-correlation map of the analysis group;
│                                  # CSV row/column labels RESNAME<resnum>_<chain>
├── dccm_residues.csv              # Residue-averaged DCCM (identical to dccm.csv for a C-alpha analysis group)
├── dccm_state_A.npy, …            # DCCM from the frames of each state (lower-case `state_<letter>` file names)
├── state_residue_differences.csv  # chain, residue_number, residue_name, n_atoms, rmsf_<A>, n_frames_<A>, rmsf_<B>, n_frames_<B>,
│                                  # delta_rmsf, mean_rmsd (between state-mean positions), centroid_displacement, state_a, state_b (Å)
├── ligand_contacts_per_state.csv  # (--lig / --ligand-selection) chain, residue_number, residue_name, n_atoms, persistence_all,
│                                  # mean_min_distance_all, persistence_<state>, mean_min_distance_<state>, n_frames_<state>, max_persistence
└── binding_site_rmsf.csv          # (--lig) residues with persistence ≥ 0.5 in any state: rmsf_all, rmsf_<state> (Å)
```

Persistence is the fraction of a state's frames in which any heavy-atom pair
of the residue and the ligand is closer than `--contact-cutoff` (4.5 Å).
Residue RMSF values come from the analysis group only (NaN for residues
without analysis atoms).

## HDF5 container (`--save-hdf5`)

`results.h5` holds the complete `AnalysisResults` object: groups `config`,
`trajectories_info`, `pca` (`eigenvalues`, `eigenvectors`, `mean_coords`,
`projections`, `explained_variance`, `cumulative_variance`, `atom_names`,
`residue_numbers`), `fel`, `states`, `frame_assignments`, `convergence`,
`replicate_statistics`, `run_metadata`, `extras` (bootstrap, barriers,
kinetics, descriptors, state summary, MSM …) and `warnings`; file attributes
`format_version` (1), `conformatlas_version`, `output_dir`,
`unassigned_population`. Tables are stored as groups of column datasets and
dataclasses as groups with a `__type__` attribute. Read it back with
`conformatlas.results_io.load_results(path)`; the object is the one the
pipeline returned. Without the container, `load_results_dir(path)` rebuilds an
`AnalysisResults` from the CSV/NPY files (eigenvectors are then empty).

## Reweighting outputs (`conformatlas reweight`)

```text
reweight/
├── weights_<COLVAR or window>.dat # "# time weight" text file, one row per COLVAR row (or per --frame-times entry);
│                                  # mean weight 1 over kept frames, 0 for frames before --skip-time; read by analyze --weights-file
├── pmf.csv                        # center, pmf_kJ_mol, probability, weight_sum, count — 1-D PMF of the (first) CV from the weights
├── reweight_summary.json          # metad: method, temperature_K, hills, cvs, skip_time_ps, grid_bins, hill_shape, files[{colvar, weights,
│                                  #   frames_kept, kish_n_eff, bias_factor}], pmf
│                                  # umbrella: method, temperature_K, cv, centers, kappas, period, skip_time_ps, bins, converged, n_iter,
│                                  #   window_free_energies_kJ_mol, files
├── rct.csv                        # metad: hill_time_ps, c_t_kJ_mol (Tiwary–Parrinello c(t) after every hill)
├── bias_final.csv                 # metad: <cv>, bias_kJ_mol — the end-of-run bias on the default grid
└── window_free_energies.csv       # umbrella: window, n_samples, free_energy_kJ_mol (f_0 = 0)
```

## Interactive report components

`report.html` references no external resource: the Plotly library (≈ 4.8 MB),
the NGL viewer (≈ 1.3 MB), every figure (base64) and every PDB shown in the
viewer are inline, which makes the file ≈ 11–15 MB; `--plotly-js cdn` loads
Plotly from `cdn.plot.ly` instead. The interactive landscape shows ΔG, the
bin probability and the watershed label of every bin on hover, basin
outlines, the state minima, time-coloured paths per replicate (subsampled
evenly beyond 5000 points) and a time slider (≤ 200 positions). The NGL
viewer shows the representative structure of every state and the average
structure with the per-residue RMSF (Å) written to the B-factor column.
`report.pdf` (`--pdf`) contains the static figures only and says so.

Axis labels follow the collective variables in use (PC, dPC, IC, RMSD/Rg, or the names given with `--cv-labels`); the DataFrame/CSV columns keep the names `PC1..PCn` for every decomposition so that downstream scripts work unchanged. Cartesian projections from the Python backend are in Å; projections from the GROMACS backend (`--pca-backend gromacs`) are in nm.

## Comparison outputs (`conformatlas compare`)

```text
comparison/
├── state_population_comparison.csv  # Per-state populations per system, replicate SD, bootstrap CI,
│                                    # difference to the reference with CI, bootstrap p, Welch t/p
├── state_free_energy_comparison.csv # State minima per system and ΔΔG vs reference with CI
├── fel_diff_<SYS>_vs_<REF>.csv/.npy # ΔΔG map on the shared grid (+ bootstrap SD column)
├── fel_overlap.csv                  # Bhattacharyya, Hellinger, JS divergence, sampled-bin Jaccard
├── subspace_overlap.csv             # RMSIP between each system's own essential subspace
├── state_count_distribution.csv     # (--bootstrap-mode resegment) n_states, count, fraction over pooled-landscape resamples
├── state_populations_resegmented.csv  # (--bootstrap-mode resegment) per pooled state: match_rate and population CI under re-segmentation
├── comparison_summary.json
└── <SYS>/                           # Per-system FEL on the shared grid
comparison_interactive.html          # (plotly installed / --interactive) side-by-side Plotly landscapes, linked axes, shared colour scale
figures/
├── comparison_shared_pca.png, comparison_populations.png (bootstrap CI error bars)
├── comparison_fel_<SYS>.png, condition_comparison.png
└── ddg_map_<SYS>_vs_<REF>.png       # ΔΔG map with uncertainty panel; hatched where |ΔΔG| < 2 SD
```
