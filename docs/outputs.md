# Output Directory Structure & File Manifest

ConformAtlas creates an organized, reproducible, and machine-readable output directory:

```text
results/
├── report.html                  # Standalone HTML report with embedded figures (report.pdf with --pdf)
├── run_metadata.json            # Software versions, command line, options, superposition, warnings
├── summary.json                 # States, populations (+ bootstrap CI), convergence, N_eff
│
├── pca/
│   ├── eigenvalues.csv          # Eigenvalues (TICA: autocorrelations and implied timescales)
│   ├── explained_variance.csv   # Individual and cumulative variance (%)
│   ├── projections.csv          # system, replicate, frame, time_ps, PC1..PCn (dPC/IC for other types)
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
│   └── equilibration.csv        # Detected t0, frames discarded, g and N_eff of production segment
│
├── statistics/
│   ├── state_populations.csv    # Mean, block SD/SEM, 95 % CI, bootstrap 95 % CI per state
│   └── replicate_statistics.csv # Per-replicate state populations
│
└── figures/
    ├── pca.png, explained_variance.png
    ├── fel_2d_3d.png, fel_3d.png, fel_1d.png      # Landscape; 1-D projections with bootstrap bands
    ├── fel_uncertainty.png                       # ΔG next to its bootstrap SD
    ├── basin_map.png, state_populations.png
    ├── state_timeseries.png                      # State sequence per replicate
    ├── descriptors.png                           # RMSD, Rg, RMSF, RMSD distribution
    ├── mode_contributions.png                    # Per-residue mode weights
    └── convergence.png, convergence_progressive.png, convergence_cosine.png
```

Annotated figures are also written without labels as `<name>_unannotated.png` for manual annotation in a figure editor.

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
├── comparison_summary.json
└── <SYS>/                           # Per-system FEL on the shared grid
figures/
├── comparison_shared_pca.png, comparison_populations.png (bootstrap CI error bars)
├── comparison_fel_<SYS>.png, condition_comparison.png
└── ddg_map_<SYS>_vs_<REF>.png       # ΔΔG map with uncertainty panel; hatched where |ΔΔG| < 2 SD
```
