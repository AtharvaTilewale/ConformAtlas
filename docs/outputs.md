# Output Directory Structure & File Manifest

ConformAtlas creates an organized, reproducible, and machine-readable output directory:

```text
results/
├── report.html                  # Standalone interactive HTML report with embedded figures
├── run_metadata.json            # Machine-readable metadata (software versions, paths, command)
├── summary.json                 # High-level summary of states, populations, and convergence
│
├── pca/
│   ├── eigenvalues.csv          # Eigenvalues for each principal mode
│   ├── explained_variance.csv   # Individual and cumulative variance (%)
│   ├── projections.csv          # Frame projections: system, replicate, frame, time_ps, PC1..PCn
│   └── common_atom_mapping.csv  # Atom selection mapping (for condition comparison)
│
├── fel/
│   ├── free_energy.csv          # Tabular 4-column FEL: PC1, PC2, FreeEnergy_kJ_mol, Probability
│   ├── free_energy.npy          # 2D NumPy array of free energy values (NaN where unobserved)
│   ├── probability.npy          # 2D NumPy array of normalized probability density
│   ├── x_edges.npy              # PC1 bin edges
│   └── y_edges.npy              # PC2 bin edges
│
├── states/
│   ├── basins.csv               # Summary of detected states, populations, minima coordinates
│   ├── frame_assignments.csv    # Trajectory frames with assigned state labels and basin IDs
│   ├── basin_map.npy            # 2D integer matrix of watershed basin IDs
│   ├── State_A/
│   │   ├── representative.pdb   # Real full-atom PDB structure extracted at basin minimum
│   │   └── metadata.json        # Frame provenance metadata (replicate, frame, time, distance)
│   └── State_B/
│       ├── representative.pdb
│       └── metadata.json
│
├── convergence/
│   ├── rmsip.csv                # First-half vs second-half subspace overlap
│   ├── cosine_content.csv       # Cosine content for PC1 through PC5
│   ├── progressive_overlap.csv  # Subspace similarity at 25%, 50%, 75%, 100%
│   ├── population_stability.csv # State population progression over cumulative time
│   └── pairwise_rmsip.csv       # Inter-replicate subspace overlap matrix
│
├── statistics/
│   ├── state_populations.csv    # State populations with Mean, SD, SEM, and 95% CI
│   └── replicate_statistics.csv # Detailed per-replicate state population breakdown
│
└── figures/
    ├── pca.png                  # Scatter and density plots of PC1/PC2 projections
    ├── explained_variance.png   # Scree plot of eigenvalues and cumulative variance
    ├── fel_2d_3d.png            # Publication-grade 2D contour and 3D surface FEL
    ├── basin_map.png            # Side-by-side FEL and discrete watershed basin segmentation
    ├── state_populations.png    # Bar chart of conformational populations with error bars
    └── convergence.png          # 4-panel convergence diagnostics figure
```
