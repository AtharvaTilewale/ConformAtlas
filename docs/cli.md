# Command-line reference

Every option below is taken from `conformatlas <command> --help` of version
1.0.2. Options marked *YAML* can also be given in the configuration file read
by `--config` (key names in parentheses where they differ from the flag;
`conformatlas init-config --type analyze` writes a template with every key).
Command-line flags override the file.

```{contents}
:local:
:depth: 2
```

## Exit codes and global options

| Code | Meaning |
|---|---|
| 0 | Run completed. Warnings (sparse grid, suggested equilibration, skipped optional steps) are printed, listed in the report and stored in `run_metadata.json`. |
| 1 | Analysis failure. A short message is printed; the traceback is written only to the log file `conformatlas_<date>_<time>.log` in the output directory. |
| 2 | Usage or configuration problem, reported before any trajectory is read: malformed YAML (with line and column), missing input file, unknown group name (the available groups are listed), invalid colormap / figure format / font scale / dpi, or a missing optional dependency for a requested feature (`--pdf` without WeasyPrint, `--interactive` without plotly, `--save-hdf5` without h5py); the `pip install` command is printed. |

Root-level options: `--version`, `-q/--quiet` (no banner, no progress bars;
the banner is shown only on an interactive terminal and never for `--help` or
`--version`), `-h/--help`. The legacy root-level `-s/-f/-T/-o` options run
`analyze` directly and are kept for backward compatibility. Progress bars use
`rich` when it is installed and the output is a terminal; they are silent in
pipes and log files, with `--no-progress`, `-q`, or the environment variable
`CONFORMATLAS_NO_PROGRESS=1`.

## `conformatlas analyze`

PCA / free-energy-landscape analysis of one system (single or multi-replicate).

### Input

| Option | Default | Meaning |
|---|---|---|
| `--config FILE` | — | YAML file with any of the options below (command-line flags override it). |
| `-s, --structure, --topology FILE` | required | Topology/structure file (`.tpr`, `.gro`, `.pdb`). A `.gro`/`.pdb` also serves as the superposition reference; with a `.tpr` the first trajectory frame is used. *YAML* `topology` |
| `-f, --trajectory FILE` | required | Trajectory file (`.xtc`/`.trr`/`.dcd`). Repeat for replicates: `-f rep1.xtc -f rep2.xtc`. *YAML* `trajectories` |
| `-T, --temperature FLOAT` | 300 | Simulation temperature in Kelvin (> 0). |
| `-o, --output DIRECTORY` | `Outputs` | Output directory. *YAML* `output_dir` |
| `-b, --begin FLOAT`, `-e, --end FLOAT` | — | First / last time to read (ps). *YAML* `begin_ps`, `end_ps` |
| `--stride INTEGER` | 1 | Frame stride. |
| `-n, --index FILE` | — | GROMACS index file; group names given to `-l/-c/-g` are looked up here first. *YAML* `index_file` |
| `-l, --lsq-group TEXT` | `Backbone` | Least-squares fit group. |
| `-c, --cov-group TEXT` | `Backbone` | Analysis (covariance) group. |
| `-g, --lig TEXT` | — | Ligand residue/group name (e.g. `ZMR`, `ATP`): added to the analysis group and given its own RMSD. *YAML* `lig_group` |
| `--fit / --no-fit` | `--fit` | Superpose frames on `--lsq-group` before PCA. `--no-fit` only for pre-fitted trajectories. |
| `--mass-weighted-fit` | off | Mass-weight the superposition. |
| `--from DIRECTORY` | — | Re-analyse an existing output directory (rebin, re-segment, bootstrap, kinetics, MSM) without reading trajectories. The PCA tables of the source directory are reused; steps that need coordinates (descriptors, state-ensemble structures, mode structures, subspace-overlap diagnostics, residue analyses) are not recomputed. *YAML* `from_results` |

### Equilibration

| Option | Default | Meaning |
|---|---|---|
| `--equilibration [none\|auto]` | `none` | `auto`: discard the initial transient detected per replicate (Chodera max-N_eff criterion). `none`: detect and report only. |
| `--equilibration-on [rmsd\|cv]` | `rmsd` | Observable for the detection: fit-group RMSD, or both landscape CVs (the later of the two t₀ is adopted; see [Methods](methods.md)). With `cv` and `--equilibration auto` the decomposition is refitted on the production window. |

### Decomposition and collective variables

| Option | Default | Meaning |
|---|---|---|
| `--pca-type [cartesian\|dihedral\|tica]` | `cartesian` | Cartesian PCA, dihedral PCA, or TICA on fitted coordinates. |
| `--dihedral-set [backbone\|backbone+chi\|chi\|nucleic]` | `backbone` | Torsions used by `--pca-type dihedral`: backbone φ/ψ, φ/ψ plus side-chain χ₁/χ₂, χ₁/χ₂ only, or nucleic-acid α–ζ and χ. |
| `--tica-lag INTEGER` | 10 | TICA lag time in frames. |
| `--n-components INTEGER` | 10 | Components to retain. |
| `--pca-backend [auto\|python\|gromacs]` | `auto` | `auto`/`python`: in-memory MDAnalysis route (Å); `gromacs`: `gmx covar`/`anaeig` (nm). |
| `--pca-weighting [equal-replicate\|frames]` | `equal-replicate` | Weighting of replicates in PCA. |
| `--cv [pc\|rmsd-rg\|file]` | `pc` | Landscape coordinates: two components, RMSD × Rg, or columns of `--cv-file`. |
| `--fel-pcs TEXT` | `1,2` | Components spanning the landscape, e.g. `1,3`. |
| `--cv-file FILE` | — | External CV file per trajectory (`.xvg`/`.dat`/`.csv`; time + columns), same order as `-f`. *YAML* `cv_files` |
| `--cv-cols TEXT` | `1,2` | 0-based columns of `--cv-file` holding CV1, CV2. *YAML* `cv_columns` |
| `--cv-labels TEXT` | — | Axis names for `--cv file`, e.g. `'d1 (nm),d2 (nm)'`. |
| `--weights-file FILE` | — | Per-frame statistical weights per trajectory (biased-sampling reweighting), same order as `-f`; produced by `conformatlas reweight`. *YAML* `weights_files` |

### Engine (memory and speed)

| Option | Default | Meaning |
|---|---|---|
| `--engine, --pca-engine [auto\|memory\|streaming]` | `auto` | `auto`: in-memory PCA unless the memory estimate exceeds 2 GB (then the streaming engine is used and a warning is recorded); `streaming`: chunked two-pass PCA that never holds the trajectory in memory. The streaming engine supports Cartesian PCA only (not `--cv rmsd-rg`), and skips the steps that need all coordinates at once (descriptors, state-ensemble structures, mode structures). The threshold is the YAML key `streaming_threshold_gb` (2.0). |
| `--precision [float64\|float32]` | `float64` | Storage dtype of the projections with the streaming engine (`pca/projections.npy`). Covariance accumulation is always float64. |
| `--n-workers INTEGER` | 1 | Worker processes for the streaming engine (one per trajectory). |
| `--chunk-size INTEGER` | 256 | Frames per chunk (streaming engine). |
| `--no-progress` | — | Disable progress bars. |

### Landscape, states and uncertainty

| Option | Default | Meaning |
|---|---|---|
| `--bins TEXT` | `auto` | FEL bins per axis, or `auto` (Freedman–Diaconis, clipped to 16–128). |
| `--smooth-sigma FLOAT` | 0 | Gaussian smoothing SD in bins (0 = raw). |
| `--min-state-population FLOAT` | 0.05 | Minimum fraction for a major state. |
| `--max-states INTEGER` | 10 | Maximum number of major states. |
| `--min-barrier FLOAT` | 1 kT | Minimum barrier (kJ/mol) between basins; shallower ones are merged. |
| `--bootstrap INTEGER` | 200 | Block-bootstrap resamples for ΔG, populations and barriers (0 disables). *YAML* `n_bootstrap` |
| `--bootstrap-mode [fixed\|resegment]` | `fixed` | `resegment` re-runs the watershed on every resample and reports the state-count distribution and population intervals that include segmentation uncertainty. |
| `--seed INTEGER` | 0 | Random seed for the bootstrap (and the MSM clustering). |
| `--n-representatives INTEGER` | 1 | Structures written per state. |
| `--representative [minimum\|medoid]` | `minimum` | Closest to the basin minimum, or medoid in PC space. *YAML* `representative_method` |
| `--fel-backend [python\|gromacs]` | `python` | FEL backend. |
| `--gmx-command TEXT` | — | Path to a GROMACS executable. |

### Markov state model

| Option | Default | Meaning |
|---|---|---|
| `--msm / --no-msm` | off | Build a Markov state model on the leading components after basin detection (outputs in `msm/`). |
| `--msm-lag INTEGER` | 10 | MSM lag time in frames. The lag is always the user's choice; the plateau scan only reports. |
| `--msm-lags TEXT` | geometric 1..5·lag | Comma-separated lags (frames) for the implied-timescale scan. |
| `--msm-clusters INTEGER` | 100 | k-means microstates (capped at one tenth of the frame count). *YAML* `msm_n_clusters` |
| `--msm-clustering [kmeans\|regular-space]` | `kmeans` | Microstate clustering. |
| `--msm-dmin FLOAT` | — | Regular-space clustering radius (component units); required for `regular-space`. |
| `--msm-components INTEGER` | 3 | Leading components used for clustering (the two landscape components come first). *YAML* `msm_n_components` |
| `--msm-states INTEGER` | number of landscape states (≥ 2) | Number of PCCA+ metastable sets. *YAML* `msm_n_metastable` |
| `--msm-nonreversible` | off | Row-normalised estimator instead of the reversible maximum-likelihood estimator. *YAML* `msm_reversible: false` |
| `--msm-ck-k INTEGER` | 5 | Chapman–Kolmogorov multiples of the lag. |

### Residue-level analyses

| Option | Default | Meaning |
|---|---|---|
| `--residue-analysis / --no-residue-analysis` | off | DCCM, per-residue state differences and (with `--lig`) ligand-contact persistence; outputs in `residues/`. Forces the in-memory engine. |
| `--ligand-selection TEXT` | atoms of `--lig` | MDAnalysis selection of the ligand for the contact analysis. |
| `--contact-cutoff FLOAT` | 4.5 | Heavy-atom contact cutoff in Å. |

### Figures and report

| Option | Default | Meaning |
|---|---|---|
| `--fig-format TEXT` | `png` | Figure format(s): `png`, `svg`, `pdf`, `all`, or a comma list such as `png,svg`. One file per format; the report embeds the first. |
| `--colormap TEXT` | `turbo` | Matplotlib colormap for landscapes / density maps. |
| `--font-scale FLOAT` | 1.0 | Multiply all figure font sizes (0.3–4). |
| `--fig-dpi INTEGER` | 300 | Raster figure dpi. |
| `--path-overlay` | off | Also draw each replicate's time-coloured path on the static 2-D landscape (`figures/fel_2d_path`). |
| `--interactive / --no-interactive` | when plotly is installed | Embed the Plotly landscape and the NGL structure viewer in `report.html`. `--interactive` without plotly exits with code 2. |
| `--plotly-js [inline\|cdn]` | `inline` | Embed `plotly.min.js` in the report (self-contained, ≈ 5 MB larger) or load it from the CDN. |
| `--pdf` | off | Also export `report.pdf` (requires WeasyPrint; static figures only). |
| `--save-hdf5 / --no-save-hdf5` | off | Also write `results.h5`, a single HDF5 container of every table and array (requires h5py). |
| `-q, --quiet` | off | No banner or progress output. |

## `conformatlas compare`

Compare systems in a shared PCA coordinate space.

| Option | Default | Meaning |
|---|---|---|
| `-c, --config PATH` | required | YAML configuration file for the comparison (`conformatlas init-config --type compare`). |
| `-o, --output PATH` | from YAML | Override output directory. |
| `--bootstrap-mode [fixed\|resegment]` | `fixed` | `resegment` re-runs the watershed on every pooled-landscape resample (YAML key `bootstrap_mode`); writes `comparison/state_count_distribution.csv` and `comparison/state_populations_resegmented.csv`. |
| `--interactive / --no-interactive` | when plotly is installed | Write `comparison_interactive.html` (side-by-side Plotly landscapes, linked axes, shared colour scale). |
| `--fig-format`, `--colormap`, `--font-scale`, `--fig-dpi`, `--plotly-js`, `-q` | as above | Figure and report control, same semantics as `analyze`. |

An MSM is not built by `compare`: systems have different kinetics, and a
shared microstate set would need its own validation.

## `conformatlas reweight`

Per-frame weights from biased sampling (metadynamics, umbrella sampling). The
weight files are consumed unchanged by `conformatlas analyze --weights-file`
(one per trajectory; frame counts must match the analysed frames, hence
`--frame-times` or a COLVAR stride equal to the trajectory output stride).
Formulas and validation: [Methods](methods.md), *Enhanced-sampling
reweighting*.

### `conformatlas reweight metad`

Metadynamics reweighting from PLUMED `HILLS` + `COLVAR`.

| Option | Default | Meaning |
|---|---|---|
| `--hills FILE` | required | PLUMED HILLS file (`multivariate false`; well-tempered heights are rescaled on read). |
| `--colvar FILE` | required | PLUMED COLVAR file(s), one per trajectory (repeatable). |
| `--cv TEXT` | required | Biased CV column name(s) in the COLVAR file (repeatable for several CVs). |
| `-T, --temperature FLOAT` | required | Temperature in Kelvin. |
| `--method [tiwary-parrinello\|final-bias]` | `tiwary-parrinello` | Time-dependent c(t) reweighting (Tiwary & Parrinello 2015) or static end-of-run bias. |
| `--skip-time FLOAT` | 0 | Frames before this time (ps) get weight 0. |
| `--grid-bins INTEGER` | 200 | Grid points per CV for the c(t) integrals. |
| `--hill-shape [plumed\|gaussian]` | `plumed` | PLUMED ≥ 2.5 stretched Gaussian, or a plain Gaussian (HILLS from PLUMED < 2.5). |
| `--frame-times FILE` | — | File(s) with the trajectory frame times (ps), one per COLVAR, to pick the matching COLVAR rows. |
| `-o, --output DIRECTORY` | `reweight` | Output directory (`weights_<colvar>.dat`, `pmf.csv`, `rct.csv`, `bias_final.csv`, `reweight_summary.json`). |

### `conformatlas reweight umbrella`

Umbrella-sampling weights and PMF by MBAR or WHAM.

| Option | Default | Meaning |
|---|---|---|
| `--colvar FILE` | required | One COLVAR file per umbrella window (repeatable, in the order of `--center`). |
| `--cv TEXT` | required | Restrained CV column name. |
| `--center FLOAT` | required | Restraint centre per window (repeatable). |
| `--kappa FLOAT` | required | Force constant κ in kJ/mol per CV unit², V = ½ κ (x − x₀)²; one value or one per window. |
| `-T, --temperature FLOAT` | required | Temperature in Kelvin. |
| `--method [mbar\|wham]` | `mbar` | Estimator for the window free energies. |
| `--bins INTEGER` | 100 | PMF bins. |
| `--range FLOAT FLOAT` | data range | PMF range lo hi. |
| `--skip-time FLOAT` | 0 | Discard COLVAR rows before this time (ps). |
| `--period FLOAT` | — | Period of a periodic CV (e.g. 6.283185 for radians). |
| `-o, --output DIRECTORY` | `reweight` | Output directory (`weights_<window>.dat`, `pmf.csv`, `window_free_energies.csv`, `reweight_summary.json`). |

## `conformatlas doctor`

Checks the Python version, GROMACS, the required libraries and the optional
ones (`weasyprint`, `plotly`, `h5py`, `rich`), and prints their versions. No
options.

## `conformatlas example`

Generates the 5-residue polyalanine demonstration project (WT and MUT, two
replicates each) and a `comparison.yaml`. `-o, --output PATH` (default
`example_project`).

## `conformatlas init-config`

Writes a starter YAML template: `-t, --type [compare|analyze]` (default
`compare`), `-o, --output PATH` (prints to the console when omitted). The
`analyze` template lists every key of the tables above with its default.
