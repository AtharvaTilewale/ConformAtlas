# Using ConformAtlas from Python

The command line is a thin wrapper around library functions. Everything
`conformatlas analyze` does is available as
`conformatlas.pipeline.run_analysis(config)`, and the individual steps (PCA,
landscape, basins, bootstrap, kinetics, convergence) are plain functions on NumPy
arrays that can be used on their own. The generated reference for every module is
in [API Reference](api.rst); this page explains how the pieces fit together.

```{contents}
:local:
:depth: 2
```

## The full workflow: `run_analysis`

```python
from pathlib import Path

from conformatlas.models import AnalysisConfig
from conformatlas.pipeline import run_analysis

cfg = AnalysisConfig(
    topology="tests/data/h1n1_ca/neuraminidase_ca.pdb",
    trajectories=["tests/data/h1n1_ca/neuraminidase_ca.xtc"],
    temperature=300.0,
    index_file="tests/data/h1n1_ca/index.ndx",
    lsq_group="C-alpha",
    cov_group="C-alpha",
    lig_group="ZMR",
    bins=12,
    n_bootstrap=50,
    output_dir=Path("tutorial/api_run"),
    system_name="NA-zanamivir",
)
results = run_analysis(cfg)

print(results.pca.explained_variance[:2].round(1))
for state in results.states:
    print(state.label, round(100 * state.population, 1), round(state.min_free_energy, 2))
print(results.convergence.sampling_assessment, round(sum(results.convergence.n_eff.values()), 1))
print(sorted(p.name for p in results.output_dir.iterdir() if p.is_dir()))
```

```text
[25.9 12.1]
State A 46.9 0.63
State B 39.5 -0.0
Insufficient evidence 3.1
['convergence', 'descriptors', 'fel', 'figures', 'kinetics', 'pca', 'states', 'statistics']
```

`run_analysis` writes the complete output directory (tables, structures, figures
and `report.html`) exactly as the CLI does and returns an `AnalysisResults`. An
optional `echo` callable receives the progress messages that the CLI prints.

### `AnalysisConfig`

`conformatlas.models.AnalysisConfig` is a dataclass with one field per
`conformatlas analyze` option (the YAML keys of `conformatlas init-config --type
analyze` use the same names). Validation happens in `__post_init__`: a
non-positive temperature, fewer than 5 bins, an unknown `pca_type`, identical
`fel_pcs`, or `end_ps <= begin_ps` raise `ValueError` before any file is read.

| Field (default) | Meaning |
|---|---|
| `topology`, `trajectories` | Structure (`.pdb`/`.gro`/`.tpr`) and one or more trajectories (replicates) |
| `temperature` (300.0) | Kelvin; enters only via kT |
| `lsq_group`, `cov_group` (`"Backbone"`), `lig_group` (None), `index_file` (None) | Fit group, analysis group, ligand group; GROMACS index file for group names |
| `fit` (True), `mass_weighted_fit` (False) | Least-squares superposition before PCA |
| `begin_ps`, `end_ps` (None), `stride` (1), `equilibration` (`"none"`) | Time window; `"auto"` discards the detected transient |
| `pca_type` (`"cartesian"`), `tica_lag` (10), `n_components` (10), `pca_backend` (`"auto"`), `pca_weighting` (`"equal-replicate"`) | Decomposition |
| `cv` (`"pc"`), `fel_pcs` ((1, 2)), `cv_files`, `cv_columns`, `cv_labels`, `weights_files` | Collective variables of the landscape and optional per-frame weights |
| `bins` (None → Freedman–Diaconis), `smooth_sigma` (0.0) | Grid |
| `min_state_population` (0.05), `max_states` (10), `min_barrier` (None → 1 kT) | State definition |
| `n_bootstrap` (200), `seed` (0) | Block bootstrap |
| `n_representatives` (1), `representative_method` (`"minimum"`) | Structures per state |
| `output_dir` (`Outputs`), `system_name` (`"System"`), `pdf` (False), `fel_backend` (`"python"`), `gmx_command` (None) | Output and backends |

`conformatlas.config.build_analysis_config(options, config_file=None)` performs
the YAML + command-line merge the CLI uses, if you want to reuse a YAML file from
Python.

### `AnalysisResults`

| Attribute | Type | Content |
|---|---|---|
| `config` | `AnalysisConfig` | The configuration that was run |
| `trajectories_info` | `list[TrajectoryInfo]` | Frames, start/end time and timestep per replicate |
| `pca` | `PCAResults` | `eigenvalues`, `eigenvectors` (3N × k), `explained_variance` and `cumulative_variance` (%), `projections` (DataFrame: `system, replicate, frame, time_ps, PC1…PCk`), `mean_coords`, `atom_names`, `residue_numbers` |
| `fel` | `FELGrid` | `x_edges`, `y_edges`, `x_centers`, `y_centers`, `probability`, `free_energy` (kJ/mol, NaN where unsampled), `temperature`, `smooth_sigma` |
| `states` | `list[BasinState]` | `label`, `population` (fraction), `min_pc1`, `min_pc2`, `min_free_energy`, representative replicate/frame/time, `representative_pdb`, `basin_id` |
| `unassigned_population` | float | Fraction of frames in minor basins |
| `frame_assignments` | DataFrame | Every frame with its CVs, `basin_id` and `state` |
| `convergence` | `ConvergenceResults` | `rmsip_halves`, `progressive_overlap`, `cosine_content`, `population_stability`, `statistical_inefficiency`, `n_eff`, `equilibration`, `sampling_assessment`, `assessment_notes` |
| `replicate_statistics` | DataFrame | Population mean, block SD/SEM, CI per state (`statistics/state_populations.csv`) |
| `run_metadata` | dict | Contents of `run_metadata.json` |
| `extras` | dict | Objects not covered above; keys present today: `bootstrap` (`BootstrapResult` with `fe_std`, `fe_ci_low`, `fe_ci_high`, `state_table`, `barrier_table`, `block_lengths`), `barriers` (DataFrame), `kinetics` (dict of DataFrames), `descriptors` (dict of DataFrames), `state_summary` (DataFrame), `tica_timescales_ps` (array, TICA only) |
| `warnings` | `list[str]` | Warnings emitted during the run (also in `run_metadata.json`) |
| `output_dir` | `Path` | Where everything was written |

Units: coordinates and projections in Å (Python backend) or nm (GROMACS backend),
time in ps, free energies in kJ/mol, R = 0.008314462618 kJ/(mol K).

## Comparisons: `run_condition_comparison`

```python
from conformatlas.config import parse_comparison_config
from conformatlas.comparison import run_condition_comparison

comp = parse_comparison_config("comparison.yaml")
res = run_condition_comparison(comp)
res["population_comparison"]      # DataFrame: per-state populations, CI, bootstrap p, Welch t/p
res["landscape_overlap"]          # DataFrame: Bhattacharyya, Hellinger, JS, Jaccard
res["subspace_overlap"]           # DataFrame: RMSIP between the systems' own subspaces
res["ddg_maps"]["MUT"]["ddg"]     # ΔΔG array on the shared grid (vs the reference system)
```

`run_condition_comparison` returns a dictionary (not a dataclass) whose keys mirror
the files listed under *Comparison outputs* in [Outputs](outputs.md); the figures
are produced by the CLI from this dictionary with the `plot_comparison_*` functions
of `conformatlas.plotting`.

## Building blocks

The functions below operate on arrays and DataFrames and are the units the
validation suite tests against independent implementations
([Benchmarks](benchmarks.md)).

| Step | Function | Validated against |
|---|---|---|
| Superposition | `superpose.kabsch_rotation(mobile, reference, weights=None)`, `superpose.superpose_frame(fit_coords, apply_coords, reference_fit_coords, weights=None)` | MDAnalysis QCP RMSD |
| Streaming | `trajectory.stream_trajectory_coordinates(topology_path, trajectory_path, atom_indices=None, group="Backbone", stride=1, fit=True, fit_atom_indices=None, fit_group=None, reference_fit_coords=None, mass_weighted=False, begin_ps=None, end_ps=None)` → coordinates (n_frames, n_atoms, 3) and times | — |
| PCA | `pca.compute_cartesian_pca(coords_dict, weighting="equal-replicate", n_components=10)` → eigenvalues, eigenvectors, explained and cumulative variance, mean coordinates | MDAnalysis PCA, `gmx covar` |
| TICA | `pca.compute_tica(coords_dict, lag=10, n_components=10, weighting="equal-replicate")`, `pca.implied_timescales(eigenvalues, lag_frames, dt_ps)` | deeptime, AR(1) theory |
| Dihedral features | `features.stream_backbone_dihedrals(topology_path, trajectory_path, selection="protein", stride=1, begin_ps=None, end_ps=None)` → features, times, feature table | MDAnalysis Ramachandran |
| Landscape | `fel.calculate_fel(pc1, pc2, temperature=300.0, bins=64, smooth_sigma=0.0, x_range=None, y_range=None, weights=None)` → `FELGrid`; `fel.choose_bins(cv1, cv2)` (Freedman–Diaconis, 16–128) | Exact Gaussian, `gmx sham` |
| Basins | `basins.detect_energy_basins(fel_grid, projections, min_state_population=0.05, max_states=10, min_barrier=None, weights=None)` → states, unassigned fraction, frame assignments, basin map; `basins.compute_state_barriers(basin_labels, free_energy, states)` | Analytic double well |
| Time series | `timeseries.statistical_inefficiency(x, mintime=3, fast=False)`, `timeseries.effective_sample_size(x)`, `timeseries.detect_equilibration(x, nskip=1, max_discard=0.9)`, `timeseries.block_length_from_g(g, n, min_blocks=5)` | pymbar |
| Bootstrap | `bootstrap.bootstrap_landscape(cv1, cv2, replicate_ids, fel_grid, basin_labels, states, block_lengths, n_boot=200, weights=None, seed=0)` → `BootstrapResult` | Coverage on known Markov systems |
| Kinetics | `kinetics.compute_state_kinetics(frame_assignments, state_labels, dt_ps=None)` (lag-1 counts within replicates) | Known Markov chain |
| Convergence | `convergence.calculate_subspace_rmsip(u, v)`, `convergence.calculate_cosine_content(projections_1d, period_k=1)`, `convergence.compute_convergence_diagnostics(coords_dict, projections_df, frame_assignments_df, n_modes=10)` | Amadei / Hess definitions |
| Descriptors | `descriptors.rmsd_series(coords, reference, subset=None)`, `descriptors.radius_of_gyration(coords, subset=None, masses=None)`, `descriptors.rmsf_per_atom(coords)` | MDAnalysis, `gmx gyrate` |
| Comparison statistics | `comparison_stats.landscape_overlap(p, q)`, `comparison_stats.population_difference_table(states, frame_assignments, system_names, reference, boots)`, `comparison_stats.ddg_map(fel_ref, fel_sys, boot_ref, boot_sys)`, `comparison_stats.subspace_overlap_matrix(eigvecs, n_modes=10)` | SciPy, null-hypothesis experiments |
| Provenance | `provenance.collect_environment()` → versions of Python, NumPy, SciPy, pandas, MDAnalysis, scikit-image, matplotlib, GROMACS and the platform | `importlib.metadata` |

A minimal landscape from two arrays:

```python
import numpy as np
from conformatlas.fel import calculate_fel

rng = np.random.default_rng(0)
x, y = rng.normal(0, 0.7, 50_000), rng.normal(0, 0.7, 50_000)
fel = calculate_fel(x, y, temperature=300.0, bins=40)
print(fel.free_energy.shape, np.nanmin(fel.free_energy), np.isnan(fel.free_energy).any())
```

```text
(40, 40) -0.0 True
```

The minimum is zero by construction (ΔG is relative to the most populated bin) and
bins that received no sample are `NaN`, not zero.

## Reloading a results directory

`conformatlas.results_io.load_results_dir(path)` rebuilds an `AnalysisResults`
from the files `analyze` wrote (it is what `conformatlas analyze --from DIR`
uses). Frame coordinates are re-binned against the saved grid edges, which only
works when the CSV floats are read back exactly: the default `pandas.read_csv`
parser can differ from the written float64 value by one or two units in the
last place, enough to move a frame that sits on the outermost grid edge out of
the grid. The loader therefore reads every CSV with
`float_precision="round_trip"`. Do the same in your own scripts whenever you
re-bin `pca/projections.csv` or `states/frame_assignments.csv` against
`fel/x_edges.npy` / `fel/y_edges.npy`:

```python
import pandas as pd

frames = pd.read_csv("results/states/frame_assignments.csv", float_precision="round_trip")
```

## High-level `conformatlas.api`

`conformatlas.api` exposes the same building blocks as keyword-argument functions
that return in-memory objects, for scripts and notebooks. It re-implements no
numerics: every function calls the engines above, and the chain
`load_trajectories → pca → landscape → states → kinetics` reproduces
`api.analyze` exactly (identical eigenvalues, ΔG, states, dwell summary and
RMSIP; `validation/checks_performance.py`). The signatures below were generated
with `inspect.signature` from the installed module.

```text
api.load_trajectories(topology, trajectories,
    *, group='Backbone', fit_group=None, lig_group=None, index_file=None, stride=1, begin_ps=None, end_ps=None, fit=True, mass_weighted_fit=False, system_name='System')
api.pca(data,
    *, n_components=10, weighting='equal-replicate', method='cartesian', lag=10, times=None, system_name=None)
api.landscape(source,
    *, temperature=300.0, bins=None, smooth_sigma=0.0, pcs=(1, 2), cv_columns=None, cv_names=None, cv_unit='Å', weights=None)
api.states(land,
    *, min_population=0.05, max_states=10, min_barrier=None, min_peak_distance=3)
api.kinetics(assignment, dt_ps=None)
api.convergence(data, pcs, assignment=None,
    *, n_modes=10)
api.analyze(topology, trajectories,
    *, output_dir='Outputs', echo=None, **options)
api.plot_landscape(source, ax=None,
    *, states=None, vmax=None, cmap='turbo', levels=40, contour=True, annotate=True, cv_names=None, cv_unit=None)
```

| Function | Returns |
|---|---|
| `api.load_trajectories` | `Ensemble` — fitted coordinates and times per replicate; `replicate_ids`, `n_frames`, `n_atoms`, `dt_ps` |
| `api.pca` | `PCAResults` (`method="tica"` with `lag` frames gives TICA) |
| `api.landscape` | `Landscape` — `fel` (`FELGrid`), `cv` table, `cv_names`, `cv_unit`, `bins`, `weights` |
| `api.states` | `StateAssignment` — `states`, `unassigned_population`, `frame_assignments`, `basin_map`, `barriers`, `landscape`; `.labels`, `.populations()` |
| `api.kinetics` | dict of DataFrames `transition_counts`, `transition_counts_per_replicate`, `transition_matrix`, `dwell_times`, `dwell_summary`, `mfpt`, `summary`, plus `dt_ps` (raises for weighted ensembles) |
| `api.convergence` | `ConvergenceResults` |
| `api.analyze` | `AnalysisResults` — the full pipeline; every `AnalysisConfig` field is accepted as a keyword argument, unknown ones raise `TypeError` |
| `api.plot_landscape` | `(fig, ax, mappable)` |

Type annotations (omitted above): paths are `str | Path`; `data` is an `Ensemble` or a mapping replicate → `(n_frames, n_atoms, 3)` array; `source` of `landscape` is a `PCAResults`, a projections DataFrame or an `AnalysisResults`; `source` of `plot_landscape` is a `Landscape`, `FELGrid` or `AnalysisResults`.

Run on the neuraminidase fixture (81 frames, 387 Cα atoms, 250 ps between frames):

```python
from conformatlas import api

ens = api.load_trajectories(
    "tests/data/h1n1_ca/neuraminidase_ca.pdb",
    ["tests/data/h1n1_ca/neuraminidase_ca.xtc"],
    group="C-alpha", index_file="tests/data/h1n1_ca/index.ndx",
)
pcs = api.pca(ens, n_components=10)
land = api.landscape(pcs, temperature=300.0, bins=12)
st = api.states(land, min_population=0.05)
kin = api.kinetics(st, dt_ps=ens.dt_ps)
print(ens.n_frames, ens.n_atoms, round(ens.dt_ps, 1))
print(pcs.explained_variance[:2].round(1))
print([(s.label, round(100 * s.population, 1)) for s in st.states])
print(sorted(kin))
```

```text
81 387 250.0
[26.  12.3]
[('State A', 44.4), ('State B', 28.4), ('State C', 12.3), ('State D', 8.6)]
['dt_ps', 'dwell_summary', 'dwell_times', 'mfpt', 'summary', 'transition_counts', 'transition_counts_per_replicate', 'transition_matrix']
```

`api.plot_landscape(land, states=st.states)` draws the landscape on a
matplotlib axis and returns `(fig, ax, mappable)`; `conformatlas.plotting.figure_style(...)`
is a context manager for the colormap, font scale, dpi and formats used by all
figure functions. The executed notebook
`examples/notebooks/neuraminidase_walkthrough.ipynb` walks through the same
chain.

## Results container: `conformatlas.results_io`

`save_results(results, path)` writes one HDF5 file (h5py; `conformatlas[hdf5]`)
holding the complete `AnalysisResults`; `load_results(path)` is its inverse.
`load_results_dir(path)` rebuilds the object from a results *directory*
(eigenvectors are then empty because the pipeline does not write them) and is
what `conformatlas analyze --from` uses. `rebin(results, bins, smooth_sigma,
*, min_state_population=None, max_states=None, min_barrier=None, weights=None,
temperature=None)` recomputes the landscape, basins, barriers and population
table on a new grid from the stored CV columns, leaving the PCA untouched; the
bootstrap, kinetics and representatives of the source are dropped, not
recomputed, and `extras` keeps `descriptors` and gains `barriers` and
`basin_map`. Rebinned results are bitwise identical to a fresh pipeline run with
the same grid (validation check).

```python
from conformatlas import api
from conformatlas.results_io import load_results, rebin, save_results

res = api.analyze(
    "tests/data/h1n1_ca/neuraminidase_ca.pdb",
    ["tests/data/h1n1_ca/neuraminidase_ca.xtc"],
    output_dir="tutorial/api_io", index_file="tests/data/h1n1_ca/index.ndx",
    lsq_group="C-alpha", cov_group="C-alpha", bins=12, n_bootstrap=0, interactive=False,
)
path = save_results(res, "tutorial/api_io/results.h5")
back = load_results(path)
print(back.fel.free_energy.shape, (back.pca.projections["PC1"] == res.pca.projections["PC1"]).all())
fine = rebin(res, bins=16, smooth_sigma=1.0)
print([(s.label, round(100 * s.population, 1)) for s in res.states])
print([(s.label, round(100 * s.population, 1)) for s in fine.states])
print(sorted(fine.extras))
```

```text
(12, 12) True
[('State A', 44.4), ('State B', 28.4), ('State C', 12.3), ('State D', 8.6)]
[('State A', 46.9), ('State B', 40.7), ('State C', 12.3)]
['barriers', 'basin_map', 'descriptors']
```

## Streaming PCA: `conformatlas.streaming`

`streaming_pca(topology, trajectories, atom_indices, *, fit=True,
fit_atom_indices=None, reference_fit_coords=None, mass_weighted=False, stride=1,
begin_ps=None, end_ps=None, weighting="equal-replicate", n_components=10,
chunk_size=256, n_workers=1, precision="float64", projection_store=None,
system_name="System", quiet=False, keep_covariance=False)` is the two-pass
engine behind `--engine streaming` ([Methods](methods.md) §19). `trajectories`
is a mapping replicate → path (or a sequence); `atom_indices` are 0-based atom
indices into the topology. It returns a `StreamingPCAResult` with
`eigenvalues`, `eigenvectors`, `explained_variance`, `cumulative_variance`,
`mean_coords`, `projection_array` (an array or, with `projection_store`, a
`.npy` memmap), `times`, `fit_rmsd`, and `to_pca_results()` for a drop-in
`PCAResults`. The building blocks — `CovarianceAccumulator`,
`accumulate_replicate`, `combine_replicate_moments`,
`eigendecompose_covariance`, `project_replicate`, `iter_frame_chunks` — are
public. Eigenvalues agree with the in-memory `compute_cartesian_pca` to
machine precision:

```python
import numpy as np
import MDAnalysis as mda
from conformatlas.streaming import streaming_pca
from conformatlas.pca import compute_cartesian_pca
from conformatlas.trajectory import stream_trajectory_coordinates

top, trj = "tests/data/h1n1_ca/neuraminidase_ca.pdb", "tests/data/h1n1_ca/neuraminidase_ca.xtc"
idx = mda.Universe(top).select_atoms("name CA").indices
res = streaming_pca(top, {"Rep_1": trj}, idx, n_components=5, chunk_size=32, quiet=True)
coords, times = stream_trajectory_coordinates(top, trj, atom_indices=idx)
ref_eigenvalues = compute_cartesian_pca({"Rep_1": coords}, n_components=5)[0]
print(res.n_frames, res.projection_array.shape, res.precision)
print(np.allclose(res.eigenvalues, ref_eigenvalues, rtol=1e-10), res.explained_variance[:2].round(1))
print(type(res.to_pca_results()).__name__)
```

```text
{'Rep_1': 81} (81, 5) float64
True [26.  12.3]
PCAResults
```

Worker processes use `fork` on Linux (`CONFORMATLAS_MP_START=spawn|forkserver`
overrides it) and fall back to serial execution with a logged warning when
processes cannot be started. Progress bars (`conformatlas.progress`) are
silent off a terminal and with `CONFORMATLAS_NO_PROGRESS=1`.

## MDAnalysis interface: `conformatlas.mda_analysis`

`ConformAtlasPCA(atomgroup, fit_atomgroup=None, reference=None, fit=True,
mass_weighted=False, n_components=10, precision="float64", chunk_size=256,
**AnalysisBase kwargs)` is an MDAnalysis `AnalysisBase` subclass around the
same accumulator, so `start/stop/step` slicing and combination with other
MDAnalysis analyses work as usual. After `.run()`, `results` holds
`eigenvalues` (Å²), `eigenvectors`, `explained_variance` (%),
`cumulative_variance`, `mean`, `projections` (n_frames × k, Å),
`covariance` and `n_frames`. A single trajectory is one replicate, so the
`equal-replicate` and `frames` weightings coincide. Validated against
`MDAnalysis.analysis.pca.PCA` (9.4·10⁻⁹ relative after its (N−1)/N
normalisation) and against the in-memory path on a sliced run.

```python
import MDAnalysis as mda
from conformatlas.mda_analysis import ConformAtlasPCA

u = mda.Universe("tests/data/h1n1_ca/neuraminidase_ca.pdb", "tests/data/h1n1_ca/neuraminidase_ca.xtc")
pca = ConformAtlasPCA(u.select_atoms("name CA"), n_components=5).run(step=2)
print(pca.results.n_frames, pca.results.projections.shape)
print(pca.results.explained_variance[:3].round(1), pca.results.eigenvalues[:2].round(2))
```

```text
41 (41, 5)
[26.5 13.   7.9] [103.29  50.8 ]
```

## Reweighting: `conformatlas.reweighting`

The module behind `conformatlas reweight` ([Methods](methods.md) §17).
Metadynamics: `read_hills(path)`, `read_colvar(path, columns)`,
`metad_weights(cv_values, cv_times, hills, temperature, method="tiwary-parrinello"|"final-bias",
skip_time=0.0, grid_axes=None, bins=200, bias_factor=None, hill_shape="plumed")`
→ `MetadReweightResult` (`times`, `bias`, `rct`, `log_weights`, `weights` with
mean 1 over kept frames, `kept`, `hill_times`, `rct_history`),
`hills_bias`, `hills_bias_at_times`, `metad_rct`, `match_times`,
`write_weights_file`, `pmf_1d`. Umbrella sampling: `UmbrellaWindow(samples,
center, kappa, name="", period=None)`, `wham(windows, temperature, bins=100,
...)`, `mbar(windows, temperature, bins=100, ...)` → `UmbrellaResult`
(`window_free_energies` in kJ/mol with f₀ = 0, `weights` per window summing to
1 overall, `pmf`, `n_iter`, `converged`), plus `mbar_solve`, `mbar_weights`,
`umbrella_windows_from_colvars`, `write_umbrella_outputs`. Three harmonic
windows drawn from their own restraint distributions (flat underlying PMF):

```python
import numpy as np
from conformatlas.reweighting import UmbrellaWindow, mbar, wham

rng = np.random.default_rng(0)
kT = 0.008314462618 * 300.0
centers, kappa = [0.3, 0.5, 0.7], 500.0           # nm, kJ/mol/nm^2
windows = [UmbrellaWindow(rng.normal(c, np.sqrt(kT / kappa), 4000), c, kappa, name=f"w{i}") for i, c in enumerate(centers)]
res = mbar(windows, temperature=300.0, bins=40)
ref = wham(windows, temperature=300.0, bins=40)
print(res.converged, res.n_iter, res.window_free_energies.round(2))
print(np.abs(res.window_free_energies - ref.window_free_energies).max().round(3))
print(res.pmf.columns.tolist(), len(res.pmf))
print([round(float(w.sum()), 3) for w in res.weights], round(sum(float(w.sum()) for w in res.weights), 6))
```

```text
True 13 [ 0.   -0.02 -0.21]
0.014
['center', 'pmf_kJ_mol', 'probability', 'weight_sum', 'count'] 40
[0.404, 0.173, 0.423] 1.0
```

For metadynamics, `read_hills` → `read_colvar` → `metad_weights` →
`write_weights_file(path, times, weights)` produces the file that
`analyze --weights-file` reads; `rw.kept` marks the frames after `skip_time`
and `rw.weights[rw.kept].mean()` is 1 by construction.

## Residue-level analyses: `conformatlas.residue_analysis`

`dccm(coords, weights=None)` (atoms × atoms), `dccm_per_state(coords,
frame_states)`, `collapse_to_residues(matrix, atom_table)`,
`per_residue_rmsf`, `state_residue_differences(coords, atom_table,
frame_states, state_a, state_b)`, `all_state_residue_differences`,
`ligand_residue_contacts(topology, trajectory, ligand_selection,
protein_selection="protein", cutoff=4.5, ...)` → `ContactResult`,
`contact_persistence(result, frame_states=None, weights=None)`,
`binding_site_residues(persistence, threshold=0.5)`, `binding_site_rmsf`,
`atom_table_from_universe(atomgroup)`, `write_residue_outputs` and the
pipeline driver `analyze_residues`. Coordinates must already be superposed
(the arrays returned by `stream_trajectory_coordinates` with `fit=True`);
nothing is re-fitted per state ([Methods](methods.md) §18).

```python
import MDAnalysis as mda
from conformatlas.residue_analysis import (atom_table_from_universe, binding_site_residues, collapse_to_residues,
                                           contact_persistence, dccm, ligand_residue_contacts)
from conformatlas.trajectory import stream_trajectory_coordinates

top, trj = "tests/data/h1n1_ca/neuraminidase_ca.pdb", "tests/data/h1n1_ca/neuraminidase_ca.xtc"
ca = mda.Universe(top).select_atoms("name CA")
coords, times = stream_trajectory_coordinates(top, trj, atom_indices=ca.indices)
C = dccm(coords)
C_res, table = collapse_to_residues(C, atom_table_from_universe(ca))
print(C.shape, C_res.shape, round(float(C.diagonal().mean()), 3), round(float(C.min()), 2))
contacts = ligand_residue_contacts(top, trj, "resname ZMR", cutoff=4.5)
pers = contact_persistence(contacts)
site = binding_site_residues(pers, threshold=0.5)
print(pers.columns.tolist())
print(site[["residue_number", "residue_name", "persistence_all"]].round(2).to_string(index=False))
```

```text
(387, 387) (387, 387) 1.0 -0.64
['chain', 'residue_number', 'residue_name', 'n_atoms', 'persistence_all', 'mean_min_distance_all', 'max_persistence']
 residue_number residue_name  persistence_all
             70          ASP             0.96
             71          ARG             0.94
```

(The fixture carries only Cα atoms of the protein, so contacts are counted
between ligand heavy atoms and Cα atoms; on an all-atom topology every heavy
atom of a residue counts.)

## Markov state models: `conformatlas.msm`

`build_msm(projections, columns, lag, lags=None, n_clusters=100,
clustering="kmeans", dmin=None, n_metastable=2, reversible=True, dt_ps=None,
n_timescales=5, ck_k_max=5, seed=0, basin_states=None)` runs the whole layer
of [Methods](methods.md) §15 on a projections DataFrame (`replicate`, `frame`,
`time_ps` and the component columns) and returns an `MSMResult`
(`clustering`, `msm` with `active_set`, `transition_matrix`, `stationary`,
`implied_timescales`, `lag_report`, `memberships`, `metastable_sets`,
`frame_table`, `ck_test`, `ck_summary`, `mfpt_ps`,
`coarse_transition_matrix`, `coarse_stationary`, `basin_overlap`, `notes`);
`save_msm(result, output_dir)` writes `msm/`. The individual steps —
`kmeans_cluster`, `regular_space_cluster`, `count_matrix`,
`transition_matrix_reversible`, `stationary_distribution`,
`implied_timescales_scan`, `chapman_kolmogorov_test`, `pcca_plus`,
`coarse_grain`, `mfpt_between_sets`, `basin_overlap_table` — are public and
operate on NumPy arrays. A two-state chain with T = [[0.95, 0.05], [0.10, 0.90]]
(slowest implied timescale −Δt/ln 0.85 = 61.5 ps at Δt = 10 ps, π = (2/3, 1/3)):

```python
import numpy as np
import pandas as pd
from conformatlas.msm import build_msm

rng = np.random.default_rng(0)
T_true = np.array([[0.95, 0.05], [0.10, 0.90]])
wells = np.array([[-1.0, 0.0], [1.0, 0.0]])
frames = []
for rep in ("Rep_1", "Rep_2"):
    s = 0
    for i in range(3000):
        s = rng.choice(2, p=T_true[s])
        x = wells[s] + rng.normal(0, 0.25, 2)
        frames.append((rep, i, 10.0 * i, x[0], x[1]))
proj = pd.DataFrame(frames, columns=["replicate", "frame", "time_ps", "PC1", "PC2"])
msm = build_msm(proj, ["PC1", "PC2"], lag=5, lags=[1, 2, 5, 10, 20], n_clusters=20, n_metastable=2, dt_ps=10.0)
print(msm.msm.active_set.size, msm.coarse_stationary.round(3))
print(msm.implied_timescales[["lag_frames", "ts_1"]].round(1).to_string(index=False))
print(msm.ck_summary["passed"], msm.lag_report["suggested_lag_frames"])
print(msm.mfpt_ps.round(0))
```

```text
20   metastable_set  stationary_probability
0            MS1                    0.33
1            MS2                    0.67
 lag_frames  ts_1
          1  57.1
          2  58.6
          5  58.1
         10  55.1
         20  78.3
True 1
          to_MS1  to_MS2
from_set                
MS1          0.0   130.0
MS2        256.0     0.0
```

The timescale of 57–58 ps at lags 1–5 is the finite-sample estimate of the
61.5 ps reference; the set populations recover π (labels are assigned by PCCA+,
so MS1/MS2 need not follow the order of the generating states). The MSM layer
does not accept per-frame weights.
