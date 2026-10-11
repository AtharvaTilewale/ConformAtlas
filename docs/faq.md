# Frequently asked questions and pitfalls

Each entry states what goes wrong, how to recognise it in the output, and what to
do. Where the answer depends on a convention, the convention is stated.

```{contents}
:local:
:depth: 1
```

## My landscape has frames at huge PC values and the RMSD jumps between frames

**Cause.** The molecule was broken across the periodic boundary or jumped between
periodic images in the trajectory. Cartesian PCA sees a chain that is split across
the box as an enormous displacement, and a single such frame dominates PC1.

**Symptoms.** `check_inputs` prints a warning that consecutive Cα atoms are farther
apart than a peptide bond allows (the check uses 4.5 Å between neighbouring Cα
atoms) and the first eigenvalue is orders of magnitude larger than the second.

**Fix.** Make molecules whole and remove jumps before analysing, e.g.

```shell
gmx trjconv -s md.tpr -f md.xtc -o md_whole.xtc -pbc mol -center -ur compact
```

(choose the protein as the centring group and the system as output). The
least-squares superposition performed by ConformAtlas removes rigid-body motion
but cannot repair a broken molecule.

## I passed a `.tpr` and the fit looks wrong / the RMSD to the reference is large at frame 0

A run-input file (`.tpr`) carries the coordinates of the *pre-processing input*
structure, which is generally not in the same frame of reference as a trajectory
that has been treated with `trjconv -pbc -center`. ConformAtlas therefore never
uses `.tpr` coordinates as the superposition reference: with a `.tpr` topology the
reference is the **first frame of the first trajectory**, and `run_metadata.json`
records the reference file under `superposition.reference`. Pass a `.gro` or `.pdb`
(for example the frame you want to report RMSD against) as `-s` if you want an
explicit reference; the file must contain the same atoms as the trajectory.

## Which atoms are "Backbone"? My GROMACS and Python results differ

`Backbone` means **N, CA, C**, the GROMACS definition, in both backends since 1.0.0. The MDAnalysis selection keyword `backbone` means N, CA, C, **O**; that set is
available under the name `MainChain`. Mixing the two definitions changes the
eigenvalues by tens of percent, which is how the discrepancy was found. The
mappings are:

| Group name | Atoms |
|---|---|
| `Backbone` | N, CA, C (GROMACS `Backbone`) |
| `MainChain` | N, CA, C, O (GROMACS `MainChain`) |
| `C-alpha` | CA |
| `Protein` | all protein atoms |
| anything else | passed to MDAnalysis as a selection string, or looked up in `-n index.ndx` |

Groups from an index file take precedence over the built-in names, and the same
group must refer to the same atoms in every system of a comparison.

## My ΔG values differ from `gmx sham` by a constant

`gmx sham` uses **298.15 K** unless `-tsham` is given; ConformAtlas has no default
temperature and uses the `-T` value you pass. A landscape at 300 K is scaled by
300/298.15 relative to sham's default, i.e. by 0.6 % — 0.03 kJ/mol at 5 kJ/mol.
Larger differences are usually the second convention: `gmx sham` fills
**unsampled bins with the maximum sampled free energy**, whereas ConformAtlas leaves
them undefined (`NaN`, grey in the figures) because a bin that was never visited has
no measured free energy, only a lower bound. On identical projections, bin edges
and temperature the two agree bin-for-bin to the resolution of sham's XPM output
(validation check "ΔG grid vs gmx sham", median |Δ| 0.006 kJ/mol).

## Why does `compare` print two p-values, and why do they disagree?

The bootstrap p-value resamples blocks of frames within each replicate; it asks
whether the population difference could be produced by frame-to-frame noise given
the trajectories you have. Welch's t-test uses the per-replicate populations as
the samples; it asks whether the difference would reproduce in new, independent
simulations. With two replicates per system the t-test has at most two degrees of
freedom and almost no power, so a tiny bootstrap p together with a large Welch p is
the expected outcome of under-replicated sampling (the
[tutorial](tutorial_neuraminidase.md) shows exactly this). For a claim about a
mutation the replicate-level test is the relevant one; plan for three or more
replicates per system. Welch's t-test is `scipy.stats.ttest_ind(equal_var=False)`
on the replicate populations and assumes those populations are approximately
normal across replicates.

## N_eff is tiny although I have thousands of frames

N_eff = N / g where g = 1 + 2τ is the statistical inefficiency of the PC1/PC2
projections (Chodera 2007/2016, computed exactly from the autocorrelation
function). g is measured in frames, so N_eff is the trajectory length divided by
the correlation time of the slowest mode in the plane, independent of how often
you saved frames. Saving every 1 ps instead of every 100 ps multiplies N by 100 and
g by 100. To raise N_eff, simulate longer or run more independent replicates; see
`convergence/effective_samples.csv` for g per replicate and the discussion in the
{ref}`tutorial <tutorial-why-neff>`.

## `--equilibration auto` discarded half of my trajectory

The criterion (Chodera 2016) chooses the start time t₀ that maximises the
effective sample size of the production segment, evaluated on the fit-group RMSD.
If RMSD drifts for most of the run, the optimal t₀ is late and N_eff of the
remainder is still small. The detection is reported in
`convergence/equilibration.csv` whether or not it is applied; a late t₀ is a
statement about the trajectory, not an error. Compare the `N_eff_full` and
`N_eff_production` columns before deciding.
`--equilibration-on cv` runs the same criterion on both landscape CVs as well
and adopts the later start (a second row per replicate in the table, observable
`PC1+PC2 (latest t0)`); with `--equilibration auto` the decomposition is then
refitted on the production window so that the PCA basis excludes the
transient.

## When should I use dihedral PCA or TICA instead of Cartesian PCA?

- **Cartesian PCA** (default) describes collective displacements of the chosen
  atoms after superposition. It is appropriate for folded proteins with a stable
  core; the fit removes rigid-body motion but makes the coordinates depend on the
  fit group.
- **Dihedral PCA** (`--pca-type dihedral`; sin/cos of backbone φ/ψ, Mu et al. 2005)
  needs no superposition and is the better choice for peptides and intrinsically
  disordered or multi-domain systems, where no rigid core exists and Cartesian
  PC1 mostly reflects the choice of fit atoms. It requires a topology with full
  backbone atoms (N, CA, C of consecutive residues).
- **TICA** (`--pca-type tica --tica-lag N`) finds the *slowest* linear
  combinations rather than the largest-amplitude ones (Pérez-Hernández 2013). Use it
  when the interesting transition is slow but small in amplitude, or as the input
  space for state definition. Check `pca/tica_timescales.csv`: the implied
  timescales should be insensitive to the lag time; if they grow with the lag the
  process is not Markovian at that lag and the result should not be interpreted
  kinetically. TICA on a single short trajectory is unreliable.

In every case the landscape is only as meaningful as the two coordinates it is
built on; inspect the explained variance (Cartesian) or the implied timescale gap
(TICA) before interpreting basins.

## How many bins should the landscape have?

The default (`--bins auto`) applies the Freedman–Diaconis rule to each CV and clips
the result to 16–128 bins per axis. The rule balances resolution against counting
noise for the given number of frames; it is a heuristic, not a convergence
criterion. Two checks worth making:

1. The pipeline warns when the occupied bins hold fewer than two frames on
   average; at that point each ΔG value carries ≈ kT of counting noise and the
   basin segmentation follows the noise.
2. State populations should be stable when the bin count is changed by ±30 %. If
   they are not, the states are artefacts of the grid; smooth (`--smooth-sigma 1`)
   or use fewer bins, and report that you did.

For short trajectories such as the 81-frame tutorial fixture, the automatic rule
would give 16 bins (the floor) and a fixed smaller grid is more honest.

## `barriers.csv` is empty

Barriers are measured at the saddle between *adjacent* basins on the grid. If two
states are separated by unsampled bins there is no path through sampled space and
no saddle can be measured; the file is then empty rather than containing a guess.
Smoothing or coarser bins can connect the basins, but the barrier would then rest
on very few frames; the bootstrap interval in the table reports how few.

## Can I use ConformAtlas with biased (metadynamics, umbrella) trajectories?

Only through `--weights-file`, which supplies per-frame statistical weights
that enter the histogram, the populations and the bootstrap. Kinetics (and the
MSM layer) are skipped for weighted frames because transition counts of a
biased trajectory are not those of the unbiased system. The weights can be
produced by `conformatlas reweight metad` (PLUMED `HILLS` + `COLVAR`,
Tiwary–Parrinello c(t) or final-bias) and `conformatlas reweight umbrella`
(MBAR or WHAM over harmonic windows); one weight file per trajectory, with as
many rows as analysed frames — use `--frame-times` or a COLVAR stride equal to
the trajectory output stride. Formulas, conventions and the validation against
`plumed sum_hills` and pymbar are in [Methods](methods.md) §17; only
one-dimensional metadynamics was validated against a known potential.

## When can I trust the Markov state model?

`--msm` builds the model at the lag *you* give (`--msm-lag`); nothing is chosen
automatically. Before quoting MFPTs or set populations check, in this order:

1. `msm/implied_timescales.csv` and `figures/msm_implied_timescales.png`: the
   resolved timescales (those longer than the lag) should be flat in the lag.
   `msm_summary.json` reports `plateau_found` and `suggested_lag_frames` using a
   10 % relative-change rule, which is a reporting convention, not a test.
2. `active_count_fraction` close to 1: a large fraction of counts outside the
   largest connected set means poorly connected microstates.
3. The Chapman–Kolmogorov test (`ck_test.passed`, `figures/msm_ck_test.png`):
   the model at lag τ must predict the directly estimated kinetics at 2τ … kτ.
   The pass threshold `max(0.05, 3 SE)` is a convention; the deviations are exact.
4. `msm/basin_overlap.csv`: `row_fraction_max` near 1 means the kinetic sets
   coincide with the landscape basins; otherwise the basins are not kinetically
   separated at that lag.

Short runs such as the bundled demonstration data fail point 1 by
construction (no plateau); the MSM outputs then illustrate the file formats and
nothing more. The MSM uses sliding-window counts within replicates and does not
accept per-frame weights.

## The run says it switched to the streaming engine / my machine ran out of memory

The in-memory path stores every fitted frame of the analysis group (float32,
3·n_atoms·n_frames·4 bytes). When the estimate exceeds 2 GB (`run_metadata.json`
→ `memory_estimate_gb`), `--engine auto` switches to the two-pass streaming
engine, which holds only a 3N × 3N covariance per replicate and reads each
trajectory twice; `--n-workers` processes one replicate per worker. PCA
results are identical to the in-memory path (validated to 10⁻¹⁴ relative), but
the steps that need all coordinates at once — descriptors, per-state average
structures and RMSF, mode structures, pairwise replicate RMSIP — are skipped
and a warning is recorded. Force either path with `--engine memory` or
`--engine streaming`; reduce memory further with `--stride`, a smaller
analysis group (`--cov-group C-alpha`) or `--precision float32` for the stored
projections. The streaming engine supports Cartesian PCA only and has no
Gram/dual route: for more than ≈ 7 000 analysis atoms use a coarser selection.

## Can I change the grid or the state thresholds without re-reading the trajectories?

Yes: `conformatlas analyze --from results/ --bins 48 --smooth-sigma 1.0 -o
results_48/` reloads the stored projections and recomputes the landscape,
basins, barriers, bootstrap, kinetics and (with `--msm`) the MSM with the new
`--bins/--smooth-sigma/--min-state-population/--max-states/--min-barrier`.
The PCA tables are copied; descriptors and structures that need coordinates
are copied or skipped. In Python the same step is
`conformatlas.results_io.rebin(results, bins, smooth_sigma)`.

## How many states are there really? The bootstrap keeps the labels fixed

By default the bootstrap holds the watershed labels fixed and resamples the
frames, so its intervals do not include the uncertainty of the segmentation
itself. `--bootstrap-mode resegment` re-runs the watershed on every resample
and writes `states/state_count_distribution.csv` (how often 1, 2, 3 … major
states were found) and `states/state_populations_resegmented.csv`
(`match_rate` per state and population intervals that include the
segmentation step). If the state count varies across resamples, the number of
states is not resolved by the data and should be reported as such. The
matching rule between resampled and reference states (≥ 50 % frame overlap, or
the minimum inside the reference basin) is a definition; the coverage of the
resulting intervals was measured at 0.89 for a nominal 0.95 on synthetic
two- and three-well systems, slightly anti-conservative like the fixed mode.

## `report.html` is 10–15 MB. Can it be smaller?

The report embeds the Plotly library (≈ 4.8 MB), the NGL viewer (≈ 1.3 MB),
every figure and every PDB it displays, so that it works without network
access. `--plotly-js cdn` loads Plotly from `cdn.plot.ly` instead (≈ 5 MB
less, needs a connection when opened); `--no-interactive` drops the Plotly
landscape and the viewer altogether. `--pdf` writes a static-figure PDF next
to it.

## The command exited with code 2 before reading anything

Exit code 2 means a usage or configuration problem detected up front:
malformed YAML (the message gives line and column), a missing input file, a
group name that exists neither as a built-in nor in the index file (the
available groups are listed), an invalid colormap / figure format / font scale
/ dpi, or a feature whose optional dependency is not installed (`--pdf`,
`--interactive`, `--save-hdf5`; the `pip install` command is printed). Exit
code 1 is an analysis failure; the traceback is in the log file of the output
directory.

## What exactly is validated?

`validation/VALIDATION.md` lists every check with its reference and tolerance;
[Benchmarks](benchmarks.md) summarises it. Quantities not listed there (the state
labelling heuristics, the sampling verdict thresholds, the Freedman–Diaconis grid
choice) are documented as heuristics in [Methods](methods.md).
