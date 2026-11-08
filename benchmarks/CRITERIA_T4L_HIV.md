# Benchmark pass criteria — T4 lysozyme replicates and HIV-1 protease Markov model

**Status:** fixed and committed *before any T4L or HIV-1 protease result was computed* (only data integrity,
file formats and the authors' published settings were inspected). Tolerances are not changed after seeing
results; departures go to `DEVIATIONS.md`; a failed gate is reported as failed.

Lessons from the ADK benchmark applied here: the NumPy reference uses the exact SI value
k_B = 1.380649e-23 J/K × 6.02214076e23 mol⁻¹ / 1000 = 0.00831446261815324 kJ mol⁻¹ K⁻¹ (ADK D4); ProDy
normalises its covariance by N (ADK D2); GROMACS projections are not comparable at Level 1 (ADK D1).

## Code versions
* **Released code**: ConformAtlas at the base of the `benchmark` branch (1.0.2 plus documentation).
* **Timescale fix**: branch `fix/msm-timescale-ordering` (implied timescales ranked by |λ|, the deeptime /
  PyEMMA convention). Every gate on implied timescales uses this code and says so.
* **Lanczos PCA**: branch `perf/truncated-pca`, used only in the performance tables, labelled.
* Note added 2026-11-08 (after T0–T7, before T4-R): both branches were merged into `dev` with their code unchanged (only changelog and documentation conflicts resolved)
  (merge commits `d6bcbc2`, `a1dac8e`); T4-R uses `dev`.

## Data (public, not in the repository)

| Item | Source | Integrity gate |
|---|---|---|
| T4L, cysteine-free pseudo-wild type (T54, A97), 5 × 1 µs, AMBER ff99SB*-ILDN + TIP4P/2005, protein-only xtc at 1 ps | Zenodo 10.5281/zenodo.3989057 (Hoffmann, Mulder & Schäfer) | md5; 2,612 protein atoms; 162 residues |
| T4L, same protein, 3 × ~1.07 µs, AMBER ff15ipq + SPC/Eb | Zenodo 10.5281/zenodo.3991429 | md5; 2,612 protein atoms |
| HIV-1 protease, 1,000 × 130 ns apo simulations, lambda features, the authors' PyEMMA k-means, MSM (lag 300 frames = 30 ns) and 5-state HMM | Zenodo 10.5281/zenodo.5006613 (Sadiq, Muñiz Chicharro, Friedrich & Wade 2021, CC-BY 4.0) | md5; 1,000 feature files × 1,300 frames × 3 values |

T4L analysis copies are written once with `gmx trjconv -dt 100` (100 ps spacing) and a protein-only run file with
`gmx convert-tpr` (group Protein). **T0 (gate):** every 100 ps copy has dt = 100 ps and ⌊length/100 ps⌋ + 1 frames,
and every frame of every copy has all consecutive Cα–Cα distances ≤ 4.2 Å (no molecule broken across the periodic
boundary). The analysis temperature is the thermostat reference temperature read from the run file.

## T4 lysozyme

### T1 — PCA numerics, Level 1 (gate)
All 5 ff99SB*-ILDN replicates at 100 ps, pooled (frames weighting). Every frame is fitted once, before any tool
sees it, by NumPy Kabsch superposition on the Cα atoms to the first frame of md1us4; tools get these identical
coordinates and apply no further fit. Selections: Cα and protein heavy atoms. Reference: independent NumPy
covariance (N-normalised) + `eigh`. Gates exactly as ADK G1: double-precision tools (ConformAtlas, MDAnalysis,
ProDy, MDTraj + scikit-learn full solver) ≤ 1e-6 on relative eigenvalue error, explained-variance error
(percentage points), 1 − |cos| for modes separated by ≥ 1 % from both neighbours, and projection error / SD;
GROMACS ≤ 1e-4 (eigenvalues, eigenvectors), ≤ 1e-3 (explained variance), projections not applicable.
ConformAtlas must also meet its row against the NumPy reference by itself.

### T2 — free-energy landscape (gate)
Cα PC1–PC2 of T1, analysis temperature, 32×32 and 64×64 bins with `gmx sham`'s range convention, smoothing off.
ConformAtlas vs NumPy histogram inversion (exact SI k_B): ≤ 1e-9 kJ/mol over occupied bins and identical occupied
bins. ConformAtlas vs `gmx sham` (`-dist` output, same text input): ≤ 1e-3 kJ/mol, identical occupied bins.

### T3 — kinetics vs deeptime (gate; timescales use the timescale-fix code)
Discretisation computed once: deeptime k-means (100 clusters, k-means++, fixed seed 0) on PC1–PC3 of T1; the
5 replicates are 5 separate discrete trajectories (no transition counted across a replicate boundary).
* TICA eigenvalues (top 5, input top 10 PCs, lags 10 and 50 frames) on the single trajectory md1us4:
  relative error vs deeptime ≤ 1e-8. The pooled 5-trajectory TICA is reported, not gated (mean-removal conventions
  for multiple trajectories differ between implementations).
* Reversible MSM on the 5 discrete trajectories, lags 10 and 50 frames: same active set; max |Δπ| ≤ 1e-7;
  max |ΔT| ≤ 1e-7; top-5 implied timescales relative error ≤ 1e-5.

### T4 — null test of `conformatlas compare` (gate; the main test of this phase)
All 10 ways of splitting the 5 ff99SB*-ILDN replicates into a group of 2 and a group of 3. Each split is analysed
with `conformatlas compare` (systems "A" = 2 replicates, "B" = 3 replicates; Cα fit and analysis groups; analysis
temperature; all other settings default). The two groups come from the same force field, protocol and protein,
so every true population difference is zero. Measured over all (split, state) pairs:
* **bootstrap false-positive rate** = fraction whose 95 % bootstrap CI of the population difference excludes 0;
* Welch false-positive rate = fraction with Welch p < 0.05 (reported, not gated).

**Gate: bootstrap false-positive rate ≤ 0.15** (nominal 0.05; the allowance covers binomial noise of ~30
correlated tests). Declared before running: the bootstrap resamples frames *within* replicates, so if replicates
sample different regions its intervals miss the between-replicate variance and the gate will fail; such a failure
is a real limitation and is reported as one.

### T4-R — re-test of the null after the `compare` fix (gate; added 2026-11-08, before the re-run)
T4 failed with the 1.0.2 code (bootstrap false-positive rate 1.00). The fix on branch `dev` makes the default
bootstrap hierarchical (`bootstrap_level: replicate`: replicates are resampled with replacement, then blocks of frames
inside each drawn replicate), adds a Welch 95 % interval on per-replicate populations as the primary test, and warns
when replicates disagree. T4-R repeats T4 exactly — same 10 splits, same trajectories, same YAML, all other settings
default — with the `dev` code. Measured over all (split, state) pairs:
* **bootstrap false-positive rate** (95 % replicate-level bootstrap CI of the difference excludes 0). **Gate ≤ 0.15**,
  unchanged from T4.
* Welch-CI false-positive rate and Welch p < 0.05 rate (reported, not gated), median bootstrap CI half-width.

Expected limitation, declared before running: with 2 and 3 replicates per group a replicate-level bootstrap has very
few distinct resamples and underestimates the between-replicate variance by the factor (k − 1)/k, so it can still be
too narrow; if T4-R fails, the failure is reported as such. The original T4 result stays in the report unchanged.
T5 is re-run with the same code for information (`t5_force_field_dev.json`).

### T5 — force-field comparison (informational, no gate)
`conformatlas compare` on 5 ff99SB*-ILDN vs 3 ff15ipq replicates (default settings, Cα): population differences with
CIs and Welch p, RMSIP between the two force fields' essential subspaces, landscape overlap. Reported as a
force-field difference; there is no ground truth.

### T6 — replicate agreement (informational)
Pairwise RMSIP (top 10 Cα modes, each replicate's own PCA after the T1 fit) between all 8 replicates, and the
sampling/convergence verdict of `conformatlas analyze` per replicate.

### T7 — performance (reported, never gated)
Pooled 5 ff99SB*-ILDN replicates at 100 ps (50,005 frames), Cα and heavy atoms: the same tools and protocol as the
ADK performance tables (median of 3 after one discarded warm-up, fresh process, 8 BLAS threads). Scaling on
md1us4 at 1 ps (≈ 1,000,001 frames): Cα with every tool that fits in memory; heavy atoms with the ConformAtlas
streaming engine and GROMACS only (the in-memory float64 matrix would need ≈ 31 GB); a single run per point,
declared here because one run takes minutes.

## HIV-1 protease

### H0 — integrity (gate)
The authors' k-means object yields 964 discrete trajectories of 1,300 frames with labels in 0–99 (their selection,
`traj/sims_array.dat`: the 964 simulations with at least 1,300 frames, truncated to the first 1,300); 100 cluster
centres in 3 cylindrical lambda coordinates; the saved MSM has lag 300 frames. Settings read from the saved objects
before this file was committed (no model quantity was computed): `BayesianMSM`, reversible, count mode
"effective", connectivity "largest", 100 of 100 states active. H2 is therefore informational (see below).

### H1 — ConformAtlas MSM on the authors' discretisation vs PyEMMA and deeptime (gate; timescale-fix code)
The authors' discrete trajectories (their k-means, unchanged), lag 300 frames, reversible maximum-likelihood
estimator, sliding counts, largest connected set. Compared with a fresh `pyemma.msm.estimate_markov_model` and with
deeptime using identical options: same active set; max |Δπ| ≤ 1e-6; max |ΔT| ≤ 1e-6; top-5 implied timescales
relative error ≤ 1e-5.

### H2 — against the authors' saved model (conditional gate)
If the saved MSM was estimated with sliding counts, the H1 tolerances apply to ConformAtlas vs the saved model.
If it was estimated with PyEMMA's "effective" counts (the default of `bayesian_markov_model`), its point estimate
differs from a sliding-count estimate by construction; the comparison is then informational: timescale ratios and
max |Δπ| are reported, no gate.

### H3 — metastable states (informational)
ConformAtlas PCCA+ with 5 sets on the H1 model vs the authors' 5-state HMM: microstate-to-set agreement after the
best permutation (weighted by the stationary distribution) and set stationary probabilities side by side. PCCA+ and
an HMM are different methods; no gate.

### H4 — implied-timescale scan (informational)
ConformAtlas implied timescales over lags vs the authors' saved implied-timescale object at the same lags.

## Reporting rules
As for ADK: every gate is reported with its measured value; tools that fail to run are listed; trajectories are not
stored in the repository; nothing is pushed (local branches only, at the user's request). Update 2026-11-08: the
user allowed pushing to branches other than `main`; this work is consolidated on `dev`.
