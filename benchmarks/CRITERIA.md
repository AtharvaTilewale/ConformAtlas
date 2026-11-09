# Benchmark pass criteria — adenylate kinase (ADK) equilibrium trajectory

**Status:** fixed and committed *before any ADK result was computed.* Tolerances below are not
changed after seeing results. Any departure from this file is recorded in
`benchmarks/DEVIATIONS.md` with the reason; a gate that fails is reported as a failure.

## Data (all public, not stored in this repository)

| Item | Source | Integrity gate |
|---|---|---|
| ADK equilibrium trajectory, apo, CHARMM27, explicit water, 300 K, protein only, fitted on CORE | figshare 10.6084/m9.figshare.5108170 (Seyler & Beckstein 2017, CC-BY 4.0): `adk4AKE.psf`, `1ake_007-nowater-core-dt240ps.dcd` | md5 equals the publisher's; 3,341 atoms, 214 residues, 4,187 frames, Δt = 240 ps |
| Reference apo PMF on the NMP-CORE and LID-CORE angles | Zenodo 10.5281/zenodo.29617 (Beckstein et al. 2009, CC-BY-SA 4.0) | md5 equals the publisher's |
| Crystal structures 4AKE (open), 1AKE (closed) | RCSB PDB | downloaded by ID |

Version pins (tool versions, GROMACS version, BLAS threads, machine) are written to
`benchmarks/ENVIRONMENT.md` by the run script. BLAS/OpenMP threads are fixed to 8 for every tool.

## Conventions declared up front

* Eigenvalues of MDAnalysis, ProDy and scikit-learn use the N−1 covariance; they are multiplied by
  (N−1)/N before comparison with ConformAtlas, GROMACS and the NumPy reference (N).
* Eigenvector and projection signs are arbitrary; each is aligned to the reference before comparison.
* "Double-precision tools" = NumPy reference, ConformAtlas, MDAnalysis, ProDy, MDTraj + scikit-learn.
  GROMACS is single precision and has its own tolerance.
* Selections: Cα (214 atoms), backbone N/CA/C (642), protein heavy atoms (not hydrogen). Atom counts are
  recorded.
* Level 1 (numerics) feeds every tool the same pre-fitted coordinates and disables any further fit.
  Level 2 (end-to-end) lets every tool use its own default fit.

## Gates

### G0 — input integrity
Pass if every integrity gate in the data table passes.

### G1 — PCA numerics, Level 1 (Cα, backbone, heavy atoms; frames weighting)
Reference: independent NumPy code (explicit float64 covariance, N-normalised, `numpy.linalg.eigh`).

| Quantity (modes 1–10) | Double-precision tools | GROMACS |
|---|---|---|
| eigenvalues, relative error | ≤ 1e-6 | ≤ 1e-4 |
| explained variance, absolute (percentage points) | ≤ 1e-6 | ≤ 1e-3 |
| eigenvectors, 1 − \|cos\|, modes whose relative eigenvalue gap to both neighbours is ≥ 1 % | ≤ 1e-6 | ≤ 1e-4 |
| projections after sign alignment, max \|Δ\| / SD of that PC | ≤ 1e-6 | ≤ 1e-3 |

Modes in near-degenerate clusters (gap < 1 %) are not compared vector by vector; the sine of the largest
principal angle between the cluster's subspaces is reported. ConformAtlas must also meet the
double-precision row against the NumPy reference **by itself** (not only against other tools).

### G2 — free-energy landscape, Level 1 (Cα PC1–PC2, 300 K, k_B = 0.0083144626 kJ mol⁻¹ K⁻¹)
Same bin edges for every comparison, smoothing off, ΔG zeroed at the lowest occupied bin.

| Comparison | Gate (max \|ΔΔG\| over occupied bins) |
|---|---|
| ConformAtlas vs independent NumPy histogram Boltzmann inversion | ≤ 1e-9 kJ/mol |
| ConformAtlas vs `gmx sham` on identical projections, 32×32 and 64×64 bins | ≤ 1e-3 kJ/mol |

### G3 — angle coordinates and the published PMF
Angles use Cα centres of mass, with the definition of Beckstein et al. (J Mol Biol 394:160, 2009) as
restated in PLoS Comput Biol (doi 10.1371/journal.pcbi.1003721, Methods, "Inter-domain angle
definitions"): θ_NMP is the angle between (CORE 90–100 → 115–125) and (CORE 90–100 → NMP 35–55);
θ_LID is the angle between (115–125 → CORE 179–185) and (115–125 → LID 125–153). Residue numbers are the
PSF/PDB numbers. One follow-up paper describes the same angles with backbone N/Cα/C centres instead of Cα
only; that variant is computed as a sensitivity check and does not enter the gates.

* **G3a (definition check).** The published PMF, interpolated at the angles computed from 4AKE and from
  1AKE, must lie within 10 kJ/mol of the published global minimum at both points. This tests that the
  coordinate frame reproduces the published one.
* **G3b (tool check).** ConformAtlas's landscape on (θ_NMP, θ_LID) equals the NumPy histogram on the same
  bins to ≤ 1e-9 kJ/mol (smoothing off).
* **G3c (informational, no pass/fail).** Comparison of the trajectory's landscape with the published PMF
  (converted kcal → kJ): Spearman ρ over bins occupied in both, and the separation of the two minima. The
  published PMF is implicit-solvent umbrella sampling, the trajectory is explicit-water CHARMM27, so
  agreement is not expected to be quantitative.

### G4 — kinetics, one shared discretisation
One k-means discretisation (100 clusters, fixed seed) of PC1–PC3 is computed once and given to both
ConformAtlas and deeptime. Lags 10 and 50 frames.

| Quantity | Gate |
|---|---|
| TICA eigenvalues (top 5), relative error vs deeptime | ≤ 1e-8 |
| MSM stationary distribution, max \|Δπ\| | ≤ 1e-7 |
| MSM implied timescales (top 5), relative error | ≤ 1e-5 |

If ConformAtlas cannot be given an external discrete trajectory through its public API, that is recorded as
a deviation and the comparison is made on ConformAtlas's own discretisation with deeptime run on the same
cluster assignments exported from ConformAtlas.

### G5 — Level 2 end-to-end (informational expectation)
RMSIP of the top 10 modes between ConformAtlas's default workflow and each other tool's default workflow is
expected to be ≥ 0.95. This is reported with any value, not gated, because default fits differ by design.

### Informational, no gates
* Projection of the 1AKE and 4AKE crystal structures into the trajectory's PC space.
* Sampled states, populations and barriers from `conformatlas analyze` with bootstrap intervals.

## Performance (reported, never gated)
Wall time and peak resident memory (`/usr/bin/time -v`) for load → covariance → eigendecomposition →
projections on each selection, median of 3 runs after one discarded warm-up run, 8 BLAS threads. No speed
claim is made unless the measured medians support it. ConformAtlas is run from the released code
(version of this branch) and, separately, from the `perf/truncated-pca` branch; the two are labelled.

## Reporting rules
1. Every gate is reported with its measured value, pass or fail.
2. Tools that fail to run or time out are listed, not dropped.
3. Scripts, pins and the data DOIs are the only things shipped in the repository; trajectories are not.
