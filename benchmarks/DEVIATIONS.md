# Deviations from `benchmarks/CRITERIA.md`

Each entry states what differs, why, and what was done instead. Entries are added when they are found;
none of them changes a tolerance after seeing a result.

## D1 — GROMACS projections excluded from the Level 1 gate (found while debugging the script, before any result was read)
`gmx anaeig` always least-squares fits every frame to the structure stored in the eigenvector file when it
projects a trajectory, and has no option to switch that off. The pre-registered Level 1 comparison feeds
every tool pre-fitted coordinates with no further fit, so GROMACS projections cannot be compared on equal
terms. The projection cell of the GROMACS row is therefore not applicable; GROMACS eigenvalues, explained
variance and eigenvectors remain gated at the pre-registered tolerances. For the free-energy comparison with
`gmx sham` (G2) this does not matter, because `gmx sham` is given ConformAtlas's own projections.

## D2 — ProDy normalises its covariance by N, not N−1 (found after the first Cα run showed a 2.39e-4 = 1/N eigenvalue offset)
CRITERIA.md declared that ProDy uses the N−1 covariance and multiplies its eigenvalues by (N−1)/N. That was
an unverified assumption and it was wrong: `prody.PCA.buildCovariance` ends with `cov /= n_confs`, and on
random data its covariance equals the N-normalised one to 2.2e-16 (the N−1 version differs by 2 %). The
conversion is therefore not applied to ProDy. Tolerances are unchanged. The first-run failure is kept here
for transparency.

## D3 — script error reading `gmx covar` eigenvectors (found by the first Cα run, 1 − |cos| = 1.0)
`eigenvec.trr` from `gmx covar -nofit` holds a placeholder frame (t = −1), the average structure, and then the
eigenvectors, so eigenvectors start at frame index 2. The script read from index 1. Verified directly: with
the corrected offset the overlap matrix with the NumPy reference eigenvectors is the identity. This was a
bug in the benchmark script, not a tool disagreement.

## D4 — G2 (ConformAtlas vs NumPy histogram) FAILS as registered; cause is the k_B value written in CRITERIA.md
CRITERIA.md fixed k_B = 0.0083144626 kJ mol⁻¹ K⁻¹ and a 1e-9 kJ/mol gate. ConformAtlas uses 0.008314462618 and
the exact SI value is 0.00831446261815…, so the registered constant is 2.2e-9 (relative) too small. Measured
with the registered constant: max |ΔΔG| = 2.12e-8 kJ/mol (32×32) and 1.56e-8 kJ/mol (64×64) — **the gate as
registered is not met.** Diagnosis (not a replacement gate): recomputing the NumPy reference with the exact SI
value gives 1.81e-10 and 1.33e-10 kJ/mol, i.e. within the 1e-9 tolerance; the occupied bins are identical in
both cases (0 bins occupied in one landscape only). The gate result is reported as FAIL in the summary, with
this diagnosis next to it. The `gmx sham` comparison (gate 1e-3 kJ/mol) is unaffected and passes at
5.2e-6 kJ/mol.

## D5 — script error in the first G4 run (active set read from the count model instead of the fitted MSM)
The first G4 run compared ConformAtlas's active set (96 states) with `TransitionCountModel.states` of
deeptime (100 states, before restriction to the largest connected set) and therefore reported different
active sets. The fitted deeptime MSM restricts itself to the largest connected set (it logged the skipped
state sets); the script now reads `MaximumLikelihoodMSM.state_symbols()`. No tolerance was changed.

## D6 — G4 implied timescales FAIL as registered; cause is the ordering convention (a behavioural difference, not a numerical error)
Measured with one shared discretisation (96 active states in both implementations): TICA eigenvalues agree
with deeptime to ≤ 5.2e-15 relative, the stationary distribution to ≤ 2.3e-10 and the transition matrix to
≤ 1.5e-9 — all within the registered gates. The registered gate on the top-5 implied timescales (relative error
≤ 1e-5) is **not met**: 2.0e-2 at lag 10 and 1.7e-3 at lag 50. The mismatch is confined to the fifth
timescale. Diagnosis: the reversible transition matrix has a negative eigenvalue (λ ≈ −0.268 at lag 10) whose
magnitude exceeds the sixth positive eigenvalue. deeptime ranks timescales by |λ| (t = −τ / ln|λ|, the
convention of Prinz et al. 2011), so its fifth timescale belongs to that negative eigenvalue; ConformAtlas ranks
by signed eigenvalue and its fifth timescale is the next positive one. A dense eigendecomposition of
ConformAtlas's own T ordered by |λ| reproduces deeptime's list (see `msm_timescale_ordering_diagnosis_only` in
`results/g4_kinetics.json`). The gate is reported as FAIL. Whether to rank by |λ| or to flag negative
eigenvalues in the output is a design question for ConformAtlas; it is not changed on this branch.
