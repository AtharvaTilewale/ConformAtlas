# Mathematical & Algorithmic Methods

This document provides a comprehensive description of the physical, statistical, and algorithmic methods implemented in **ConformAtlas** (ConformAtlas).

---

## 1. Molar Free Energy Landscape Formulation

The Free Energy Landscape (FEL), or Potential of Mean Force (PMF), is evaluated by Boltzmann inversion of the sampled probability distribution along collective variables (the first two principal components $PC_1$ and $PC_2$):

$$\Delta G(x, y) = -R \cdot T \cdot \ln\left[ \frac{P(x, y)}{P_{\max}} \right]$$

where:
* $R = 0.008314462618\ \text{kJ}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$ is the universal molar gas constant.
* $T$ is the simulation temperature in Kelvin ($T > 0$ K).
* $P(x, y)$ is the normalized two-dimensional probability density evaluated across $N_{\text{bins}} \times N_{\text{bins}}$ histogram bins:
  $$P(x, y) = \frac{N(x, y)}{\sum_{i, j} N(i, j)}$$
* $P_{\max} = \max_{(x,y) \in \text{occupied}} P(x, y)$ is the maximum observed probability density, ensuring that the global free energy minimum satisfies:
  $$\Delta G_{\min} = 0.0\ \text{kJ/mol}$$

### Zero-Probability Handling
Bins with zero counts ($N(x, y) = 0$) represent unobserved regions of conformational space. In ConformAtlas, these bins are strictly assigned **$\text{NaN}$** (masked). They are **never** assigned $0.0\ \text{kJ/mol}$ (which would incorrectly denote a global energy minimum) or subjected to unconstrained $\ln(0)$ arithmetic.

---

## 2. Principal Component Analysis (PCA) & Coordinate Projection

### 2.0 Least-Squares Superposition (Removal of Rigid-Body Motion)
Cartesian coordinates taken directly from a trajectory contain the overall translation and rotation of the molecule in the simulation box. Unless this rigid-body motion is removed, the leading principal components describe how the molecule drifts and tumbles rather than how it changes shape. ConformAtlas therefore superposes every frame on a reference structure before building the covariance matrix, exactly as `gmx covar -fit` does.

For each frame the optimal proper rotation $\mathbf{R}$ and translation are obtained by the Kabsch algorithm on the **fit group** (`--lsq-group`, default `Backbone`): with both coordinate sets centred on their centroids, the SVD of the cross-covariance $\mathbf{H} = \mathbf{X}_{\text{mobile}}^T \mathbf{X}_{\text{ref}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ gives

$$\mathbf{R} = \mathbf{U} \, \mathrm{diag}(1, 1, \det(\mathbf{U}\mathbf{V}^T)) \, \mathbf{V}^T$$

where the determinant correction excludes reflections so that chirality is preserved. The same transform is then applied to the **analysis group** (`--cov-group`), which may differ from the fit group. The reference is the structure file passed with `-s` (or the first trajectory frame when the structure carries no coordinates, e.g. `.tpr`). In condition comparisons every system is superposed on the *same* reference (the reference system's structure, restricted to the common fit atoms), so projections of different systems are directly comparable.

The fit is unweighted by default; `--mass-weighted-fit` uses topology masses. `--no-fit` skips the step and should only be used for trajectories already fitted with `gmx trjconv -fit rot+trans`. Molecules must be whole across periodic boundaries (`gmx trjconv -pbc mol`) before analysis; a broken molecule cannot be superposed meaningfully.

### 2.1 Coordinate Centering
Let $\mathbf{x}_{k, i} \in \mathbb{R}^{3M}$ denote the Cartesian coordinates of $M$ selected atoms (e.g. Backbone or $\text{C}\alpha$) in frame $i$ of replicate $k \in \{1, \dots, K\}$.

The ensemble mean structure $\mathbf{\bar{x}}$ is computed based on the user-selected weighting scheme:
* **Equal-Replicate Weighting** (`--pca-weighting equal-replicate`, recommended default):
  $$\mathbf{\bar{x}} = \frac{1}{K} \sum_{k=1}^K \left( \frac{1}{N_k} \sum_{i=1}^{N_k} \mathbf{x}_{k, i} \right)$$
* **Frame-Weighted** (`--pca-weighting frames`):
  $$\mathbf{\bar{x}} = \frac{1}{N_{\text{total}}} \sum_{k=1}^K \sum_{i=1}^{N_k} \mathbf{x}_{k, i}, \quad N_{\text{total}} = \sum_{k=1}^K N_k$$

### 2.2 Covariance Matrix and Eigen-Decomposition
The centered displacement vectors are $\mathbf{y}_{k, i} = \mathbf{x}_{k, i} - \mathbf{\bar{x}}$. The $3M \times 3M$ symmetric covariance matrix $\mathbf{C}$ is:

$$\mathbf{C} = \sum_{k=1}^K \sum_{i=1}^{N_k} w_{k, i} \, \mathbf{y}_{k, i} \, \mathbf{y}_{k, i}^T$$

where $w_{k, i} = \frac{1}{K \cdot N_k}$ for equal-replicate weighting, or $w_{k, i} = \frac{1}{N_{\text{total}}}$ for frame weighting.

Eigen-decomposition yields orthogonal eigenvectors $\mathbf{v}_j$ and non-negative eigenvalues $\lambda_j$:

$$\mathbf{C} \, \mathbf{v}_j = \lambda_j \, \mathbf{v}_j, \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{3M} \ge 0$$

### 2.3 Explained Variance and Projections
The individual and cumulative explained variances are:

$$\text{Var}_j = 100 \times \frac{\lambda_j}{\sum_{m=1}^{3M} \lambda_m} \%, \quad \text{CumVar}_j = \sum_{m=1}^j \text{Var}_m \%$$

Each frame is projected onto the principal modes:

$$p_{k, i, j} = \mathbf{y}_{k, i}^T \, \mathbf{v}_j$$

---

## 3. Shared PCA Space for Condition Comparisons

When comparing two systems (e.g. Wild-Type vs Mutant), Cartesian PCA coordinates cannot be compared unless both trajectories are evaluated on the exact same basis $\mathbf{V}$.

1. **Common Atom Intersection**: Since point mutations alter side-chain atom composition, the analysis group is filtered to invariant atoms (typically `Backbone` or `C-alpha`) matching by `(chain, residue_number, atom_name)`.
2. **PCA Reference Modes**:
   - `combined`: A balanced covariance matrix is computed across WT and Mutant ensembles.
   - `reference-system`: The PCA basis $\mathbf{V}_{\text{WT}}$ is computed solely from WT, and Mutant frames are projected onto $\mathbf{V}_{\text{WT}}$.
3. **Grid Alignment**: Both systems are evaluated on an identical PC1/PC2 grid with shared bounds and bin edges, enabling direct calculation of free energy differences:
   $$\Delta\Delta G(x, y) = \Delta G_{\text{mutant}}(x, y) - \Delta G_{\text{WT}}(x, y)$$

---

## 4. Topological Basin Detection & State Segmentation

Rather than selecting arbitrary pixel minima, conformational states are segmented using topological watershed analysis:

1. **Local Minima Identification**: Candidate energy minima are identified by locating local maxima in the probability density $P(x, y)$ separated by at least $d_{\min}$ grid bins.
2. **Watershed Flooding**: Local minima serve as marker seeds. The watershed algorithm floods upward along the free energy surface $\Delta G(x, y)$, segmenting the landscape into catchment basins separated by energy ridges.
3. **Barrier-Based Merging**: Watershed over-segments any broad minimum whose surface is rippled by counting noise. Adjacent basins are therefore merged iteratively, starting from the shallowest separation, whenever the lowest saddle on their shared boundary lies less than $\Delta G_{\text{barrier}}$ above the minimum of the shallower basin. The default threshold is $1\,k_B T = RT$ (2.49 kJ/mol at 300 K) and can be changed with `--min-barrier` (kJ/mol).
4. **Frame-to-Basin Assignment**: Every trajectory frame $(p_1, p_2)$ is mapped to its corresponding spatial grid bin $(x_{\text{bin}}, y_{\text{bin}})$ and assigned to the underlying basin. Occupied bins that watershed could not reach from any marker (isolated islands of sampled space) inherit the label of the nearest labelled bin, so every sampled frame belongs to a basin. Density maxima on the border of the grid are valid markers (a basin at the extreme of PC1 or PC2 sits on the border by construction).
5. **Major State Filtering**: Basins containing $\ge \text{min\_state\_population}$ (default: 5%) of total frames are designated as major conformational states (ordered by population as `State A`, `State B`, etc.).
6. **Representative Frame**: For each basin, the real trajectory frame minimizing the Euclidean distance to the basin's free energy minimum in $(PC_1, PC_2)$ space is extracted as the representative structure:
   $$d = \sqrt{(PC_{1, \text{frame}} - PC_{1, \min})^2 + (PC_{2, \text{frame}} - PC_{2, \min})^2}$$

---

## 5. Convergence & Sampling Diagnostics

### 5.1 Root Mean Square Inner Product (RMSIP)
Subspace overlap between two sets of $s$ orthonormal eigenvectors $\mathbf{U} = [\mathbf{u}_1, \dots, \mathbf{u}_s]$ and $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_s]$ is calculated as:

$$\text{RMSIP} = \sqrt{\frac{1}{s} \sum_{i=1}^s \sum_{j=1}^s (\mathbf{u}_i \cdot \mathbf{v}_j)^2} = \sqrt{\frac{1}{s} \|\mathbf{U}^T \mathbf{V}\|_F^2}$$

* $\text{RMSIP} = 1.0$: Identical subspaces.
* $\text{RMSIP} = 0.0$: Mutually orthogonal subspaces.

In ConformAtlas, RMSIP is evaluated:
* Between the pooled first halves and the pooled second halves of all replicates (the headline value), and separately for each replicate (`rmsip.csv`, one row per replicate).
* Progressively, between the first 25%, 50%, 75% and 100% of every replicate and the full data set. The 100% point is the full data set compared with itself and equals 1 by construction; it is the end point of the curve, not evidence.
* Pairwise between independent replicates.

State population stability is evaluated on the same cumulative fractions, taken *per replicate* (the first 25% of every replicate pooled, and so on), so that the curve reflects cumulative simulation time rather than the order in which replicates were listed.

### 5.2 Cosine Content
To detect whether motion along large-amplitude modes represents genuine conformational transitions or random diffusive drift (random walk), the cosine content $c_k$ is evaluated (Hess, 2000, 2002):

$$c_k = \frac{2}{N} \frac{\left( \sum_{t=0}^{N-1} p(t) \cos\left[ \frac{k \pi t}{N - 1} \right] \right)^2}{\sum_{t=0}^{N-1} p(t)^2}$$

* $c_1 \approx 1.0$: Motion along this PC matches a half-cosine wave, characteristic of un-converged diffusive drift.
* $c_1 \ll 0.5$: Motion exhibits non-diffusive fluctuations or transitions among multiple discrete states.

---

## 6. Uncertainty Estimation

When $K \ge 2$ independent replicates are provided, between-replicate statistics serve as the primary measure of uncertainty for state populations $x_k$:

$$\bar{x} = \frac{1}{K} \sum_{k=1}^K x_k, \quad \text{SD} = \sqrt{\frac{1}{K - 1} \sum_{k=1}^K (x_k - \bar{x})^2}$$

$$\text{SEM} = \frac{\text{SD}}{\sqrt{K}}$$

The 95% Confidence Interval is constructed using the Student's $t$-distribution with $K - 1$ degrees of freedom:

$$\text{CI}_{95\%} = \left[ \bar{x} - t_{0.025, K-1} \cdot \text{SEM},\quad \bar{x} + t_{0.025, K-1} \cdot \text{SEM} \right]$$

If only a single replicate is supplied, the trajectory is partitioned into 5 sequential blocks to assess intra-trajectory block variability.


---

## 7. Time Window, Equilibration and Effective Sample Size

Frames are restricted to `-b ≤ t ≤ -e` (ps) before any analysis. The **statistical inefficiency** of a series $x_t$ is

$$g = 1 + 2\sum_{t=1}^{T} \left(1 - \frac{t}{N}\right) C(t), \qquad N_\text{eff} = N/g,$$

where $C(t)$ is the normalised autocorrelation function, summed until it first becomes non-positive (Chodera et al. 2007). $C(t)$ is evaluated from the full FFT autocorrelation (exact; identical to pymbar's non-fast estimator). ConformAtlas reports $g$ of PC1 and PC2 for every replicate and uses the larger value; $N_\text{eff}$ over all replicates is printed in the convergence notes. Fewer than 50 effective frames force the sampling verdict to *Insufficient evidence*.

The **equilibration point** $t_0$ is chosen as the start that maximises $N_\text{eff}$ of the remaining production segment (Chodera 2016), evaluated on the fit RMSD to the reference. With `--equilibration auto` the frames before $t_0$ are discarded per replicate; by default the detected $t_0$ is only reported (`convergence/equilibration.csv`). On a monotonically drifting RMSD the criterion discards aggressively; inspect the descriptor figure before accepting it. `--equilibration-on cv` applies the same criterion to both landscape CVs and adopts the later start (§16).

## 8. Alternative Collective Variables

* **Dihedral PCA** (`--pca-type dihedral`): backbone φ/ψ angles are mapped to $(\cos\theta, \sin\theta)$ so that periodicity is respected (Mu, Nguyen & Stock 2005); PCA is run on this feature matrix. No superposition is needed.
* **TICA** (`--pca-type tica`): the generalised eigenproblem $C(\tau)\,v = \lambda\,C(0)\,v$ is solved with the symmetrised time-lagged covariance $C(\tau)$ at lag $\tau$ frames (Molgedey & Schuster 1994; Pérez-Hernández et al. 2013). Features are first whitened in the PCA basis; directions with variance below $10^{-6}$ of the largest are discarded, and when the number of frames is not much larger than the number of coordinates the problem is pre-reduced to the leading principal components. Eigenvalues are autocorrelations at lag τ and the implied timescale is $t_i = -\tau\,\Delta t/\ln\lambda_i$. Lag times are computed within each replicate, never across the junction between replicates.
* **RMSD / Rg** (`--cv rmsd-rg`): backbone RMSD to the reference after superposition and the radius of gyration of the analysis group.
* **External CVs** (`--cv file`): two columns of a text file (PLUMED `COLVAR`, `gmx` `.xvg`, CSV); one file per trajectory, rows matched to frames after striding and windowing.
* **Component choice** (`--fel-pcs i,j`): any two retained components may span the landscape.

## 9. Reweighting

With `--weights-file`, per-frame weights $w_t$ (for example $\exp(\beta V_\text{bias})$ from metadynamics, or Boltzmann factors from replica exchange) enter the histogram, state populations and bootstrap: $P(x,y) \propto \sum_t w_t\, \mathbb{1}[(x_t,y_t)\in\text{bin}]$. Kinetic quantities (transition counts, dwell times, MFPT) are not computed for weighted frames because the time ordering of a biased trajectory is not physical. Weight files for metadynamics and umbrella-sampling runs are produced by `conformatlas reweight` (§17).

## 10. Grid Resolution

Without `--bins`, the bin width follows the Freedman–Diaconis rule $h = 2\,\text{IQR}\,N^{-1/3}$ on each CV; the finer of the two axes sets a square grid, clipped to 16–128 bins. The result is recorded in `run_metadata.json` (`bins`, `bins_mode`). A warning is issued when the occupied bins hold fewer than two frames on average.

## 11. Barriers and Kinetics

Two basins are *adjacent* when their label regions share a grid edge. The **saddle** between them is the minimum, over the shared boundary, of the larger of the two neighbouring free energies, and the barrier from state $i$ is saddle $-\,\Delta G_{\min,i}$ (`states/barriers.csv`). Barriers are not reported for non-adjacent basins. Transition counts are taken between consecutive assigned frames within each replicate (unassigned frames are skipped, replicate junctions are never counted), the transition-probability matrix is the row-normalised count matrix at a lag of one frame, dwell times are lengths of uninterrupted visits (visits touching the start or end of a replicate are flagged as censored), and mean first-passage times are the Markov estimate from the one-frame transition matrix — a lower bound that assumes memoryless hopping between the detected states.

## 12. Block Bootstrap of the Landscape

For each of `--bootstrap` resamples, frames are drawn per replicate as circular blocks of length $L = 2g$ (at least five blocks per replicate), the histogram, free energy, basin populations (basin labels are held fixed), state minima and saddles are recomputed, and the per-bin standard deviation, 2.5 % and 97.5 % quantiles are stored (`fel/free_energy_bootstrap_*.npy`, `statistics/state_populations.csv`, `states/state_free_energies.csv`, `states/barriers.csv`). Bins sampled in fewer than half the resamples are left undefined. `--bootstrap-mode resegment` (§16) additionally re-runs the watershed on every resample.

## 13. Comparison Statistics

All systems share one grid. For every system $S$ and the reference $R$: the **ΔΔG map** $\Delta G_S(x,y) - \Delta G_R(x,y)$ (each landscape to its own minimum; NaN where either is unsampled) with the bootstrap standard deviation of the difference obtained from independent per-system block bootstraps; **population differences** with the bootstrap 95 % interval and a two-sided bootstrap p-value (twice the smaller tail fraction of the difference distribution, floored at $1/B$); **Welch's t-test** on per-replicate populations, reported only when every system has at least two replicates, with the replicate counts stated because two or three replicates give very low power; **state ΔΔG** from the state minima; the **Bhattacharyya coefficient** $\sum\sqrt{p_S\,q_R}$, Hellinger distance and **Jensen–Shannon divergence** (bits) between the normalised sampled densities; and the **RMSIP** between the leading eigenvectors of each system's own PCA (`comparison/subspace_overlap.csv`).


---

## 14. What is validated, what is heuristic

`validation/validate.py` reproduces the checks listed in `validation/VALIDATION.md`. The following are **validated** against analytical results or independent codes: Boltzmann inversion (exact bin-integrated Gaussian, counting-noise level), reweighting (importance-sampling identity), Cartesian PCA (MDAnalysis to 10⁻³ after the N−1 → N rescaling; `gmx covar` to 10⁻⁴), the SVD route, Kabsch superposition (MDAnalysis QCP, 10⁻⁵ Å), statistical inefficiency and equilibration (pymbar, machine precision / identical t₀), basin populations (Boltzmann integral, 0.001), saddle barriers (analytic potential, ≤ 0.5 kJ/mol at 60 bins), block-bootstrap coverage (0.92 for a nominal 0.95 on strongly correlated data), transition matrix / dwell times / MFPT (known Markov chain, ≤ 2 %), TICA (AR(1) theory, deeptime), dihedral features (MDAnalysis), RMSD/Rg/RMSF (MDAnalysis, `gmx gyrate`), JS/Bhattacharyya (SciPy), and the type-I error of the population-difference test (0.09 at α = 0.05 with 2 + 2 correlated replicates — mildly anti-conservative, as block bootstraps are).

The following are **heuristics** and should be read as such: the sampling verdict (threshold values for RMSIP, cosine content and N_eff), the Freedman–Diaconis grid size, the `--min-state-population` and `--min-barrier` thresholds, the choice of fit RMSD as the observable for equilibration detection, the lag-1 Markov assumption behind MFPT (a lower bound), and the fixed-basin assumption in the default bootstrap (`--bootstrap-mode resegment`, §16, includes the re-segmentation uncertainty). Covariance normalisation uses N (GROMACS convention), not N−1.

The documentation itself is checked (`validation/checks_docs.py`): every command of the tutorial is executed, every number quoted in it is re-read from the output file it came from, the benchmark table is regenerated from `results.json`, every DOI in the JOSS bibliography is resolved, and the Sphinx build must pass with warnings as errors.

### Classification of the components added in 1.0.0

The table lists every component of §15–§21 with its status. *Validated* means an analytical result or an independent implementation was reproduced within a stated tolerance by the check named in the last column (the check scripts are called by `validation/validate.py`; the tolerances and measured values are in `validation/VALIDATION.md`). *Heuristic / convention* means a definition or threshold that has no reference value; its consequences may be validated, the rule itself is not.

| Component | Status | Reference / check |
|---|---|---|
| MSM count matrix (sliding window, within replicates) | validated | identical to deeptime (`checks_msm.py`) |
| Reversible maximum-likelihood T and π | validated | deeptime `MaximumLikelihoodMSM(reversible=True)`, 10⁻⁶; generating matrix of a 10⁶-step chain, 0.0007 abs |
| Row-normalised (non-reversible) T | validated | deeptime |
| Eigen-decomposition, implied timescales | validated | −τ/ln λ analytical, 0.6 % at lags 1–10 |
| PCCA+ memberships, coarse-grained matrix | validated | `deeptime.markov.pcca`, 5·10⁻⁵ / 10⁻⁵ |
| MFPT between sets | validated | analytical two-state 1/p; deeptime (machine precision) |
| Chapman–Kolmogorov deviations | validated | pass on a Markov chain (max 2·10⁻⁴), fail on a hidden-state construction (0.12) |
| CK pass criterion `max(0.05, 3 SE)`, SE from N_A/(kτ) independent pairs | heuristic / convention | sliding counts are correlated; the verdict threshold is a convention |
| Implied-timescale plateau rule (first lag with < 10 % change) | heuristic / convention | reporting rule only; the check confirms lag 1 for a Markov chain; never applied automatically |
| k-means, regular-space clustering | validated | `scipy.cluster.vq.kmeans2` (identical from the same seeds); regular-space properties exact |
| End-to-end MSM on a 3-well landscape | validated | slow timescales within 2 %, set populations within 0.006 |
| PCCA+ on a non-reversible T (real parts of eigenvectors) | unverified | same simplification as deeptime; a warning is emitted for complex spectra |
| Re-segmentation bootstrap: state matching (≥ 50 % overlap or minimum in basin) | heuristic / convention | definition |
| Re-segmentation bootstrap: population-interval coverage | validated (0.89 for nominal 0.95) | 120 two-well + 100 three-well Markov systems; slightly anti-conservative like the fixed mode |
| Re-segmentation bootstrap: state-count recovery | validated | > 0.99 for wells 6σ apart, mean 0.79 (min 0.48) at 2σ |
| `detect_equilibration_multi` (latest t₀; joint max–min N_eff) | validated | `pymbar.timeseries.detect_equilibration` per observable (identical t₀ within one `nskip`); brute force with pymbar's g |
| Per-replicate landscape panels | descriptive | shape comparison only; no statistic attached |
| COLVAR / HILLS readers, well-tempered height rescaling | validated | `numpy.loadtxt`; PLUMED source convention |
| V(s_t, t) from HILLS (PLUMED ≥ 2.5 stretched Gaussian) | validated | `metad.bias` of a synthetic PLUMED-format run, 5·10⁻⁹ kJ/mol; `plumed sum_hills` (PLUMED 2.10), 5·10⁻¹⁰ kJ/mol |
| Tiwary–Parrinello c(t) and weights | validated (1-D) | untempered limit 10⁻¹⁰; Boltzmann distribution of a double well, max 0.24 kJ/mol |
| Final-bias weights | validated | analytical, max 0.10 kJ/mol on an equilibrium sample of U + V_final; 0.51 kJ/mol on the unconverged trajectory (documented limitation) |
| Metadynamics with ≥ 2 CVs | unverified | code is dimension-general; only 1-D validated; c(t) grid resolution is a user choice |
| HILLS from PLUMED < 2.5 (`--hill-shape gaussian`) | unverified | not tested against such a version; `multivariate true` HILLS are rejected |
| WHAM PMF and window free energies | validated | analytical bin-integrated free energy, max 0.15 kJ/mol; WHAM vs MBAR f_k 0.01 kT |
| MBAR f_k, weights, PMF | validated | `pymbar.MBAR` 4.0.3, 10⁻¹³ kT / 3·10⁻¹⁴ relative; analytical PMF max 0.15 kJ/mol |
| Harmonic-restraint convention ½κ(x−x₀)², no inefficiency subsampling by default, no PMF uncertainties | convention / limitation | stated |
| DCCM (overall, per state, weighted) | validated | explicit NumPy loop; ProDy `calcCrossCorr`; 10⁻¹⁵ |
| Per-residue state differences | validated | synthetic ensemble with a known 3 Å shift, 0.01–0.02 Å |
| Ligand–residue minimum distances and contacts | validated | brute-force NumPy on the neuraminidase fixture, 3·10⁻⁶ Å, 0 mismatches, PBC on/off identical |
| Contact persistence | validated | boolean mean, exact |
| Binding-site definition (persistence ≥ 0.5) and 4.5 Å heavy-atom cutoff | heuristic / convention | conventional choices |
| Per-residue differences in the global superposition frame | convention | no per-state re-fit (stated in the module docstring) |
| Dihedral angle, χ₁/χ₂, explicit φ/ψ | validated | hand geometry; `MDAnalysis.calc_dihedrals`, `Janin` (6·10⁻¹⁴°), `Ramachandran` (0°) |
| Nucleic-acid torsions | validated on a constructed geometry | hand computation, 10⁻⁴°; CHARMM residue names untested on a real structure |
| Chain-aware common-atom mapping | validated | constructed two-chain topology with repeated residue numbers, 24/24 atoms paired |
| Streaming covariance accumulator | validated | `numpy.cov`, chunked and merged, 2.6·10⁻¹³ abs |
| Streaming PCA vs in-memory path | validated | eigenvalues / eigenvectors / projections ≤ 3·10⁻¹⁴ relative, both weightings, both routes; 2 workers + float32 memmap 3.2·10⁻⁸ |
| `ConformAtlasPCA` (MDAnalysis AnalysisBase) | validated | `MDAnalysis.analysis.pca.PCA`, 9.4·10⁻⁹ relative after the (N−1)/N rescaling; sliced run vs in-memory 2·10⁻¹⁵ |
| HDF5 round trip, `load_results_dir`, `rebin` | validated | 0 mismatches over 38 arrays/tables; rebinned ΔG, states, assignments, basin map and saddles bitwise identical to fresh runs |
| `api.*` chain vs `api.analyze` | validated | identical eigenvalues, ΔG, states, dwell summary, RMSIP |
| Benchmark timings | measurement | one machine; not a claim |
| Interactive heatmap, hover values, basin outlines, slider end points, 1-D profiles | validated | equal to `fel/free_energy.npy`, `fel/probability.npy`, `states/basin_map.npy`, last frame per replicate, `numpy.histogram` on the saved edges (10⁻¹²) |
| Interpretive sentences (quoted numbers, binomial SE, population-ratio ΔG) | validated | CSV values; analytical formulas (10⁻⁹) |
| RMSIP / cosine-content / N_eff wording thresholds | heuristic / convention | same thresholds as the sampling verdict |
| Browser rendering of the Plotly figure and the NGL viewer | unverified | no browser/WebGL in the validation environment; embedded data verified, pixels not |
| Provenance (`collect_environment`) | validated | `importlib.metadata` versions, JSON round trip, `gmx --version` parsing |

## 15. Markov State Models

The MSM layer (`--msm`, `conformatlas.msm`) follows Prinz et al. (J. Chem. Phys. 2011) and is cross-checked against deeptime, SciPy and analytical Markov chains in `validation/checks_msm.py`. The lag time is always the user's (`--msm-lag`); the plateau scan reports, it never chooses.

**Discretisation.** Frames are clustered in the leading `--msm-components` retained components (PCs, dihedral PCs or TICA ICs; the two landscape components come first) with k-means (Lloyd iterations, k-means++ seeding, fixed seed, best of 5 seedings by within-cluster sum of squares) or regular-space clustering (Prinz et al. 2011: a frame becomes a new centre when it is farther than `--msm-dmin` from every existing centre). Microstates are Voronoi cells of the centres. The number of k-means microstates is capped at one tenth of the number of frames.

**Counting.** Transition counts at lag τ (frames) are accumulated with a sliding window separately in every replicate; a pair of frames never straddles a trajectory junction. Replicates are pooled with equal weight per transition, i.e. they are assumed to sample the same equilibrium. Per-frame weights (reweighted ensembles) are not supported by the kinetic layer.

**Estimation.** The model is estimated on the largest strongly connected set of the count graph (microstates outside it are listed in `msm_summary.json` → `notes`). The default estimator is the reversible maximum-likelihood estimate under detailed balance, obtained by the fixed-point iteration

$$X_{ij} \leftarrow \frac{C_{ij} + C_{ji}}{C_i / X_i + C_j / X_j}, \qquad \pi_i = X_i / \textstyle\sum_k X_k$$

(Bowman et al. 2009; Prinz et al. 2011), with $C_i = \sum_j C_{ij}$ and $X_i = \sum_j X_{ij}$. The row-normalised estimator $T_{ij} = C_{ij}/C_i$ is available with `--msm-nonreversible`; its eigenvalues may be complex, in which case real parts are reported with a warning.

**Implied timescales.** $t_i(\tau) = -\tau\,\Delta t / \ln \lambda_i(\tau)$; for a Markovian discretisation they are independent of τ. The lags of `--msm-lags` (default: geometric 1 … 5 τ) are scanned and the first lag from which all resolved timescales ($t_i > \tau$) change by less than 10 % to the next lag is *reported* (`msm/implied_timescale_changes.csv`, `suggested_lag_frames` and `plateau_found` in `msm_summary.json`). This 10 % rule is a reporting convention. Timescales shorter than the lag are shaded in the figure and are not reliable.

**Chapman–Kolmogorov test.** For each PCCA+ set, the probability of remaining in (or moving to) a set after $k\tau$, started from π restricted to the origin set, is predicted from $T(\tau)^k$ and compared with the same quantity from a model estimated directly at lag $k\tau$, for $k = 1 … $ `--msm-ck-k`. The approximate standard error treats $N_A/(k\tau)$ transition pairs as independent (sliding counts are correlated, so this is an approximation). The test is reported as passed when every absolute deviation is below $\max(0.05,\ 3\,\mathrm{SE})$; the deviations themselves are exact, the verdict threshold is a convention.

**Metastable sets.** PCCA+ (Deuflhard & Weber 2005; Röblitz & Weber 2013): the $m$ leading right eigenvectors span a simplex whose vertices are located by the inner simplex algorithm; the linear transformation is then refined by maximising the crispness $\sum_{ij} A_{ij}^2 / A_{1j}$ under $\chi \ge 0$, $\chi\mathbf{1} = \mathbf{1}$ (Nelder–Mead, as in deeptime). Crisp sets are the arg-max memberships. The coarse-grained matrix uses the Kube & Weber (2007) projection $(\chi^T\chi)^{-1}\chi^T T \chi$ with symmetrised flux (deeptime convention); it is descriptive — rates should be read from the MFPTs and the microstate timescales. The default number of sets equals the number of watershed states (at least 2); `--msm-states` overrides it.

**Mean first-passage times** between sets solve $m_i = \tau + \sum_j T_{ij} m_j$ outside the target set ($m = 0$ inside) and average over the origin set with weights π (deeptime convention). MFPTs are in ps ($\tau\,\Delta t$ units) and inherit the Markov assumption at τ.

**Landscape versus kinetics.** `msm/basin_overlap.csv` cross-tabulates frames by PCCA+ set and watershed state; `row_fraction_max` near 1 means a kinetic set coincides with one landscape basin. Disagreement indicates that the landscape basins are not kinetically separated at the chosen lag, or that the discretisation is too coarse.

**When to trust the model.** Read the implied-timescale figure first: if no plateau is found (`plateau_found: false`), or `active_count_fraction` is well below 1, or the CK test fails, the discretisation or the trajectory length is insufficient and the MFPTs should not be quoted. A passing CK test with resolved timescales at least several τ long is the minimum condition; with two or three short replicates (as in the bundled demonstration data) the MSM layer illustrates the outputs, it does not produce publishable rates.

## 16. Re-segmentation Bootstrap and Multi-Observable Equilibration

**Re-segmentation mode** (`--bootstrap-mode resegment`). On every block-bootstrap resample the watershed segmentation is re-run on the resampled landscape (same grid, same `--min-state-population`, `--max-states`, `--min-barrier`). The number of major states found is recorded (`states/state_count_distribution.csv`) and resampled states are matched to the reference states by maximum weighted frame overlap (Hungarian assignment), accepting a match when ≥ 50 % of the resampled state's frames belong to the reference basin or when its minimum lies inside the reference basin. Population intervals under re-segmentation are the 2.5–97.5 percentiles over the matched resamples (`states/state_populations_resegmented.csv`, `Resegment_CI_95_*` columns of `statistics/state_populations.csv`); `match_rate` reports how often each state was recovered. The matching rule is a definition, not a validated quantity; what was validated is the downstream coverage: the interval covers the true population in 0.89 of 120 two-well and 100 three-well Markov systems (the block bootstrap with block length ≈ 2g is slightly anti-conservative in both modes), and the state count is recovered with probability > 0.99 for wells 6σ apart and spreads (mean 0.79, minimum 0.48) for centres 2σ apart. In `compare`, the same option applies to the pooled landscape.

**Equilibration on the landscape CVs** (`--equilibration-on cv`). The Chodera criterion of §7 is applied to both landscape CVs per replicate (`timeseries.detect_equilibration_multi`, method `max_neff`) and the latest production start among the observables is adopted, so that neither CV is still relaxing at t₀. The fit-RMSD detection still runs first; the CV-based result is written as a second row per replicate in `convergence/equilibration.csv` (observable `PC1+PC2 (latest t0)`, with the deciding CV). With `--equilibration auto` the later of the two t₀ is applied and the decomposition is refitted on the production window, so the PCA basis does not include the discarded transient. Validated against `pymbar.timeseries.detect_equilibration` per observable (identical t₀ within one `nskip`); the `joint` variant of the function (maximise the smallest N_eff over observables) agrees with a brute-force evaluation using pymbar's g.

## 17. Enhanced-Sampling Reweighting

`conformatlas reweight` (`conformatlas.reweighting`) turns biased simulations into per-frame weights for the `--weights-file` pathway of §9. All checks are in `validation/checks_enhanced.py`.

**Metadynamics.** The bias is rebuilt from the PLUMED `HILLS` file with PLUMED's conventions: for PLUMED ≥ 2.5 a hill is the stretched Gaussian $h\,[A\,e^{-d^2/2\sigma^2} + B]$ truncated at $d^2/2\sigma^2 = 6.25$ with $A = 1/(1-e^{-6.25})$, $B = 1 - A$ (`--hill-shape gaussian` gives a plain Gaussian for older HILLS files); well-tempered `height` columns, which PLUMED writes multiplied by $\gamma/(\gamma-1)$, are divided back on read; periodic CVs declared by `#! SET min_<cv>`/`max_<cv>` are wrapped; $V(s_t, t)$ uses the hills deposited strictly before $t$, which reproduces PLUMED's `metad.bias` column. Two weighting schemes:

* `final-bias`: $w_t = \exp[\beta V(s_t, t_\text{final})]$ — the end-of-run bias as a static bias. Exact for a converged bias; frames from the filling transient are over-weighted, so combine it with `--skip-time`.
* `tiwary-parrinello` (default; Tiwary & Parrinello, J. Phys. Chem. B 2015): $w_t = \exp\{\beta[V(s_t, t) - c(t)]\}$ with
  $$c(t) = k_BT \ln \frac{\int e^{\gamma\beta V(s,t)/(\gamma-1)}\,ds}{\int e^{\beta V(s,t)/(\gamma-1)}\,ds}$$
  evaluated on a regular CV grid after every hill and held piecewise constant between hills (untempered limit $k_BT\ln\langle e^{\beta V}\rangle_\text{grid}$). The integrals are grid sums over the hill-centre range ± 4σ with `--grid-bins` points per CV (or the declared period); the grid must cover the sampled region, which is a user choice for runs with two or more CVs.

Weights are normalised to mean 1 over the kept frames (0 before `--skip-time`) so that one file per trajectory can be given to `analyze --weights-file` without rescaling; `pmf.csv` is $-k_BT\ln$ of the weighted histogram of the first CV, and `rct.csv` holds $c(t)$. Validation: $V(s_t,t)$ against the `metad.bias` column of a synthetic PLUMED-format run (2000 hills, 40 000 frames) to 5·10⁻⁹ kJ/mol; the final bias against `plumed sum_hills` (PLUMED 2.10) to 5·10⁻¹⁰ kJ/mol over 46 kJ/mol of bias (a plain Gaussian would be off by 0.2); Tiwary–Parrinello weights reproduce the Boltzmann distribution of a known double well to max 0.24 kJ/mol (RMS 0.09 below 8 kJ/mol); final-bias weights to max 0.10 kJ/mol on an equilibrium sample of $U + V_\text{final}$ and 0.51 kJ/mol on the unconverged trajectory (the static-bias limitation; the unweighted histogram is off by 3.3). Only one-dimensional runs were validated against a known potential; `multivariate true` (adaptive-σ) HILLS are rejected.

**Umbrella sampling.** Windows with harmonic restraints $V_i(x) = \tfrac12\kappa_i(x - x_i)^2$ (κ in kJ/mol per CV unit², the PLUMED `RESTRAINT` and `gmx wham` convention). Window free energies $f_i$ ($f_0 = 0$) are obtained by

* **WHAM** (Kumar et al. 1992): bias evaluated at bin centres, iterated to a tolerance of 10⁻⁸ kT; or
* **MBAR** (Shirts & Chodera 2008): self-consistent iterations followed by Newton–Raphson on the convex objective (gradient $N_i(\sum_n W_{in} - 1)$, Hessian $\delta_{ij}N_i\sum_n W_{in} - N_iN_j\sum_n W_{in}W_{jn}$); unsampled states are allowed.

Unbiased frame weights are $w_n = 1 / \sum_k N_k \exp[f_k - \beta V_k(x_n)]$ (normalised to sum 1 overall, then written with mean 1 per window file), and the 1-D PMF is their histogram. No statistical-inefficiency subsampling is applied by default and PMF uncertainties are not computed. Validation: WHAM PMF against the analytical bin-integrated free energy of a double well to max 0.15 kJ/mol (RMS 0.06); MBAR PMF to max 0.15 kJ/mol; MBAR $f_k$ and unbiased weights against `pymbar.MBAR` 4.0.3 to 10⁻¹³ kT and 3·10⁻¹⁴ relative; WHAM and MBAR $f_k$ agree to 0.01 kT.

## 18. Residue-Level Analyses

`--residue-analysis` (`conformatlas.residue_analysis`) works on the globally superposed analysis-group coordinates of the pipeline; nothing is re-fitted per state, so all quantities describe motion relative to the global fit — the same assumption the Cartesian PCA makes. Checks: `validation/checks_enhanced.py`.

**Dynamic cross-correlation map** (Ichiye & Karplus 1991):

$$C_{ij} = \frac{\langle \Delta\mathbf{r}_i \cdot \Delta\mathbf{r}_j \rangle}{\sqrt{\langle \Delta\mathbf{r}_i^2\rangle\langle \Delta\mathbf{r}_j^2\rangle}}, \qquad \Delta\mathbf{r}_i = \mathbf{r}_i - \langle\mathbf{r}_i\rangle$$

with (optionally weighted) averages over frames; $C_{ij}$ is the normalised trace of the 3 × 3 atomic covariance block, not the Pearson correlation of concatenated xyz components. It is computed over all frames and from the frames of each state (`residues/dccm_state_A.npy` …) and averaged to residues. Validated against an explicit NumPy loop and ProDy's `calcCrossCorr` to 10⁻¹⁵; the weighted form against replicated frames.

**Per-residue state differences.** For every pair of states: the RMSF of each residue within each state, $\sqrt{\langle|\Delta\mathbf{r}|^2\rangle_\text{state}}$ averaged over the residue's analysis atoms; their difference `delta_rmsf`; the RMSD between the state-mean positions, $\sqrt{\text{mean}_\text{atoms}|\langle\mathbf{r}\rangle_B - \langle\mathbf{r}\rangle_A|^2}$ (`mean_rmsd`); and the displacement of the residue centroid between state means. Validated on a synthetic ensemble with a known 3 Å shift and known noise to 0.01–0.02 Å.

**Ligand–residue contacts** (with `--lig` or `--ligand-selection`). A residue is in contact in a frame when any heavy-atom pair (`not name H* and not type H`) between residue and ligand is closer than `--contact-cutoff` (4.5 Å), with minimum-image distances when the frame carries a box. Persistence is the (weighted) fraction of a state's frames in contact; the binding site is the set of residues with persistence ≥ 0.5 in any state, and its per-residue RMSF is reported from the analysis group. Minimum distances and contacts were checked against brute-force NumPy on the neuraminidase fixture (3·10⁻⁶ Å, 0 mismatches, PBC on/off identical); the 0.5 threshold and the 4.5 Å cutoff are conventions.

## 19. Streaming Covariance (two-pass PCA)

With `--engine streaming` (or automatically when the memory estimate of the in-memory path exceeds `streaming_threshold_gb`, 2 GB) the Cartesian covariance is accumulated chunk by chunk with the parallel update of Chan, Golub & LeVeque (1979) — the chunked form of Welford's algorithm — holding only the running mean and the scatter matrix $M_2 = \sum_i (\mathbf{x}_i - \mu)(\mathbf{x}_i - \mu)^T$ per replicate. Replicates are merged with the identity $\sum_i (\mathbf{x}_i - \mu)(\mathbf{x}_i - \mu)^T = M_2^{(k)} + n_k(\mu_k - \mu)(\mu_k - \mu)^T$, so both weightings of §2.1 are reproduced exactly. Superposition is applied frame by frame before accumulation, exactly as in the in-memory path, and frames are quantised to float32 before accumulation, as the in-memory path stores them, so both paths see identical input numbers. The trajectory is read twice (moments, then projections); memory is $O(D^2)$ with $D = 3\,n_\text{atoms}$ and independent of the number of frames; one worker process per trajectory can be used (`--n-workers`). There is no dual/Gram route in the streaming engine: for $D \gtrsim 20\,000$ (≈ 7 000 atoms, a 3.2 GB matrix) use a coarser selection or the in-memory engine, which switches to the SVD route for $N < D$. `--precision float32` affects only the stored projections (`pca/projections.npy`); covariance accumulation is always float64, and the float32 storage introduces a relative rounding error of about 6·10⁻⁸ (3·10⁻⁸ measured). Validation (`validation/checks_performance.py`): accumulator against `numpy.cov` (2.6·10⁻¹³ absolute), merged moments against `compute_cartesian_pca` (10⁻¹⁵ relative), eigenvalues/eigenvectors/projections against the in-memory path on the demonstration and neuraminidase data for both weightings (≤ 3·10⁻¹⁴ relative), and `ConformAtlasPCA` against `MDAnalysis.analysis.pca.PCA` (9.4·10⁻⁹ relative after the (N−1)/N rescaling).

## 20. Side-Chain and Nucleic-Acid Dihedrals

`--dihedral-set` extends the dihedral PCA of §8 beyond backbone φ/ψ (`features.stream_dihedrals`). Angles are computed with the IUPAC sign convention; each angle enters the feature matrix as (cos θ, sin θ), cos block then sin block per angle type, so the `backbone` set is identical to the previous feature layout. Definitions: χ₁ = N–CA–CB–{CG | CG1 | OG | OG1 | SG}; χ₂ = CA–CB–{CG | CG1}–{CD | CD1 | OD1 | ND1 | SD} (isoleucine accepts `CD1` or the GROMACS/AMBER `CD`); nucleic acids α β γ δ ε ζ with the standard O3′(i−1)/P(i+1) cross-residue atoms and χ = O4′–C1′–N9–C4 (purines) / O4′–C1′–N1–C2 (pyrimidines), with `*` and `′` atom names treated alike. Residues missing any atom are skipped silently. Validated against hand geometry and `MDAnalysis.calc_dihedrals` (10⁻⁹°), χ₁/χ₂ against `MDAnalysis Janin` on an all-atom neuraminidase (206 residues, 6·10⁻¹⁴°), φ/ψ against `MDAnalysis Ramachandran` (385 residues, 0°), and nucleic torsions against a hand computation on constructed coordinates (10⁻⁴°); the CHARMM residue names (`ADE/GUA/CYT/THY/URA`) are included but untested on a real structure.

## 21. Report Text and Interactive Views

**Interpretive sentences.** The report derives short statements from the computed tables; every number is taken from the same objects that produce the CSV files, and `report_interpretation.json` lists them with their sources. The effective sample size $N_\text{eff} = \sum_k N_k / g_k$ uses the statistical inefficiency along the landscape coordinates; its consequence is quoted as the binomial standard error of the dominant state's population, $\text{SE} = 100\sqrt{p(1-p)/N_\text{eff}}$ percentage points. RMSIP wording follows the heuristic ≥ 0.70 high, 0.50–0.70 moderate, < 0.50 low; PC1 cosine content ≥ 0.50 is called diffusive. Confidence-interval sentences quote the block-bootstrap 95 % interval when it exists (between-replicate otherwise) and its width as the smallest population difference the data can resolve. ΔG between the two most populated states is $-k_BT\ln(p_2/p_1)$. Projected barriers are described as lower bounds of the true barrier. The wording thresholds are the same heuristics as the sampling verdict of §5.

**Interactive landscape.** The Plotly heatmap is `FELGrid.free_energy` (kJ/mol) on the bin centres, with `FELGrid.probability` and the watershed label of the bin in the hover text; basin outlines are the 0.5 iso-contours of the per-state indicator on `basin_map`. The 1-D profiles are $-RT\ln(P_i/\max P)$ of the marginals of the 2-D probability (identical to a 1-D histogram on the same edges for an unsmoothed grid). Trajectory paths are the frames in time order per replicate (subsampled evenly beyond 5000 points); the time slider snaps each replicate's marker to the frame nearest the selected time (≤ 200 positions). The heatmap itself is never subsampled.

**Structure viewer.** NGL (MIT) is embedded; each representative structure and the average structure carry the per-residue RMSF (Å) in the B-factor column (state-ensemble RMSF for representatives, all-frame RMSF for the average) and are coloured on a 0 → max-RMSF scale. Rendering in a browser was not verified in the validation environment (no WebGL); the embedded PDB text, the B-factor values and the embedded figure data are (`validation/checks_report.py`).
