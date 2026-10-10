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

The **equilibration point** $t_0$ is chosen as the start that maximises $N_\text{eff}$ of the remaining production segment (Chodera 2016), evaluated on the fit RMSD to the reference. With `--equilibration auto` the frames before $t_0$ are discarded per replicate; by default the detected $t_0$ is only reported (`convergence/equilibration.csv`). On a monotonically drifting RMSD the criterion discards aggressively; inspect the descriptor figure before accepting it.

## 8. Alternative Collective Variables

* **Dihedral PCA** (`--pca-type dihedral`): backbone φ/ψ angles are mapped to $(\cos\theta, \sin\theta)$ so that periodicity is respected (Mu, Nguyen & Stock 2005); PCA is run on this feature matrix. No superposition is needed.
* **TICA** (`--pca-type tica`): the generalised eigenproblem $C(\tau)\,v = \lambda\,C(0)\,v$ is solved with the symmetrised time-lagged covariance $C(\tau)$ at lag $\tau$ frames (Molgedey & Schuster 1994; Pérez-Hernández et al. 2013). Features are first whitened in the PCA basis; directions with variance below $10^{-6}$ of the largest are discarded, and when the number of frames is not much larger than the number of coordinates the problem is pre-reduced to the leading principal components. Eigenvalues are autocorrelations at lag τ and the implied timescale is $t_i = -\tau\,\Delta t/\ln\lambda_i$. Lag times are computed within each replicate, never across the junction between replicates.
* **RMSD / Rg** (`--cv rmsd-rg`): backbone RMSD to the reference after superposition and the radius of gyration of the analysis group.
* **External CVs** (`--cv file`): two columns of a text file (PLUMED `COLVAR`, `gmx` `.xvg`, CSV); one file per trajectory, rows matched to frames after striding and windowing.
* **Component choice** (`--fel-pcs i,j`): any two retained components may span the landscape.

## 9. Reweighting

With `--weights-file`, per-frame weights $w_t$ (for example $\exp(\beta V_\text{bias})$ from metadynamics, or Boltzmann factors from replica exchange) enter the histogram, state populations and bootstrap: $P(x,y) \propto \sum_t w_t\, \mathbb{1}[(x_t,y_t)\in\text{bin}]$. Kinetic quantities (transition counts, dwell times, MFPT) are not computed for weighted frames because the time ordering of a biased trajectory is not physical.

## 10. Grid Resolution

Without `--bins`, the bin width follows the Freedman–Diaconis rule $h = 2\,\text{IQR}\,N^{-1/3}$ on each CV; the finer of the two axes sets a square grid, clipped to 16–128 bins. The result is recorded in `run_metadata.json` (`bins`, `bins_mode`). A warning is issued when the occupied bins hold fewer than two frames on average.

## 11. Barriers and Kinetics

Two basins are *adjacent* when their label regions share a grid edge. The **saddle** between them is the minimum, over the shared boundary, of the larger of the two neighbouring free energies, and the barrier from state $i$ is saddle $-\,\Delta G_{\min,i}$ (`states/barriers.csv`). Barriers are not reported for non-adjacent basins. Transition counts are taken between consecutive assigned frames within each replicate (unassigned frames are skipped, replicate junctions are never counted), the transition-probability matrix is the row-normalised count matrix at a lag of one frame, dwell times are lengths of uninterrupted visits (visits touching the start or end of a replicate are flagged as censored), and mean first-passage times are the Markov estimate from the one-frame transition matrix — a lower bound that assumes memoryless hopping between the detected states.

## 12. Block Bootstrap of the Landscape

For each of `--bootstrap` resamples, frames are drawn per replicate as circular blocks of length $L = 2g$ (at least five blocks per replicate), the histogram, free energy, basin populations (basin labels are held fixed), state minima and saddles are recomputed, and the per-bin standard deviation, 2.5 % and 97.5 % quantiles are stored (`fel/free_energy_bootstrap_*.npy`, `statistics/state_populations.csv`, `states/state_free_energies.csv`, `states/barriers.csv`). Bins sampled in fewer than half the resamples are left undefined.

## 13. Comparison Statistics

All systems share one grid. For every system $S$ and the reference $R$: the **ΔΔG map** $\Delta G_S(x,y) - \Delta G_R(x,y)$ (each landscape to its own minimum; NaN where either is unsampled) with the bootstrap standard deviation of the difference obtained from independent per-system block bootstraps; **population differences** with the bootstrap 95 % interval and a two-sided bootstrap p-value (twice the smaller tail fraction of the difference distribution, floored at $1/B$); **Welch's t-test** on per-replicate populations, reported only when every system has at least two replicates, with the replicate counts stated because two or three replicates give very low power; **state ΔΔG** from the state minima; the **Bhattacharyya coefficient** $\sum\sqrt{p_S\,q_R}$, Hellinger distance and **Jensen–Shannon divergence** (bits) between the normalised sampled densities; and the **RMSIP** between the leading eigenvectors of each system's own PCA (`comparison/subspace_overlap.csv`).


---

## 14. What is validated, what is heuristic

`validation/validate.py` reproduces the checks listed in `validation/VALIDATION.md`. The following are **validated** against analytical results or independent codes: Boltzmann inversion (exact bin-integrated Gaussian, counting-noise level), reweighting (importance-sampling identity), Cartesian PCA (MDAnalysis to 10⁻³ after the N−1 → N rescaling; `gmx covar` to 10⁻⁴), the SVD route, Kabsch superposition (MDAnalysis QCP, 10⁻⁵ Å), statistical inefficiency and equilibration (pymbar, machine precision / identical t₀), basin populations (Boltzmann integral, 0.001), saddle barriers (analytic potential, ≤ 0.5 kJ/mol at 60 bins), block-bootstrap coverage (0.92 for a nominal 0.95 on strongly correlated data), transition matrix / dwell times / MFPT (known Markov chain, ≤ 2 %), TICA (AR(1) theory, deeptime), dihedral features (MDAnalysis), RMSD/Rg/RMSF (MDAnalysis, `gmx gyrate`), JS/Bhattacharyya (SciPy), and the type-I error of the population-difference test (0.09 at α = 0.05 with 2 + 2 correlated replicates — mildly anti-conservative, as block bootstraps are).

The following are **heuristics** and should be read as such: the sampling verdict (threshold values for RMSIP, cosine content and N_eff), the Freedman–Diaconis grid size, the `--min-state-population` and `--min-barrier` thresholds, the choice of fit RMSD as the observable for equilibration detection, the lag-1 Markov assumption behind MFPT (a lower bound), and the fixed-basin assumption in the bootstrap (uncertainty from re-segmenting the landscape is not included). Covariance normalisation uses N (GROMACS convention), not N−1.
