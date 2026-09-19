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
3. **Frame-to-Basin Assignment**: Every trajectory frame $(p_1, p_2)$ is mapped to its corresponding spatial grid bin $(x_{\text{bin}}, y_{\text{bin}})$ and assigned to the underlying basin.
4. **Major State Filtering**: Basins containing $\ge \text{min\_state\_population}$ (default: 5%) of total frames are designated as major conformational states (ordered by population as `State A`, `State B`, etc.).
5. **Representative Frame**: For each basin, the real trajectory frame minimizing the Euclidean distance to the basin's free energy minimum in $(PC_1, PC_2)$ space is extracted as the representative structure:
   $$d = \sqrt{(PC_{1, \text{frame}} - PC_{1, \min})^2 + (PC_{2, \text{frame}} - PC_{2, \min})^2}$$

---

## 5. Convergence & Sampling Diagnostics

### 5.1 Root Mean Square Inner Product (RMSIP)
Subspace overlap between two sets of $s$ orthonormal eigenvectors $\mathbf{U} = [\mathbf{u}_1, \dots, \mathbf{u}_s]$ and $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_s]$ is calculated as:

$$\text{RMSIP} = \sqrt{\frac{1}{s} \sum_{i=1}^s \sum_{j=1}^s (\mathbf{u}_i \cdot \mathbf{v}_j)^2} = \sqrt{\frac{1}{s} \|\mathbf{U}^T \mathbf{V}\|_F^2}$$

* $\text{RMSIP} = 1.0$: Identical subspaces.
* $\text{RMSIP} = 0.0$: Mutually orthogonal subspaces.

In ConformAtlas, RMSIP is evaluated:
* Between the first half and second half of each trajectory.
* Progressively across 25%, 50%, 75%, and 100% of trajectory time.
* Pairwise between independent replicates.

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
