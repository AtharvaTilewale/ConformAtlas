# Multi-Replicate Analysis Guide

Modern biomolecular MD simulations rely on multiple independent replicates to ensure statistical reproducibility and avoid premature convergence traps.

---

## 1. Running Multi-Replicate Analysis

Pass repeated `-f` flags to `conformatlas analyze`:

```bash
conformatlas analyze \
    -s WT/topology.tpr \
    -f WT/rep1.xtc \
    -f WT/rep2.xtc \
    -f WT/rep3.xtc \
    -T 310 \
    --pca-weighting equal-replicate \
    --output results/WT_pooled
```

---

## 2. Replicate Preservation & PCA Weighting

### 2.1 Replicate Identity Preservation
ConformAtlas preserves replicate labels (`Rep_1`, `Rep_2`, `Rep_3`) throughout:
* Coordinate streaming
* Principal component projections (`pca/projections.csv`)
* Watershed basin assignments (`states/frame_assignments.csv`)
* Replicate-level population statistics (`statistics/replicate_statistics.csv`)

### 2.2 Replicate Weighting Modes
Trajectories often differ in length due to cluster walltimes or simulation crashes. ConformAtlas provides two weighting modes:
* **`equal-replicate`** (Default): Each replicate contributes equally ($1/K$) to the mean structure and covariance matrix, regardless of frame count. This prevents an unusually long simulation from biasing the principal components.
* **`frames`**: Every frame is weighted equally ($1/N_{\text{total}}$). Recommended only when all replicates have identical lengths.

---

## 3. Replicate Uncertainty Statistics

State populations are evaluated independently across replicates to calculate:
* **Sample Mean**: $\bar{x} = \frac{1}{K} \sum x_k$
* **Sample Standard Deviation**: $\text{SD}$ (degrees of freedom $K-1$)
* **Standard Error of the Mean**: $\text{SEM} = \text{SD} / \sqrt{K}$
* **95% Confidence Interval**: Using Student's $t$ critical value $t_{0.025, K-1} \times \text{SEM}$.
