# Sampling & Convergence Diagnostics

Determining whether a molecular dynamics simulation has sampled sufficient conformational phase space is a central challenge in structural biophysics.

---

## 1. Multi-Evidence Framework

ConformAtlas rejects simplistic single-metric convergence thresholds. Instead, it evaluates four complementary diagnostics:

1. **First-Half vs Second-Half Subspace Overlap (RMSIP)**:
   Measures whether the dominant PCA subspace remains stable across the simulation halves. The headline value pools the first halves of all replicates against their second halves; `rmsip.csv` also lists the value for each replicate on its own.
2. **Progressive Overlap (25%, 50%, 75%, 100%)**:
   Tracks the approach of the subspace spanned by the first part of every replicate towards the full-data subspace. The 100% point equals 1 by construction.
3. **Cosine Content ($c_1$) of Dominant Modes**:
   Distinguishes genuine equilibrium fluctuations from random diffusive drift (Hess diffusion test).
4. **State Population Stability**:
   Monitors whether the occupancy of conformational states stabilizes over cumulative trajectory time. Fractions are taken per replicate (the first 25% of every replicate, pooled), not from the concatenated frame table.

---

## 2. Heuristic Interpretation

| Diagnostic | Good Stability | Mixed Evidence | Insufficient Sampling |
|---|---|---|---|
| **RMSIP (halves)** | $\ge 0.70$ | $0.50 \text{--} 0.70$ | $< 0.50$ |
| **PC1 Cosine Content** | $< 0.50$ | $0.50 \text{--} 0.70$ | $\ge 0.70$ |
| **Progressive Overlap** | Plateaus near 1.0 | Ascending steadily | Erratic fluctuations |
| **State Populations** | Stable ($\pm 5\%$) | Shifting ($\pm 15\%$) | Novel states appearing at end |

---

## 3. Heuristic Disclaimer
All convergence diagnostics are empirical heuristics. High subspace overlap demonstrates that the simulation is not undergoing rapid drift along its observed modes, but **does not guarantee** that all biologically relevant conformational states were visited.
