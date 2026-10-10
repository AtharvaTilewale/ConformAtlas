# ConformAtlas Example Project

This directory contains a miniature 5-residue polyalanine dataset with 2 replicates (100 frames each) for WT and MUT.

The peptide switches between a bent (State A) and a twisted (State B) conformation through a two-state Markov process (WT ≈ 65 % A, MUT ≈ 30 % A), and every frame carries a random rigid-body rotation and translation, as a molecule tumbling in a box would. The two states are only resolved after superposition, which `conformatlas analyze` performs by default. With 200 frames a 64 × 64 grid is too sparse; use `--bins 32 --smooth-sigma 1.0`.

### 1. Analyze WT Multi-Replicate Trajectories
```bash
conformatlas analyze \
    -s WT/topology.pdb \
    -f WT/rep1.xtc \
    -f WT/rep2.xtc \
    -T 300 \
    --bins 32 --smooth-sigma 1.0 \
    -o results_wt
```

### 2. Compare WT vs MUT in Shared PCA Space
```bash
conformatlas compare --config comparison.yaml -o results_comparison
```
