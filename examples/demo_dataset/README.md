# ConformAtlas Example Project

This directory contains a miniature 5-residue polyalanine MD dataset with 2 replicates for WT and MUT.

### 1. Analyze WT Multi-Replicate Trajectories
```bash
conformatlas analyze \
    -s WT/topology.pdb \
    -f WT/rep1.xtc \
    -f WT/rep2.xtc \
    -T 300 \
    -o results_wt
```

### 2. Compare WT vs MUT in Shared PCA Space
```bash
conformatlas compare --config comparison.yaml -o results_comparison
```
