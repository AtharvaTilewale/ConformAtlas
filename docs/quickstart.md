# Quickstart Guide

Get up and running with **ConformAtlas** in 5 minutes.

---

## 1. Generate an Example Project

Generate a lightweight 5-residue polyalanine dataset containing WT and Mutant replicates:

```bash
conformatlas example --output demo_project
cd demo_project
```

---

## 2. Single System & Multi-Replicate Analysis

Analyze multi-replicate molecular dynamics trajectories:

```bash
conformatlas analyze \
    -s WT/topology.pdb \
    -f WT/rep1.xtc \
    -f WT/rep2.xtc \
    -T 300 \
    --bins 32 \
    --output results_wt
```

### Key CLI Options:
* `-s, --structure, --topology`: Topology/structure file (`.tpr`, `.gro`, `.pdb`).
* `-f, --trajectory`: Trajectory file (`.xtc`). Repeat this flag to include multiple replicates.
* `-T, --temperature`: Absolute temperature in Kelvin (e.g. `300`, `310`, `330`).
* `--stride`: Frame stride for large trajectories (e.g. `--stride 5`).
* `--smooth-sigma`: Gaussian smoothing sigma for free energy surface (e.g. `--smooth-sigma 1.0`).
* `--pca-weighting`: Weighting mode (`equal-replicate` or `frames`).

---

## 3. WT vs Mutant Comparison in Shared PCA Space

To compare Wild-Type and Mutant ensembles in an identical PCA coordinate system:

1. Create a configuration file (or use `demo_project/comparison.yaml`):

```yaml
project: WT_vs_MUT_Comparison
temperature: 300.0
output_dir: results_comparison

pca:
  fit_group: Backbone
  analysis_group: Backbone
  reference: combined       # Options: 'combined' or 'reference-system'
  components: 10

bins: 64
smooth_sigma: 1.0
min_state_population: 0.05

systems:
  - name: WT
    topology: WT/topology.pdb
    trajectories:
      - WT/rep1.xtc
      - WT/rep2.xtc

  - name: MUT
    topology: MUT/topology.pdb
    trajectories:
      - MUT/rep1.xtc
      - MUT/rep2.xtc
```

2. Run comparative analysis:

```bash
conformatlas compare --config comparison.yaml
```

---

## 4. Inspecting Outputs

Open the comprehensive interactive HTML report:

```bash
# In your web browser
xdg-open results_wt/report.html
```
