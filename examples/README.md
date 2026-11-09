# ConformAtlas Examples

This directory contains examples and templates for running single-replicate, multi-replicate, and WT vs Mutant comparative analyses with **ConformAtlas**.

---

## 1. Generating the Miniature Test Dataset

To avoid distributing multi-gigabyte trajectory binaries in the repository, generate a verified lightweight 5-residue polyalanine dataset with 2 replicates each for WT and Mutant:

```bash
conformatlas example --output examples/polyalanine_demo
```

This creates:
```text
examples/polyalanine_demo/
├── WT/
│   ├── topology.pdb
│   ├── rep1.xtc
│   └── rep2.xtc
├── MUT/
│   ├── topology.pdb
│   ├── rep1.xtc
│   └── rep2.xtc
├── comparison.yaml
└── README.md
```

---

## 2. Running Example Commands

### Multi-Replicate WT Analysis
```bash
conformatlas analyze \
    -s examples/polyalanine_demo/WT/topology.pdb \
    -f examples/polyalanine_demo/WT/rep1.xtc \
    -f examples/polyalanine_demo/WT/rep2.xtc \
    -T 300 \
    --bins 32 \
    --smooth-sigma 1.0 \
    -o examples/polyalanine_demo/results_wt
```

### WT vs Mutant Comparison in Shared PCA Space
```bash
conformatlas compare \
    --config examples/polyalanine_demo/comparison.yaml \
    -o examples/polyalanine_demo/results_comparison
```


## Adenylate kinase (`adk/`)

The Cα atoms of a published 1 µs equilibrium simulation of apo *E. coli* adenylate kinase (Seyler & Beckstein, figshare, doi:10.6084/m9.figshare.5108170, CC BY 4.0): 4,187 frames, domain angles per frame, and the open (4AKE) and closed (1AKE) crystal structures. Used by the [tutorial](../docs/tutorial_adk.md) and the executed notebook [`notebooks/adk_walkthrough.ipynb`](notebooks/adk_walkthrough.ipynb); see [`adk/README.md`](adk/README.md) for provenance and `adk/make_adk_data.py` to rebuild it.

```bash
conformatlas analyze -s examples/adk/adk_ca.pdb -f examples/adk/adk_ca.xtc -n examples/adk/index.ndx \
    -l C-alpha -c C-alpha -T 300 --bins 30 --bootstrap 200 --msm -o adk_results
```

## Regression fixture

`tests/data/adk_ca/` holds every 40th frame of the same trajectory (105 frames) with an Ap5A ligand whose coordinates are synthetic (placed from 1AKE), plus the heavy atoms of 4AKE. The integration tests use it for the ligand, index-file, time-window, bootstrap and alternative-CV code paths.
