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
