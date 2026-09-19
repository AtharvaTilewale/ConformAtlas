"""Synthetic / miniature example dataset generator for quick demonstrations."""

import logging
from pathlib import Path

import numpy as np

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

logger = logging.getLogger("conformatlas.examples")


def generate_mini_example_dataset(base_dir: str | Path):
    """Generate lightweight PDB topology and mini XTC trajectories for WT and MUT."""
    base_dir = Path(base_dir)
    wt_dir = base_dir / "WT"
    mut_dir = base_dir / "MUT"
    wt_dir.mkdir(parents=True, exist_ok=True)
    mut_dir.mkdir(parents=True, exist_ok=True)

    # 5-residue polyalanine alpha-helix PDB template
    pdb_lines = [
        "HEADER    POLYALANINE MINI MODEL",
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C",
        "ATOM      3  C   ALA A   1       2.009   1.428   0.000  1.00 20.00           C",
        "ATOM      4  O   ALA A   1       1.246   2.395   0.000  1.00 20.00           O",
        "ATOM      5  CB  ALA A   1       2.019  -0.776  -1.206  1.00 20.00           C",
        "ATOM      6  N   ALA A   2       3.328   1.545   0.000  1.00 20.00           N",
        "ATOM      7  CA  ALA A   2       4.015   2.831   0.000  1.00 20.00           C",
        "ATOM      8  C   ALA A   2       3.724   3.677   1.238  1.00 20.00           C",
        "ATOM      9  O   ALA A   2       4.521   4.562   1.564  1.00 20.00           O",
        "ATOM     10  CB  ALA A   2       5.526   2.607  -0.081  1.00 20.00           C",
        "ATOM     11  N   ALA A   3       2.569   3.407   1.889  1.00 20.00           N",
        "ATOM     12  CA  ALA A   3       2.128   4.108   3.096  1.00 20.00           C",
        "ATOM     13  C   ALA A   3       2.894   3.578   4.305  1.00 20.00           C",
        "ATOM     14  O   ALA A   3       2.568   3.927   5.441  1.00 20.00           O",
        "ATOM     15  CB  ALA A   3       0.627   3.931   3.272  1.00 20.00           C",
        "ATOM     16  N   ALA A   4       3.921   2.753   4.048  1.00 20.00           N",
        "ATOM     17  CA  ALA A   4       4.779   2.179   5.093  1.00 20.00           C",
        "ATOM     18  C   ALA A   4       3.993   1.222   6.002  1.00 20.00           C",
        "ATOM     19  O   ALA A   4       4.331   1.042   7.176  1.00 20.00           O",
        "ATOM     20  CB  ALA A   4       5.955   1.439   4.464  1.00 20.00           C",
        "ATOM     21  N   ALA A   5       2.934   0.628   5.457  1.00 20.00           N",
        "ATOM     22  CA  ALA A   5       2.091  -0.340   6.166  1.00 20.00           C",
        "ATOM     23  C   ALA A   5       2.915  -1.579   6.577  1.00 20.00           C",
        "ATOM     24  O   ALA A   5       2.483  -2.428   7.368  1.00 20.00           O",
        "ATOM     25  CB  ALA A   5       0.916  -0.781   5.297  1.00 20.00           C",
        "END",
    ]
    pdb_content = "\n".join(pdb_lines) + "\n"

    wt_topo = wt_dir / "topology.pdb"
    mut_topo = mut_dir / "topology.pdb"
    wt_topo.write_text(pdb_content, encoding="utf-8")
    mut_topo.write_text(pdb_content, encoding="utf-8")

    if not HAS_MDA:
        logger.warning("MDAnalysis not installed; cannot generate synthetic binary XTC trajectories.")
        return

    # Generate synthetic conformational ensembles with 2 distinct basins for WT and MUT
    u = mda.Universe(str(wt_topo))
    base_pos = u.atoms.positions.copy()
    n_atoms = len(base_pos)

    # Replicate trajectories (100 frames each)
    n_frames = 100
    np.random.seed(42)

    # WT: mostly Basin A (65%), some Basin B (35%)
    for rep_idx, rep_name in enumerate(["rep1.xtc", "rep2.xtc"]):
        out_xtc = wt_dir / rep_name
        with mda.Writer(str(out_xtc), n_atoms=n_atoms) as w:
            for f in range(n_frames):
                u.trajectory.ts.time = float(f * 10)  # 10 ps timestep
                noise = np.random.normal(0, 0.15, size=base_pos.shape)
                if f < 65:
                    # State A: bending motion along mode 1
                    shift = np.sin(f / 10.0) * 0.8
                    pos = base_pos + noise
                    pos[:, 0] += shift
                else:
                    # State B: twisting motion along mode 2
                    shift = np.cos(f / 8.0) * 0.6
                    pos = base_pos + noise
                    pos[:, 1] += shift
                u.atoms.positions = pos
                w.write(u)

    # MUT: shifted populations, mostly Basin B (70%), some Basin A (30%)
    for rep_idx, rep_name in enumerate(["rep1.xtc", "rep2.xtc"]):
        out_xtc = mut_dir / rep_name
        with mda.Writer(str(out_xtc), n_atoms=n_atoms) as w:
            for f in range(n_frames):
                u.trajectory.ts.time = float(f * 10)
                noise = np.random.normal(0, 0.15, size=base_pos.shape)
                if f < 30:
                    shift = np.sin(f / 10.0) * 0.8
                    pos = base_pos + noise
                    pos[:, 0] += shift
                else:
                    shift = np.cos(f / 8.0) * 0.6
                    pos = base_pos + noise
                    pos[:, 1] += shift
                u.atoms.positions = pos
                w.write(u)

    # Create comparison.yaml
    yaml_text = f"""# Example Comparison Configuration
project: WT_vs_MUT_Example
temperature: 300.0
output_dir: {base_dir.resolve()}/results_comparison

pca:
  fit_group: Backbone
  analysis_group: Backbone
  reference: combined
  components: 5

bins: 32
smooth_sigma: 1.0
min_state_population: 0.05
max_states: 5
stride: 1

systems:
  - name: WT
    topology: {wt_topo.resolve()}
    trajectories:
      - {wt_dir.resolve()}/rep1.xtc
      - {wt_dir.resolve()}/rep2.xtc

  - name: MUT
    topology: {mut_topo.resolve()}
    trajectories:
      - {mut_dir.resolve()}/rep1.xtc
      - {mut_dir.resolve()}/rep2.xtc
"""
    (base_dir / "comparison.yaml").write_text(yaml_text, encoding="utf-8")

    readme_text = """# ConformAtlas Example Project

This directory contains a miniature 5-residue polyalanine MD dataset with 2 replicates for WT and MUT.

### 1. Analyze WT Multi-Replicate Trajectories
```bash
conformatlas analyze \\
    -s WT/topology.pdb \\
    -f WT/rep1.xtc \\
    -f WT/rep2.xtc \\
    -T 300 \\
    -o results_wt
```

### 2. Compare WT vs MUT in Shared PCA Space
```bash
conformatlas compare --config comparison.yaml -o results_comparison
```
"""
    (base_dir / "README.md").write_text(readme_text, encoding="utf-8")
