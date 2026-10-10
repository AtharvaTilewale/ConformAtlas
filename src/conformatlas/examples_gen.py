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
        logger.warning(
            "MDAnalysis not installed; cannot generate synthetic binary XTC trajectories."
        )
        return

    # ------------------------------------------------------------------
    # Synthetic two-state ensembles with GENUINE internal motion.
    #
    # The peptide is split at a hinge (residues 1-2 vs 3-5). State A bends the
    # C-terminal half about an axis perpendicular to the helix; State B twists
    # it about the helix axis. Each frame additionally receives a random
    # rigid-body rotation and translation, exactly as a molecule tumbling in a
    # simulation box would, so the dataset only yields two basins once frames
    # have been least-squares superposed (which ConformAtlas does before PCA).
    # Replicates switch between states through a two-state Markov chain whose
    # stationary distribution gives the target populations.
    # ------------------------------------------------------------------
    u = mda.Universe(str(wt_topo))
    base_pos = u.atoms.positions.copy().astype(np.float64)
    n_atoms = len(base_pos)
    n_frames = 100

    hinge_atom = 11  # CA of residue 3 (0-based index)
    moving = np.arange(n_atoms) >= 10  # residues 3-5 move relative to residues 1-2
    hinge = base_pos[hinge_atom]
    helix_axis = base_pos[21] - base_pos[1]  # CA(5) - CA(1)
    helix_axis /= np.linalg.norm(helix_axis)
    bend_axis = np.cross(helix_axis, [0.0, 0.0, 1.0])
    bend_axis /= np.linalg.norm(bend_axis)

    def _rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle_rad) * k + (1 - np.cos(angle_rad)) * (k @ k)

    def _random_rotation(rng: np.random.Generator) -> np.ndarray:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def _conformation(state: str, rng: np.random.Generator) -> np.ndarray:
        pos = base_pos.copy()
        if state == "A":
            angle = np.deg2rad(40.0 + rng.normal(0.0, 4.0))
            rot = _rotation(bend_axis, angle)
        else:
            angle = np.deg2rad(70.0 + rng.normal(0.0, 5.0))
            rot = _rotation(helix_axis, angle)
        pos[moving] = (pos[moving] - hinge) @ rot.T + hinge
        pos += rng.normal(0.0, 0.15, size=pos.shape)  # thermal noise, Angstrom
        # Rigid-body tumbling and drift in the box (removed by superposition).
        pos = (pos - pos.mean(axis=0)) @ _random_rotation(rng).T
        pos += rng.uniform(-8.0, 8.0, size=3) + 20.0
        return pos

    def _write_replicate(out_xtc: Path, p_state_a: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # Two-state Markov chain: pi_A = b / (a + b) with persistence a + b = 0.2
        a_to_b = 0.2 * (1.0 - p_state_a)
        b_to_a = 0.2 * p_state_a
        state = "A" if rng.random() < p_state_a else "B"
        with mda.Writer(str(out_xtc), n_atoms=n_atoms) as w:
            for f in range(n_frames):
                u.trajectory.ts.time = float(f * 10)  # 10 ps timestep
                u.atoms.positions = _conformation(state, rng).astype(np.float32)
                w.write(u)
                flip = a_to_b if state == "A" else b_to_a
                if rng.random() < flip:
                    state = "B" if state == "A" else "A"

    # WT: mostly State A (65%); MUT: mostly State B (70%)
    for rep_idx, rep_name in enumerate(["rep1.xtc", "rep2.xtc"]):
        _write_replicate(wt_dir / rep_name, p_state_a=0.65, seed=100 + rep_idx)
        _write_replicate(mut_dir / rep_name, p_state_a=0.30, seed=200 + rep_idx)

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

This directory contains a miniature 5-residue polyalanine dataset with 2 replicates (100 frames each) for WT and MUT.

The peptide switches between a bent (State A) and a twisted (State B) conformation
(WT about 65 % A, MUT about 30 % A) and every frame carries a random rigid-body rotation
and translation. The two states are only resolved after superposition, which
`conformatlas analyze` performs by default. Use `--bins 32 --smooth-sigma 1.0`:
200 frames do not populate a 64 x 64 grid.

### 1. Analyze WT Multi-Replicate Trajectories
```bash
conformatlas analyze \\
    -s WT/topology.pdb \\
    -f WT/rep1.xtc \\
    -f WT/rep2.xtc \\
    -T 300 \\
    --bins 32 --smooth-sigma 1.0 \\
    -o results_wt
```

### 2. Compare WT vs MUT in Shared PCA Space
```bash
conformatlas compare --config comparison.yaml -o results_comparison
```
"""
    (base_dir / "README.md").write_text(readme_text, encoding="utf-8")
