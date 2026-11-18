"""Generate inputs for the ConformAtlas container end-to-end tests.

Usage (inside the image):
    python make_test_data.py SRC_DIR MDA_DATA_DIR OUT_DIR [N_LARGE_FRAMES]

SRC_DIR       a checkout of the ConformAtlas release under test (tests/, validation/)
MDA_DATA_DIR  directory with adk.psf, adk_dims.dcd from MDAnalysisTests
OUT_DIR       where the generated files are written
N_LARGE_FRAMES  frames in the long synthetic AdK trajectory (default 0 = skip)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def main() -> None:
    src, mda_data, out = (Path(a) for a in sys.argv[1:4])
    n_large = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(src / "validation"))

    import MDAnalysis as mda
    from checks_enhanced import sample_umbrella_windows, simulate_metadynamics

    from conformatlas.reweighting import write_weights_file

    # Metadynamics: PLUMED HILLS + COLVAR from a 1-D double well.
    simulate_metadynamics(out / "HILLS", out / "COLVAR", n_steps=150_000, pace=150)

    # Umbrella sampling: one COLVAR per window.
    centers = np.arange(-1.6, 1.61, 0.2)
    samples = sample_umbrella_windows(centers, 250.0, 300.0, 4000)
    for k, s in enumerate(samples):
        with open(out / f"COLVAR_{k:02d}", "w", encoding="utf-8") as fh:
            fh.write("#! FIELDS time d\n")
            for i, v in enumerate(s):
                fh.write(f"{i * 0.5:.3f} {v:.8f}\n")
    (out / "centers.txt").write_text(" ".join(f"{c:.1f}" for c in centers))

    # Uniform per-frame weights for the real neuraminidase trajectory.
    h1 = src / "tests" / "data" / "h1n1_ca"
    u = mda.Universe(str(h1 / "neuraminidase_ca.pdb"), str(h1 / "neuraminidase_ca.xtc"))
    times = np.array([ts.time for ts in u.trajectory])
    write_weights_file(out / "weights_uniform.dat", times, np.ones_like(times))

    # Full-atom adenylate kinase (closed -> open transition, 98 frames) as PDB + XTC.
    adk = mda.Universe(str(mda_data / "adk.psf"), str(mda_data / "adk_dims.dcd"))
    adk.trajectory[0]
    adk.atoms.write(str(out / "adk.pdb"))
    with mda.Writer(str(out / "adk_dims.xtc"), adk.atoms.n_atoms) as w:
        for i, ts in enumerate(adk.trajectory):
            ts.time = i * 10.0
            w.write(adk.atoms)

    # Long trajectory: the 98-frame path played back and forth with thermal noise.
    if n_large:
        rng = np.random.default_rng(7)
        frames = np.array([adk.atoms.positions.copy() for _ in adk.trajectory])
        order = np.concatenate([np.arange(len(frames)), np.arange(len(frames) - 2, 0, -1)])
        with mda.Writer(str(out / "adk_long.xtc"), adk.atoms.n_atoms) as w:
            for i in range(n_large):
                adk.atoms.positions = frames[order[i % len(order)]] + rng.normal(0.0, 0.3, frames.shape[1:])
                adk.trajectory.ts.time = i * 10.0
                w.write(adk.atoms)
    print(f"test data written to {out}")


if __name__ == "__main__":
    main()
