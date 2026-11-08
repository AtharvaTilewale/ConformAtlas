"""G5 (Level 2, informational expectation) of benchmarks/CRITERIA.md: every tool runs its own default
superposition + PCA on the Cα atoms of the (unfitted-by-the-tool) ADK trajectory; RMSIP of the top 10 modes
between ConformAtlas's default ``conformatlas analyze`` run and each other tool is reported.

Expectation (not a gate): RMSIP >= 0.95.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DCD, N_MODES, PSF, SELECTIONS, coordinates, save_json, universe  # noqa: E402

GMX = "/home/atharva/.claude-science/conda/envs/gmx-ref/bin/gmx"
SEL = SELECTIONS["calpha"]


def rmsip(a: np.ndarray, b: np.ndarray, k: int = N_MODES) -> float:
    a, b = a[:, :k], b[:, :k]
    a = a / np.linalg.norm(a, axis=0)
    b = b / np.linalg.norm(b, axis=0)
    return float(np.sqrt(np.sum((a.T @ b) ** 2) / k))


def conformatlas_default(tmp: Path) -> np.ndarray:
    out = tmp / "ca_out"
    cmd = [
        sys.executable,
        "-m",
        "conformatlas",
        "analyze",
        "-s",
        str(PSF),
        "-f",
        str(DCD),
        "-T",
        "300",
        "-c",
        "C-alpha",
        "-l",
        "Backbone",
        "--engine",
        "memory",
        "-o",
        str(out),
    ]
    r = subprocess.run(
        cmd, capture_output=True, text=True, env={**os.environ, "CONFORMATLAS_NO_PROGRESS": "1"}
    )
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[-500:])
    vec = np.load(out / "pca" / "eigenvectors.npy")
    return vec


def mdanalysis_default() -> np.ndarray:
    from MDAnalysis.analysis.pca import PCA

    return np.asarray(
        PCA(universe(), select=SEL, align=True, n_components=N_MODES).run().results.p_components
    )[:, :N_MODES]


def prody_default() -> np.ndarray:
    import prody

    prody.confProDy(verbosity="none")
    coords = coordinates("calpha")
    ens = prody.Ensemble("adk")
    ens.setCoords(coords[0])
    ens.addCoordset(coords)
    ens.superpose()
    pca = prody.PCA("adk")
    pca.buildCovariance(ens)
    pca.calcModes(n_modes=N_MODES)
    return np.asarray(pca.getEigvecs())


def mdtraj_sklearn_default() -> np.ndarray:
    import mdtraj
    from sklearn.decomposition import PCA

    u = universe()
    idx = u.select_atoms(SEL).indices
    with tempfile.TemporaryDirectory() as tmp:
        pdb = Path(tmp) / "adk.pdb"
        u.atoms.write(str(pdb))
        traj = mdtraj.load_dcd(str(DCD), top=str(pdb), atom_indices=idx)
    traj.superpose(traj, 0)
    x = traj.xyz.reshape(traj.n_frames, -1)
    return PCA(n_components=N_MODES, random_state=0).fit(x).components_.T


def gromacs_default() -> np.ndarray:
    import MDAnalysis as mda

    u = universe()
    ag = u.select_atoms(SEL)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ag.write(str(tmp / "sel.pdb"))
        with mda.Writer(str(tmp / "sel.trr"), ag.n_atoms) as w:
            for _ in u.trajectory:
                w.write(ag)
        r = subprocess.run(
            [
                GMX,
                "covar",
                "-f",
                "sel.trr",
                "-s",
                "sel.pdb",
                "-o",
                "eigenval.xvg",
                "-v",
                "eigenvec.trr",
                "-nomwa",
                "-nopbc",
            ],
            input="0\n0\n",
            text=True,
            capture_output=True,
            cwd=tmp,
            env={"GMX_MAXBACKUP": "-1", "PATH": "/usr/bin:/bin"},
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:])
        ev = mda.Universe(str(tmp / "sel.pdb"), str(tmp / "eigenvec.trr"))
        # fit group == analysis group: frame 0 = fit reference (t=-1), frame 1 = average, eigenvectors from frame 2
        frames = np.array(
            [ev.trajectory[i].positions.astype(float).ravel() for i in range(2, N_MODES + 2)]
        ).T
    return frames


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = conformatlas_default(Path(tmp))
    out = {
        "n_modes": N_MODES,
        "reference": "ConformAtlas default (`analyze -c C-alpha -l Backbone --engine memory`)",
        "expectation": ">= 0.95",
        "tools": {},
    }
    for name, fn in {
        "MDAnalysis (align=True)": mdanalysis_default,
        "ProDy (Ensemble.superpose)": prody_default,
        "MDTraj superpose + sklearn PCA": mdtraj_sklearn_default,
        "GROMACS covar (default fit)": gromacs_default,
    }.items():
        try:
            v = rmsip(ref, fn())
            out["tools"][name] = {"rmsip": v, "meets_expectation": bool(v >= 0.95)}
        except Exception as exc:
            out["tools"][name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(name, out["tools"][name], flush=True)
    save_json("g5_default_workflows.json", out)


if __name__ == "__main__":
    main()
