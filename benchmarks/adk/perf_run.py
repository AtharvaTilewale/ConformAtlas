"""One timed run of one tool on one selection (Level 1: no extra fit), in a fresh process.

Stage timed: load the selection from the DCD -> covariance -> eigendecomposition -> top-10 projections.
Interpreter start-up and module imports are excluded (they are in the wall time recorded by the orchestrator).

    python perf_run.py <tool> <selection>
Tools: conformatlas (code taken from $CA_SRC if set), numpy_dense, mdanalysis, prody, mdtraj_sklearn_auto,
       mdtraj_sklearn_full, gromacs
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if os.environ.get("CA_SRC"):
    sys.path.insert(0, os.environ["CA_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from common import DCD, N_MODES, SELECTIONS, universe  # noqa: E402

GMX = os.environ.get("GMX") or shutil.which("gmx") or "gmx"  # $GMX or gmx on PATH


def _load(selection: str) -> np.ndarray:
    u = universe()
    ag = u.select_atoms(SELECTIONS[selection])
    return u.trajectory.timeseries(ag, order="fac").astype(np.float64)


def t_conformatlas(sel: str):
    from conformatlas.pca import compute_cartesian_pca, project_coordinates

    c = _load(sel)
    ev, vec, _, _, mean = compute_cartesian_pca({"r": c}, weighting="frames", n_components=N_MODES)
    project_coordinates(c, mean, vec)
    return ev


def t_numpy_dense(sel: str):
    c = _load(sel)
    x = c.reshape(len(c), -1)
    xc = x - x.mean(0)
    w, v = np.linalg.eigh(xc.T @ xc / len(x))
    return (xc @ v[:, ::-1][:, :N_MODES], w[::-1][:N_MODES])[1]


def t_mdanalysis(sel: str):
    from MDAnalysis.analysis.pca import PCA

    u = universe()
    p = PCA(u, select=SELECTIONS[sel], align=False, n_components=N_MODES).run()
    p.transform(u.select_atoms(SELECTIONS[sel]), n_components=N_MODES)
    return p.results.variance[:N_MODES]


def t_prody(sel: str):
    import prody

    prody.confProDy(verbosity="none")
    idx = universe().select_atoms(SELECTIONS[sel]).indices
    ens = prody.parseDCD(str(DCD))  # ProDy's own DCD parser (all atoms), then the selection
    coords = ens.getCoordsets()[:, idx]
    e = prody.Ensemble("a")
    e.setCoords(coords[0])
    e.addCoordset(coords)
    pca = prody.PCA("a")
    pca.buildCovariance(e)
    pca.calcModes(n_modes=N_MODES)
    e.setCoords(coords.mean(0))
    prody.calcProjection(e, pca[:N_MODES], rmsd=False)
    return pca.getEigvals()


def _mdtraj_sklearn(sel: str, solver: str):
    import mdtraj
    from sklearn.decomposition import PCA

    u = universe()
    idx = u.select_atoms(SELECTIONS[sel]).indices
    with tempfile.TemporaryDirectory() as tmp:
        pdb = Path(tmp) / "a.pdb"
        u.atoms.write(str(pdb))
        t = mdtraj.load_dcd(str(DCD), top=str(pdb), atom_indices=idx)
    x = t.xyz.reshape(t.n_frames, -1)
    p = PCA(n_components=N_MODES, svd_solver=solver, random_state=0).fit(x)
    p.transform(x)
    return p.explained_variance_


def t_gromacs(sel: str, prepared: dict):
    d = Path(prepared["dir"])
    env = {"GMX_MAXBACKUP": "-1", "PATH": "/usr/bin:/bin"}
    for args in (
        [
            "covar",
            "-f",
            "sel.trr",
            "-s",
            "sel.pdb",
            "-o",
            "ev.xvg",
            "-v",
            "evec.trr",
            "-nofit",
            "-nomwa",
            "-nopbc",
        ],
        [
            "anaeig",
            "-f",
            "sel.trr",
            "-s",
            "sel.pdb",
            "-v",
            "evec.trr",
            "-first",
            "1",
            "-last",
            str(N_MODES),
            "-proj",
            "proj.xvg",
        ],
    ):
        r = subprocess.run(
            [GMX, *args], input="0\n0\n", text=True, capture_output=True, cwd=d, env=env
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-300:])
    return None


def prepare_gromacs(sel: str) -> dict:
    """Convert the selection to TRR + PDB once, outside the timed stage."""
    import MDAnalysis as mda

    tmp = Path(tempfile.mkdtemp(prefix="gmxperf_"))
    u = universe()
    ag = u.select_atoms(SELECTIONS[sel])
    ag.write(str(tmp / "sel.pdb"))
    with mda.Writer(str(tmp / "sel.trr"), ag.n_atoms) as w:
        for _ in u.trajectory:
            w.write(ag)
    return {"dir": str(tmp)}


TOOLS = {
    "conformatlas": t_conformatlas,
    "numpy_dense": t_numpy_dense,
    "mdanalysis": t_mdanalysis,
    "prody": t_prody,
    "mdtraj_sklearn_auto": lambda s: _mdtraj_sklearn(s, "auto"),
    "mdtraj_sklearn_full": lambda s: _mdtraj_sklearn(s, "full"),
}


def main() -> None:
    tool, sel = sys.argv[1], sys.argv[2]
    prepared = prepare_gromacs(sel) if tool == "gromacs" else None
    t0 = time.perf_counter()
    result = t_gromacs(sel, prepared) if tool == "gromacs" else TOOLS[tool](sel)
    dt = time.perf_counter() - t0
    top = None if result is None else float(np.asarray(result)[0])
    if prepared:
        import shutil

        shutil.rmtree(prepared["dir"], ignore_errors=True)
    children_gb = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    )  # largest waited-for child (gmx)
    print(
        json.dumps(
            {
                "tool": tool,
                "selection": sel,
                "compute_seconds": dt,
                "top_eigenvalue": top,
                "children_peak_rss_gb": children_gb,
            }
        )
    )


if __name__ == "__main__":
    main()
