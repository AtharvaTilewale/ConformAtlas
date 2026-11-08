"""One timed run of one tool on one T4L dataset and selection, in a fresh process (T7, reported only).

Datasets: ``pooled100`` = the 5 ff99SB*-ILDN replicates at 100 ps (whole molecules, 50,005 frames);
``md1us4_1ps`` = the deposited md1us4 trajectory at 1 ps (1,000,001 frames).
Stage timed: load -> covariance -> eigendecomposition -> top-10 projections, no superposition (as for ADK).

    python perf_run_t4l.py <tool> <dataset> <selection>
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if os.environ.get("CA_SRC"):
    sys.path.insert(0, os.environ["CA_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from common_t4l import DATA, SELECTIONS, SETS, replicates, tpr, xtc  # noqa: E402

GMX = "/home/atharva/.claude-science/conda/envs/gmx-ref/bin/gmx"
K = 10


def files(dataset: str) -> list[str]:
    if dataset == "pooled100":
        return [str(xtc("ff99SBstar", r)) for r in replicates("ff99SBstar")]
    return [str(DATA / "t4l" / SETS["ff99SBstar"][0] / "md1us4.xtc")]


def indices(sel: str) -> np.ndarray:
    import MDAnalysis as mda

    return mda.Universe(str(tpr("ff99SBstar"))).select_atoms(SELECTIONS[sel]).indices


def load_mda(dataset: str, sel: str) -> np.ndarray:
    import MDAnalysis as mda

    parts = []
    for f in files(dataset):
        u = mda.Universe(str(tpr("ff99SBstar")), f)
        parts.append(u.trajectory.timeseries(u.select_atoms(SELECTIONS[sel]), order="fac"))
    return np.concatenate(parts).astype(np.float64)


def t_conformatlas(dataset, sel):
    from conformatlas.pca import compute_cartesian_pca, project_coordinates

    c = load_mda(dataset, sel)
    ev, vec, _, _, mean = compute_cartesian_pca({"r": c}, weighting="frames", n_components=K)
    project_coordinates(c, mean, vec)
    return ev[0]


def t_conformatlas_streaming(dataset, sel):
    from conformatlas.streaming import streaming_pca

    trajs = {f"r{i}": f for i, f in enumerate(files(dataset))}
    res = streaming_pca(
        str(tpr("ff99SBstar")),
        trajs,
        indices(sel).tolist(),
        fit=False,
        weighting="frames",
        n_components=K,
        quiet=True,
    )
    return float(res.eigenvalues[0])


def t_numpy_dense(dataset, sel):
    c = load_mda(dataset, sel)
    x = c.reshape(len(c), -1)
    xc = x - x.mean(0)
    w, v = np.linalg.eigh(xc.T @ xc / len(x))
    xc @ v[:, ::-1][:, :K]
    return w[-1]


def t_mdanalysis(dataset, sel):
    import MDAnalysis as mda
    from MDAnalysis.analysis.pca import PCA

    u = mda.Universe(str(tpr("ff99SBstar")), files(dataset))
    p = PCA(u, select=SELECTIONS[sel], align=False, n_components=K).run()
    p.transform(u.select_atoms(SELECTIONS[sel]), n_components=K)
    return p.results.variance[0]


def t_prody(dataset, sel):
    import prody

    prody.confProDy(verbosity="none")
    c = load_mda(dataset, sel)  # ProDy has no XTC reader; coordinates loaded with MDAnalysis
    e = prody.Ensemble("t")
    e.setCoords(c[0])
    e.addCoordset(c)
    p = prody.PCA("t")
    p.buildCovariance(e)
    p.calcModes(n_modes=K)
    e.setCoords(c.mean(0))
    prody.calcProjection(e, p[:K], rmsd=False)
    return p.getEigvals()[0]


def _sk(dataset, sel, solver):
    import mdtraj
    from sklearn.decomposition import PCA

    with tempfile.TemporaryDirectory() as tmp:
        import MDAnalysis as mda

        pdb = Path(tmp) / "t.pdb"
        mda.Universe(str(tpr("ff99SBstar")), files(dataset)[0]).atoms.write(str(pdb))
        t = mdtraj.load(files(dataset), top=str(pdb), atom_indices=indices(sel))
    x = t.xyz.reshape(t.n_frames, -1)
    p = PCA(n_components=K, svd_solver=solver, random_state=0).fit(x)
    p.transform(x)
    return p.explained_variance_[0]


def prepare_gromacs(dataset, sel) -> dict:
    d = Path(tempfile.mkdtemp(prefix="gmxperf_"))
    env = {"GMX_MAXBACKUP": "-1", "PATH": "/usr/bin:/bin"}
    group = "C-alpha" if sel == "calpha" else "Protein-H"
    fs = files(dataset)
    if len(fs) > 1:
        subprocess.run(
            [GMX, "trjcat", "-f", *fs, "-o", str(d / "in.xtc"), "-cat"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        traj = d / "in.xtc"
    else:
        traj = Path(fs[0])
    return {"dir": str(d), "traj": str(traj), "group": group, "env": env}


def t_gromacs(prep: dict):
    d = Path(prep["dir"])
    sel = f"{prep['group']}\n{prep['group']}\n"
    for args in (
        [
            "covar",
            "-f",
            prep["traj"],
            "-s",
            str(tpr("ff99SBstar")),
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
            prep["traj"],
            "-s",
            str(tpr("ff99SBstar")),
            "-v",
            "evec.trr",
            "-first",
            "1",
            "-last",
            str(K),
            "-proj",
            "proj.xvg",
        ],
    ):
        r = subprocess.run(
            [GMX, *args], input=sel, text=True, capture_output=True, cwd=d, env=prep["env"]
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-300:])
    return None


TOOLS = {
    "conformatlas": t_conformatlas,
    "conformatlas_streaming": t_conformatlas_streaming,
    "numpy_dense": t_numpy_dense,
    "mdanalysis": t_mdanalysis,
    "prody": t_prody,
    "sklearn_auto": lambda d, s: _sk(d, s, "auto"),
    "sklearn_full": lambda d, s: _sk(d, s, "full"),
}


def main() -> None:
    tool, dataset, sel = sys.argv[1:4]
    prep = prepare_gromacs(dataset, sel) if tool == "gromacs" else None
    t0 = time.perf_counter()
    top = t_gromacs(prep) if tool == "gromacs" else TOOLS[tool](dataset, sel)
    dt = time.perf_counter() - t0
    if prep:
        import shutil

        shutil.rmtree(prep["dir"], ignore_errors=True)
    print(
        json.dumps(
            {
                "tool": tool,
                "dataset": dataset,
                "selection": sel,
                "compute_seconds": dt,
                "top_eigenvalue": None if top is None else float(top),
                "children_peak_rss_gb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                / 1e6,
            }
        )
    )


if __name__ == "__main__":
    main()
