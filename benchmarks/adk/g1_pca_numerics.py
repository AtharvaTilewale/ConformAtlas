"""G0 (input integrity) and G1 (PCA numerics, Level 1) of benchmarks/CRITERIA.md.

Every tool receives the same pre-fitted coordinates and no further superposition is applied.

    python benchmarks/adk/g1_pca_numerics.py [calpha backbone heavy]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ADK_EQ,
    DCD,
    N_MODES,
    PSF,
    SELECTIONS,
    compare_pca,
    coordinates,
    md5,
    reference_pca,
    save_json,
    universe,
)

GMX = os.environ.get("GMX") or shutil.which("gmx") or "gmx"  # $GMX or gmx on PATH
TOL_DOUBLE = {"eig": 1e-6, "expl": 1e-6, "vec": 1e-6, "proj": 1e-6}
TOL_GMX = {"eig": 1e-4, "expl": 1e-3, "vec": 1e-4, "proj": 1e-3}


def g0() -> dict:
    rec = json.loads((ADK_EQ / "RECORD.json").read_text())["files"]
    u = universe()
    checks = {
        "md5_dcd": md5(DCD) == rec[DCD.name]["md5"],
        "md5_psf": md5(PSF) == rec[PSF.name]["md5"],
        "n_atoms": u.atoms.n_atoms == 3341,
        "n_residues": u.residues.n_residues == 214,
        "n_frames": len(u.trajectory) == 4187,
        "dt_ps": abs(u.trajectory.dt - 240.0) < 1e-3,
    }
    return {"checks": checks, "pass": all(checks.values())}


def run_conformatlas(coords: np.ndarray) -> dict:
    from conformatlas.pca import compute_cartesian_pca, project_coordinates

    ev, vec, expl, _, mean = compute_cartesian_pca(
        {"r": coords}, weighting="frames", n_components=N_MODES
    )
    return {
        "eigenvalues": ev,
        "eigenvectors": vec,
        "explained": expl,
        "projections": project_coordinates(coords, mean, vec),
    }


def run_mdanalysis(selection: str) -> dict:
    from MDAnalysis.analysis.pca import PCA

    u = universe()
    pca = PCA(u, select=SELECTIONS[selection], align=False, n_components=N_MODES).run()
    n = len(u.trajectory)
    var = np.asarray(pca.results.variance) * (n - 1) / n  # N-1 -> N
    total = float(np.sum(var)) if len(var) > N_MODES else None
    ag = u.select_atoms(SELECTIONS[selection])
    proj = pca.transform(ag, n_components=N_MODES)
    cum = np.asarray(pca.results.cumulated_variance) * 100.0
    return {
        "eigenvalues": var[:N_MODES],
        "eigenvectors": np.asarray(pca.results.p_components)[:, :N_MODES],
        "explained": np.diff(np.concatenate([[0.0], cum[:N_MODES]])),
        "projections": np.asarray(proj),
        "note": f"variance array length {len(var)} (total {total})",
    }


def run_prody(coords: np.ndarray) -> dict:
    import prody

    prody.confProDy(verbosity="none")
    ens = prody.Ensemble("adk")
    ens.setCoords(coords[0])
    ens.addCoordset(coords)
    pca = prody.PCA("adk")
    pca.buildCovariance(ens)
    pca.calcModes(n_modes=N_MODES)
    ev = np.asarray(pca.getEigvals())  # ProDy divides by N (D2), no conversion
    vec = np.asarray(pca.getEigvecs())
    ens.setCoords(coords.mean(axis=0))
    proj = np.asarray(prody.calcProjection(ens, pca[:N_MODES], rmsd=False))
    return {
        "eigenvalues": ev,
        "eigenvectors": vec,
        "explained": 100.0 * np.asarray(pca.getEigvals()) / np.trace(pca.getCovariance()),
        "projections": proj,
    }


def run_mdtraj_sklearn(selection: str, solver: str) -> dict:
    import mdtraj
    from sklearn.decomposition import PCA

    u = universe()
    idx = u.select_atoms(SELECTIONS[selection]).indices
    with tempfile.TemporaryDirectory() as tmp:
        pdb = Path(tmp) / "adk.pdb"
        u.atoms.write(str(pdb))
        traj = mdtraj.load_dcd(str(DCD), top=str(pdb), atom_indices=idx)
    x = traj.xyz.reshape(traj.n_frames, -1).astype(np.float64) * 10.0  # nm -> Å
    n = x.shape[0]
    pca = PCA(n_components=N_MODES, svd_solver=solver, random_state=0).fit(x)
    return {
        "eigenvalues": pca.explained_variance_ * (n - 1) / n,
        "eigenvectors": pca.components_.T,
        "explained": 100.0 * pca.explained_variance_ratio_,
        "projections": pca.transform(x),
    }


def _read_xvg(path: Path) -> np.ndarray:
    rows = [ln.split() for ln in path.read_text().splitlines() if ln and ln[0] not in "#@"]
    return np.array(rows, dtype=float)


def run_gromacs(selection: str) -> dict:
    import MDAnalysis as mda

    u = universe()
    ag = u.select_atoms(SELECTIONS[selection])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ag.write(str(tmp / "sel.pdb"))
        with mda.Writer(str(tmp / "sel.trr"), ag.n_atoms) as w:
            for _ in u.trajectory:
                w.write(ag)
        env = {"GMX_MAXBACKUP": "-1", "PATH": "/usr/bin:/bin"}

        def run(args: list[str]) -> None:
            r = subprocess.run(
                [GMX, *args], input="0\n0\n", text=True, capture_output=True, cwd=tmp, env=env
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-600:])

        run(
            [
                "covar",
                "-f",
                "sel.trr",
                "-s",
                "sel.pdb",
                "-o",
                "eigenval.xvg",
                "-v",
                "eigenvec.trr",
                "-nofit",
                "-nomwa",
                "-nopbc",
            ]
        )
        # gmx anaeig always re-fits frames to the structure in the eigenvector file, so its projections are
        # not comparable at Level 1 (no further fit); see benchmarks/DEVIATIONS.md D1.
        eigenval = _read_xvg(tmp / "eigenval.xvg")[:, 1]
        evec = mda.Universe(str(tmp / "sel.pdb"), str(tmp / "eigenvec.trr"))
        # frame 0 is a placeholder (t=-1), frame 1 the average structure, eigenvectors start at frame 2
        vec = np.array(
            [evec.trajectory[i].positions.astype(float).ravel() for i in range(2, N_MODES + 2)]
        ).T
        vec /= np.linalg.norm(vec, axis=0)
    return {
        "eigenvalues": eigenval[:N_MODES] * 100.0,  # nm^2 -> Å^2
        "eigenvectors": vec,
        "explained": 100.0 * eigenval[:N_MODES] / eigenval.sum(),
        "projections": None,
    }


def main(selections: list[str]) -> None:
    out: dict = {"g0": g0()}
    print("G0", out["g0"], flush=True)
    if not out["g0"]["pass"]:
        save_json("g1_pca_numerics.json", out)
        sys.exit("G0 failed: input integrity")
    for sel in selections:
        coords = coordinates(sel)
        t0 = time.perf_counter()
        ref = reference_pca(coords)
        print(
            f"[{sel}] N={coords.shape[0]} D={coords.shape[1] * 3}  reference eigh {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        tools = {
            "ConformAtlas": lambda: run_conformatlas(coords),
            "MDAnalysis": lambda: run_mdanalysis(sel),
            "ProDy": lambda: run_prody(coords),
            "MDTraj+sklearn(full)": lambda: run_mdtraj_sklearn(sel, "full"),
            "MDTraj+sklearn(auto)": lambda: run_mdtraj_sklearn(sel, "auto"),
            "GROMACS": lambda: run_gromacs(sel),
        }
        res = {
            "N": coords.shape[0],
            "D": coords.shape[1] * 3,
            "reference_top": ref["eigenvalues"][:N_MODES].tolist(),
            "tools": {},
        }
        for name, fn in tools.items():
            t = time.perf_counter()
            try:
                m = compare_pca(fn(), ref)
                tol = TOL_GMX if name == "GROMACS" else TOL_DOUBLE
                if "(auto)" in name:
                    m["gate"] = "informational (randomised solver)"
                else:
                    m["pass"] = bool(
                        m["max_rel_eigenvalue_error"] <= tol["eig"]
                        and m["max_explained_abs_error_pct"] <= tol["expl"]
                        and m["max_one_minus_cos_separated"] <= tol["vec"]
                        and (
                            np.isnan(m["max_projection_err_over_sd_separated"])
                            and name == "GROMACS"
                            or m["max_projection_err_over_sd_separated"] <= tol["proj"]
                        )
                    )
                m["seconds"] = round(time.perf_counter() - t, 1)
            except Exception as exc:  # reported, not hidden
                m = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            res["tools"][name] = m
            print(
                f"  {name:22s}",
                {
                    k: (f"{v:.2e}" if isinstance(v, float) else v)
                    for k, v in m.items()
                    if k
                    in (
                        "max_rel_eigenvalue_error",
                        "max_explained_abs_error_pct",
                        "max_one_minus_cos_separated",
                        "max_projection_err_over_sd_separated",
                        "pass",
                        "error",
                        "seconds",
                    )
                },
                flush=True,
            )
        out[sel] = res
        save_json("g1_pca_numerics.json", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["calpha", "backbone", "heavy"])
