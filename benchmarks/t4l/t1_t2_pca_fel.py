"""T0 (integrity), T1 (PCA numerics, Level 1) and T2 (free-energy landscape) of benchmarks/CRITERIA_T4L_HIV.md.

CA_SRC=<released src> python benchmarks/t4l/t1_t2_pca_fel.py [calpha heavy]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adk"))
from common import compare_pca, reference_pca  # noqa: E402  (benchmarks/adk/common.py)
from common_t4l import (  # noqa: E402
    KB_SI,
    N_MODES,
    SELECTIONS,
    TEMPERATURE,
    pooled,
    raw,
    replicates,
    save_json,
    tpr,
    universe,
)

GMX = "/home/atharva/.claude-science/conda/envs/gmx-ref/bin/gmx"
TOL_DOUBLE = {"eig": 1e-6, "expl": 1e-6, "vec": 1e-6, "proj": 1e-6}
TOL_GMX = {"eig": 1e-4, "expl": 1e-3, "vec": 1e-4}


# ------------------------------------------------------------------ T0
def t0() -> dict:
    out = {}
    for ff in ("ff99SBstar", "ff15ipq"):
        for rep in replicates(ff):
            u = universe(ff, rep)
            n = len(u.trajectory)
            t_last = u.trajectory[-1].time
            dt = u.trajectory.dt
            ca = raw(ff, rep, "calpha")
            d = np.linalg.norm(np.diff(ca, axis=1), axis=2)
            expected = int(np.floor(round(t_last - u.trajectory[0].time) / 100.0)) + 1
            out[f"{ff}/{rep}"] = {
                "frames": n,
                "expected_frames": expected,
                "dt_ps": float(dt),
                "atoms": int(u.atoms.n_atoms),
                "max_consecutive_ca_A": float(d.max()),
                "pass": bool(
                    n == expected
                    and abs(dt - 100.0) < 1e-3
                    and u.atoms.n_atoms == 2612
                    and d.max() <= 4.2
                ),
            }
            print("T0", ff, rep, out[f"{ff}/{rep}"], flush=True)
    return {"replicates": out, "pass": all(v["pass"] for v in out.values())}


# ------------------------------------------------------------------ T1 tools (identical pre-fitted coordinates)
def run_conformatlas(c: np.ndarray) -> dict:
    from conformatlas.pca import compute_cartesian_pca, project_coordinates

    ev, vec, expl, _, mean = compute_cartesian_pca(
        {"r": c}, weighting="frames", n_components=N_MODES
    )
    return {
        "eigenvalues": ev,
        "eigenvectors": vec,
        "explained": expl,
        "projections": project_coordinates(c, mean, vec),
    }


def run_mdanalysis(c: np.ndarray, selection: str) -> dict:
    import MDAnalysis as mda
    from MDAnalysis.analysis.pca import PCA
    from MDAnalysis.coordinates.memory import MemoryReader

    u0 = mda.Universe(str(tpr("ff99SBstar")))
    u = mda.Merge(u0.select_atoms(SELECTIONS[selection]))
    u.load_new(c.astype(np.float32), format=MemoryReader, order="fac")
    p = PCA(u, select="all", align=False, n_components=N_MODES).run()
    n = len(c)
    var = np.asarray(p.results.variance) * (n - 1) / n
    cum = np.asarray(p.results.cumulated_variance) * 100.0
    return {
        "eigenvalues": var[:N_MODES],
        "eigenvectors": np.asarray(p.results.p_components)[:, :N_MODES],
        "explained": np.diff(np.concatenate([[0.0], cum[:N_MODES]])),
        "projections": np.asarray(p.transform(u.atoms, n_components=N_MODES)),
        "note": "MemoryReader holds float32 coordinates",
    }


def run_prody(c: np.ndarray) -> dict:
    import prody

    prody.confProDy(verbosity="none")
    ens = prody.Ensemble("t4l")
    ens.setCoords(c[0])
    ens.addCoordset(c)
    pca = prody.PCA("t4l")
    pca.buildCovariance(ens)
    pca.calcModes(n_modes=N_MODES)
    ens.setCoords(c.mean(axis=0))
    return {
        "eigenvalues": np.asarray(pca.getEigvals()),
        "eigenvectors": np.asarray(pca.getEigvecs()),
        "explained": 100.0 * np.asarray(pca.getEigvals()) / np.trace(pca.getCovariance()),
        "projections": np.asarray(prody.calcProjection(ens, pca[:N_MODES], rmsd=False)),
    }


def run_sklearn(c: np.ndarray, solver: str) -> dict:
    from sklearn.decomposition import PCA

    x = c.reshape(len(c), -1)
    n = len(x)
    p = PCA(n_components=N_MODES, svd_solver=solver, random_state=0).fit(x)
    return {
        "eigenvalues": p.explained_variance_ * (n - 1) / n,
        "eigenvectors": p.components_.T,
        "explained": 100.0 * p.explained_variance_ratio_,
        "projections": p.transform(x),
    }


def _read_xvg(path: Path) -> np.ndarray:
    return np.array(
        [ln.split() for ln in path.read_text().splitlines() if ln and ln[0] not in "#@"],
        dtype=float,
    )


def run_gromacs(c: np.ndarray, selection: str) -> dict:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    u0 = mda.Universe(str(tpr("ff99SBstar")))
    u = mda.Merge(u0.select_atoms(SELECTIONS[selection]))
    u.load_new(c.astype(np.float32), format=MemoryReader, order="fac")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        u.atoms.write(str(tmp / "sel.pdb"))
        with mda.Writer(str(tmp / "sel.trr"), u.atoms.n_atoms) as w:
            for _ in u.trajectory:
                w.write(u.atoms)
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
                "-nofit",
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
            raise RuntimeError(r.stderr[-500:])
        ev = _read_xvg(tmp / "eigenval.xvg")[:, 1]
        e = mda.Universe(str(tmp / "sel.pdb"), str(tmp / "eigenvec.trr"))
        vec = np.array(
            [e.trajectory[i].positions.astype(float).ravel() for i in range(2, N_MODES + 2)]
        ).T
        vec /= np.linalg.norm(vec, axis=0)
    return {
        "eigenvalues": ev[:N_MODES] * 100.0,
        "eigenvectors": vec,
        "explained": 100.0 * ev[:N_MODES] / ev.sum(),
        "projections": None,
    }


def t1(selection: str) -> tuple[dict, np.ndarray]:
    c, lens = pooled("ff99SBstar", selection)
    ref = reference_pca(c)
    res = {"N": int(len(c)), "D": int(c.shape[1] * 3), "replicate_frames": lens, "tools": {}}
    tools = {
        "ConformAtlas": lambda: run_conformatlas(c),
        "MDAnalysis": lambda: run_mdanalysis(c, selection),
        "ProDy": lambda: run_prody(c),
        "scikit-learn PCA (full)": lambda: run_sklearn(c, "full"),
        "scikit-learn PCA (auto)": lambda: run_sklearn(c, "auto"),
        "GROMACS": lambda: run_gromacs(c, selection),
    }
    ca_proj = None
    for name, fn in tools.items():
        t = time.perf_counter()
        try:
            out = fn()
            if name == "ConformAtlas":
                ca_proj = out["projections"]
            m = compare_pca(out, ref)
            if "(auto)" in name:
                m["gate"] = "informational (randomised solver)"
            else:
                tol = TOL_GMX if name == "GROMACS" else TOL_DOUBLE
                ok = (
                    m["max_rel_eigenvalue_error"] <= tol["eig"]
                    and m["max_explained_abs_error_pct"] <= tol["expl"]
                    and m["max_one_minus_cos_separated"] <= tol["vec"]
                )
                if name != "GROMACS":
                    ok = ok and m["max_projection_err_over_sd_separated"] <= tol["proj"]
                m["pass"] = bool(ok)
            if out.get("note"):
                m["note"] = out["note"]
        except Exception as exc:
            m = {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "pass": False}
        m["seconds"] = round(time.perf_counter() - t, 1)
        res["tools"][name] = m
        print(
            f"T1 {selection} {name}",
            {
                k: (f"{v:.2e}" if isinstance(v, float) else v)
                for k, v in m.items()
                if k
                in (
                    "max_rel_eigenvalue_error",
                    "max_one_minus_cos_separated",
                    "max_projection_err_over_sd_separated",
                    "pass",
                    "error",
                    "seconds",
                )
            },
            flush=True,
        )
    return res, ca_proj


# ------------------------------------------------------------------ T2
def sham_range(v, n):
    lo, hi = float(np.min(v)), float(np.max(v))
    d = (hi - lo) / (2.0 * n)
    return lo - d, hi + d


def numpy_fel(x, y, n, xr, yr):
    ix = np.floor((x - xr[0]) / (xr[1] - xr[0]) * n).astype(int)
    iy = np.floor((y - yr[0]) / (yr[1] - yr[0]) * n).astype(int)
    counts = np.zeros((n, n))
    np.add.at(counts, (iy, ix), 1.0)
    p = counts / counts.sum()
    with np.errstate(divide="ignore"):
        g = -KB_SI * TEMPERATURE * np.log(p / p.max())
    g[counts == 0] = np.nan
    return g - np.nanmin(g)


def diff(a, b):
    both = np.isfinite(a) & np.isfinite(b)
    a0, b0 = a - np.nanmin(a), b - np.nanmin(b)
    return {
        "occupied_in_both": int(both.sum()),
        "occupied_in_one_only": int((np.isfinite(a) ^ np.isfinite(b)).sum()),
        "max_abs_diff_kJ_mol": float(np.max(np.abs(a0[both] - b0[both]))),
    }


def t2(proj: np.ndarray) -> dict:
    from conformatlas.fel import calculate_fel

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with open(tmp / "pc12.xvg", "w") as fh:
            fh.write("# PC1 PC2\n")
            for a, b in proj[:, :2]:
                fh.write(f"{a:.9e} {b:.9e}\n")
        xy = np.loadtxt(tmp / "pc12.xvg")
        x, y = xy[:, 0], xy[:, 1]
        for n in (32, 64):
            xr, yr = sham_range(x, n), sham_range(y, n)
            g = calculate_fel(
                x, y, temperature=TEMPERATURE, bins=n, x_range=xr, y_range=yr, smooth_sigma=0.0
            ).free_energy
            dn = diff(g, numpy_fel(x, y, n, xr, yr))
            wd = tmp / f"s{n}"
            wd.mkdir()
            r = subprocess.run(
                [
                    GMX,
                    "sham",
                    "-f",
                    str(tmp / "pc12.xvg"),
                    "-notime",
                    "-tsham",
                    str(TEMPERATURE),
                    "-ngrid",
                    str(n),
                    "-nlevels",
                    "400",
                    "-dist",
                    "ener.xvg",
                ],
                cwd=wd,
                capture_output=True,
                text=True,
                env={"GMX_MAXBACKUP": "-1", "PATH": "/usr/bin:/bin"},
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-400:])
            gs = np.full((n, n), np.nan)
            for ln in (wd / "ener.xvg").read_text().splitlines():
                if ln.strip() and ln[0] not in "#@":
                    f = ln.split()
                    gs[int(f[0]) // n, int(f[0]) % n] = float(f[1])
            ds = diff(g, gs.T)
            out[str(n)] = {
                "numpy": {
                    **dn,
                    "pass": bool(
                        dn["max_abs_diff_kJ_mol"] <= 1e-9 and dn["occupied_in_one_only"] == 0
                    ),
                },
                "gmx_sham": {
                    **ds,
                    "pass": bool(
                        ds["max_abs_diff_kJ_mol"] <= 1e-3 and ds["occupied_in_one_only"] == 0
                    ),
                },
            }
            print("T2", n, out[str(n)], flush=True)
    return out


def main(selections: list[str]) -> None:
    out = {"T0": t0()}
    save_json("t1_t2.json", out)
    if not out["T0"]["pass"]:
        sys.exit("T0 failed")
    for sel in selections:
        res, proj = t1(sel)
        out[f"T1_{sel}"] = res
        if sel == "calpha" and proj is not None:
            np.save(
                Path(__file__).resolve().parent
                / "results"
                / "pooled_ff99SBstar_calpha_projections.npy",
                proj,
            )
            out["T2"] = t2(proj)
        save_json("t1_t2.json", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["calpha", "heavy"])
