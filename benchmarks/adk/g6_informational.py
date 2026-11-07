"""Informational items of benchmarks/CRITERIA.md (no gates):
(a) projection of the 1AKE and 4AKE crystal structures into the trajectory's Cα PC space,
(b) the default ``conformatlas analyze`` run: states, populations, barriers and sampling diagnostics.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, DCD, PSF, coordinates, save_json, universe  # noqa: E402

CORE = [(1, 29), (60, 121), (160, 214)]


def crystal_ca(pdb: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for ln in pdb.read_text().splitlines():
        if (
            ln.startswith("ATOM")
            and ln[21] == "A"
            and ln[12:16].strip() == "CA"
            and ln[16] in (" ", "A")
        ):
            rows.append((int(ln[22:26]), float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    a = np.array(rows)
    return a[:, 0].astype(int), a[:, 1:]


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation r and translations so that (p - pc) @ r ~ (q - qc)."""
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return r, pc, qc


def main() -> None:
    from conformatlas.pca import compute_cartesian_pca, project_coordinates

    out: dict = {}
    coords = coordinates("calpha")
    ev, vec, expl, _, mean = compute_cartesian_pca(
        {"r": coords}, weighting="frames", n_components=10
    )
    proj = project_coordinates(coords, mean, vec)
    resid = universe().select_atoms("name CA").resids
    core_mask = np.zeros(len(resid), bool)
    for lo, hi in CORE:
        core_mask |= (resid >= lo) & (resid <= hi)
    mean_xyz = mean.reshape(-1, 3)
    out["trajectory_PC_ranges"] = {
        f"PC{i + 1}": [float(proj[:, i].min()), float(proj[:, i].max()), float(proj[:, i].std())]
        for i in range(3)
    }
    out["explained_variance_pct_top5"] = [float(x) for x in expl[:5]]
    cry = {}
    for name in ("4AKE", "1AKE"):
        r_id, xyz = crystal_ca(DATA / "adk" / "crystal" / f"{name}.pdb")
        idx = {int(r): i for i, r in enumerate(r_id)}
        sel_traj = [i for i, r in enumerate(resid) if core_mask[i] and int(r) in idx]
        sel_cry = [idx[int(resid[i])] for i in sel_traj]
        rot, pc, qc = kabsch(xyz[sel_cry], mean_xyz[sel_traj])
        fitted = (xyz - pc) @ rot + qc
        have = [i for i, r in enumerate(resid) if int(r) in idx]
        flat = np.zeros_like(mean_xyz)
        for i in have:
            flat[i] = fitted[idx[int(resid[i])]]
        p = project_coordinates(flat[None], mean, vec)[0]
        pct = [float((proj[:, i] < p[i]).mean() * 100) for i in range(3)]
        cry[name] = {
            "n_residues_in_common": len(have),
            "PC1_PC2_PC3": [float(x) for x in p[:3]],
            "percentile_in_trajectory": pct,
            "rmsd_to_mean_structure_A": float(
                np.sqrt(np.mean(np.sum((flat[have] - mean_xyz[have]) ** 2, axis=1)))
            ),
        }
    out["crystal_projection"] = cry
    first = proj[0, :3]
    out["trajectory_first_frame_PC1_PC2_PC3"] = [float(x) for x in first]
    print(json.dumps(out, indent=1))

    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "analyze"
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
            str(res),
        ]
        r = subprocess.run(
            cmd, capture_output=True, text=True, env={**os.environ, "CONFORMATLAS_NO_PROGRESS": "1"}
        )
        out["analyze_returncode"] = r.returncode
        if r.returncode == 0:
            for rel in (
                "states/state_ensemble_summary.csv",
                "states/barriers.csv",
                "states/state_free_energies.csv",
            ):
                f = res / rel
                if f.exists():
                    out[f"analyze:{rel}"] = pd.read_csv(f).round(4).to_dict(orient="records")
            meta = json.loads((res / "run_metadata.json").read_text())
            out["analyze_warnings"] = meta.get("warnings", [])
        else:
            out["analyze_error"] = (r.stderr or r.stdout)[-500:]
    save_json("g6_informational.json", out)


if __name__ == "__main__":
    main()
