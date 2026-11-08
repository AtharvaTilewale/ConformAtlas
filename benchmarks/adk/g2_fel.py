"""G2 of benchmarks/CRITERIA.md: free-energy landscape on Cα PC1-PC2, ConformAtlas vs an independent NumPy
histogram Boltzmann inversion and vs ``gmx sham`` on identical projections and bin edges.

The comparison recipe (identical CV values through a text round-trip, sham's bin convention
[min - d, max + d] with d = range / (2 n), ``-dist`` output) follows gmx_sham.cpp of GROMACS 2026.3.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KB, TEMPERATURE, coordinates, save_json  # noqa: E402

GMX = os.environ.get("GMX") or shutil.which("gmx") or "gmx"  # $GMX or gmx on PATH
GATE_NUMPY = 1e-9
GATE_SHAM = 1e-3
GRIDS = (32, 64)


def sham_range(v: np.ndarray, n: int) -> tuple[float, float]:
    lo, hi = float(np.min(v)), float(np.max(v))
    d = (hi - lo) / (2.0 * n)
    return lo - d, hi + d


KB_SI = 1.380649e-23 * 6.02214076e23 / 1000.0  # exact SI value, kJ mol^-1 K^-1


def numpy_fel(x, y, n, xr, yr, temperature=TEMPERATURE, kb=KB) -> np.ndarray:
    """Independent Boltzmann inversion, layout [iy, ix]; unoccupied bins NaN."""
    ix = np.floor((x - xr[0]) / (xr[1] - xr[0]) * n).astype(int)
    iy = np.floor((y - yr[0]) / (yr[1] - yr[0]) * n).astype(int)
    counts = np.zeros((n, n))
    np.add.at(counts, (iy, ix), 1.0)
    p = counts / counts.sum()
    with np.errstate(divide="ignore"):
        g = -kb * temperature * np.log(p / p.max())
    g[counts == 0] = np.nan
    return g - np.nanmin(g)


def read_sham(path: Path, n: int) -> np.ndarray:
    g = np.full((n, n), np.nan)  # [ix, iy]
    for ln in path.read_text().splitlines():
        if ln.strip() and ln[0] not in "#@":
            f = ln.split()
            i = int(f[0])
            g[i // n, i % n] = float(f[1])
    return g.T


def max_diff(a: np.ndarray, b: np.ndarray) -> dict:
    both = np.isfinite(a) & np.isfinite(b)
    a0, b0 = a - np.nanmin(a), b - np.nanmin(b)
    return {
        "occupied_in_both": int(both.sum()),
        "occupied_in_one_only": int((np.isfinite(a) ^ np.isfinite(b)).sum()),
        "max_abs_diff_kJ_mol": float(np.max(np.abs(a0[both] - b0[both])))
        if both.any()
        else float("nan"),
    }


def main() -> None:
    from conformatlas.fel import calculate_fel
    from conformatlas.pca import compute_cartesian_pca, project_coordinates

    coords = coordinates("calpha")
    _, vec, _, _, mean = compute_cartesian_pca({"r": coords}, weighting="frames", n_components=2)
    proj = project_coordinates(coords, mean, vec)
    out: dict = {"n_frames": int(len(proj)), "temperature_K": TEMPERATURE, "kB": KB, "grids": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with open(
            tmp / "pc12.xvg", "w", encoding="utf-8"
        ) as fh:  # identical CV values for every tool
            fh.write("# PC1 PC2 (no time column)\n")
            for a, b in proj[:, :2]:
                fh.write(f"{a:.9e} {b:.9e}\n")
        xy = np.loadtxt(tmp / "pc12.xvg")
        x, y = xy[:, 0], xy[:, 1]
        for n in GRIDS:
            xr, yr = sham_range(x, n), sham_range(y, n)
            fel = calculate_fel(
                x, y, temperature=TEMPERATURE, bins=n, x_range=xr, y_range=yr, smooth_sigma=0.0
            )
            g_ca = fel.free_energy
            d_np = max_diff(g_ca, numpy_fel(x, y, n, xr, yr))  # registered constant (criteria file)
            d_np_si = max_diff(
                g_ca, numpy_fel(x, y, n, xr, yr, kb=KB_SI)
            )  # diagnosis only, see DEVIATIONS D4
            wd = tmp / f"sham{n}"
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
                    "-bin",
                    "bindex.ndx",
                    "-g",
                    "shamlog.log",
                ],
                cwd=wd,
                capture_output=True,
                text=True,
                env={"GMX_MAXBACKUP": "-1", "PATH": "/usr/bin:/bin"},
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-500:])
            d_sh = max_diff(g_ca, read_sham(wd / "ener.xvg", n))
            out["grids"][str(n)] = {
                "numpy": {
                    **d_np,
                    "gate_kJ_mol": GATE_NUMPY,
                    "pass": bool(
                        d_np["max_abs_diff_kJ_mol"] <= GATE_NUMPY
                        and d_np["occupied_in_one_only"] == 0
                    ),
                },
                "numpy_with_exact_SI_kB_diagnosis_only": {**d_np_si, "kB_SI": KB_SI},
                "gmx_sham": {
                    **d_sh,
                    "gate_kJ_mol": GATE_SHAM,
                    "pass": bool(
                        d_sh["max_abs_diff_kJ_mol"] <= GATE_SHAM
                        and d_sh["occupied_in_one_only"] == 0
                    ),
                },
            }
            print(n, out["grids"][str(n)], flush=True)
    save_json("g2_fel.json", out)


if __name__ == "__main__":
    main()
