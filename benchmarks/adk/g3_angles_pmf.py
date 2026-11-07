"""G3 of benchmarks/CRITERIA.md: NMP-CORE / LID-CORE angle coordinates, the published apo PMF
(Beckstein et al. 2009, Zenodo 29617), and ConformAtlas's landscape on those coordinates.

Angles (Cα centres of mass; Beckstein et al. 2009 as restated in PLoS Comput Biol 10:e1003721):
  theta_NMP: angle between (CORE 90-100 -> 115-125) and (CORE 90-100 -> NMP 35-55), vertex at CORE 90-100
  theta_LID: angle between (115-125 -> CORE 179-185) and (115-125 -> LID 125-153), vertex at 115-125
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, KB, TEMPERATURE, save_json, universe  # noqa: E402
from g2_fel import KB_SI, max_diff, numpy_fel  # noqa: E402

PMF = DATA / "adk" / "reference_PMF_apo" / "free_bw2_tol1e-05.dat"
CRYSTAL = DATA / "adk" / "crystal"
KCAL = 4.184
RES = {
    "core_a": (90, 100),
    "hinge": (115, 125),
    "nmp": (35, 55),
    "core_b": (179, 185),
    "lid": (125, 153),
}


def _angle(vertex: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v1, v2 = a - vertex, b - vertex
    cosang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def angles(centre) -> tuple[np.ndarray, np.ndarray]:
    """``centre(lo, hi)`` -> (..., 3) centre of residues lo..hi. Returns (theta_NMP, theta_LID)."""
    c = {k: centre(*v) for k, v in RES.items()}
    theta_nmp = _angle(c["core_a"], c["hinge"], c["nmp"])
    theta_lid = _angle(c["hinge"], c["core_b"], c["lid"])
    return theta_nmp, theta_lid


def trajectory_angles(group: str) -> tuple[np.ndarray, np.ndarray]:
    u = universe()
    sel = {"calpha": "name CA", "backbone": "name N CA C"}[group]
    ag = u.select_atoms(sel)
    resid = ag.resids
    xyz = u.trajectory.timeseries(ag, order="fac").astype(np.float64)

    def centre(lo, hi):
        return xyz[:, (resid >= lo) & (resid <= hi), :].mean(axis=1)

    return angles(centre)


def crystal_angles(pdb: Path, group: str) -> tuple[float, float]:
    rows = []
    for ln in pdb.read_text().splitlines():
        if (
            ln.startswith("ATOM")
            and ln[21] == "A"
            and ln[12:16].strip() in (("CA",) if group == "calpha" else ("N", "CA", "C"))
        ):
            if ln[16] not in (" ", "A"):  # first alternate location only
                continue
            rows.append((int(ln[22:26]), float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    arr = np.array(rows)

    def centre(lo, hi):
        m = (arr[:, 0] >= lo) & (arr[:, 0] <= hi)
        return arr[m, 1:].mean(axis=0)

    return tuple(float(v) for v in angles(centre))


def load_pmf():
    d = np.loadtxt(PMF, comments="#")
    xs, ys = np.unique(d[:, 0]), np.unique(d[:, 1])
    g = np.full((len(ys), len(xs)), np.nan)  # [iy (LID), ix (NMP)]
    for x, y, f in zip(d[:, 0], d[:, 1], d[:, 2]):
        if np.isfinite(f):
            g[np.searchsorted(ys, y), np.searchsorted(xs, x)] = f * KCAL
    return xs, ys, g - np.nanmin(g)


def pmf_at(xs, ys, g, x, y) -> float:
    """Bilinear interpolation; NaN if outside the grid or any neighbouring node is undefined."""
    if not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]):
        return float("nan")
    i = min(np.searchsorted(xs, x, side="right") - 1, len(xs) - 2)
    j = min(np.searchsorted(ys, y, side="right") - 1, len(ys) - 2)
    tx, ty = (x - xs[i]) / (xs[i + 1] - xs[i]), (y - ys[j]) / (ys[j + 1] - ys[j])
    nodes = g[j : j + 2, i : i + 2]
    if not np.all(np.isfinite(nodes)):
        return float("nan")
    return float(
        (1 - ty) * ((1 - tx) * nodes[0, 0] + tx * nodes[0, 1])
        + ty * ((1 - tx) * nodes[1, 0] + tx * nodes[1, 1])
    )


def main() -> None:
    from conformatlas.fel import calculate_fel

    xs, ys, gpmf = load_pmf()
    out: dict = {
        "pmf_grid": {
            "NMP_deg": [float(xs[0]), float(xs[-1])],
            "LID_deg": [float(ys[0]), float(ys[-1])],
            "step_deg": 2.0,
        }
    }

    # ---- G3a: definition check on the crystal structures
    cry = {}
    for name in ("4AKE", "1AKE"):
        for grp in ("calpha", "backbone"):
            cry[f"{name}_{grp}"] = crystal_angles(CRYSTAL / f"{name}.pdb", grp)
    g3a = {}
    for name in ("4AKE", "1AKE"):
        th_n, th_l = cry[f"{name}_calpha"]
        val = pmf_at(xs, ys, gpmf, th_n, th_l)
        g3a[name] = {
            "theta_NMP": th_n,
            "theta_LID": th_l,
            "pmf_kJ_mol": val,
            "backbone_variant_angles": cry[f"{name}_backbone"],
            "pass": bool(np.isfinite(val) and val <= 10.0),
        }
    out["G3a"] = {"points": g3a, "gate_kJ_mol": 10.0, "pass": all(v["pass"] for v in g3a.values())}
    print("G3a", out["G3a"], flush=True)

    # ---- trajectory angles
    t_n, t_l = trajectory_angles("calpha")
    tb_n, tb_l = trajectory_angles("backbone")
    out["trajectory_angles"] = {
        "NMP_deg": {"min": float(t_n.min()), "max": float(t_n.max()), "mean": float(t_n.mean())},
        "LID_deg": {"min": float(t_l.min()), "max": float(t_l.max()), "mean": float(t_l.mean())},
        "backbone_variant_max_abs_diff_deg": [
            float(np.max(np.abs(t_n - tb_n))),
            float(np.max(np.abs(t_l - tb_l))),
        ],
        "first_frame": [float(t_n[0]), float(t_l[0])],
    }
    print("trajectory angles", out["trajectory_angles"], flush=True)

    # informational: is the trajectory stationary? mean angles in 10 consecutive time blocks
    blocks = np.array_split(np.arange(len(t_n)), 10)
    out["trajectory_drift_informational"] = {
        "block_mean_NMP_deg": [round(float(t_n[b].mean()), 1) for b in blocks],
        "block_mean_LID_deg": [round(float(t_l[b].mean()), 1) for b in blocks],
        "block_ns": round(len(blocks[0]) * 0.24, 1),
    }
    print("drift", out["trajectory_drift_informational"], flush=True)
    # informational: local minima of the published PMF (finite nodes, 8-neighbourhood)
    mins = []
    for jy in range(gpmf.shape[0]):
        for jx in range(gpmf.shape[1]):
            v = gpmf[jy, jx]
            if not np.isfinite(v):
                continue
            nb = gpmf[max(jy - 1, 0) : jy + 2, max(jx - 1, 0) : jx + 2]
            if v <= np.nanmin(nb):
                mins.append((float(xs[jx]), float(ys[jy]), round(float(v), 1)))
    out["pmf_local_minima_informational"] = {
        "minima_(NMP_deg,LID_deg,kJ_mol)": sorted(mins, key=lambda m: m[2])[:8]
    }
    print("PMF minima", out["pmf_local_minima_informational"], flush=True)

    # ---- G3b: ConformAtlas vs NumPy on the PMF's 2-degree grid; G3c: vs the published PMF (informational)
    x_range, y_range = (
        (xs[0] - 1.0, xs[0] - 1.0 + 64.0),
        (ys[0] - 1.0, ys[0] - 1.0 + 64.0),
    )  # 32 bins x 2 deg
    inside = (t_n >= x_range[0]) & (t_n < x_range[1]) & (t_l >= y_range[0]) & (t_l < y_range[1])
    n = 32
    fel = calculate_fel(
        t_n[inside],
        t_l[inside],
        temperature=TEMPERATURE,
        bins=n,
        x_range=x_range,
        y_range=y_range,
        smooth_sigma=0.0,
    )
    g_ca = fel.free_energy
    d_np = max_diff(g_ca, numpy_fel(t_n[inside], t_l[inside], n, x_range, y_range))
    d_si = max_diff(g_ca, numpy_fel(t_n[inside], t_l[inside], n, x_range, y_range, kb=KB_SI))
    out["G3b"] = {
        "frames_inside_grid": int(inside.sum()),
        "frames_total": int(len(t_n)),
        "registered_kB": {
            **d_np,
            "gate_kJ_mol": 1e-9,
            "pass": bool(d_np["max_abs_diff_kJ_mol"] <= 1e-9 and d_np["occupied_in_one_only"] == 0),
        },
        "exact_SI_kB_diagnosis_only": d_si,
    }
    print("G3b", out["G3b"], flush=True)

    # G3c: align ConformAtlas bins (edges at 34,36,... / 94,96,...) with PMF nodes (centres 35,37,... / 95,97,...)
    ca_x = 0.5 * (fel.x_edges[:-1] + fel.x_edges[1:])
    ca_y = 0.5 * (fel.y_edges[:-1] + fel.y_edges[1:])
    pairs = []
    for iy, yc in enumerate(ca_y):
        for ix, xc in enumerate(ca_x):
            if np.isfinite(g_ca[iy, ix]):
                jx, jy = np.argmin(np.abs(xs - xc)), np.argmin(np.abs(ys - yc))
                if (
                    abs(xs[jx] - xc) < 1e-6
                    and abs(ys[jy] - yc) < 1e-6
                    and np.isfinite(gpmf[jy, jx])
                ):
                    pairs.append((xc, yc, g_ca[iy, ix], gpmf[jy, jx]))
    pairs = np.array(pairs)
    rho = spearmanr(pairs[:, 2], pairs[:, 3]) if len(pairs) > 3 else None
    imin_t = np.nanargmin(g_ca)
    iy0, ix0 = np.unravel_index(imin_t, g_ca.shape)
    jy, jx = np.unravel_index(np.nanargmin(gpmf), gpmf.shape)
    out["G3c_informational"] = {
        "bins_occupied_in_trajectory": int(np.isfinite(g_ca).sum()),
        "bins_compared_(occupied_and_finite_in_PMF)": int(len(pairs)),
        "spearman_rho": float(rho.statistic) if rho else None,
        "spearman_p": float(rho.pvalue) if rho else None,
        "trajectory_minimum_deg_(NMP,LID)": [float(ca_x[ix0]), float(ca_y[iy0])],
        "published_minimum_deg_(NMP,LID)": [float(xs[jx]), float(ys[jy])],
        "minimum_separation_deg": float(np.hypot(ca_x[ix0] - xs[jx], ca_y[iy0] - ys[jy])),
        "note": "published PMF: implicit-solvent umbrella sampling (2009); trajectory: explicit-water CHARMM27; not expected to agree quantitatively",
    }
    print("G3c", out["G3c_informational"], flush=True)
    out["kB_registered"] = KB
    save_json("g3_angles_pmf.json", out)


if __name__ == "__main__":
    main()
