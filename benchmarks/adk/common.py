"""Shared helpers for the ADK benchmark (see benchmarks/CRITERIA.md).

Data location is taken from the environment variable ``CA_BENCH_DATA`` (default: the directory used when
the benchmark was run). Trajectories are never stored in the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

DATA = Path(os.environ.get("CA_BENCH_DATA", "/media/atharva/3EA6170E34C679E4/ca_benchmark_data"))
ADK_EQ = DATA / "adk" / "equilibrium_1us"
PSF = ADK_EQ / "adk4AKE.psf"
DCD = ADK_EQ / "1ake_007-nowater-core-dt240ps.dcd"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

KB = 0.0083144626  # kJ mol^-1 K^-1
TEMPERATURE = 300.0
SELECTIONS = {"calpha": "name CA", "backbone": "name N CA C", "heavy": "not name H*"}
N_MODES = 10


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def universe():
    import MDAnalysis as mda

    return mda.Universe(str(PSF), str(DCD))


def coordinates(selection: str) -> np.ndarray:
    """(n_frames, n_atoms, 3) float64 copy of the float32 trajectory coordinates (Å)."""
    u = universe()
    ag = u.select_atoms(SELECTIONS.get(selection, selection))
    return u.trajectory.timeseries(ag, order="fac").astype(np.float64)


def reference_pca(coords: np.ndarray, n_modes: int = N_MODES + 2) -> dict:
    """Independent NumPy covariance PCA: explicit float64 covariance, N-normalised, eigh."""
    n = coords.shape[0]
    x = coords.reshape(n, -1)
    mean = x.mean(axis=0)
    xc = x - mean
    cov = xc.T @ xc / n
    w, v = np.linalg.eigh(cov)
    w, v = w[::-1], v[:, ::-1]
    return {
        "eigenvalues": w[:n_modes],
        "eigenvectors": v[:, :n_modes],
        "trace": float(np.trace(cov)),
        "mean": mean,
        "projections": xc @ v[:, :n_modes],
    }


def well_separated(
    eigenvalues: np.ndarray, n_modes: int = N_MODES, gap: float = 0.01
) -> np.ndarray:
    """Boolean mask over modes 1..n_modes whose relative gap to both neighbours is >= gap."""
    w = np.asarray(eigenvalues, dtype=float)
    ok = np.zeros(n_modes, dtype=bool)
    for i in range(n_modes):
        up = abs(w[i] - w[i - 1]) / w[i] if i > 0 else np.inf
        down = abs(w[i] - w[i + 1]) / w[i]
        ok[i] = min(up, down) >= gap
    return ok


def compare_pca(tool: dict, ref: dict, n_modes: int = N_MODES) -> dict:
    """Metrics of the G1 gate table. ``tool`` holds eigenvalues, eigenvectors (D, k), explained (%),
    projections (N, k), already converted to Å units and N-normalisation."""
    ev_ref = ref["eigenvalues"][:n_modes]
    rel_ev = np.abs(tool["eigenvalues"][:n_modes] - ev_ref) / ev_ref
    expl_ref = 100.0 * ev_ref / ref["trace"]
    d_expl = np.abs(np.asarray(tool["explained"][:n_modes]) - expl_ref)
    sep = well_separated(ref["eigenvalues"], n_modes)
    dots = np.sum(tool["eigenvectors"][:, :n_modes] * ref["eigenvectors"][:, :n_modes], axis=0)
    one_minus_cos = 1.0 - np.minimum(np.abs(dots), 1.0)
    sign = np.where(dots < 0, -1.0, 1.0)
    proj_err = np.full(n_modes, np.nan)
    if tool.get("projections") is not None:
        p, pr = tool["projections"][:, :n_modes] * sign, ref["projections"][:, :n_modes]
        proj_err = np.max(np.abs(p - pr), axis=0) / pr.std(axis=0)
    # subspace distance of the non-separated modes (sine of largest principal angle)
    ns = ~sep
    sub = float("nan")
    if ns.any():
        a = np.linalg.qr(tool["eigenvectors"][:, :n_modes][:, ns])[0]
        b = np.linalg.qr(ref["eigenvectors"][:, :n_modes][:, ns])[0]
        s = np.linalg.svd(a.T @ b, compute_uv=False)
        sub = float(np.sqrt(max(0.0, 1.0 - s.min() ** 2)))
    return {
        "max_rel_eigenvalue_error": float(rel_ev.max()),
        "max_explained_abs_error_pct": float(d_expl.max()),
        "max_one_minus_cos_separated": float(one_minus_cos[sep].max())
        if sep.any()
        else float("nan"),
        "max_projection_err_over_sd_separated": float(np.nanmax(proj_err[sep]))
        if sep.any() and not np.all(np.isnan(proj_err[sep]))
        else float("nan"),
        "n_separated_modes": int(sep.sum()),
        "separated_modes": [int(i + 1) for i in np.flatnonzero(sep)],
        "subspace_sine_nonseparated": sub,
    }


def save_json(name: str, obj) -> Path:
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, default=float))
    return path
