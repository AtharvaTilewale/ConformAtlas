"""Alternative collective variables: backbone dihedral features and external CV files.

Dihedral PCA (dPCA; Mu, Nguyen & Stock, Proteins 2005) replaces Cartesian
coordinates by (cos phi, sin phi, cos psi, sin psi) of every residue, which removes
the periodicity problem and needs no superposition. The resulting feature matrix
is used exactly like a coordinate matrix by the PCA routines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("conformatlas.features")

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Ramachandran

    HAS_MDANALYSIS = True
except ImportError:  # pragma: no cover
    HAS_MDANALYSIS = False


def _frame_window(u, stride: int, begin_ps: float | None, end_ps: float | None):
    """Return (start, stop, step) frame indices for a time window."""
    times = (
        np.array([ts.time for ts in u.trajectory[::1]])
        if (begin_ps is not None or end_ps is not None)
        else None
    )
    start, stop = 0, len(u.trajectory)
    if times is not None:
        if begin_ps is not None:
            start = int(np.searchsorted(times, begin_ps, side="left"))
        if end_ps is not None:
            stop = int(np.searchsorted(times, end_ps, side="right"))
    return start, stop, max(1, stride)


def stream_backbone_dihedrals(
    topology_path: str | Path,
    trajectory_path: str | Path,
    selection: str = "protein",
    stride: int = 1,
    begin_ps: float | None = None,
    end_ps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Backbone phi/psi features for every frame.

    Returns
    -------
    features : (n_frames, 4 * n_residues) array of cos/sin phi/psi
    times : (n_frames,) ps
    labels : DataFrame describing each feature column (residue, angle, component)
    """
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required for dihedral features.")
    u = mda.Universe(str(topology_path), str(trajectory_path))
    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"Selection '{selection}' matched 0 atoms for dihedral analysis.")
    start, stop, step = _frame_window(u, stride, begin_ps, end_ps)
    rama = Ramachandran(ag).run(start=start, stop=stop, step=step)
    angles = np.deg2rad(rama.results.angles)  # (n_frames, n_res, 2) phi, psi
    n_frames, n_res, _ = angles.shape
    feats = np.concatenate(
        [
            np.cos(angles[:, :, 0]),
            np.sin(angles[:, :, 0]),
            np.cos(angles[:, :, 1]),
            np.sin(angles[:, :, 1]),
        ],
        axis=1,
    ).astype(np.float32)
    times = np.array([u.trajectory[i].time for i in range(start, stop, step)], dtype=np.float64)
    # Residues with both phi and psi defined (Ramachandran drops termini).
    resids = (
        [res.resid for res in rama.ag1.residues] if hasattr(rama, "ag1") else list(range(n_res))
    )
    resnames = [res.resname for res in rama.ag1.residues] if hasattr(rama, "ag1") else [""] * n_res
    rows = []
    for comp, angle in (("cos", "phi"), ("sin", "phi"), ("cos", "psi"), ("sin", "psi")):
        for r_id, r_name in zip(resids, resnames):
            rows.append(
                {"residue_number": r_id, "residue_name": r_name, "angle": angle, "component": comp}
            )
    labels = pd.DataFrame(rows)
    if len(labels) != feats.shape[1]:
        labels = pd.DataFrame({"feature": np.arange(feats.shape[1])})
    return feats, times, labels


def read_cv_file(
    path: str | Path,
    columns: tuple[int, int] = (1, 2),
    time_column: int | None = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read two collective variables from an .xvg/.dat/.csv/.txt file.

    Comment lines starting with ``#`` or ``@`` (GROMACS xvg) are skipped. Column
    indices are 0-based; the default takes time from column 0 and the CVs from
    columns 1 and 2. Returns (cv array of shape (n, 2), time array or None).
    """
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in "#@;":
                continue
            s = s.replace(",", " ")
            try:
                rows.append([float(tok) for tok in s.split()])
            except ValueError:
                continue  # header line
    if not rows:
        raise ValueError(f"No numeric data found in CV file {path}")
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=np.float64)
    c1, c2 = columns
    if max(c1, c2) >= arr.shape[1]:
        raise ValueError(f"CV file {path} has {arr.shape[1]} columns; requested {columns}.")
    cv = arr[:, [c1, c2]]
    t = arr[:, time_column] if time_column is not None and time_column < arr.shape[1] else None
    return cv, t


def read_weights_file(path: str | Path, column: int | None = None) -> np.ndarray:
    """Read per-frame statistical weights (e.g. exp(beta V_bias) from a reweighting
    scheme). Uses the last column by default; comment lines are skipped."""
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in "#@;":
                continue
            try:
                rows.append([float(tok) for tok in s.replace(",", " ").split()])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric data in weights file {path}")
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=np.float64)
    col = (arr.shape[1] - 1) if column is None else column
    w = arr[:, col]
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("Weights must be finite and non-negative.")
    return w
