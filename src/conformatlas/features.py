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

from conformatlas.errors import InputDataError

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
    # Residues with both phi and psi defined (Ramachandran drops termini). ``ag2``
    # holds the N atoms of those residues; ``ag1`` holds the *preceding* C atoms,
    # so its residues would be shifted by one.
    resids = (
        [res.resid for res in rama.ag2.residues] if hasattr(rama, "ag2") else list(range(n_res))
    )
    resnames = [res.resname for res in rama.ag2.residues] if hasattr(rama, "ag2") else [""] * n_res
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


# =============================================================================
# Extended dihedral sets: side-chain chi1/chi2 and nucleic-acid torsions
# =============================================================================

#: Fourth atom of chi1 (N-CA-CB-X) per residue name.
CHI1_ATOM4: dict[str, str] = {
    **dict.fromkeys(
        ["ARG", "ASN", "ASP", "GLN", "GLU", "HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"], "CG"
    ),
    **dict.fromkeys(["LEU", "LYS", "LYN", "MET", "PHE", "PRO", "TRP", "TYR", "ASH", "GLH"], "CG"),
    "ILE": "CG1",
    "VAL": "CG1",
    "SER": "OG",
    "THR": "OG1",
    "CYS": "SG",
    "CYX": "SG",
    "CYM": "SG",
}
#: (third, fourth) atoms of chi2 (CA-CB-X3-X4) per residue name.
CHI2_ATOMS34: dict[str, tuple[str, str]] = {
    **dict.fromkeys(["ARG", "GLN", "GLU", "LYS", "LYN", "PRO", "GLH"], ("CG", "CD")),
    "ILE": ("CG1", "CD1|CD"),  # CD1 (PDB) or CD (GROMACS/AMBER)
    **dict.fromkeys(["LEU", "PHE", "TRP", "TYR"], ("CG", "CD1")),
    **dict.fromkeys(["ASN", "ASP", "ASH"], ("CG", "OD1")),
    **dict.fromkeys(["HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"], ("CG", "ND1")),
    "MET": ("CG", "SD"),
}
#: Residue names of purine / pyrimidine nucleotides (PDB, AMBER 3'/5' terminal variants).
PURINES = set("A G DA DG RA RG ADE GUA A3 A5 G3 G5 DA3 DA5 DG3 DG5".split())
PYRIMIDINES = set("C U T DC DT DU RC RU CYT URA THY C3 C5 U3 U5 DC3 DC5 DT3 DT5".split())

#: Nucleic-acid torsions as (atom names, residue offsets) -- offset -1/+1 = previous/next residue.
NUCLEIC_BACKBONE: dict[str, tuple[tuple[str, int], ...]] = {
    "alpha": (("O3'", -1), ("P", 0), ("O5'", 0), ("C5'", 0)),
    "beta": (("P", 0), ("O5'", 0), ("C5'", 0), ("C4'", 0)),
    "gamma": (("O5'", 0), ("C5'", 0), ("C4'", 0), ("C3'", 0)),
    "delta": (("C5'", 0), ("C4'", 0), ("C3'", 0), ("O3'", 0)),
    "epsilon": (("C4'", 0), ("C3'", 0), ("O3'", 0), ("P", 1)),
    "zeta": (("C3'", 0), ("O3'", 0), ("P", 1), ("O5'", 1)),
}
PROTEIN_BACKBONE: dict[str, tuple[tuple[str, int], ...]] = {
    "phi": (("C", -1), ("N", 0), ("CA", 0), ("C", 0)),
    "psi": (("N", 0), ("CA", 0), ("C", 0), ("N", 1)),
}
DIHEDRAL_SETS = ("backbone", "backbone+chi", "chi", "nucleic")
#: Order of angle types in the feature matrix.
ANGLE_ORDER = (
    "phi",
    "psi",
    "chi1",
    "chi2",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "chi",
)


def compute_dihedral(p0, p1, p2, p3) -> np.ndarray:
    """Dihedral angle(s) in radians, IUPAC sign convention (as MDAnalysis).

    Accepts single points or ``(n, 3)`` arrays. ``0`` = cis, ``pi`` = trans; the
    angle is positive when, looking along p1->p2, the bond p2->p3 is rotated
    clockwise relative to p1->p0.
    """
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.arctan2(y, x)


def _atom_name_normalised(name: str) -> str:
    return name.replace("*", "'")


def _residue_atom(res, name: str):
    """First atom of ``res`` whose name matches one of the ``|``-separated
    alternatives in ``name`` (``*`` and ``'`` treated alike), or None."""
    options = name.split("|")
    for opt in options:
        for a in res.atoms:
            if _atom_name_normalised(a.name) == opt:
                return a
    return None


def _neighbour(residues, i: int, offset: int):
    """Residue ``i + offset`` if it is a sequence neighbour (same segment, resid +/- 1)."""
    j = i + offset
    if j < 0 or j >= len(residues):
        return None
    r, n = residues[i], residues[j]
    if r.segid != n.segid or int(n.resid) - int(r.resid) != offset:
        return None
    return n


def dihedral_definitions(residues, dihedral_set: str) -> tuple[list[dict], list]:
    """Atom quadruplets for ``dihedral_set`` over an MDAnalysis ResidueGroup.

    Returns ``(labels, atomgroups)``; each label has ``chain, residue_number,
    residue_name, angle``. Residues lacking any of the four atoms (termini,
    GLY/ALA for chi1, ...) are skipped silently.
    """
    if dihedral_set not in DIHEDRAL_SETS:
        raise ValueError(f"dihedral_set must be one of {DIHEDRAL_SETS}, got '{dihedral_set}'.")
    res_list = list(residues)
    labels: list[dict] = []
    groups = []

    def _add(i, res, angle, spec):
        atoms = []
        for name, off in spec:
            target = res if off == 0 else _neighbour(res_list, i, off)
            if target is None:
                return
            a = _residue_atom(target, name)
            if a is None:
                return
            atoms.append(a)
        chain = getattr(res.atoms[0], "chainID", "") or res.segid or "A"
        labels.append(
            {
                "chain": str(chain).strip() or "A",
                "residue_number": int(res.resid),
                "residue_name": str(res.resname),
                "angle": angle,
            }
        )
        groups.append(mda.AtomGroup(atoms))

    if dihedral_set == "nucleic":
        for i, res in enumerate(res_list):
            for angle, spec in NUCLEIC_BACKBONE.items():
                _add(i, res, angle, spec)
            rn = res.resname.strip()
            if rn in PURINES:
                _add(i, res, "chi", (("O4'", 0), ("C1'", 0), ("N9", 0), ("C4", 0)))
            elif rn in PYRIMIDINES:
                _add(i, res, "chi", (("O4'", 0), ("C1'", 0), ("N1", 0), ("C2", 0)))
        return labels, groups

    if dihedral_set in ("backbone", "backbone+chi"):
        # keep only residues with both phi and psi (as MDAnalysis Ramachandran does)
        for i, res in enumerate(res_list):
            before = len(groups)
            _add(i, res, "phi", PROTEIN_BACKBONE["phi"])
            if len(groups) == before:
                continue
            _add(i, res, "psi", PROTEIN_BACKBONE["psi"])
            if len(groups) == before + 1:  # psi missing -> drop phi too
                groups.pop()
                labels.pop()
    if dihedral_set in ("chi", "backbone+chi"):
        for i, res in enumerate(res_list):
            rn = res.resname.strip()
            if rn in CHI1_ATOM4:
                _add(i, res, "chi1", (("N", 0), ("CA", 0), ("CB", 0), (CHI1_ATOM4[rn], 0)))
            if rn in CHI2_ATOMS34:
                a3, a4 = CHI2_ATOMS34[rn]
                _add(i, res, "chi2", (("CA", 0), ("CB", 0), (a3, 0), (a4, 0)))
    return labels, groups


def stream_dihedrals(
    topology_path: str | Path,
    trajectory_path: str | Path,
    selection: str = "protein",
    dihedral_set: str = "backbone",
    stride: int = 1,
    begin_ps: float | None = None,
    end_ps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """cos/sin features of a dihedral set computed with explicit atom-name definitions.

    ``dihedral_set``: ``backbone`` (phi, psi), ``backbone+chi`` (plus chi1, chi2),
    ``chi`` or ``nucleic`` (alpha, beta, gamma, delta, epsilon, zeta, chi). The
    selection must be residue-complete (whole residues; ``"protein"`` or
    ``"nucleic"``). Angles are computed with ``MDAnalysis.analysis.dihedrals.Dihedral``.

    Returns
    -------
    features : (n_frames, 2 * n_dihedrals) float32 -- for each angle type in
        ``ANGLE_ORDER``, the cos block of all its residues then the sin block
        (the layout of ``stream_backbone_dihedrals`` for the backbone set)
    times : (n_frames,) ps
    labels : one row per feature column (chain, residue_number, residue_name, angle, component)
    angles_deg : (n_frames, n_dihedrals) raw dihedrals in degrees, columns in
        ``labels`` order of the cos block
    """
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required for dihedral features.")
    from MDAnalysis.analysis.dihedrals import Dihedral

    u = mda.Universe(str(topology_path), str(trajectory_path))
    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"Selection '{selection}' matched 0 atoms for dihedral analysis.")
    labels, groups = dihedral_definitions(ag.residues, dihedral_set)
    if not groups:
        raise InputDataError(
            f"No '{dihedral_set}' dihedrals could be defined in selection '{selection}' "
            f"({len(ag)} atoms, {len(ag.residues)} residues): the selection lacks the atoms that "
            "define these torsions (N, CA, C for phi/psi; side-chain heavy atoms for chi; "
            "sugar-phosphate backbone and glycosidic atoms for nucleic). A C-alpha-only or "
            "coarse-grained trajectory cannot be used with --pca-type dihedral; use "
            "--pca-type cartesian, or supply a topology/trajectory with the full backbone."
        )
    start, stop, step = _frame_window(u, stride, begin_ps, end_ps)
    dih = Dihedral(groups).run(start=start, stop=stop, step=step)
    angles_deg = np.asarray(dih.results.angles, dtype=np.float64)  # (n_frames, n_dih)
    times = np.array([u.trajectory[i].time for i in range(start, stop, step)], dtype=np.float64)
    lab = pd.DataFrame(labels)
    present = set(lab["angle"].tolist())
    angle_order = [a for a in ANGLE_ORDER if a in present]
    col_order = np.concatenate([np.flatnonzero(lab["angle"].to_numpy() == a) for a in angle_order])
    angles_deg = angles_deg[:, col_order]
    lab = lab.iloc[col_order].reset_index(drop=True)
    rad = np.deg2rad(angles_deg)
    blocks, rows = [], []
    for a in angle_order:
        idx = np.flatnonzero(lab["angle"].to_numpy() == a)
        for comp, fn in (("cos", np.cos), ("sin", np.sin)):
            blocks.append(fn(rad[:, idx]))
            for i in idx:
                r = dict(lab.iloc[i])
                r["component"] = comp
                rows.append(r)
    feats = np.concatenate(blocks, axis=1).astype(np.float32)
    return feats, times, pd.DataFrame(rows), angles_deg
