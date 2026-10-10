"""Interpretation of principal modes: atom/residue contributions, NMD export,
extreme-projection structures and the average structure.

Only meaningful for Cartesian modes (eigenvectors of length 3 x n_atoms). For
dihedral PCA the contribution table is per dihedral feature instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("conformatlas.modes")

try:
    import MDAnalysis as mda

    HAS_MDANALYSIS = True
except ImportError:  # pragma: no cover
    HAS_MDANALYSIS = False


def mode_atom_contributions(
    eigenvectors: np.ndarray, atom_info: pd.DataFrame, n_modes: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-atom and per-residue contribution of each of the first ``n_modes`` modes.

    The contribution of atom a to mode j is the norm of the three Cartesian
    components of eigenvector j belonging to that atom; the squares sum to 1 over
    atoms. The per-residue value is the square root of the summed squares of its
    atoms, so residue contributions also sum (in squares) to 1.
    """
    n_atoms = len(atom_info)
    if eigenvectors.shape[0] != 3 * n_atoms:
        raise ValueError(
            f"Eigenvectors have {eigenvectors.shape[0]} rows, expected 3 x {n_atoms} atoms."
        )
    k = min(n_modes, eigenvectors.shape[1])
    atom_df = atom_info.copy().reset_index(drop=True)
    for j in range(k):
        v = eigenvectors[:, j].reshape(n_atoms, 3)
        atom_df[f"PC{j + 1}"] = np.linalg.norm(v, axis=1)
    cols = [f"PC{j + 1}" for j in range(k)]
    sq = atom_df[cols] ** 2
    sq[["chain", "residue_number", "residue_name"]] = atom_df[
        ["chain", "residue_number", "residue_name"]
    ]
    res_df = sq.groupby(["chain", "residue_number", "residue_name"], sort=False)[cols].sum()
    res_df = np.sqrt(res_df).reset_index()
    return atom_df, res_df


def write_nmd(
    path: str | Path,
    atom_info: pd.DataFrame,
    mean_coords: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    n_modes: int = 10,
    title: str = "ConformAtlas PCA",
) -> Path:
    """Write modes in NMWiz (.nmd) format for VMD / ProDy visualisation.

    Mode vectors are scaled by sqrt(eigenvalue) so that arrow lengths reflect the
    amplitude of each mode (same convention as ProDy's writeNMD for PCA).
    """
    path = Path(path)
    n_atoms = len(atom_info)
    k = min(n_modes, eigenvectors.shape[1])
    mean = np.asarray(mean_coords, dtype=np.float64).reshape(n_atoms, 3)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("nmwiz_load " + path.name + "\n")
        fh.write(f"name {title}\n")
        fh.write("atomnames " + " ".join(str(a) for a in atom_info["atom_name"]) + "\n")
        fh.write("resnames " + " ".join(str(r) for r in atom_info["residue_name"]) + "\n")
        fh.write("resids " + " ".join(str(int(r)) for r in atom_info["residue_number"]) + "\n")
        fh.write("chainids " + " ".join(str(c)[:1] or "A" for c in atom_info["chain"]) + "\n")
        fh.write("coordinates " + " ".join(f"{x:.3f}" for x in mean.ravel()) + "\n")
        for j in range(k):
            scale = float(np.sqrt(max(eigenvalues[j], 0.0)))
            vec = eigenvectors[:, j]
            fh.write(f"mode {j + 1} {scale:.4f} " + " ".join(f"{x:.5f}" for x in vec) + "\n")
    return path


def _atomgroup(
    topology: str | Path, atom_indices: np.ndarray, trajectory: str | Path | None = None
):
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required to write structures.")
    u = (
        mda.Universe(str(topology))
        if trajectory is None
        else mda.Universe(str(topology), str(trajectory))
    )
    return u, u.atoms[np.asarray(atom_indices, dtype=int)]


def write_structure(
    path: str | Path,
    topology: str | Path,
    atom_indices: np.ndarray,
    coords: np.ndarray,
    trajectory: str | Path | None = None,
) -> Path:
    """Write one set of coordinates for the selected atoms as PDB."""
    path = Path(path)
    _, ag = _atomgroup(topology, atom_indices, trajectory)
    ag.positions = np.asarray(coords, dtype=np.float32).reshape(len(ag), 3)
    ag.write(str(path))
    return path


def write_extreme_structures(
    path: str | Path,
    topology: str | Path,
    atom_indices: np.ndarray,
    mean_coords: np.ndarray,
    eigenvector: np.ndarray,
    proj_min: float,
    proj_max: float,
    n_frames: int = 11,
    trajectory: str | Path | None = None,
) -> Path:
    """Multi-model PDB interpolating along one mode from its minimum to maximum
    projection (the equivalent of ``gmx anaeig -extr -nframes``)."""
    path = Path(path)
    u, ag = _atomgroup(topology, atom_indices, trajectory)
    n = len(ag)
    mean = np.asarray(mean_coords, dtype=np.float64).reshape(n, 3)
    vec = np.asarray(eigenvector, dtype=np.float64).reshape(n, 3)
    with mda.Writer(str(path), n_atoms=n, multiframe=True) as w:
        for s in np.linspace(proj_min, proj_max, n_frames):
            ag.positions = (mean + s * vec).astype(np.float32)
            w.write(ag)
    return path


def save_mode_interpretation(
    output_dir: str | Path,
    topology: str | Path,
    atom_indices: np.ndarray,
    atom_info: pd.DataFrame,
    mean_coords: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    projections: pd.DataFrame,
    n_modes: int = 3,
    trajectory: str | Path | None = None,
) -> dict[str, Path]:
    """Write contributions, NMD, average and extreme structures into pca/."""
    pca_dir = Path(output_dir) / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    atom_df, res_df = mode_atom_contributions(eigenvectors, atom_info, n_modes=n_modes)
    atom_df.to_csv(pca_dir / "mode_contributions_atoms.csv", index=False)
    res_df.to_csv(pca_dir / "mode_contributions_residues.csv", index=False)
    out["contributions_atoms"] = pca_dir / "mode_contributions_atoms.csv"
    out["contributions_residues"] = pca_dir / "mode_contributions_residues.csv"
    np.save(pca_dir / "eigenvectors.npy", eigenvectors)
    np.save(pca_dir / "mean_coordinates.npy", mean_coords)

    if HAS_MDANALYSIS:
        try:
            out["nmd"] = write_nmd(
                pca_dir / "modes.nmd", atom_info, mean_coords, eigenvectors, eigenvalues
            )
            out["average_pdb"] = write_structure(
                pca_dir / "average.pdb", topology, atom_indices, mean_coords, trajectory
            )
            for j in range(min(n_modes, eigenvectors.shape[1])):
                col = f"PC{j + 1}"
                if col not in projections.columns:
                    continue
                pmin, pmax = float(projections[col].min()), float(projections[col].max())
                out[f"extremes_{col}"] = write_extreme_structures(
                    pca_dir / f"{col}_extremes.pdb",
                    topology,
                    atom_indices,
                    mean_coords,
                    eigenvectors[:, j],
                    pmin,
                    pmax,
                    trajectory=trajectory,
                )
        except Exception as e:  # structure writing must never abort the analysis
            logger.warning(f"Could not write mode structures: {e}")
    return out
