"""Classical ensemble descriptors from in-memory (fitted) coordinates.

RMSD to the reference, radius of gyration, per-atom and per-residue RMSF, computed
on the analysis-group coordinates after superposition. An optional ligand subset
(atom positions inside the analysis selection) receives its own RMSD, i.e. ligand
displacement in the protein frame after fitting on the protein.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def rmsd_series(
    coords: np.ndarray, reference: np.ndarray, subset: np.ndarray | None = None
) -> np.ndarray:
    """Per-frame RMSD (no re-fitting) of ``coords`` (n_frames, n_atoms, 3) to ``reference``."""
    c = coords if subset is None else coords[:, subset, :]
    r = reference if subset is None else reference[subset]
    d = c.astype(np.float64) - r[None, :, :].astype(np.float64)
    return np.sqrt(np.mean(np.sum(d * d, axis=2), axis=1))


def radius_of_gyration(
    coords: np.ndarray, subset: np.ndarray | None = None, masses: np.ndarray | None = None
) -> np.ndarray:
    """Per-frame radius of gyration (mass-weighted when masses are given)."""
    c = coords if subset is None else coords[:, subset, :]
    c = c.astype(np.float64)
    if masses is not None:
        m = np.asarray(masses, dtype=np.float64)
        if subset is not None:
            m = m[subset]
        w = m / m.sum()
    else:
        w = np.full(c.shape[1], 1.0 / c.shape[1])
    com = np.einsum("fai,a->fi", c, w)
    d = c - com[:, None, :]
    return np.sqrt(np.einsum("fai,fai,a->f", d, d, w))


def rmsf_per_atom(coords: np.ndarray) -> np.ndarray:
    """Per-atom RMSF about the mean structure over all frames."""
    c = coords.astype(np.float64)
    mean = c.mean(axis=0)
    d = c - mean[None]
    return np.sqrt(np.mean(np.sum(d * d, axis=2), axis=0))


def rmsf_per_residue(atom_rmsf: np.ndarray, atom_info: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-atom RMSF to residues as sqrt(mean of squares)."""
    df = atom_info.copy()
    df["rmsf2"] = atom_rmsf**2
    grouped = (
        df.groupby(["chain", "residue_number", "residue_name"], sort=False)["rmsf2"]
        .mean()
        .reset_index()
    )
    grouped["RMSF"] = np.sqrt(grouped.pop("rmsf2"))
    return grouped


def compute_descriptors(
    coords_dict: dict[str, np.ndarray],
    times_dict: dict[str, np.ndarray],
    reference: np.ndarray,
    atom_info: pd.DataFrame,
    ligand_mask: np.ndarray | None = None,
    masses: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """RMSD(t), Rg(t), RMSF for every replicate.

    Parameters
    ----------
    coords_dict : replicate -> (n_frames, n_atoms, 3) fitted analysis coordinates
    times_dict : replicate -> times (ps)
    reference : (n_atoms, 3) reference coordinates of the analysis atoms
    atom_info : DataFrame with atom_index, chain, residue_number, residue_name, atom_name
    ligand_mask : boolean mask over analysis atoms marking ligand atoms (optional)
    masses : per-atom masses for Rg (optional)
    """
    protein_mask = None if ligand_mask is None else ~ligand_mask
    ts_rows = []
    for rep, coords in coords_dict.items():
        t = times_dict[rep]
        rmsd_all = rmsd_series(
            coords, reference, protein_mask if protein_mask is not None else None
        )
        rg = radius_of_gyration(coords, protein_mask, masses)
        df = pd.DataFrame({"replicate": rep, "time_ps": t, "RMSD": rmsd_all, "Rg": rg})
        if ligand_mask is not None and ligand_mask.any():
            df["Ligand_RMSD"] = rmsd_series(coords, reference, np.where(ligand_mask)[0])
        ts_rows.append(df)
    timeseries = pd.concat(ts_rows, ignore_index=True)

    all_coords = np.concatenate(list(coords_dict.values()), axis=0)
    atom_rmsf = rmsf_per_atom(all_coords)
    atom_df = atom_info.copy().reset_index(drop=True)
    atom_df["RMSF"] = atom_rmsf
    if ligand_mask is not None:
        atom_df["is_ligand"] = ligand_mask
    per_rep_rmsf = {rep: rmsf_per_atom(c) for rep, c in coords_dict.items()}
    for rep, vals in per_rep_rmsf.items():
        atom_df[f"RMSF_{rep}"] = vals
    residue_df = rmsf_per_residue(atom_rmsf, atom_info.reset_index(drop=True))

    summary = (
        timeseries.groupby("replicate", sort=False)
        .agg(
            frames=("RMSD", "size"),
            RMSD_mean=("RMSD", "mean"),
            RMSD_sd=("RMSD", "std"),
            RMSD_max=("RMSD", "max"),
            Rg_mean=("Rg", "mean"),
            Rg_sd=("Rg", "std"),
        )
        .reset_index()
    )
    if "Ligand_RMSD" in timeseries.columns:
        lig = timeseries.groupby("replicate", sort=False)["Ligand_RMSD"].agg(["mean", "std", "max"])
        summary["Ligand_RMSD_mean"] = lig["mean"].to_numpy()
        summary["Ligand_RMSD_sd"] = lig["std"].to_numpy()
        summary["Ligand_RMSD_max"] = lig["max"].to_numpy()

    return {
        "timeseries": timeseries,
        "rmsf_atoms": atom_df,
        "rmsf_residues": residue_df,
        "summary": summary,
    }


def save_descriptors(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    d = Path(output_dir) / "descriptors"
    d.mkdir(parents=True, exist_ok=True)
    results["timeseries"].to_csv(d / "rmsd_rg_timeseries.csv", index=False)
    results["rmsf_atoms"].to_csv(d / "rmsf_atoms.csv", index=False)
    results["rmsf_residues"].to_csv(d / "rmsf_residues.csv", index=False)
    results["summary"].to_csv(d / "summary.csv", index=False)
    return d
