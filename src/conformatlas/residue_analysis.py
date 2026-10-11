"""Residue-level analyses on fitted coordinates: dynamic cross-correlation maps,
per-residue differences between conformational states, ligand-residue contact
persistence and binding-site fluctuations.

All coordinate inputs are ``(n_frames, n_atoms, 3)`` arrays **already superposed
on a common reference** (the arrays returned by
``conformatlas.trajectory.stream_trajectory_coordinates`` with ``fit=True``).
Nothing here re-fits: the DCCM and the per-residue displacements therefore
describe motion relative to that global fit (the same assumption made by the
Cartesian PCA of the pipeline). Units are those of the coordinates (A for the
Python backend).

Definitions
-----------
* DCCM (Ichiye & Karplus, Proteins 1991; McCammon & Harvey 1987):

      C_ij = <dr_i . dr_j> / sqrt(<dr_i . dr_i> <dr_j . dr_j>),  dr_i = r_i - <r_i>

  with (optionally weighted) ensemble averages over frames. ``C_ii = 1``;
  ``C_ij`` is the normalised trace of the 3x3 atomic covariance block, not the
  Pearson correlation of concatenated xyz components.
* Per-residue RMSF (Angstrom) in a state: ``sqrt(mean_atoms <|dr|^2>_state)``.
* Per-residue RMSD between state means: ``sqrt(mean_atoms |<r>_B - <r>_A|^2)``;
  ``centroid_displacement`` is ``|c_B - c_A|`` of the residue centroids.
* Contact: a residue is in contact with the ligand in a frame when any
  (heavy-atom) ligand/residue pair is closer than ``cutoff`` (default 4.5 A;
  minimum-image distances when the frame carries a box and ``use_pbc``).
  Persistence per state = fraction of that state's frames with the contact.
* Binding-site RMSF: per-state RMSF of residues whose persistence reaches
  ``persistence_threshold`` in at least one state.

The 4.5 A cutoff and the 0.5 persistence threshold are conventional choices
(heuristic, not validated quantities); the DCCM, RMSF, displacement and
distance arithmetic are validated against explicit numpy/ProDy/MDAnalysis
references in validation/checks_enhanced.py.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("conformatlas.residue_analysis")

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array

    HAS_MDANALYSIS = True
except ImportError:  # pragma: no cover
    HAS_MDANALYSIS = False

RESIDUE_KEY_COLUMNS = ["chain", "residue_number", "residue_name"]


# =============================================================================
# Helpers
# =============================================================================


def state_file_slug(label: str) -> str:
    """File-name-safe form of a state label: ``"State A"`` -> ``"state_A"``.

    Lower-cases the word ``State``, replaces whitespace with ``_`` and drops
    every character outside ``[A-Za-z0-9_.-]`` so that output names contain no
    spaces (``residues/dccm_state_A.npy``).
    """
    text = re.sub(r"^State\b", "state", str(label).strip())
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"[^A-Za-z0-9_.-]", "", text) or "state"


def _as_frames(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim == 2:  # (n_frames, 3 n_atoms)
        if c.shape[1] % 3:
            raise ValueError("Flattened coordinates must have 3*n_atoms columns.")
        c = c.reshape(c.shape[0], -1, 3)
    if c.ndim != 3 or c.shape[2] != 3:
        raise ValueError("coords must be (n_frames, n_atoms, 3).")
    return c


def _norm_weights(n: int, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError(
            "weights must be a finite, non-negative (n_frames,) array with positive sum."
        )
    return w / w.sum()


def _residue_groups(atom_table: pd.DataFrame) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Residue table (in order of first appearance) and atom-index lists per residue."""
    df = atom_table.reset_index(drop=True)
    if "chain" not in df.columns:
        df = df.assign(chain="A")
    keys = list(zip(df["chain"].astype(str), df["residue_number"].astype(int)))
    order: dict[tuple, int] = {}
    groups: list[list[int]] = []
    for i, k in enumerate(keys):
        if k not in order:
            order[k] = len(groups)
            groups.append([])
        groups[order[k]].append(i)
    rows = []
    for k, idx in zip(order.keys(), groups):
        rows.append(
            {
                "chain": k[0],
                "residue_number": k[1],
                "residue_name": str(df.loc[idx[0], "residue_name"]),
                "n_atoms": len(idx),
            }
        )
    return pd.DataFrame(rows), [np.asarray(g, dtype=int) for g in groups]


def atom_table_from_universe(ag) -> pd.DataFrame:
    """Atom table (atom_index, chain, residue_number, residue_name, atom_name) of an AtomGroup."""
    rows = []
    for a in ag:
        chain = getattr(a, "chainID", "") or getattr(a, "segid", "") or "A"
        rows.append(
            {
                "atom_index": int(a.index),
                "chain": str(chain).strip() or "A",
                "residue_number": int(getattr(a, "resnum", a.resid)),
                "residue_name": str(a.resname),
                "atom_name": str(a.name),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Dynamic cross-correlation map
# =============================================================================


def dccm(coords: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Dynamic cross-correlation matrix of atoms from fitted coordinates.

    ``C_ij = <dr_i . dr_j> / sqrt(<|dr_i|^2> <|dr_j|^2>)`` with frame weights
    ``weights`` (uniform by default). Atoms that do not move return NaN rows.
    """
    c = _as_frames(coords)
    n_frames, n_atoms, _ = c.shape
    w = _norm_weights(n_frames, weights)
    mean = np.einsum("f,fai->ai", w, c)
    d = c - mean[None]
    # covariance of the displacement vectors: sum over xyz of weighted frame products
    dw = d * np.sqrt(w)[:, None, None]
    flat = dw.reshape(n_frames, n_atoms * 3)
    cov_full = flat.T @ flat  # (3n, 3n)
    cov = cov_full.reshape(n_atoms, 3, n_atoms, 3)
    trace_cov = np.einsum("iaja->ij", cov)  # sum over the diagonal of each 3x3 block
    var = np.diag(trace_cov).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = trace_cov / np.sqrt(np.outer(var, var))
    corr[var <= 0, :] = np.nan
    corr[:, var <= 0] = np.nan
    np.fill_diagonal(corr, np.where(var > 0, 1.0, np.nan))
    return corr


def collapse_to_residues(
    matrix: np.ndarray, atom_table: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Average an atom-level matrix over the atoms of each residue pair.

    For one atom per residue (e.g. C-alpha) this is the identity.
    """
    res_df, groups = _residue_groups(atom_table)
    n = len(groups)
    out = np.empty((n, n))
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            out[i, j] = np.nanmean(matrix[np.ix_(gi, gj)])
    return out, res_df


def dccm_per_state(
    coords: np.ndarray,
    frame_states: np.ndarray,
    states: list[str] | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """DCCM computed separately from the frames of each state."""
    c = _as_frames(coords)
    fs = np.asarray(frame_states)
    if fs.shape[0] != c.shape[0]:
        raise ValueError("frame_states must have one label per frame.")
    labels = states if states is not None else list(dict.fromkeys(fs.tolist()))
    out = {}
    for s in labels:
        m = fs == s
        if m.sum() < 2:
            logger.warning(f"State {s}: fewer than 2 frames, DCCM skipped.")
            continue
        out[str(s)] = dccm(c[m], None if weights is None else np.asarray(weights)[m])
    return out


# =============================================================================
# Per-residue fluctuations and state differences
# =============================================================================


def per_residue_rmsf(
    coords: np.ndarray, atom_table: pd.DataFrame, weights: np.ndarray | None = None
) -> pd.DataFrame:
    """RMSF per residue: ``sqrt(mean over residue atoms of <|r - <r>|^2>)``."""
    c = _as_frames(coords)
    res_df, groups = _residue_groups(atom_table)
    if len(atom_table) != c.shape[1]:
        raise ValueError(
            f"atom_table has {len(atom_table)} rows but coords has {c.shape[1]} atoms."
        )
    w = _norm_weights(c.shape[0], weights)
    mean = np.einsum("f,fai->ai", w, c)
    msf_atom = np.einsum("f,fa->a", w, ((c - mean[None]) ** 2).sum(axis=2))
    res_df = res_df.copy()
    res_df["rmsf"] = [float(np.sqrt(msf_atom[g].mean())) for g in groups]
    return res_df


def state_residue_differences(
    coords: np.ndarray,
    atom_table: pd.DataFrame,
    frame_states: np.ndarray,
    state_a: str,
    state_b: str,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-residue comparison of two states sampled in the same fitted coordinates.

    Columns: ``chain, residue_number, residue_name, n_atoms, rmsf_<A>, rmsf_<B>,
    delta_rmsf`` (B - A), ``mean_rmsd`` (RMSD between the state-average positions
    of the residue's atoms), ``centroid_displacement``, ``n_frames_<A>``,
    ``n_frames_<B>``. Frame ``weights`` (if given) are applied within each state.
    """
    c = _as_frames(coords)
    fs = np.asarray(frame_states)
    if fs.shape[0] != c.shape[0]:
        raise ValueError("frame_states must have one label per frame.")
    res_df, groups = _residue_groups(atom_table)
    if len(atom_table) != c.shape[1]:
        raise ValueError(
            f"atom_table has {len(atom_table)} rows but coords has {c.shape[1]} atoms."
        )
    out = res_df.copy()
    means = {}
    for label in (state_a, state_b):
        m = fs == label
        if m.sum() == 0:
            raise ValueError(f"State '{label}' has no frames.")
        w = _norm_weights(int(m.sum()), None if weights is None else np.asarray(weights)[m])
        cs = c[m]
        mean = np.einsum("f,fai->ai", w, cs)
        msf_atom = np.einsum("f,fa->a", w, ((cs - mean[None]) ** 2).sum(axis=2))
        out[f"rmsf_{label}"] = [float(np.sqrt(msf_atom[g].mean())) for g in groups]
        out[f"n_frames_{label}"] = int(m.sum())
        means[label] = mean
    out["delta_rmsf"] = out[f"rmsf_{state_b}"] - out[f"rmsf_{state_a}"]
    dmean = means[state_b] - means[state_a]
    out["mean_rmsd"] = [float(np.sqrt((dmean[g] ** 2).sum(axis=1).mean())) for g in groups]
    out["centroid_displacement"] = [
        float(np.linalg.norm(means[state_b][g].mean(axis=0) - means[state_a][g].mean(axis=0)))
        for g in groups
    ]
    out["state_a"] = state_a
    out["state_b"] = state_b
    return out


def all_state_residue_differences(
    coords: np.ndarray,
    atom_table: pd.DataFrame,
    frame_states: np.ndarray,
    states: list[str] | None = None,
    reference_state: str | None = None,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """``state_residue_differences`` of every state against ``reference_state``
    (default: the most populated state), stacked in one table."""
    fs = np.asarray(frame_states)
    labels = states if states is not None else list(dict.fromkeys(fs.tolist()))
    if reference_state is None:
        counts = {s: int((fs == s).sum()) for s in labels}
        reference_state = max(counts, key=counts.get)
    parts = [
        state_residue_differences(coords, atom_table, fs, reference_state, s, weights)
        for s in labels
        if s != reference_state
    ]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# Ligand-residue contacts
# =============================================================================


@dataclass
class ContactResult:
    """Per-frame ligand-residue contact matrix."""

    contacts: np.ndarray  # (n_frames, n_residues) bool
    min_distance: np.ndarray  # (n_frames, n_residues) minimum heavy-atom distance
    residues: pd.DataFrame  # chain, residue_number, residue_name, n_atoms
    times: np.ndarray
    cutoff: float
    ligand_selection: str
    protein_selection: str
    used_pbc: bool


def _heavy(ag):
    try:
        return ag.select_atoms("not name H* and not type H")
    except Exception:  # topology without types
        return ag.select_atoms("not name H*")


def ligand_residue_contacts(
    topology_path: str | Path,
    trajectory_path: str | Path,
    ligand_selection: str,
    protein_selection: str = "protein",
    cutoff: float = 4.5,
    stride: int = 1,
    begin_ps: float | None = None,
    end_ps: float | None = None,
    heavy_atoms_only: bool = True,
    use_pbc: bool = True,
) -> ContactResult:
    """Per-frame ligand-residue contacts (any atom pair within ``cutoff`` A).

    Distances are computed with ``MDAnalysis.lib.distances.distance_array``;
    when the frame has box dimensions and ``use_pbc`` is True the minimum-image
    convention is applied (triclinic boxes supported).
    """
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required for contact analysis.")
    u = mda.Universe(str(topology_path), str(trajectory_path))
    lig = u.select_atoms(ligand_selection)
    prot = u.select_atoms(protein_selection)
    if heavy_atoms_only:
        lig, prot = _heavy(lig), _heavy(prot)
    if len(lig) == 0:
        raise ValueError(f"Ligand selection '{ligand_selection}' matched 0 atoms.")
    if len(prot) == 0:
        raise ValueError(f"Protein selection '{protein_selection}' matched 0 atoms.")
    if len(lig & prot) > 0:
        raise ValueError("Ligand and protein selections overlap.")
    atom_table = atom_table_from_universe(prot)
    res_df, groups = _residue_groups(atom_table)
    # residue index per protein atom
    res_index = np.empty(len(prot), dtype=int)
    for r, g in enumerate(groups):
        res_index[g] = r
    n_res = len(groups)
    contacts, mind, times = [], [], []
    used_pbc = False
    for ts in u.trajectory[:: max(1, stride)]:
        t = float(ts.time)
        if begin_ps is not None and t < begin_ps:
            continue
        if end_ps is not None and t > end_ps:
            break
        box = (
            ts.dimensions
            if (use_pbc and ts.dimensions is not None and np.all(ts.dimensions[:3] > 0))
            else None
        )
        used_pbc = used_pbc or box is not None
        d = distance_array(lig.positions, prot.positions, box=box)  # (n_lig, n_prot)
        d_atom = d.min(axis=0)
        # minimum over the atoms of each residue
        md = np.full(n_res, np.inf)
        np.minimum.at(md, res_index, d_atom)
        mind.append(md)
        contacts.append(md < cutoff)
        times.append(t)
    if not contacts:
        raise ValueError("No frames selected for contact analysis.")
    return ContactResult(
        contacts=np.array(contacts, dtype=bool),
        min_distance=np.array(mind, dtype=np.float64),
        residues=res_df,
        times=np.array(times, dtype=np.float64),
        cutoff=cutoff,
        ligand_selection=ligand_selection,
        protein_selection=protein_selection,
        used_pbc=used_pbc,
    )


def contact_persistence(
    result: ContactResult,
    frame_states: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    min_persistence: float = 0.0,
) -> pd.DataFrame:
    """Fraction of frames (per state) in which each residue contacts the ligand.

    Returns one row per residue with ``persistence_all`` and one
    ``persistence_<state>`` column per state (weighted fractions when ``weights``
    are given), plus ``n_frames_<state>`` and the mean minimum distance per
    state. Rows whose maximum persistence is below ``min_persistence`` are dropped.
    """
    n_frames = result.contacts.shape[0]
    w = _norm_weights(n_frames, weights)
    out = result.residues.copy()
    out["persistence_all"] = (result.contacts * w[:, None]).sum(axis=0)
    out["mean_min_distance_all"] = (result.min_distance * w[:, None]).sum(axis=0)
    states: list[str] = []
    if frame_states is not None:
        fs = np.asarray(frame_states)
        if fs.shape[0] != n_frames:
            raise ValueError("frame_states must have one label per contact frame.")
        states = [str(s) for s in dict.fromkeys(fs.tolist())]
        for s in states:
            m = fs == s
            ws = w[m] / w[m].sum() if w[m].sum() > 0 else w[m]
            out[f"persistence_{s}"] = (result.contacts[m] * ws[:, None]).sum(axis=0)
            out[f"mean_min_distance_{s}"] = (result.min_distance[m] * ws[:, None]).sum(axis=0)
            out[f"n_frames_{s}"] = int(m.sum())
    pers_cols = ["persistence_all"] + [f"persistence_{s}" for s in states]
    out["max_persistence"] = out[pers_cols].max(axis=1)
    out = out[out["max_persistence"] >= min_persistence].reset_index(drop=True)
    out.attrs["cutoff"] = result.cutoff
    out.attrs["states"] = states
    return out


def binding_site_residues(persistence: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Residues whose contact persistence reaches ``threshold`` in any state."""
    return persistence[persistence["max_persistence"] >= threshold].reset_index(drop=True)


def binding_site_rmsf(
    coords: np.ndarray,
    atom_table: pd.DataFrame,
    site: pd.DataFrame,
    frame_states: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-residue RMSF of binding-site residues, overall and per state.

    ``site`` must carry ``chain`` and ``residue_number`` columns (e.g. the output
    of ``binding_site_residues``). Residues of the site absent from ``atom_table``
    (for instance when the analysis group is C-alpha only) are reported with NaN.
    """
    c = _as_frames(coords)
    keys = set(zip(site["chain"].astype(str), site["residue_number"].astype(int)))
    all_res = per_residue_rmsf(c, atom_table, weights)
    all_res = all_res.rename(columns={"rmsf": "rmsf_all"})
    if frame_states is not None:
        fs = np.asarray(frame_states)
        for s in dict.fromkeys(fs.tolist()):
            m = fs == s
            r = per_residue_rmsf(
                c[m], atom_table, None if weights is None else np.asarray(weights)[m]
            )
            all_res[f"rmsf_{s}"] = r["rmsf"].to_numpy()
    mask = [
        (ch, rn) in keys
        for ch, rn in zip(all_res["chain"].astype(str), all_res["residue_number"].astype(int))
    ]
    found = all_res[mask].reset_index(drop=True)
    missing = keys - set(zip(found["chain"].astype(str), found["residue_number"].astype(int)))
    if missing:
        extra = site[
            [
                (ch, rn) in missing
                for ch, rn in zip(site["chain"].astype(str), site["residue_number"].astype(int))
            ]
        ]
        extra = extra[RESIDUE_KEY_COLUMNS].copy()
        found = pd.concat([found, extra], ignore_index=True)
    return found


# =============================================================================
# Convenience: write the residues/ directory
# =============================================================================


def write_residue_outputs(
    out_dir: str | Path,
    atom_table: pd.DataFrame,
    dccm_matrix: np.ndarray | None = None,
    dccm_states: dict[str, np.ndarray] | None = None,
    state_differences: pd.DataFrame | None = None,
    contacts: pd.DataFrame | None = None,
    site_rmsf: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write ``residues/`` files: ``dccm.csv`` / ``dccm.npy`` (atom order of
    ``atom_table``; residue-collapsed ``dccm_residues.csv``), per-state
    ``dccm_<state slug>.npy`` (``dccm_state_A.npy`` for ``State A``), ``state_residue_differences.csv``,
    ``ligand_contacts_per_state.csv`` and ``binding_site_rmsf.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    labels = [
        f"{n}{r}_{c}"
        for c, r, n in zip(
            atom_table["chain"].astype(str) if "chain" in atom_table else ["A"] * len(atom_table),
            atom_table["residue_number"].astype(int),
            atom_table["residue_name"].astype(str),
        )
    ]
    if dccm_matrix is not None:
        np.save(out_dir / "dccm.npy", dccm_matrix)
        pd.DataFrame(dccm_matrix, index=labels, columns=labels).to_csv(out_dir / "dccm.csv")
        res_mat, res_df = collapse_to_residues(dccm_matrix, atom_table)
        rlab = [
            f"{n}{r}_{c}"
            for c, r, n in zip(res_df["chain"], res_df["residue_number"], res_df["residue_name"])
        ]
        pd.DataFrame(res_mat, index=rlab, columns=rlab).to_csv(out_dir / "dccm_residues.csv")
        written["dccm"] = out_dir / "dccm.csv"
        written["dccm_npy"] = out_dir / "dccm.npy"
        written["dccm_residues"] = out_dir / "dccm_residues.csv"
    if dccm_states:
        for s, m in dccm_states.items():
            p = out_dir / f"dccm_{state_file_slug(s)}.npy"
            np.save(p, m)
            written[f"dccm_{s}"] = p
    if state_differences is not None and len(state_differences):
        p = out_dir / "state_residue_differences.csv"
        state_differences.to_csv(p, index=False)
        written["state_residue_differences"] = p
    if contacts is not None:
        p = out_dir / "ligand_contacts_per_state.csv"
        contacts.to_csv(p, index=False)
        written["ligand_contacts_per_state"] = p
    if site_rmsf is not None:
        p = out_dir / "binding_site_rmsf.csv"
        site_rmsf.to_csv(p, index=False)
        written["binding_site_rmsf"] = p
    return written


# =============================================================================
# Orchestration: the whole residues/ step from the pipeline's inputs
# =============================================================================


def analyze_residues(
    structure: str | Path,
    trajectories: dict[str, str | Path],
    frame_assignments: pd.DataFrame,
    out_dir: str | Path,
    analysis_atom_indices: list[int] | np.ndarray,
    coords_by_replicate: dict[str, np.ndarray] | None = None,
    ligand_selection: str | None = None,
    protein_selection: str = "protein",
    cutoff: float = 4.5,
    stride: int = 1,
    begin_ps: float | None = None,
    end_ps: float | None = None,
    fit: bool = True,
    fit_atom_indices: list[int] | np.ndarray | None = None,
    reference_fit_coords: np.ndarray | None = None,
    mass_weighted: bool = False,
    weights: np.ndarray | None = None,
    reference_state: str | None = None,
    persistence_threshold: float = 0.5,
    make_figures: bool = True,
    figure_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Run every residue-level analysis and write ``residues/`` (+ figures).

    ``frame_assignments`` is the pipeline table (columns ``replicate``, ``state``,
    ordered replicate by replicate); ``coords_by_replicate`` are the fitted
    coordinates the pipeline already holds (``{replicate: (n_frames, n_atoms, 3)}``);
    when omitted they are streamed again with the same selection/fit arguments
    (``conformatlas.trajectory.stream_trajectory_coordinates``). Contact analysis
    runs only when ``ligand_selection`` is given and reads the trajectories with
    the same ``stride``/time window so that frames line up with ``frame_assignments``.
    """
    from conformatlas.trajectory import atom_info_for_indices, stream_trajectory_coordinates

    out_dir = Path(out_dir)
    rep_ids = list(dict.fromkeys(frame_assignments["replicate"].tolist()))
    if set(rep_ids) != set(trajectories):
        raise ValueError(
            f"frame_assignments replicates {rep_ids} do not match trajectories {list(trajectories)}."
        )
    parts = []
    for rep in rep_ids:
        if coords_by_replicate is not None and rep in coords_by_replicate:
            c = np.asarray(coords_by_replicate[rep])
        else:
            c, _ = stream_trajectory_coordinates(
                structure,
                trajectories[rep],
                atom_indices=list(np.asarray(analysis_atom_indices, dtype=int)),
                stride=stride,
                fit=fit,
                fit_atom_indices=None
                if fit_atom_indices is None
                else list(np.asarray(fit_atom_indices, dtype=int)),
                reference_fit_coords=reference_fit_coords,
                mass_weighted=mass_weighted,
                begin_ps=begin_ps,
                end_ps=end_ps,
            )
        n_assigned = int((frame_assignments["replicate"] == rep).sum())
        if c.shape[0] != n_assigned:
            raise ValueError(
                f"{rep}: {c.shape[0]} coordinate frames but {n_assigned} assigned frames."
            )
        parts.append(c)
    coords = np.concatenate(parts, axis=0)
    atom_table = atom_info_for_indices(structure, np.asarray(analysis_atom_indices, dtype=int))
    states = frame_assignments["state"].astype(str).to_numpy()
    labels = [
        s
        for s in dict.fromkeys(states.tolist())
        if s and s.lower() not in ("nan", "none", "unassigned")
    ]

    m_all = dccm(coords, weights)
    m_states = dccm_per_state(coords, states, labels, weights)
    diffs = (
        all_state_residue_differences(coords, atom_table, states, labels, reference_state, weights)
        if len(labels) >= 2
        else None
    )

    contacts_df = site_rmsf = None
    if ligand_selection:
        res_parts = []
        for rep in rep_ids:
            res_parts.append(
                ligand_residue_contacts(
                    structure,
                    trajectories[rep],
                    ligand_selection,
                    protein_selection,
                    cutoff=cutoff,
                    stride=stride,
                    begin_ps=begin_ps,
                    end_ps=end_ps,
                )
            )
        merged = ContactResult(
            contacts=np.concatenate([r.contacts for r in res_parts]),
            min_distance=np.concatenate([r.min_distance for r in res_parts]),
            residues=res_parts[0].residues,
            times=np.concatenate([r.times for r in res_parts]),
            cutoff=cutoff,
            ligand_selection=ligand_selection,
            protein_selection=protein_selection,
            used_pbc=any(r.used_pbc for r in res_parts),
        )
        if merged.contacts.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Contact analysis produced {merged.contacts.shape[0]} frames but {coords.shape[0]} frames are assigned; "
                "use the same stride/time window as the main analysis."
            )
        contacts_df = contact_persistence(merged, states, weights)
        site = binding_site_residues(contacts_df, persistence_threshold)
        site_rmsf = (
            binding_site_rmsf(coords, atom_table, site, states, weights) if len(site) else None
        )

    written = write_residue_outputs(
        out_dir, atom_table, m_all, m_states, diffs, contacts_df, site_rmsf
    )
    if make_figures:
        from conformatlas.plotting_residue import (
            plot_contact_persistence,
            plot_dccm,
            plot_residue_state_differences,
        )

        fig_dir = Path(figure_dir) if figure_dir is not None else out_dir.parent / "figures"
        res_mat, res_df = collapse_to_residues(m_all, atom_table)
        written["fig_dccm"] = plot_dccm(
            res_mat,
            res_df["residue_number"].to_numpy(),
            fig_dir / "dccm.png",
            chains=res_df["chain"].to_numpy(),
        )
        for s, m in m_states.items():
            rm, _ = collapse_to_residues(m, atom_table)
            written[f"fig_dccm_{s}"] = plot_dccm(
                rm,
                res_df["residue_number"].to_numpy(),
                fig_dir / f"dccm_{state_file_slug(s)}.png",
                title=f"DCCM, state {s}",
                chains=res_df["chain"].to_numpy(),
            )
        if diffs is not None and len(diffs):
            written["fig_state_residue_differences"] = plot_residue_state_differences(
                diffs, fig_dir / "state_residue_differences.png"
            )
        if contacts_df is not None:
            written["fig_contact_persistence"] = plot_contact_persistence(
                contacts_df, fig_dir / "ligand_contact_persistence.png"
            )
    return written
