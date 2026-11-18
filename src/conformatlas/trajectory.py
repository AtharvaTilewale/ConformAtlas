"""Trajectory handling, inspection, streaming, and atom correspondence."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from conformatlas.gromacs import GromacsRunner
from conformatlas.models import TrajectoryInfo
from conformatlas.superpose import rmsd, superpose_frame

logger = logging.getLogger("conformatlas.trajectory")

#: Structure formats whose coordinates may serve as the superposition reference.
COORDINATE_SUFFIXES = {".pdb", ".gro", ".pdbqt", ".mol2", ".xyz", ".crd", ".ent", ".cif", ".mmcif"}

try:
    import MDAnalysis as mda

    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False


def inspect_trajectory(
    topology_path: str | Path,
    trajectory_path: str | Path,
    replicate_id: str,
    system_name: str,
    gmx_runner: GromacsRunner | None = None,
) -> TrajectoryInfo:
    """Inspect trajectory frame count, time bounds, and timestep."""
    top_path = Path(topology_path)
    traj_path = Path(trajectory_path)

    if not top_path.exists():
        raise FileNotFoundError(f"Topology file not found: {top_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    if HAS_MDANALYSIS:
        try:
            u = mda.Universe(str(top_path), str(traj_path))
            n_frames = len(u.trajectory)
            start_ps = float(u.trajectory[0].time) if n_frames > 0 else 0.0
            end_ps = float(u.trajectory[-1].time) if n_frames > 0 else 0.0
            dt_ps = float(u.trajectory.dt) if n_frames > 1 else 0.0
            return TrajectoryInfo(
                path=traj_path,
                replicate_id=replicate_id,
                system_name=system_name,
                n_frames=n_frames,
                start_time_ps=start_ps,
                end_time_ps=end_ps,
                timestep_ps=dt_ps,
            )
        except Exception as e:
            logger.debug(f"MDAnalysis inspection fell back to GROMACS: {e}")

    # Fallback to GROMACS 'gmx check'
    if gmx_runner is None:
        gmx_runner = GromacsRunner()

    res = gmx_runner.run("check", ["-f", str(traj_path)], check=False)
    out = res.stdout + res.stderr
    n_frames = 0
    start_ps = 0.0
    end_ps = 0.0
    dt_ps = 0.0

    import re

    # Match: "Step       0       0" or "Last frame         500      1000"
    m_last = re.search(r"Last frame\s+(\d+)\s+time\s+([0-9\.\-]+)", out, re.IGNORECASE)
    m_step = re.search(r"Step\s+(\d+)\s+time\s+([0-9\.\-]+)", out, re.IGNORECASE)
    if m_last:
        n_frames = int(m_last.group(1)) + 1
        end_ps = float(m_last.group(2))
    elif m_step:
        n_frames = int(m_step.group(1)) + 1
        end_ps = float(m_step.group(2))

    return TrajectoryInfo(
        path=traj_path,
        replicate_id=replicate_id,
        system_name=system_name,
        n_frames=n_frames,
        start_time_ps=start_ps,
        end_time_ps=end_ps,
        timestep_ps=dt_ps,
    )


def extract_atom_selection_info(
    topology_path: str | Path,
    group: str = "Backbone",
) -> pd.DataFrame:
    """Extract table of selected atoms (index, resnum, resname, atomname) from topology."""
    top_path = Path(topology_path)
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required for direct pythonic atom selection extraction.")

    u = mda.Universe(str(top_path))
    sel_str = map_group_to_mda_selection(group)
    ag = u.select_atoms(sel_str)

    if len(ag) == 0:
        raise ValueError(f"Selection '{group}' (query: '{sel_str}') matched 0 atoms in {top_path}")

    records = []
    for atom in ag:
        # chain label: PDB chainID when set, else segment id, else 'A'
        chain = str(getattr(atom, "chainID", "") or "").strip()
        if not chain:
            chain = str(getattr(atom, "segid", "") or "").strip() or "A"
        records.append(
            {
                "atom_index": atom.index,
                "chain": chain,
                "residue_number": int(atom.resnum),
                "residue_name": str(atom.resname),
                "atom_name": str(atom.name),
            }
        )

    return pd.DataFrame(records)


def map_group_to_mda_selection(group: str) -> str:
    """Translate common GROMACS group names to MDAnalysis selection strings."""
    g_lower = group.lower().strip()
    if g_lower == "backbone":
        # GROMACS 'Backbone' = N, CA, C (no carbonyl O); 'BB' covers MARTINI beads
        return "(protein and name N CA C) or name BB"
    elif g_lower in ["mainchain", "main-chain"]:
        # GROMACS 'MainChain' = N, CA, C, O
        return "backbone or name BB"
    elif g_lower in ["c-alpha", "calpha", "ca"]:
        return "name CA or name BB"
    elif g_lower in ["protein"]:
        return "protein or (name BB SC1 SC2 SC3 SC4 SC5)"
    elif g_lower in ["system"]:
        return "all"
    elif g_lower in ["sidechain"]:
        return "(protein and not backbone) or (name SC1 SC2 SC3 SC4 SC5)"
    else:
        # Pass custom query directly or try atom name
        return group


def build_common_atom_mapping(
    topologies: dict[str, str | Path],
    group: str = "Backbone",
    ndx_groups: dict[str, np.ndarray] | None = None,
    chain_aware: bool | None = None,
) -> tuple[dict[str, list[int]], pd.DataFrame]:
    """Find common intersection of atoms across multiple topologies for shared PCA.

    Atoms are matched by ``(chain, residue_number, atom_name)``; ``chain`` is the
    PDB chain ID when present, otherwise the segment ID. When the systems do not
    share the same set of chain labels (e.g. a PDB with chain ``A`` compared with
    a ``.gro`` whose segment is ``SYSTEM``) the key falls back to
    ``(residue_number, atom_name)`` and a warning is logged, because residue
    numbers that repeat across chains can then no longer be told apart.

    If ``ndx_groups`` contains ``group``, those (0-based) atom indices are used
    for every system; a GROMACS index file therefore only makes sense in a
    comparison when all systems share the same atom numbering.

    Parameters
    ----------
    topologies : Dict[str, Union[str, Path]]
        Dictionary mapping system name to topology file path.
    group : str
        Atom selection group (e.g., 'Backbone', 'C-alpha').
    ndx_groups : dict, optional
        Parsed index-file groups (name -> 0-based atom indices).
    chain_aware : bool, optional
        ``True`` forces chain-aware keys (raises if the chain sets differ),
        ``False`` forces the legacy ``(residue_number, atom_name)`` key, ``None``
        (default) chooses automatically as described above.

    Returns
    -------
    Tuple[Dict[str, List[int]], pd.DataFrame]
        Dictionary of {system_name: list_of_atom_indices} and
        the common atom mapping DataFrame (one row per system and common atom,
        with ``key``, ``chain``, ``residue_number``, ``residue_name``, ``atom_name``).
    """
    system_dfs: dict[str, pd.DataFrame] = {}
    for sys_name, topo_path in topologies.items():
        if ndx_groups is not None and group in ndx_groups:
            df = atom_info_for_indices(topo_path, ndx_groups[group])
        else:
            df = extract_atom_selection_info(topo_path, group=group)
        if "chain" not in df.columns:
            df["chain"] = "A"
        df["chain"] = df["chain"].astype(str).str.strip().replace("", "A")
        system_dfs[sys_name] = df

    chain_sets = {name: frozenset(df["chain"].unique()) for name, df in system_dfs.items()}
    same_chains = len(set(chain_sets.values())) == 1
    if chain_aware is None:
        use_chain = same_chains
        if not same_chains:
            logger.warning(
                "Systems carry different chain/segment labels "
                f"({ {k: sorted(v) for k, v in chain_sets.items()} }); atoms are matched by "
                "(residue_number, atom_name) only. Residue numbers repeated across chains "
                "cannot be distinguished -- give all systems consistent chain IDs to enable "
                "chain-aware matching."
            )
    elif chain_aware and not same_chains:
        raise ValueError(
            "chain_aware=True but the systems have different chain labels: "
            f"{ {k: sorted(v) for k, v in chain_sets.items()} }"
        )
    else:
        use_chain = bool(chain_aware)

    for df in system_dfs.values():
        resnum = df["residue_number"].astype(int).astype(str)
        aname = df["atom_name"].astype(str).str.strip()
        df["key"] = (
            (df["chain"] + "/" + resnum + "_" + aname) if use_chain else (resnum + "_" + aname)
        )
        dup = df["key"].duplicated(keep=False)
        if dup.any():
            examples = df.loc[dup, "key"].unique()[:3].tolist()
            logger.warning(
                f"{int(dup.sum())} atoms share a key (e.g. {examples}); only the first "
                "occurrence of each key is used for the mapping."
            )

    # Find intersection of keys
    common_keys = None
    for sys_name, df in system_dfs.items():
        keys_set = set(df["key"].tolist())
        common_keys = keys_set if common_keys is None else common_keys.intersection(keys_set)

    if not common_keys:
        raise ValueError(
            f"No common atoms found across systems {list(topologies.keys())} for group '{group}'."
        )

    # Maintain sequence order from the first system
    first_df = list(system_dfs.values())[0]
    ordered_keys = list(dict.fromkeys(k for k in first_df["key"].tolist() if k in common_keys))

    mapping_indices: dict[str, list[int]] = {}
    mapping_records = []

    for sys_name, df in system_dfs.items():
        df_first = df.drop_duplicates("key", keep="first").set_index("key")
        indices = []
        for key in ordered_keys:
            if key not in df_first.index:
                raise ValueError(f"System '{sys_name}' missing key '{key}' from common set.")
            row = df_first.loc[key]
            indices.append(int(row["atom_index"]))
            mapping_records.append(
                {
                    "system": sys_name,
                    "key": key,
                    "atom_index": int(row["atom_index"]),
                    "chain": str(row["chain"]),
                    "residue_number": int(row["residue_number"]),
                    "residue_name": str(row["residue_name"]),
                    "atom_name": str(row["atom_name"]),
                }
            )
        mapping_indices[sys_name] = indices

    mapping_df = pd.DataFrame(mapping_records)
    logger.info(
        f"Built common atom mapping with {len(ordered_keys)} atoms across {len(topologies)} systems "
        f"({'chain-aware' if use_chain else 'residue-number'} keys)."
    )
    return mapping_indices, mapping_df


def load_reference_coordinates(
    topology_path: str | Path,
    atom_indices: list[int] | None = None,
    group: str = "Backbone",
    fallback_trajectory: str | Path | None = None,
) -> np.ndarray:
    """Return reference coordinates (n_atoms, 3) for a fit group.

    The structure file is used when it carries coordinates (.pdb, .gro). Topologies
    without coordinates (.tpr) fall back to the first frame of ``fallback_trajectory``.
    """
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required to load reference coordinates.")

    def _select(universe):
        if atom_indices is not None:
            return universe.atoms[atom_indices]
        return universe.select_atoms(map_group_to_mda_selection(group))

    # Only genuine coordinate files serve as the reference. Run-input files (.tpr)
    # expose the pre-processing input structure, which is not on the same frame
    # of reference as a PBC-treated trajectory, so they fall through to frame 0.
    if Path(topology_path).suffix.lower() in COORDINATE_SUFFIXES:
        try:
            u = mda.Universe(str(topology_path))
            ag = _select(u)
            if len(ag) == 0:
                raise ValueError(f"Fit selection '{group}' matched 0 atoms in {topology_path}.")
            positions = ag.positions
            if positions is not None and np.all(np.isfinite(positions)):
                return np.array(positions, dtype=np.float64)
        except ValueError:
            raise
        except Exception as e:
            logger.debug(f"Structure file carries no usable coordinates ({e}); using trajectory.")

    if fallback_trajectory is None:
        raise ValueError(
            f"{topology_path} carries no coordinates and no trajectory was given for a reference."
        )
    u = mda.Universe(str(topology_path), str(fallback_trajectory))
    ag = _select(u)
    if len(ag) == 0:
        raise ValueError(f"Fit selection '{group}' matched 0 atoms in {topology_path}.")
    u.trajectory[0]
    return np.array(ag.positions, dtype=np.float64)


def stream_trajectory_coordinates(
    topology_path: str | Path,
    trajectory_path: str | Path,
    atom_indices: list[int] | None = None,
    group: str = "Backbone",
    stride: int = 1,
    fit: bool = True,
    fit_atom_indices: list[int] | None = None,
    fit_group: str | None = None,
    reference_fit_coords: np.ndarray | None = None,
    mass_weighted: bool = False,
    begin_ps: float | None = None,
    end_ps: float | None = None,
    return_fit_rmsd: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream trajectory frames and return (optionally superposed) coordinates.

    Every frame is least-squares fitted onto a reference using the fit group, and the
    resulting rigid-body transform is applied to the analysis group before the
    coordinates are stored. This removes overall translation and rotation so that
    the subsequent Cartesian PCA describes internal motion only -- the same step
    ``gmx covar -fit`` performs. Pass ``fit=False`` for trajectories that were
    already fitted (e.g. ``gmx trjconv -fit rot+trans``).

    Parameters
    ----------
    topology_path, trajectory_path : path-like
    atom_indices : list of int, optional
        Explicit analysis-group atom indices. If None, ``group`` is used.
    group : str
        Analysis selection (GROMACS group name or MDAnalysis selection string).
    stride : int
        Frame stride.
    fit : bool
        Superpose frames onto the reference before storing coordinates.
    fit_atom_indices : list of int, optional
        Explicit fit-group atom indices. If None, ``fit_group`` is used.
    fit_group : str, optional
        Fit selection. Defaults to the analysis selection.
    reference_fit_coords : (n_fit, 3) array, optional
        Reference coordinates for the fit group. Defaults to the structure file
        (or the first frame when the structure carries no coordinates).
    mass_weighted : bool
        Mass-weight the fit (requires masses in the topology).
    begin_ps, end_ps : float, optional
        Keep only frames with begin_ps <= time <= end_ps (after stride).
    return_fit_rmsd : bool
        Also return the per-frame RMSD of the fit group to the reference after
        superposition (the classical RMSD(t) diagnostic; zeros when fit=False).

    Returns
    -------
    (coords, times) or (coords, times, fit_rmsd)
        coords has shape (n_frames, n_atoms, 3) in the trajectory's length unit
        (Angstrom for MDAnalysis), times is in ps.
    """
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required for direct streaming of trajectory coordinates.")

    u = mda.Universe(str(topology_path), str(trajectory_path))
    sel_str = None
    if atom_indices is not None:
        ag = u.atoms[atom_indices]
    else:
        sel_str = map_group_to_mda_selection(group)
        ag = u.select_atoms(sel_str)

    if len(ag) == 0:
        raise ValueError(
            f"Atom selection '{group}' (query: '{sel_str}') matched 0 atoms in {topology_path}."
        )

    fit_ag = None
    weights = None
    if fit:
        if fit_atom_indices is not None:
            fit_ag = u.atoms[fit_atom_indices]
        elif fit_group is not None:
            fit_ag = u.select_atoms(map_group_to_mda_selection(fit_group))
            if len(fit_ag) == 0:
                raise ValueError(f"Fit selection '{fit_group}' matched 0 atoms in {topology_path}.")
        else:
            fit_ag = ag

        if reference_fit_coords is None:
            u.trajectory[0]
            reference_fit_coords = np.array(fit_ag.positions, dtype=np.float64)
        else:
            reference_fit_coords = np.asarray(reference_fit_coords, dtype=np.float64)
            if reference_fit_coords.shape != (len(fit_ag), 3):
                raise ValueError(
                    f"reference_fit_coords has shape {reference_fit_coords.shape}, "
                    f"expected {(len(fit_ag), 3)} for the fit group."
                )

        if mass_weighted:
            try:
                weights = np.asarray(fit_ag.masses, dtype=np.float64)
                if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
                    raise ValueError("non-positive or missing masses")
            except Exception as e:
                logger.warning(
                    f"Mass-weighted fit requested but masses unavailable ({e}); "
                    "falling back to an unweighted fit."
                )
                weights = None

        if len(fit_ag) < 3:
            raise ValueError(
                f"Fit group has only {len(fit_ag)} atoms; at least 3 are needed for a rigid fit."
            )

    coords_list = []
    times_list = []
    rmsd_list = []

    for ts in u.trajectory[::stride]:
        t = float(ts.time)
        if begin_ps is not None and t < begin_ps:
            continue
        if end_ps is not None and t > end_ps:
            break
        if fit:
            fit_pos = fit_ag.positions
            frame_coords = superpose_frame(fit_pos, ag.positions, reference_fit_coords, weights)
            if return_fit_rmsd:
                fitted_fit = superpose_frame(fit_pos, fit_pos, reference_fit_coords, weights)
                rmsd_list.append(rmsd(fitted_fit, reference_fit_coords))
        else:
            frame_coords = ag.positions
            if return_fit_rmsd:
                rmsd_list.append(0.0)
        coords_list.append(np.array(frame_coords, dtype=np.float32))
        times_list.append(t)

    if not coords_list:
        raise ValueError(
            f"No frames of {trajectory_path} fall in the window "
            f"[{begin_ps}, {end_ps}] ps with stride {stride}."
        )
    coords = np.array(coords_list, dtype=np.float32)
    times = np.array(times_list, dtype=np.float64)
    if return_fit_rmsd:
        return coords, times, np.array(rmsd_list, dtype=np.float64)
    return coords, times


# ----------------------------------------------------------------------------
# Selection resolution (GROMACS group names, MDAnalysis strings, .ndx groups)
# ----------------------------------------------------------------------------


def resolve_selection_indices(
    topology_path: str | Path,
    group: str,
    ndx_groups: dict[str, np.ndarray] | None = None,
    extra_group: str | None = None,
    trajectory_path: str | Path | None = None,
) -> np.ndarray:
    """Atom indices (0-based) for ``group``, optionally united with ``extra_group``.

    Resolution order: a group of that name in the index file (if given), then the
    GROMACS group-name mapping / MDAnalysis selection string. ``extra_group`` is
    resolved the same way (an index group, or a residue name such as a ligand) and
    appended, preserving topology order.
    """
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required to resolve atom selections.")
    from conformatlas.ndx import lookup_group

    u = (
        mda.Universe(str(topology_path))
        if trajectory_path is None
        else mda.Universe(str(topology_path), str(trajectory_path))
    )

    def _one(name: str) -> np.ndarray:
        if ndx_groups:
            idx = lookup_group(ndx_groups, name)
            if idx is not None:
                if idx.max() >= u.atoms.n_atoms:
                    raise ValueError(
                        f"Index group '{name}' refers to atom {idx.max() + 1} but the topology "
                        f"has {u.atoms.n_atoms} atoms."
                    )
                return np.asarray(idx, dtype=np.int64)
        try:
            sel = u.select_atoms(map_group_to_mda_selection(name))
        except Exception:
            sel = u.atoms[:0]
        if len(sel) == 0 and " " not in name.strip():
            # A bare token that is not a selection keyword: treat as residue name
            # (ligands such as LIG, ATP, ZMR) or segment / molecule name.
            sel = u.select_atoms(f"resname {name}")
            if len(sel) == 0:
                sel = u.select_atoms(f"segid {name}")
        if len(sel) == 0:
            raise ValueError(f"Selection '{name}' matched 0 atoms in {topology_path}.")
        return sel.indices.astype(np.int64)

    idx = _one(group)
    if extra_group:
        idx = np.union1d(idx, _one(extra_group))
    return np.sort(idx)


def atom_info_for_indices(topology_path: str | Path, atom_indices: np.ndarray) -> pd.DataFrame:
    """Atom table (index, chain, residue number/name, atom name, mass) for given indices."""
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required.")
    u = mda.Universe(str(topology_path))
    ag = u.atoms[np.asarray(atom_indices, dtype=int)]
    try:
        masses = ag.masses.astype(float)
    except Exception:
        masses = np.full(len(ag), np.nan)
    records = []
    for atom, m in zip(ag, masses):
        chain = getattr(atom, "chainID", None) or getattr(atom, "segid", "A")
        records.append(
            {
                "atom_index": int(atom.index),
                "chain": str(chain),
                "residue_number": int(atom.resnum),
                "residue_name": str(atom.resname),
                "atom_name": str(atom.name),
                "mass": float(m),
            }
        )
    return pd.DataFrame(records)


def check_inputs(
    topology_path: str | Path,
    trajectory_paths: list[str | Path],
    protein_selection: str = "protein",
    frames_to_check: int = 3,
) -> list[str]:
    """Cheap sanity checks; returns a list of human-readable warnings.

    * atom count of every trajectory matches the topology
    * protein chains are whole across periodic boundaries (consecutive C-alpha
      atoms of the same segment closer than 4.5 A) in a few sampled frames
    """
    warnings: list[str] = []
    if not HAS_MDANALYSIS:
        return warnings
    try:
        u_top = mda.Universe(str(topology_path))
        n_top = u_top.atoms.n_atoms
    except Exception as e:
        return [f"Could not read topology {topology_path}: {e}"]

    for traj in trajectory_paths:
        try:
            u = mda.Universe(str(topology_path), str(traj))
        except Exception as e:
            warnings.append(f"{Path(traj).name}: cannot be opened with the topology ({e}).")
            continue
        if u.trajectory.n_atoms != n_top:
            warnings.append(
                f"{Path(traj).name}: {u.trajectory.n_atoms} atoms per frame but the topology has "
                f"{n_top}. Use a matching structure/.tpr or a trajectory written for this selection."
            )
            continue
        ca = u.select_atoms(f"({protein_selection}) and name CA")
        if len(ca) < 3:
            continue
        n = len(u.trajectory)
        idxs = sorted(set(int(i) for i in np.linspace(0, n - 1, min(frames_to_check, n))))
        broken = False
        for fi in idxs:
            u.trajectory[fi]
            pos = ca.positions
            same_seg = ca.segids[:-1] == ca.segids[1:]
            consecutive = (ca.resids[1:] - ca.resids[:-1]) == 1
            d = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
            if np.any(d[same_seg & consecutive] > 4.5):
                broken = True
                break
        if broken:
            warnings.append(
                f"{Path(traj).name}: consecutive C-alpha atoms more than 4.5 A apart - the molecule "
                "appears broken across the periodic boundary. Make it whole first "
                "(gmx trjconv -pbc mol -center) or the superposition is meaningless."
            )
    return warnings


def estimate_memory_gb(n_frames_total: int, n_atoms: int) -> float:
    """RAM for holding float32 coordinates of all frames (GB)."""
    return n_frames_total * n_atoms * 3 * 4 / 1e9


#: Peak resident memory of an ``analyze`` run, per engine. D = 3 * n_atoms is
#: the number of coordinates; one D x D float64 matrix takes 8 D^2 bytes.
#:
#: In-memory engine::
#:
#:     BASE + N * (PER_FRAME_BYTES + COORD_FACTOR * 12 * n_atoms)
#:          + COV_MATRICES * 8 D^2        if N >= D  (covariance + eigh)
#:          + SVD_COPIES * 8 N D          if N <  D  (dual / SVD route)
#:
#: Streaming engine (N-independent apart from small per-frame arrays)::
#:
#:     BASE + N * PER_FRAME_BYTES
#:          + 8 D^2 * (n_replicates + STREAM_MATRICES + STREAM_WORKER_MATRICES * (n_workers - 1))
#:
#: Measured peaks with ConformAtlas 1.0.2 (Python 3.11, numpy 2.4, MDAnalysis
#: 2.10) on full-atom adenylate kinase, default options, all overestimated by
#: the model (so 'auto' errs on the safe side):
#:
#: ========================  ==========  ======  ========  ========
#: selection (atoms)         engine      frames  measured  model
#: ========================  ==========  ======  ========  ========
#: Backbone N,CA,C (642)     in-memory    5 000   0.84 GB   1.10 GB
#: Backbone (642)            in-memory   10 000   1.02 GB   1.44 GB
#: Backbone (642)            in-memory   20 000   1.69 GB   2.12 GB
#: Backbone (642)            in-memory   40 000   3.02 GB   3.48 GB
#: C-alpha (214)             in-memory   20 000   0.64 GB   1.08 GB
#: C-alpha (214)             in-memory   80 000   2.10 GB   2.60 GB
#: first 1 500 atoms         in-memory    8 000   2.88 GB   2.91 GB
#: first 2 500 atoms         in-memory    3 000   2.83 GB   2.93 GB
#: first 1 500 atoms         streaming    3 000   1.11 GB   1.53 GB
#: first 2 500 atoms         streaming    3 000   3.02 GB   3.26 GB
#: first 2 500 atoms, 2 rep  streaming    3 000   3.47 GB   3.71 GB
#: ========================  ==========  ======  ========  ========
IN_MEMORY_BASE_GB = 0.55
IN_MEMORY_PER_FRAME_BYTES = 4_000
IN_MEMORY_COORD_FACTOR = 8.3
IN_MEMORY_COV_MATRICES = 7.0
IN_MEMORY_SVD_COPIES = 9.0
STREAM_MATRICES = 5.0
STREAM_WORKER_MATRICES = 2.0

#: ``--engine auto`` uses the in-memory engine while its estimated peak stays
#: below this fraction of the memory available to the process.
AUTO_MEMORY_FRACTION = 0.75

#: Threshold (estimated peak, GB) used when the available memory cannot be
#: determined, e.g. on platforms without /proc.
FALLBACK_STREAMING_THRESHOLD_GB = 8.0

#: Limits above this are treated as "no limit" (cgroup v1 reports ~2**63).
_UNLIMITED_BYTES = 1 << 60


def estimate_peak_memory_gb(n_frames_total: int, n_atoms: int) -> float:
    """Estimated peak RAM (GB) of the in-memory engine for ``n_frames_total`` frames
    of ``n_atoms`` analysis atoms (model and measurements: ``IN_MEMORY_BASE_GB``)."""
    n, d = max(0, int(n_frames_total)), 3 * int(n_atoms)
    per_frame = IN_MEMORY_PER_FRAME_BYTES + IN_MEMORY_COORD_FACTOR * 12 * n_atoms
    pca = IN_MEMORY_COV_MATRICES * 8.0 * d * d if n >= d else IN_MEMORY_SVD_COPIES * 8.0 * n * d
    return IN_MEMORY_BASE_GB + (n * per_frame + pca) / 1e9


def estimate_streaming_peak_gb(
    n_frames_total: int, n_atoms: int, n_replicates: int = 1, n_workers: int = 1
) -> float:
    """Estimated peak RAM (GB) of the streaming engine: one D x D scatter matrix per
    replicate plus merge / eigendecomposition workspace (``IN_MEMORY_BASE_GB``)."""
    d = 3 * int(n_atoms)
    matrices = (
        max(1, int(n_replicates))
        + STREAM_MATRICES
        + STREAM_WORKER_MATRICES * max(0, int(n_workers) - 1)
    )
    return (
        IN_MEMORY_BASE_GB
        + (max(0, int(n_frames_total)) * IN_MEMORY_PER_FRAME_BYTES + 8.0 * d * d * matrices) / 1e9
    )


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _own_cgroup_paths(proc_cgroup: Path) -> tuple[str | None, str | None]:
    """This process's cgroup path for v2 (``0::path``) and for the v1 memory
    controller, from ``/proc/self/cgroup`` (None where absent)."""
    v2 = v1 = None
    try:
        lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hier, controllers, path = parts
        if hier == "0" and controllers == "":
            v2 = path
        elif "memory" in controllers.split(","):
            v1 = path
    return v2, v1


def _ancestors(base: Path, rel: str | None) -> list[Path]:
    """``base/rel`` and each parent up to ``base`` (innermost first)."""
    dirs = [base]
    if rel and rel.strip("/"):
        cur = base
        for part in rel.strip("/").split("/"):
            cur = cur / part
            dirs.append(cur)
    return dirs[::-1]


def _cgroup_available_bytes(
    cgroup_root: Path, proc_cgroup: Path = Path("/proc/self/cgroup")
) -> int | None:
    """Memory left under the tightest cgroup limit that applies to this process
    (v2 ``memory.max`` or v1 ``memory.limit_in_bytes``, checked on the process's own
    cgroup and every ancestor), or None when no limit is set. Inside a container the
    process's cgroup is usually the root of the mounted hierarchy."""
    v2_rel, v1_rel = _own_cgroup_paths(proc_cgroup)
    layouts = (
        (cgroup_root, v2_rel, "memory.max", "memory.current"),
        (cgroup_root / "memory", v1_rel, "memory.limit_in_bytes", "memory.usage_in_bytes"),
    )
    best: int | None = None
    for base, rel, limit_name, usage_name in layouts:
        for d in _ancestors(base, rel):
            limit = _read_int(d / limit_name)
            if limit is None or limit >= _UNLIMITED_BYTES:
                continue
            headroom = max(0, limit - (_read_int(d / usage_name) or 0))
            best = headroom if best is None else min(best, headroom)
        if best is not None:
            return best
    return None


def _meminfo_available_bytes(meminfo: Path) -> int | None:
    try:
        lines = meminfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            parts = line.split()
            try:
                return int(parts[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def available_memory_gb(
    cgroup_root: str | Path = "/sys/fs/cgroup",
    meminfo: str | Path = "/proc/meminfo",
    proc_cgroup: str | Path = "/proc/self/cgroup",
) -> float | None:
    """Memory (GB) this process can still use: the smaller of the container
    (cgroup) headroom and the host's ``MemAvailable``. None if neither is known."""
    candidates = [
        b
        for b in (
            _cgroup_available_bytes(Path(cgroup_root), Path(proc_cgroup)),
            _meminfo_available_bytes(Path(meminfo)),
        )
        if b is not None
    ]
    return min(candidates) / 1e9 if candidates else None


def suggest_stride(n_frames_total: int, n_atoms: int, limit_gb: float) -> int | None:
    """Smallest extra frame-subsampling factor that brings the estimated in-memory
    peak for ``n_frames_total`` frames under ``limit_gb``; None if no stride can
    (the selection itself is too large). Multiply by the current ``--stride``."""
    n = int(n_frames_total)
    if n <= 1 or estimate_peak_memory_gb(n, n_atoms) <= limit_gb:
        return 1
    if estimate_peak_memory_gb(1, n_atoms) > limit_gb:
        return None
    lo, hi = 1, n  # estimate(n // hi) fits, estimate(n // lo) does not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if estimate_peak_memory_gb(n // mid, n_atoms) <= limit_gb:
            hi = mid
        else:
            lo = mid
    return hi
