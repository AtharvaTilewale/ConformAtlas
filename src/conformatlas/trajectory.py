"""Trajectory handling, inspection, streaming, and atom correspondence."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from conformatlas.gromacs import GromacsRunner
from conformatlas.models import TrajectoryInfo

logger = logging.getLogger("conformatlas.trajectory")

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
        chain = getattr(atom, "chainID", getattr(atom, "segid", "A"))
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
    if g_lower in ["backbone", "mainchain"]:
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
) -> tuple[dict[str, list[int]], pd.DataFrame]:
    """Find common intersection of atoms across multiple topologies for shared PCA.

    Parameters
    ----------
    topologies : Dict[str, Union[str, Path]]
        Dictionary mapping system name to topology file path.
    group : str
        Atom selection group (e.g., 'Backbone', 'C-alpha').

    Returns
    -------
    Tuple[Dict[str, List[int]], pd.DataFrame]
        Dictionary of {system_name: list_of_atom_indices} and
        the common atom mapping DataFrame.
    """
    system_dfs: dict[str, pd.DataFrame] = {}
    for sys_name, topo_path in topologies.items():
        df = extract_atom_selection_info(topo_path, group=group)
        # Create unique key per atom: (residue_number, atom_name)
        df["key"] = df["residue_number"].astype(str) + "_" + df["atom_name"].str.strip()
        system_dfs[sys_name] = df

    # Find intersection of keys
    common_keys = None
    for sys_name, df in system_dfs.items():
        keys_set = set(df["key"].tolist())
        if common_keys is None:
            common_keys = keys_set
        else:
            common_keys = common_keys.intersection(keys_set)

    if not common_keys:
        raise ValueError(
            f"No common atoms found across systems {list(topologies.keys())} for group '{group}'."
        )

    # Maintain sequence order from the first system
    first_df = list(system_dfs.values())[0]
    ordered_keys = [k for k in first_df["key"].tolist() if k in common_keys]

    mapping_indices: dict[str, list[int]] = {}
    mapping_records = []

    for sys_name, df in system_dfs.items():
        df_indexed = df.set_index("key")
        indices = []
        for key in ordered_keys:
            if key not in df_indexed.index:
                raise ValueError(f"System '{sys_name}' missing key '{key}' from common set.")
            row = df_indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            indices.append(int(row["atom_index"]))
            mapping_records.append(
                {
                    "system": sys_name,
                    "atom_index": int(row["atom_index"]),
                    "residue_number": int(row["residue_number"]),
                    "residue_name": str(row["residue_name"]),
                    "atom_name": str(row["atom_name"]),
                }
            )
        mapping_indices[sys_name] = indices

    mapping_df = pd.DataFrame(mapping_records)
    logger.info(
        f"Built common atom mapping with {len(ordered_keys)} atoms across {len(topologies)} systems."
    )
    return mapping_indices, mapping_df


def stream_trajectory_coordinates(
    topology_path: str | Path,
    trajectory_path: str | Path,
    atom_indices: list[int] | None = None,
    group: str = "Backbone",
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream trajectory frames and return coordinates for selected atoms.

    Parameters
    ----------
    topology_path : Union[str, Path]
    trajectory_path : Union[str, Path]
    atom_indices : Optional[List[int]]
        Explicit atom indices to extract. If None, uses group.
    group : str
        Selection string if atom_indices is None.
    stride : int
        Frame stride.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (coords, times) where coords has shape (n_frames, n_atoms, 3) in Angstroms/nm,
        and times is array of frame times in ps.
    """
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required for direct streaming of trajectory coordinates.")

    u = mda.Universe(str(topology_path), str(trajectory_path))
    if atom_indices is not None:
        ag = u.atoms[atom_indices]
    else:
        sel_str = map_group_to_mda_selection(group)
        ag = u.select_atoms(sel_str)

    if len(ag) == 0:
        raise ValueError(
            f"Atom selection '{group}' (query: '{sel_str}') matched 0 atoms in {topology_path}."
        )

    coords_list = []
    times_list = []

    for ts in u.trajectory[::stride]:
        coords_list.append(ag.positions.copy())
        times_list.append(float(ts.time))

    coords = np.array(coords_list, dtype=np.float32)
    times = np.array(times_list, dtype=np.float64)
    return coords, times
