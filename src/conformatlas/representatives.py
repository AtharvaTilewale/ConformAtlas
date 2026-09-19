"""Representative structure extraction and metadata serialization."""

import logging
from pathlib import Path

from conformatlas.gromacs import GromacsRunner
from conformatlas.models import BasinState
from conformatlas.utils import save_json

logger = logging.getLogger("conformatlas.representatives")

try:
    import MDAnalysis as mda

    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False


def extract_representative_structures(
    states: list[BasinState],
    topologies: dict[str, str | Path],
    trajectories_map: dict[str, str | Path],  # replicate_key -> traj_path
    output_dir: str | Path,
    gmx_runner: GromacsRunner | None = None,
    system_name: str = "System",
) -> list[BasinState]:
    """Extract full-atom representative PDB structures for each detected basin state.

    Parameters
    ----------
    states : List[BasinState]
        List of detected major states.
    topologies : Dict[str, Union[str, Path]]
        Mapping of system name to topology file.
    trajectories_map : Dict[str, Union[str, Path]]
        Mapping of replicate ID (e.g. 'Rep_1') to trajectory file path.
    output_dir : Union[str, Path]
        Base directory for analysis output.
    gmx_runner : Optional[GromacsRunner]
        GROMACS runner instance for trjconv fallback.
    system_name : str
        System identifier.

    Returns
    -------
    List[BasinState]
        Updated BasinState list with representative_pdb paths populated.
    """
    states_dir = Path(output_dir) / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    topo_path = Path(topologies.get(system_name, list(topologies.values())[0]))

    for state in states:
        folder_name = state.label.replace(" ", "_")
        state_dir = states_dir / folder_name
        state_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = state_dir / "representative.pdb"
        meta_path = state_dir / "metadata.json"

        rep_id = state.representative_replicate
        traj_path = trajectories_map.get(rep_id)
        if traj_path is None:
            # Fallback: take the first trajectory
            traj_path = list(trajectories_map.values())[0]

        traj_path = Path(traj_path)
        extracted = False

        if HAS_MDANALYSIS and traj_path.exists() and topo_path.exists():
            try:
                u = mda.Universe(str(topo_path), str(traj_path))
                frame_idx = min(state.representative_frame, len(u.trajectory) - 1)
                u.trajectory[frame_idx]
                u.atoms.write(str(pdb_path))
                extracted = True
                logger.info(f"Extracted {state.label} representative via MDAnalysis to {pdb_path}")
            except Exception as e:
                logger.debug(f"MDAnalysis structure extraction failed, trying GROMACS: {e}")

        if not extracted and gmx_runner is not None and traj_path.exists():
            try:
                # Use gmx trjconv with -dump
                gmx_runner.run(
                    "trjconv",
                    [
                        "-s",
                        str(topo_path.resolve()),
                        "-f",
                        str(traj_path.resolve()),
                        "-dump",
                        str(state.representative_time_ps),
                        "-o",
                        str(pdb_path.resolve()),
                    ],
                    input_text="0\n",  # Select System
                    cwd=state_dir,
                )
                extracted = True
                logger.info(
                    f"Extracted {state.label} representative via GROMACS trjconv to {pdb_path}"
                )
            except Exception as e:
                logger.warning(f"GROMACS trjconv extraction failed for {state.label}: {e}")

        state.representative_pdb = pdb_path if extracted else None

        # Write metadata.json
        meta_data = {
            "state": state.label,
            "system": system_name,
            "replicate": rep_id,
            "frame_index": state.representative_frame,
            "time_ps": state.representative_time_ps,
            "time_ns": state.representative_time_ps / 1000.0,
            "min_pc1": state.min_pc1,
            "min_pc2": state.min_pc2,
            "min_free_energy_kJ_mol": state.min_free_energy,
            "distance_from_basin_minimum": state.representative_distance,
            "population_percent": state.population * 100.0,
            "pdb_file": str(pdb_path.name) if extracted else None,
        }
        save_json(meta_data, meta_path)

    return states
