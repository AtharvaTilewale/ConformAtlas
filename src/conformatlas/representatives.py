"""Representative structure extraction and per-state ensemble summaries.

Representatives are real trajectory frames. They are located by *simulation time*,
not by position in the analysed frame list, so striding and time windows cannot
select the wrong frame. Two selection rules are available:

* ``minimum``: frames closest to the basin's free-energy minimum in the CV plane
  (the classical choice);
* ``medoid``: the frame with the smallest summed distance to all other frames of
  the state in the full retained PC space (the most central conformation).

Per-state ensemble summaries (average structure of the analysis atoms, per-atom
and per-residue RMSF, mean Rg / RMSD) are computed from the in-memory fitted
coordinates and therefore cost nothing extra.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from conformatlas.gromacs import GromacsRunner
from conformatlas.models import BasinState
from conformatlas.utils import save_json

logger = logging.getLogger("conformatlas.representatives")

try:
    import MDAnalysis as mda

    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False


# ----------------------------------------------------------------------------
# Frame location by time
# ----------------------------------------------------------------------------


def _frame_index_for_time(universe, time_ps: float) -> int:
    """Index of the trajectory frame whose time is closest to ``time_ps``."""
    traj = universe.trajectory
    n = len(traj)
    if n == 1:
        return 0
    t0 = float(traj[0].time)
    dt = float(traj.dt) if traj.dt and traj.dt > 0 else None
    if dt:
        guess = int(round((time_ps - t0) / dt))
        guess = min(max(guess, 0), n - 1)
        if abs(float(traj[guess].time) - time_ps) <= 0.5 * dt:
            return guess
    # Irregular spacing: scan.
    times = np.array([ts.time for ts in traj])
    return int(np.argmin(np.abs(times - time_ps)))


# ----------------------------------------------------------------------------
# Candidate selection
# ----------------------------------------------------------------------------


def select_representative_frames(
    state: BasinState,
    frame_assignments: pd.DataFrame,
    n_representatives: int = 1,
    method: str = "minimum",
    cv_columns: tuple[str, str] = ("PC1", "PC2"),
    max_medoid_frames: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    """Rows of ``frame_assignments`` chosen as representatives of ``state``.

    Returns a DataFrame (subset of frame_assignments) with an extra column
    ``selection_score`` (distance to the minimum, or mean distance to the other
    state frames for medoids), ordered best first.
    """
    frames = frame_assignments[frame_assignments["state"] == state.label]
    if frames.empty:
        return frames.assign(selection_score=np.nan)
    n = min(n_representatives, len(frames))
    if method == "medoid":
        pc_cols = [c for c in frames.columns if c.startswith("PC") and c[2:].isdigit()]
        if not pc_cols:
            pc_cols = list(cv_columns)
        x = frames[pc_cols].to_numpy(dtype=np.float64)
        if len(x) > max_medoid_frames:
            rng = np.random.default_rng(seed)
            sub = rng.choice(len(x), size=max_medoid_frames, replace=False)
            ref = x[sub]
        else:
            ref = x
        # mean distance of each frame to the (sub)sample of state frames
        score = np.empty(len(x))
        chunk = 2000
        for i in range(0, len(x), chunk):
            d = np.sqrt(((x[i : i + chunk, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
            score[i : i + chunk] = d.mean(axis=1)
    else:
        cx, cy = cv_columns
        score = np.sqrt(
            (frames[cx].to_numpy() - state.min_pc1) ** 2
            + (frames[cy].to_numpy() - state.min_pc2) ** 2
        )
    out = frames.assign(selection_score=score).sort_values("selection_score").head(n)
    return out


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------


def extract_representative_structures(
    states: list[BasinState],
    topologies: dict[str, str | Path],
    trajectories_map: dict[str, str | Path],
    output_dir: str | Path,
    gmx_runner: GromacsRunner | None = None,
    system_name: str = "System",
    frame_assignments: pd.DataFrame | None = None,
    n_representatives: int = 1,
    method: str = "minimum",
    cv_columns: tuple[str, str] = ("PC1", "PC2"),
    cv_names: tuple[str, str] = ("PC1", "PC2"),
) -> list[BasinState]:
    """Write ``states/State_X/representative.pdb`` (and ``representative_02.pdb`` ...)
    for every state, plus ``metadata.json`` with full provenance.

    When ``frame_assignments`` is given, candidates are chosen with
    :func:`select_representative_frames`; otherwise the single frame recorded on
    the state (closest to the minimum) is used.
    """
    states_dir = Path(output_dir) / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    topo_path = Path(topologies.get(system_name, list(topologies.values())[0]))
    universes: dict[Path, object] = {}

    def _universe(traj: Path):
        if traj not in universes:
            universes[traj] = mda.Universe(str(topo_path), str(traj))
        return universes[traj]

    for state in states:
        folder = states_dir / state.label.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)

        if frame_assignments is not None:
            cands = select_representative_frames(
                state, frame_assignments, n_representatives, method, cv_columns
            )
            candidates = [
                (
                    str(r["replicate"]),
                    int(r["frame"]),
                    float(r["time_ps"]),
                    float(r["selection_score"]),
                )
                for _, r in cands.iterrows()
            ]
        else:
            candidates = [
                (
                    state.representative_replicate,
                    state.representative_frame,
                    state.representative_time_ps,
                    state.representative_distance,
                )
            ]
        if candidates and method == "medoid":
            # Keep the state's bookkeeping consistent with what was written.
            rep, frame, t, score = candidates[0]
            state.representative_replicate = rep
            state.representative_frame = frame
            state.representative_time_ps = t
            state.representative_distance = score

        written = []
        for rank, (rep_id, frame_in_analysis, time_ps, score) in enumerate(candidates, start=1):
            pdb_path = folder / (
                "representative.pdb" if rank == 1 else f"representative_{rank:02d}.pdb"
            )
            traj_path = Path(trajectories_map.get(rep_id, list(trajectories_map.values())[0]))
            extracted = False
            actual_frame = None
            if HAS_MDANALYSIS and traj_path.exists() and topo_path.exists():
                try:
                    u = _universe(traj_path)
                    actual_frame = _frame_index_for_time(u, time_ps)
                    u.trajectory[actual_frame]
                    u.atoms.write(str(pdb_path))
                    extracted = True
                except Exception as e:
                    logger.debug(f"MDAnalysis extraction failed for {state.label}: {e}")
            if not extracted and gmx_runner is not None and traj_path.exists():
                try:
                    gmx_runner.run(
                        "trjconv",
                        [
                            "-s",
                            str(topo_path.resolve()),
                            "-f",
                            str(traj_path.resolve()),
                            "-dump",
                            str(time_ps),
                            "-o",
                            str(pdb_path.resolve()),
                        ],
                        input_text="0\n",
                        cwd=folder,
                    )
                    extracted = True
                except Exception as e:
                    logger.warning(f"GROMACS trjconv extraction failed for {state.label}: {e}")
            if extracted:
                logger.info(
                    f"Extracted {state.label} representative {rank} ({rep_id}, {time_ps:.1f} ps) to {pdb_path.name}"
                )
            written.append(
                {
                    "rank": rank,
                    "file": pdb_path.name if extracted else None,
                    "replicate": rep_id,
                    "time_ps": time_ps,
                    "time_ns": time_ps / 1000.0,
                    "frame_in_analysis": frame_in_analysis,
                    "frame_in_trajectory": actual_frame,
                    "selection_score": score,
                }
            )
            if rank == 1:
                state.representative_pdb = pdb_path if extracted else None

        meta = {
            "state": state.label,
            "system": system_name,
            "selection_method": method,
            "selection_score_meaning": (
                f"distance to basin minimum in ({cv_names[0]}, {cv_names[1]})"
                if method == "minimum"
                else "mean distance to the other frames of the state in retained PC space"
            ),
            "population_percent": state.population * 100.0,
            "min_cv1": state.min_pc1,
            "min_cv2": state.min_pc2,
            "cv_names": list(cv_names),
            "min_free_energy_kJ_mol": state.min_free_energy,
            "representatives": written,
            # Backward-compatible fields (first representative)
            "replicate": written[0]["replicate"] if written else None,
            "frame_index": (
                written[0]["frame_in_trajectory"]
                if written and written[0]["frame_in_trajectory"] is not None
                else (written[0]["frame_in_analysis"] if written else None)
            ),
            "time_ps": written[0]["time_ps"] if written else None,
            "time_ns": written[0]["time_ns"] if written else None,
            "distance_from_basin_minimum": (
                written[0]["selection_score"] if written and method == "minimum" else None
            ),
            "pdb_file": written[0]["file"] if written else None,
        }
        save_json(meta, folder / "metadata.json")
    return states


# ----------------------------------------------------------------------------
# Per-state ensemble summaries
# ----------------------------------------------------------------------------


def state_ensemble_summaries(
    states: list[BasinState],
    frame_assignments: pd.DataFrame,
    coords_dict: dict[str, np.ndarray],
    reference: np.ndarray | None,
    atom_info: pd.DataFrame,
    output_dir: str | Path,
    topology: str | Path | None = None,
    atom_indices: np.ndarray | None = None,
    protein_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Average structure, RMSF and Rg/RMSD summary for every state.

    ``coords_dict`` holds the fitted analysis-group coordinates in the same frame
    order as ``frame_assignments`` (replicate by replicate). Writes
    ``states/State_X/average.pdb`` (when MDAnalysis and a topology are available),
    ``rmsf_atoms.csv`` and ``rmsf_residues.csv``; returns a per-state summary table.
    """
    from conformatlas.descriptors import (
        radius_of_gyration,
        rmsd_series,
        rmsf_per_atom,
        rmsf_per_residue,
    )
    from conformatlas.modes import write_structure

    states_dir = Path(output_dir) / "states"
    # Map (replicate, frame) -> row in the concatenated coordinate array.
    offsets = {}
    total = 0
    for rep, c in coords_dict.items():
        offsets[rep] = total
        total += len(c)
    all_coords = np.concatenate(list(coords_dict.values()), axis=0)
    fa = frame_assignments
    global_idx = fa["frame"].to_numpy(dtype=int) + fa["replicate"].map(offsets).to_numpy(dtype=int)

    rows = []
    for st in states:
        sel = global_idx[(fa["state"] == st.label).to_numpy()]
        if len(sel) == 0:
            continue
        c = all_coords[sel]
        mean = c.astype(np.float64).mean(axis=0)
        atom_rmsf = rmsf_per_atom(c)
        folder = states_dir / st.label.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        adf = atom_info.copy().reset_index(drop=True)
        adf["RMSF"] = atom_rmsf
        adf.to_csv(folder / "rmsf_atoms.csv", index=False)
        rmsf_per_residue(atom_rmsf, atom_info.reset_index(drop=True)).to_csv(
            folder / "rmsf_residues.csv", index=False
        )
        avg_path = None
        if HAS_MDANALYSIS and topology is not None and atom_indices is not None:
            try:
                avg_path = write_structure(folder / "average.pdb", topology, atom_indices, mean)
            except Exception as e:
                logger.warning(f"Could not write average structure for {st.label}: {e}")
        pmask = protein_mask
        rg = radius_of_gyration(c, pmask)
        row = {
            "state": st.label,
            "frames": int(len(sel)),
            "Rg_mean": float(rg.mean()),
            "Rg_sd": float(rg.std(ddof=1)) if len(rg) > 1 else 0.0,
            "RMSF_mean": float(atom_rmsf.mean()),
            "RMSF_max": float(atom_rmsf.max()),
            "average_pdb": str(avg_path.name) if avg_path else None,
        }
        if reference is not None:
            rmsd_ref = rmsd_series(c, reference, None if pmask is None else np.where(pmask)[0])
            row["RMSD_to_reference_mean"] = float(rmsd_ref.mean())
            row["RMSD_to_reference_sd"] = float(rmsd_ref.std(ddof=1)) if len(rmsd_ref) > 1 else 0.0
            # RMSD of the state average to the reference: how far the state sits.
            m = mean if pmask is None else mean[pmask]
            r = reference if pmask is None else reference[pmask]
            row["RMSD_average_to_reference"] = float(np.sqrt(((m - r) ** 2).sum(axis=1).mean()))
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(states_dir / "state_ensemble_summary.csv", index=False)
    return df
