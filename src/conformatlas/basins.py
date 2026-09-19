"""Topological basin detection, watershed segmentation, and state assignment."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from conformatlas.models import BasinState, FELGrid

logger = logging.getLogger("conformatlas.basins")


def detect_energy_basins(
    fel_grid: FELGrid,
    projections: pd.DataFrame,
    min_state_population: float = 0.05,
    max_states: int = 10,
    min_peak_distance: int = 3,
) -> tuple[list[BasinState], float, pd.DataFrame, np.ndarray]:
    """Detect local free energy minima and segment conformational basins using watershed.

    Parameters
    ----------
    fel_grid : FELGrid
        The computed free energy landscape.
    projections : pd.DataFrame
        DataFrame containing frame projections: ['system', 'replicate', 'frame', 'time_ps', 'PC1', 'PC2', ...].
    min_state_population : float
        Minimum population fraction required to classify as a major state (default: 0.05 / 5%).
    max_states : int
        Maximum number of major states to report.
    min_peak_distance : int
        Minimum grid bin distance between adjacent local minima.

    Returns
    -------
    Tuple[List[BasinState], float, pd.DataFrame, np.ndarray]
        (states_list, unassigned_fraction, frame_assignments_df, basin_map_2d)
    """
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image is required for topological watershed basin detection.")

    prob = fel_grid.probability.copy()
    fe = fel_grid.free_energy.copy()
    occupancy = fel_grid.occupancy_mask if fel_grid.occupancy_mask is not None else ~np.isnan(fe)

    # Clean free energy for watershed: replace NaN with high energy barrier
    fe_clean = np.where(np.isnan(fe), np.nanmax(fe) * 1.5 if not np.all(np.isnan(fe)) else 1000.0, fe)

    # Detect peaks in probability density (corresponds to energy minima)
    # peak_local_max returns (row, col) = (y_idx, x_idx)
    min_dist = max(1, min_peak_distance)
    peak_coords = peak_local_max(prob, min_distance=min_dist, threshold_rel=0.01)

    if len(peak_coords) == 0:
        # Fallback: global maximum of probability
        max_idx = np.unravel_index(np.nanargmax(prob), prob.shape)
        peak_coords = np.array([max_idx])

    # Create markers image for watershed
    markers = np.zeros(prob.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(peak_coords, start=1):
        markers[r, c] = idx

    # Run watershed segmentation on clean free energy surface
    basin_labels = watershed(fe_clean, markers=markers, mask=occupancy)

    # Assign each frame in projections to its basin
    pc1_vals = projections["PC1"].to_numpy()
    pc2_vals = projections["PC2"].to_numpy()
    total_frames = len(projections)

    # Bin indices
    x_idx = np.digitize(pc1_vals, fel_grid.x_edges) - 1
    y_idx = np.digitize(pc2_vals, fel_grid.y_edges) - 1

    # Valid grid boundaries
    valid_bounds = (
        (x_idx >= 0) & (x_idx < len(fel_grid.x_centers)) &
        (y_idx >= 0) & (y_idx < len(fel_grid.y_centers))
    )

    frame_basin_ids = np.full(total_frames, -1, dtype=np.int32)
    valid_indices = np.where(valid_bounds)[0]
    frame_basin_ids[valid_indices] = basin_labels[y_idx[valid_indices], x_idx[valid_indices]]

    # Count population per basin
    unique_ids, counts = np.unique(frame_basin_ids, return_counts=True)
    basin_pop_dict = {uid: cnt / total_frames for uid, cnt in zip(unique_ids, counts) if uid > 0}

    # Sort basins by population descending
    sorted_basins = sorted(basin_pop_dict.items(), key=lambda item: item[1], reverse=True)

    # Filter major states
    major_basins = [b_id for b_id, pop in sorted_basins if pop >= min_state_population][:max_states]
    if not major_basins and sorted_basins:
        # If no basin passes threshold, retain top basin as State A
        major_basins = [sorted_basins[0][0]]

    states: list[BasinState] = []
    state_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Map frame assignments to human readable state labels
    assigned_labels = []
    state_lookup = {}
    for idx, b_id in enumerate(major_basins):
        label = f"State {state_letters[idx % len(state_letters)]}"
        state_lookup[b_id] = label

    for b_id in frame_basin_ids:
        if b_id in state_lookup:
            assigned_labels.append(state_lookup[b_id])
        else:
            assigned_labels.append("Unassigned")

    frame_assignments_df = projections.copy()
    frame_assignments_df["basin_id"] = frame_basin_ids
    frame_assignments_df["state"] = assigned_labels

    major_total_pop = 0.0

    for state_idx, b_id in enumerate(major_basins):
        label = state_lookup[b_id]
        pop_fraction = basin_pop_dict[b_id]
        major_total_pop += pop_fraction

        # Locate minimum energy within this basin
        basin_mask = (basin_labels == b_id)
        sub_fe = np.where(basin_mask, fe, np.nan)
        min_y_idx, min_x_idx = np.unravel_index(np.nanargmin(sub_fe), sub_fe.shape)

        min_pc1 = float(fel_grid.x_centers[min_x_idx])
        min_pc2 = float(fel_grid.y_centers[min_y_idx])
        min_fe_val = float(fel_grid.free_energy[min_y_idx, min_x_idx])

        # Find closest real trajectory frame
        state_frames = frame_assignments_df[frame_assignments_df["state"] == label]
        st_pc1 = state_frames["PC1"].to_numpy()
        st_pc2 = state_frames["PC2"].to_numpy()
        distances = np.sqrt((st_pc1 - min_pc1) ** 2 + (st_pc2 - min_pc2) ** 2)

        best_idx = np.argmin(distances)
        best_row = state_frames.iloc[best_idx]

        states.append(BasinState(
            state_id=state_idx + 1,
            label=label,
            population=pop_fraction,
            min_pc1=min_pc1,
            min_pc2=min_pc2,
            min_free_energy=min_fe_val,
            representative_replicate=str(best_row["replicate"]),
            representative_frame=int(best_row["frame"]),
            representative_time_ps=float(best_row["time_ps"]),
            representative_distance=float(distances[best_idx]),
        ))

    unassigned_fraction = max(0.0, 1.0 - major_total_pop)

    logger.info(
        f"Identified {len(states)} major states (pop >= {min_state_population*100:.1f}%): "
        + ", ".join([f"{s.label}: {s.population*100:.1f}%" for s in states])
        + f" (Unassigned / Minor: {unassigned_fraction*100:.1f}%)"
    )

    return states, unassigned_fraction, frame_assignments_df, basin_labels


def save_basin_results(
    states: list[BasinState],
    frame_assignments: pd.DataFrame,
    basin_map: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Save basin metadata, frame assignments, and segmentation map."""
    states_dir = Path(output_dir) / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    # Save basins.csv
    records = []
    for s in states:
        records.append({
            "State": s.label,
            "Population_Percent": s.population * 100.0,
            "Min_PC1": s.min_pc1,
            "Min_PC2": s.min_pc2,
            "Min_FreeEnergy_kJ_mol": s.min_free_energy,
            "Representative_Replicate": s.representative_replicate,
            "Representative_Frame": s.representative_frame,
            "Representative_Time_ps": s.representative_time_ps,
            "Representative_Distance": s.representative_distance,
        })
    basins_df = pd.DataFrame(records)
    basins_df.to_csv(states_dir / "basins.csv", index=False)

    # Save frame_assignments.csv
    frame_assignments.to_csv(states_dir / "frame_assignments.csv", index=False)

    # Save basin_map.npy
    np.save(states_dir / "basin_map.npy", basin_map)

    return states_dir
