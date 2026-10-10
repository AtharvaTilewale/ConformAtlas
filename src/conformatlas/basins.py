"""Topological basin detection, watershed segmentation, and state assignment."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as ndi_label

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from conformatlas.models import BasinState, FELGrid
from conformatlas.utils import R_GAS_CONSTANT

logger = logging.getLogger("conformatlas.basins")


def pairwise_saddles(labels: np.ndarray, free_energy: np.ndarray) -> dict[tuple[int, int], float]:
    """Lowest crossing height between every pair of 4-connected neighbouring basins.

    The crossing height of a boundary edge is the higher of its two bins; the saddle
    of a pair is the minimum over all its boundary edges. Keys are (low_id, high_id).
    """
    fe = np.where(np.isfinite(free_energy), free_energy, np.inf)
    saddles: dict[tuple[int, int], float] = {}
    for a, b, fa, fb in (
        (labels[:, :-1], labels[:, 1:], fe[:, :-1], fe[:, 1:]),
        (labels[:-1, :], labels[1:, :], fe[:-1, :], fe[1:, :]),
    ):
        diff = (a != b) & (a > 0) & (b > 0)
        if not np.any(diff):
            continue
        crossing = np.maximum(fa[diff], fb[diff])
        for la, lb, h in zip(a[diff], b[diff], crossing):
            key = (int(min(la, lb)), int(max(la, lb)))
            if np.isfinite(h) and h < saddles.get(key, np.inf):
                saddles[key] = float(h)
    return saddles


def merge_shallow_basins(
    basin_labels: np.ndarray,
    free_energy: np.ndarray,
    min_barrier: float,
) -> np.ndarray:
    """Merge adjacent watershed basins separated by a barrier lower than ``min_barrier``.

    Watershed segmentation splits a single broad minimum into several basins
    whenever histogram noise creates shallow ripples. Two basins are only kept
    apart if the lowest saddle on their shared boundary lies at least
    ``min_barrier`` (kJ/mol) above the minimum of the shallower basin.
    Merging proceeds iteratively from the shallowest barrier upwards.
    """
    labels = basin_labels.copy()
    fe = np.where(np.isfinite(free_energy), free_energy, np.inf)

    while True:
        ids = [int(i) for i in np.unique(labels) if i > 0]
        if len(ids) < 2:
            return labels
        minima = {i: float(np.min(fe[labels == i])) for i in ids}
        saddles = pairwise_saddles(labels, fe)
        if not saddles:
            return labels

        best_pair, best_barrier = None, np.inf
        for (i, j), h in saddles.items():
            barrier = h - max(minima[i], minima[j])
            if barrier < best_barrier:
                best_pair, best_barrier = (i, j), barrier

        if best_pair is None or best_barrier >= min_barrier:
            return labels

        i, j = best_pair
        keep, drop = (i, j) if minima[i] <= minima[j] else (j, i)
        labels[labels == drop] = keep
        logger.debug(
            f"Merged basin {drop} into {keep} (barrier {best_barrier:.2f} kJ/mol < {min_barrier:.2f})."
        )


def frame_bin_indices(
    fel_grid: FELGrid, cv1: np.ndarray, cv2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid bin (x_idx, y_idx) of every frame plus a validity mask.

    ``np.histogram2d`` treats the right-most edge as inclusive, so a value exactly
    on it belongs to the last bin.
    """
    cv1 = np.asarray(cv1, dtype=np.float64)
    cv2 = np.asarray(cv2, dtype=np.float64)
    x_idx = np.digitize(cv1, fel_grid.x_edges) - 1
    y_idx = np.digitize(cv2, fel_grid.y_edges) - 1
    nx, ny = len(fel_grid.x_centers), len(fel_grid.y_centers)
    x_idx[(x_idx == nx) & (cv1 == fel_grid.x_edges[-1])] = nx - 1
    y_idx[(y_idx == ny) & (cv2 == fel_grid.y_edges[-1])] = ny - 1
    valid = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)
    return x_idx, y_idx, valid


def compute_state_barriers(
    basin_labels: np.ndarray,
    free_energy: np.ndarray,
    states: list[BasinState],
) -> pd.DataFrame:
    """Saddle heights and barriers between every pair of ADJACENT major states.

    Columns: state_i, state_j, saddle_kJ_mol, barrier_from_i (saddle - min_i),
    barrier_from_j. Pairs without a shared boundary are absent (separated by
    unsampled space, so no barrier estimate is possible).
    """
    saddles = pairwise_saddles(basin_labels, free_energy)
    by_id = {s.basin_id: s for s in states if s.basin_id > 0}
    rows = []
    for (a, b), h in sorted(saddles.items()):
        if a in by_id and b in by_id:
            si, sj = by_id[a], by_id[b]
            rows.append(
                {
                    "state_i": si.label,
                    "state_j": sj.label,
                    "saddle_kJ_mol": h,
                    "barrier_from_i_kJ_mol": h - si.min_free_energy,
                    "barrier_from_j_kJ_mol": h - sj.min_free_energy,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "state_i",
            "state_j",
            "saddle_kJ_mol",
            "barrier_from_i_kJ_mol",
            "barrier_from_j_kJ_mol",
        ],
    )


def detect_energy_basins(
    fel_grid: FELGrid,
    projections: pd.DataFrame,
    min_state_population: float = 0.05,
    max_states: int = 10,
    min_peak_distance: int = 3,
    min_barrier: float | None = None,
    weights: np.ndarray | None = None,
    cv_columns: tuple[str, str] = ("PC1", "PC2"),
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
    min_barrier : float, optional
        Minimum free-energy barrier (kJ/mol) separating two basins; shallower
        neighbours are merged. Defaults to 1 kT at the grid temperature.
    weights : array, optional
        Per-frame statistical weights (reweighted ensembles); populations become
        weighted fractions.

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
    fe_clean = np.where(
        np.isnan(fe), np.nanmax(fe) * 1.5 if not np.all(np.isnan(fe)) else 1000.0, fe
    )

    # Detect peaks in probability density (corresponds to energy minima)
    # peak_local_max returns (row, col) = (y_idx, x_idx)
    min_dist = max(1, min_peak_distance)
    # exclude_border must be False: a basin at the extreme of PC1/PC2 sits at
    # the edge of the grid by construction and would otherwise lose its marker.
    peak_coords = peak_local_max(
        prob, min_distance=min_dist, threshold_rel=0.01, exclude_border=False
    )

    if len(peak_coords) == 0:
        # Fallback: global maximum of probability
        max_idx = np.unravel_index(np.nanargmax(prob), prob.shape)
        peak_coords = np.array([max_idx])

    # Create markers image for watershed
    support = np.isfinite(fe) | occupancy

    # Every connected region of sampled space must own at least one marker;
    # otherwise watershed cannot label it. Add the densest bin of any region
    # that has none.
    components, n_components = ndi_label(support)
    peak_list = [tuple(pc) for pc in peak_coords]
    marked = {components[r, c] for r, c in peak_list}
    for comp_id in range(1, n_components + 1):
        if comp_id in marked:
            continue
        comp_prob = np.where(components == comp_id, prob, -np.inf)
        peak_list.append(tuple(np.unravel_index(np.argmax(comp_prob), prob.shape)))

    markers = np.zeros(prob.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(peak_list, start=1):
        markers[r, c] = idx

    # Run watershed segmentation on the clean free energy surface. The mask is
    # the finite-energy support (the smoothed support when smoothing is on),
    # which is connected; the raw occupancy mask is a sparse scatter of islands
    # for short trajectories and would leave most occupied bins unlabelled.
    basin_labels = watershed(fe_clean, markers=markers, mask=support)

    # Merge basins separated by barriers below the significance threshold.
    if min_barrier is None:
        min_barrier = R_GAS_CONSTANT * fel_grid.temperature  # 1 kT
    n_before = len(np.unique(basin_labels[basin_labels > 0]))
    basin_labels = merge_shallow_basins(basin_labels, fe, min_barrier)
    n_after = len(np.unique(basin_labels[basin_labels > 0]))
    if n_after < n_before:
        logger.info(
            f"Merged {n_before - n_after} shallow watershed basins "
            f"(barrier < {min_barrier:.2f} kJ/mol); {n_after} remain."
        )

    # Any occupied bin that watershed could not reach from a marker (an island
    # disconnected from the support) inherits the label of the nearest labelled
    # bin, so every sampled frame belongs to a basin.
    orphan = occupancy & (basin_labels == 0)
    if np.any(orphan) and np.any(basin_labels > 0):
        _, (near_r, near_c) = distance_transform_edt(basin_labels == 0, return_indices=True)
        basin_labels[orphan] = basin_labels[near_r[orphan], near_c[orphan]]
    basin_labels = np.where(occupancy, basin_labels, 0)

    # Assign each frame in projections to its basin
    pc1_vals = projections[cv_columns[0]].to_numpy()
    pc2_vals = projections[cv_columns[1]].to_numpy()
    total_frames = len(projections)

    # Bin indices
    x_idx, y_idx, valid_bounds = frame_bin_indices(fel_grid, pc1_vals, pc2_vals)
    frame_basin_ids = np.full(total_frames, -1, dtype=np.int32)
    valid_indices = np.where(valid_bounds)[0]
    frame_basin_ids[valid_indices] = basin_labels[y_idx[valid_indices], x_idx[valid_indices]]

    # Count population per basin
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != total_frames:
            raise ValueError(f"weights has {len(w)} entries for {total_frames} frames.")
        w_total = w.sum()
    else:
        w = np.ones(total_frames, dtype=np.float64)
        w_total = float(total_frames)
    unique_ids = np.unique(frame_basin_ids)
    basin_pop_dict = {
        int(uid): float(w[frame_basin_ids == uid].sum() / w_total) for uid in unique_ids if uid > 0
    }

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
        basin_mask = basin_labels == b_id
        sub_fe = np.where(basin_mask, fe, np.nan)
        min_y_idx, min_x_idx = np.unravel_index(np.nanargmin(sub_fe), sub_fe.shape)

        min_pc1 = float(fel_grid.x_centers[min_x_idx])
        min_pc2 = float(fel_grid.y_centers[min_y_idx])
        min_fe_val = float(fel_grid.free_energy[min_y_idx, min_x_idx])

        # Find closest real trajectory frame
        state_frames = frame_assignments_df[frame_assignments_df["state"] == label]
        st_pc1 = state_frames[cv_columns[0]].to_numpy()
        st_pc2 = state_frames[cv_columns[1]].to_numpy()
        distances = np.sqrt((st_pc1 - min_pc1) ** 2 + (st_pc2 - min_pc2) ** 2)

        best_idx = np.argmin(distances)
        best_row = state_frames.iloc[best_idx]

        states.append(
            BasinState(
                state_id=state_idx + 1,
                basin_id=int(b_id),
                label=label,
                population=pop_fraction,
                min_pc1=min_pc1,
                min_pc2=min_pc2,
                min_free_energy=min_fe_val,
                representative_replicate=str(best_row["replicate"]),
                representative_frame=int(best_row["frame"]),
                representative_time_ps=float(best_row["time_ps"]),
                representative_distance=float(distances[best_idx]),
            )
        )

    unassigned_fraction = max(0.0, 1.0 - major_total_pop)

    logger.info(
        f"Identified {len(states)} major states (pop >= {min_state_population * 100:.1f}%): "
        + ", ".join([f"{s.label}: {s.population * 100:.1f}%" for s in states])
        + f" (Unassigned / Minor: {unassigned_fraction * 100:.1f}%)"
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
        records.append(
            {
                "State": s.label,
                "Population_Percent": s.population * 100.0,
                "Min_PC1": s.min_pc1,
                "Min_PC2": s.min_pc2,
                "Min_FreeEnergy_kJ_mol": s.min_free_energy,
                "Representative_Replicate": s.representative_replicate,
                "Representative_Frame": s.representative_frame,
                "Representative_Time_ps": s.representative_time_ps,
                "Representative_Distance": s.representative_distance,
            }
        )
    basins_df = pd.DataFrame(records)
    basins_df.to_csv(states_dir / "basins.csv", index=False)

    # Save frame_assignments.csv
    frame_assignments.to_csv(states_dir / "frame_assignments.csv", index=False)

    # Save basin_map.npy
    np.save(states_dir / "basin_map.npy", basin_map)

    return states_dir
