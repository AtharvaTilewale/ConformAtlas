"""Multi-condition / WT vs Mutant comparison in a shared PCA space."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conformatlas.basins import detect_energy_basins, save_basin_results
from conformatlas.fel import calculate_fel, save_fel_grid
from conformatlas.models import ComparisonConfig, FELGrid, PCAResults
from conformatlas.pca import (
    compute_cartesian_pca,
    project_coordinates,
    save_pca_results,
)
from conformatlas.representatives import extract_representative_structures
from conformatlas.trajectory import (
    build_common_atom_mapping,
    stream_trajectory_coordinates,
)

logger = logging.getLogger("conformatlas.comparison")


def run_condition_comparison(config: ComparisonConfig) -> dict[str, Any]:
    """Execute complete comparative analysis between WT and Mutant systems in a shared PCA space.

    Parameters
    ----------
    config : ComparisonConfig
        Parsed comparison configuration.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing comparison results, dataframes, grids, and output paths.
    """
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initiating shared-PCA condition comparison for project: {config.project}")

    # 1. Inspect topologies and build common atom mapping
    topologies: dict[str, Path] = {s.name: s.topology for s in config.systems}
    atom_indices_dict, mapping_df = build_common_atom_mapping(
        topologies, group=config.analysis_group
    )

    pca_dir = out_dir / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(pca_dir / "common_atom_mapping.csv", index=False)

    # 2. Stream coordinates for each system and replicate
    system_coords: dict[str, dict[str, np.ndarray]] = {}
    system_times: dict[str, dict[str, np.ndarray]] = {}
    trajectories_map: dict[str, Path] = {}

    for s in config.systems:
        system_coords[s.name] = {}
        system_times[s.name] = {}
        indices = atom_indices_dict[s.name]

        for rep_idx, traj_path in enumerate(s.trajectories):
            rep_id = f"{s.name}_Rep{rep_idx + 1}"
            trajectories_map[rep_id] = traj_path
            coords, times = stream_trajectory_coordinates(
                s.topology, traj_path, atom_indices=indices, stride=config.stride
            )
            system_coords[s.name][rep_id] = coords
            system_times[s.name][rep_id] = times
            logger.info(f"Loaded {s.name} replicate {rep_id}: {coords.shape[0]} frames.")

    # 3. Build PCA basis (combined or reference-system)
    if config.reference == "reference-system":
        ref_name = config.reference_system or config.systems[0].name
        if ref_name not in system_coords:
            raise ValueError(f"Reference system '{ref_name}' not found in systems.")
        logger.info(f"Computing PCA basis exclusively on reference system '{ref_name}'...")
        evals, evecs, exp_var, cum_var, mean_struct = compute_cartesian_pca(
            system_coords[ref_name],
            weighting=config.pca_weighting,
            n_components=config.components,
        )
    else:  # 'combined'
        logger.info("Computing common balanced PCA basis across all pooled systems...")
        pooled_coords_dict = {}
        for s_name, reps in system_coords.items():
            for rep_id, coords in reps.items():
                pooled_coords_dict[rep_id] = coords

        evals, evecs, exp_var, cum_var, mean_struct = compute_cartesian_pca(
            pooled_coords_dict,
            weighting=config.pca_weighting,
            n_components=config.components,
        )

    # 4. Project each system and replicate onto the shared basis
    proj_dfs = []
    for s_name, reps in system_coords.items():
        for rep_id, coords in reps.items():
            proj_matrix = project_coordinates(coords, mean_struct, evecs)
            times = system_times[s_name][rep_id]
            rep_df = pd.DataFrame(
                proj_matrix, columns=[f"PC{i+1}" for i in range(proj_matrix.shape[1])]
            )
            rep_df.insert(0, "system", s_name)
            rep_df.insert(1, "replicate", rep_id)
            rep_df.insert(2, "frame", np.arange(len(times)))
            rep_df.insert(3, "time_ps", times)
            proj_dfs.append(rep_df)

    master_projections = pd.concat(proj_dfs, ignore_index=True)
    pca_results = PCAResults(
        eigenvalues=evals,
        eigenvectors=evecs,
        explained_variance=exp_var,
        cumulative_variance=cum_var,
        projections=master_projections,
        mean_coords=mean_struct,
    )
    save_pca_results(pca_results, out_dir)

    # 5. Determine shared grid boundaries for PC1 and PC2
    pc1_all = master_projections["PC1"].to_numpy()
    pc2_all = master_projections["PC2"].to_numpy()
    pad_x = 0.05 * (np.max(pc1_all) - np.min(pc1_all))
    pad_y = 0.05 * (np.max(pc2_all) - np.min(pc2_all))
    x_range = (float(np.min(pc1_all) - pad_x), float(np.max(pc1_all) + pad_x))
    y_range = (float(np.min(pc2_all) - pad_y), float(np.max(pc2_all) + pad_y))

    # Global pooled FEL
    global_fel = calculate_fel(
        pc1_all,
        pc2_all,
        temperature=config.temperature,
        bins=config.bins,
        smooth_sigma=config.smooth_sigma,
        x_range=x_range,
        y_range=y_range,
    )
    save_fel_grid(global_fel, out_dir)

    # Detect global conformational basins
    states, unassigned_pop, frame_assignments, basin_map = detect_energy_basins(
        global_fel,
        master_projections,
        min_state_population=config.min_state_population,
        max_states=config.max_states,
    )
    save_basin_results(states, frame_assignments, basin_map, out_dir)

    # Extract representatives
    extract_representative_structures(
        states,
        topologies,
        trajectories_map,
        output_dir=out_dir,
        system_name=config.systems[0].name,
    )

    # 6. Calculate per-system FEL on the exact same grid
    comp_dir = out_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    system_fels: dict[str, FELGrid] = {}
    for s in config.systems:
        s_df = master_projections[master_projections["system"] == s.name]
        s_fel = calculate_fel(
            s_df["PC1"].to_numpy(),
            s_df["PC2"].to_numpy(),
            temperature=config.temperature,
            bins=config.bins,
            smooth_sigma=config.smooth_sigma,
            x_range=x_range,
            y_range=y_range,
        )
        system_fels[s.name] = s_fel
        s_out = comp_dir / s.name
        s_out.mkdir(parents=True, exist_ok=True)
        save_fel_grid(s_fel, s_out)

    # 7. Compare State Populations between systems
    comp_records = []
    major_state_labels = [st.label for st in states]

    for label in major_state_labels:
        row = {"State": label}
        for s in config.systems:
            s_frames = frame_assignments[frame_assignments["system"] == s.name]
            st_count = np.sum(s_frames["state"] == label)
            tot_s = len(s_frames)
            pop = (st_count / tot_s) * 100.0 if tot_s > 0 else 0.0
            row[f"{s.name}_Population_Percent"] = pop

        if len(config.systems) == 2:
            s1_name = config.systems[0].name
            s2_name = config.systems[1].name
            diff = row[f"{s2_name}_Population_Percent"] - row[f"{s1_name}_Population_Percent"]
            row["Difference_Percent"] = diff

        comp_records.append(row)

    pop_comp_df = pd.DataFrame(comp_records)
    pop_comp_df.to_csv(comp_dir / "state_population_comparison.csv", index=False)

    # 8. If exactly 2 systems, compute FEL difference map: ΔΔG = G2 - G1
    diff_grid = None
    if len(config.systems) == 2:
        s1 = config.systems[0].name
        s2 = config.systems[1].name
        fe1 = system_fels[s1].free_energy
        fe2 = system_fels[s2].free_energy
        # Mask where either is NaN / unobserved
        diff_grid = np.where(np.isnan(fe1) | np.isnan(fe2), np.nan, fe2 - fe1)
        np.save(comp_dir / "fel_diff.npy", diff_grid)

    logger.info("Shared-PCA condition comparison completed successfully.")

    return {
        "config": config,
        "pca_results": pca_results,
        "global_fel": global_fel,
        "system_fels": system_fels,
        "states": states,
        "frame_assignments": frame_assignments,
        "population_comparison": pop_comp_df,
        "diff_grid": diff_grid,
        "output_dir": out_dir,
    }
