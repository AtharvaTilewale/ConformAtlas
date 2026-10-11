"""Multi-condition / WT vs Mutant comparison in a shared PCA space."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conformatlas.basins import detect_energy_basins, save_basin_results
from conformatlas.comparison_stats import (
    SystemBootstrap,
    bootstrap_system,
    ddg_map,
    ddg_map_table,
    landscape_overlap_table,
    population_difference_table,
    state_free_energy_table,
    subspace_overlap_matrix,
)
from conformatlas.fel import calculate_fel, choose_bins, save_fel_grid
from conformatlas.models import ComparisonConfig, FELGrid, PCAResults
from conformatlas.ndx import parse_ndx
from conformatlas.pca import (
    compute_cartesian_pca,
    project_coordinates,
    save_pca_results,
)
from conformatlas.plotting import set_cv_labels, set_figure_style
from conformatlas.provenance import collect_environment
from conformatlas.representatives import extract_representative_structures
from conformatlas.trajectory import (
    build_common_atom_mapping,
    load_reference_coordinates,
    stream_trajectory_coordinates,
)

logger = logging.getLogger("conformatlas.comparison")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


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
    # figure options (None keeps the active module-level style)
    set_figure_style(
        fig_format=config.fig_format,
        colormap=config.colormap,
        font_scale=config.font_scale,
        dpi=config.fig_dpi,
    )
    if config.interactive is not None or config.plotly_js is not None:
        from conformatlas.report import set_report_options

        set_report_options(
            **{
                k: v
                for k, v in {
                    "interactive": config.interactive,
                    "plotly_js": config.plotly_js,
                }.items()
                if v is not None
            }
        )

    logger.info(f"Initiating shared-PCA condition comparison for project: {config.project}")

    # 1. Inspect topologies and build common atom mapping
    topologies: dict[str, Path] = {s.name: s.topology for s in config.systems}
    ndx_groups = parse_ndx(config.index_file) if config.index_file else None
    atom_indices_dict, mapping_df = build_common_atom_mapping(
        topologies, group=config.analysis_group, ndx_groups=ndx_groups
    )
    reference_name = config.reference_system or config.systems[0].name
    if reference_name not in topologies:
        raise ValueError(f"Reference system '{reference_name}' not found in systems.")
    set_cv_labels("Shared PC1", "Shared PC2", "Å")

    pca_dir = out_dir / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(pca_dir / "common_atom_mapping.csv", index=False)

    # 1b. Common fit-group mapping and a single shared reference structure.
    # Every frame of every system is superposed onto the SAME reference (the
    # reference system's structure file, or the first system's), so that the
    # projections of different systems are directly comparable.
    fit_indices_dict: dict[str, list[int]] | None = None
    reference_fit: np.ndarray | None = None
    if config.fit:
        if config.fit_group == config.analysis_group:
            fit_indices_dict = atom_indices_dict
        else:
            fit_indices_dict, fit_mapping_df = build_common_atom_mapping(
                topologies, group=config.fit_group, ndx_groups=ndx_groups
            )
            fit_mapping_df.to_csv(pca_dir / "common_fit_atom_mapping.csv", index=False)
        ref_sys = reference_name
        ref_cfg = next(sc for sc in config.systems if sc.name == ref_sys)
        reference_fit = load_reference_coordinates(
            ref_cfg.topology,
            atom_indices=fit_indices_dict[ref_sys],
            fallback_trajectory=ref_cfg.trajectories[0],
        )
        logger.info(
            f"Superposing all frames on '{config.fit_group}' "
            f"({len(reference_fit)} common atoms) of reference system '{ref_sys}'."
        )
    else:
        logger.warning(
            "Superposition disabled (pca.fit: false); trajectories are assumed pre-fitted "
            "to a common reference, otherwise projections are not comparable across systems."
        )

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
                s.topology,
                traj_path,
                atom_indices=indices,
                stride=config.stride,
                fit=config.fit,
                fit_atom_indices=fit_indices_dict[s.name] if fit_indices_dict else None,
                reference_fit_coords=reference_fit,
                mass_weighted=config.mass_weighted_fit,
                begin_ps=config.begin_ps,
                end_ps=config.end_ps,
            )
            if coords.shape[0] == 0:
                raise ValueError(
                    f"No frames of {traj_path} fall in the requested time window "
                    f"({config.begin_ps}-{config.end_ps} ps)."
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
                proj_matrix, columns=[f"PC{i + 1}" for i in range(proj_matrix.shape[1])]
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

    bins = config.bins if config.bins is not None else choose_bins(pc1_all, pc2_all)
    logger.info(
        f"Shared FEL grid: {bins} x {bins} bins" + (" (auto)" if config.bins is None else "")
    )

    # Global pooled FEL
    global_fel = calculate_fel(
        pc1_all,
        pc2_all,
        temperature=config.temperature,
        bins=bins,
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
        min_barrier=config.min_barrier,
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
            bins=bins,
            smooth_sigma=config.smooth_sigma,
            x_range=x_range,
            y_range=y_range,
        )
        system_fels[s.name] = s_fel
        s_out = comp_dir / s.name
        s_out.mkdir(parents=True, exist_ok=True)
        save_fel_grid(s_fel, s_out)

    system_names = [s.name for s in config.systems]

    # 7. Per-system block bootstrap on the shared grid (populations, state minima, ΔG grids)
    boots: dict[str, SystemBootstrap] = {}
    if config.n_bootstrap and config.n_bootstrap > 0:
        rng = np.random.default_rng(config.seed)
        for name in system_names:
            s_df = master_projections[master_projections["system"] == name]
            boots[name] = bootstrap_system(
                name, s_df, global_fel, basin_map, states, config.n_bootstrap, rng
            )
            logger.info(
                f"Block bootstrap for {name}: {config.n_bootstrap} resamples, "
                f"block lengths {boots[name].block_lengths}"
            )

    # 7b. Re-segmentation bootstrap of the pooled landscape (bootstrap_mode: resegment).
    # The per-system comparison statistics below keep the fixed pooled segmentation; this
    # step only reports how stable the number and populations of the pooled states are.
    resegment_tables: dict[str, pd.DataFrame] = {}
    if (
        config.n_bootstrap
        and config.n_bootstrap > 0
        and config.bootstrap_mode == "resegment"
        and states
    ):
        from conformatlas.bootstrap import bootstrap_landscape
        from conformatlas.comparison_stats import system_block_lengths

        pooled_block_lengths: dict[str, int] = {}
        for name in system_names:
            s_df = master_projections[master_projections["system"] == name]
            pooled_block_lengths.update(system_block_lengths(s_df))
        pooled_boot = bootstrap_landscape(
            pc1_all,
            pc2_all,
            master_projections["replicate"].to_numpy(),
            global_fel,
            basin_map,
            states,
            pooled_block_lengths,
            n_boot=int(config.n_bootstrap),
            seed=config.seed,
            mode="resegment",
            resegment_kwargs={
                "min_state_population": config.min_state_population,
                "max_states": config.max_states,
                "min_barrier": config.min_barrier,
            },
        )
        if pooled_boot.state_count_distribution is not None:
            pooled_boot.state_count_distribution.to_csv(
                comp_dir / "state_count_distribution.csv", index=False
            )
            pooled_boot.resegment_state_table.to_csv(
                comp_dir / "state_populations_resegmented.csv", index=False
            )
            resegment_tables = {
                "state_count_distribution": pooled_boot.state_count_distribution,
                "state_populations_resegmented": pooled_boot.resegment_state_table,
            }
            logger.info(
                "Re-segmentation bootstrap of the pooled landscape: "
                + ", ".join(
                    f"{int(r.n_states)} states in {r.fraction * 100:.0f}%"
                    for r in pooled_boot.state_count_distribution.itertuples()
                )
            )

    # 8. State populations per system with uncertainty and tests against the reference
    pop_comp_df = population_difference_table(
        states, frame_assignments, system_names, reference_name, boots
    )
    pop_comp_df.to_csv(comp_dir / "state_population_comparison.csv", index=False)

    state_fe_df = state_free_energy_table(states, basin_map, system_fels, reference_name, boots)
    state_fe_df.to_csv(comp_dir / "state_free_energy_comparison.csv", index=False)

    # 9. ΔΔG maps of every system against the reference on the shared grid
    ddg_maps: dict[str, dict[str, Any]] = {}
    diff_grid = None
    for name in system_names:
        if name == reference_name:
            continue
        ddg, sd = ddg_map(
            system_fels[reference_name],
            system_fels[name],
            boots.get(reference_name),
            boots.get(name),
        )
        ddg_maps[name] = {"ddg": ddg, "sd": sd, "reference": reference_name}
        np.save(comp_dir / f"fel_diff_{name}_vs_{reference_name}.npy", ddg)
        ddg_map_table(
            system_fels[reference_name], system_fels[name], ddg, sd, reference_name, name
        ).to_csv(comp_dir / f"fel_diff_{name}_vs_{reference_name}.csv", index=False)
        if diff_grid is None:
            diff_grid = ddg
    if len(system_names) == 2 and diff_grid is not None:
        np.save(comp_dir / "fel_diff.npy", diff_grid)  # backwards-compatible name

    # 10. Landscape overlap between all pairs of systems
    overlap_df = landscape_overlap_table(system_fels)
    overlap_df.to_csv(comp_dir / "fel_overlap.csv", index=False)

    # 11. Essential-subspace similarity: PCA of each system on its own, RMSIP between them
    own_eigvecs: dict[str, np.ndarray] = {}
    for name in system_names:
        try:
            _, ev, _, _, _ = compute_cartesian_pca(
                system_coords[name], weighting=config.pca_weighting, n_components=config.components
            )
            own_eigvecs[name] = ev
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Per-system PCA failed for {name}: {exc}")
    subspace_df = None
    if len(own_eigvecs) >= 2:
        subspace_df = subspace_overlap_matrix(own_eigvecs, n_modes=min(10, config.components))
        subspace_df.to_csv(comp_dir / "subspace_overlap.csv")

    summary = {
        "project": config.project,
        "reference_system": reference_name,
        "systems": system_names,
        "n_frames_per_system": {
            n: int((master_projections["system"] == n).sum()) for n in system_names
        },
        "bins": int(bins),
        "bins_auto": config.bins is None,
        "begin_ps": config.begin_ps,
        "end_ps": config.end_ps,
        "n_bootstrap": int(config.n_bootstrap),
        "bootstrap_mode": config.bootstrap_mode,
        "seed": int(config.seed),
        "environment": collect_environment(config.gmx_command),
        "states": [
            {"label": st.label, "population_percent": float(st.population * 100.0)} for st in states
        ],
        "landscape_overlap": overlap_df.to_dict(orient="records"),
        "subspace_rmsip": subspace_df.round(4).to_dict() if subspace_df is not None else None,
    }
    with (comp_dir / "comparison_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    logger.info("Shared-PCA condition comparison completed successfully.")

    return {
        "config": config,
        "pca_results": pca_results,
        "global_fel": global_fel,
        "system_fels": system_fels,
        "states": states,
        "frame_assignments": frame_assignments,
        "population_comparison": pop_comp_df,
        "state_free_energy_comparison": state_fe_df,
        "diff_grid": diff_grid,
        "ddg_maps": ddg_maps,
        "landscape_overlap": overlap_df,
        "subspace_overlap": subspace_df,
        "bootstraps": boots,
        "resegment_tables": resegment_tables,
        "reference_system": reference_name,
        "bins": bins,
        "output_dir": out_dir,
    }
