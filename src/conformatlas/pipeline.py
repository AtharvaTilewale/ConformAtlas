"""The single-system analysis pipeline behind ``conformatlas analyze``.

``run_analysis(config)`` performs every step from trajectory streaming to the HTML
report and returns an :class:`AnalysisResults`. The CLI and the YAML
configuration both construct an :class:`AnalysisConfig` and call this function.
"""

from __future__ import annotations

import datetime
import logging
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import conformatlas
from conformatlas.basins import compute_state_barriers, detect_energy_basins, save_basin_results
from conformatlas.bootstrap import bootstrap_landscape
from conformatlas.convergence import compute_convergence_diagnostics, save_convergence_results
from conformatlas.descriptors import compute_descriptors, save_descriptors
from conformatlas.features import read_cv_file, read_weights_file, stream_backbone_dihedrals
from conformatlas.fel import calculate_fel, choose_bins, save_fel_grid
from conformatlas.gromacs import GromacsRunner
from conformatlas.kinetics import compute_state_kinetics, save_kinetics
from conformatlas.models import AnalysisConfig, AnalysisResults, PCAResults, TrajectoryInfo
from conformatlas.modes import save_mode_interpretation
from conformatlas.ndx import parse_ndx
from conformatlas.pca import (
    compute_cartesian_pca,
    compute_tica,
    implied_timescales,
    project_coordinates,
    run_gromacs_pca,
    save_pca_results,
)
from conformatlas.plotting import (
    plot_basin_map,
    plot_convergence_cosine,
    plot_convergence_diagnostics,
    plot_convergence_progressive,
    plot_convergence_stability,
    plot_eigenvalues_and_variance,
    plot_fel,
    plot_fel_2d,
    plot_fel_3d,
    plot_pca_density,
    plot_pca_projections,
    plot_pca_scatter,
    plot_state_populations,
    set_cv_labels,
)
from conformatlas.plotting_extra import (
    plot_descriptors,
    plot_fel_1d,
    plot_fel_uncertainty,
    plot_mode_contributions,
    plot_state_timeseries,
)
from conformatlas.report import generate_html_report
from conformatlas.representatives import (
    extract_representative_structures,
    state_ensemble_summaries,
)
from conformatlas.timeseries import block_length_from_g, detect_equilibration
from conformatlas.trajectory import (
    HAS_MDANALYSIS,
    atom_info_for_indices,
    check_inputs,
    estimate_memory_gb,
    inspect_trajectory,
    load_reference_coordinates,
    resolve_selection_indices,
    stream_trajectory_coordinates,
)
from conformatlas.uncertainty import compute_replicate_statistics, save_uncertainty_results
from conformatlas.utils import R_GAS_CONSTANT, save_json

logger = logging.getLogger("conformatlas.pipeline")

# MDAnalysis emits advisory warnings for every PDB written without element /
# unit-cell records; they are not actionable for the user.
for _msg in (
    "Element information is missing",
    "Found no information for attr",
    "Unit cell dimensions not found",
    "Found missing chainIDs",
    "Supplied AtomGroup was missing",
):
    warnings.filterwarnings("ignore", message=f".*{_msg}.*")

Echo = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def run_analysis(config: AnalysisConfig, echo: Echo = _noop) -> AnalysisResults:
    """Run the complete single-system analysis described by ``config``."""
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings_: list[str] = []
    extras: dict[str, Any] = {}
    structure = config.topology
    trajectories = list(config.trajectories)
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    rep_ids = [f"Rep_{i + 1}" for i in range(len(trajectories))]
    traj_map = dict(zip(rep_ids, trajectories))

    # ------------------------------------------------------------------ GROMACS
    gmx_runner = None
    gmx_version = "None"
    try:
        gmx_runner = GromacsRunner(config.gmx_command)
        gmx_version = gmx_runner.version
        echo(f"GROMACS found: {gmx_runner.gmx_path} ({gmx_version})")
    except Exception as e:
        if config.fel_backend == "gromacs" or config.pca_backend == "gromacs":
            raise RuntimeError(f"GROMACS backend requested but no executable found: {e}") from e

    use_python = HAS_MDANALYSIS and config.pca_backend != "gromacs"
    if not use_python and gmx_runner is None:
        raise RuntimeError("Either MDAnalysis or GROMACS must be installed.")
    if config.pca_backend == "python" and not HAS_MDANALYSIS:
        raise RuntimeError("--pca-backend python requires MDAnalysis.")

    # ------------------------------------------------------------- inspection
    traj_infos: list[TrajectoryInfo] = []
    for rep_id, t_path in traj_map.items():
        info = inspect_trajectory(
            structure, t_path, rep_id, config.system_name, gmx_runner=gmx_runner
        )
        traj_infos.append(info)
        echo(
            f"  • {rep_id}: {info.n_frames} frames ({info.start_time_ps:.1f} to {info.end_time_ps:.1f} ps)"
        )

    if use_python:
        for w in check_inputs(structure, trajectories):
            warnings_.append(w)
            echo(f"  WARNING: {w}")

    # ---------------------------------------------------------- selections
    ndx_groups = parse_ndx(config.index_file) if config.index_file else None
    analysis_idx = fit_idx = None
    atom_info = None
    ligand_mask = None
    protein_mask = None
    if use_python:
        analysis_idx = resolve_selection_indices(
            structure, config.cov_group, ndx_groups, extra_group=config.lig_group
        )
        fit_idx = resolve_selection_indices(structure, config.lsq_group, ndx_groups)
        atom_info = atom_info_for_indices(structure, analysis_idx)
        if config.lig_group:
            lig_idx = resolve_selection_indices(structure, config.lig_group, ndx_groups)
            ligand_mask = np.isin(analysis_idx, lig_idx)
            protein_mask = ~ligand_mask
            echo(
                f"  • Analysis group: {config.cov_group} + {config.lig_group} "
                f"({int(protein_mask.sum())} + {int(ligand_mask.sum())} atoms); fit group: "
                f"{config.lsq_group} ({len(fit_idx)} atoms)"
            )
        else:
            echo(
                f"  • Analysis group: {config.cov_group} ({len(analysis_idx)} atoms); "
                f"fit group: {config.lsq_group} ({len(fit_idx)} atoms)"
            )
        n_frames_total = sum(max(1, info.n_frames // max(1, config.stride)) for info in traj_infos)
        mem_gb = estimate_memory_gb(n_frames_total, len(analysis_idx))
        if mem_gb > 8:
            warnings_.append(
                f"Holding {n_frames_total} frames x {len(analysis_idx)} atoms needs about {mem_gb:.1f} GB; "
                "consider --stride or a smaller --cov-group."
            )
            echo(f"  WARNING: {warnings_[-1]}")

    # ----------------------------------------------------------- streaming
    echo("Step 1: Streaming trajectories" + (" with superposition" if config.fit else ""))
    coords_dict: dict[str, np.ndarray] = {}
    times_dict: dict[str, np.ndarray] = {}
    fit_rmsd_dict: dict[str, np.ndarray] = {}
    reference_fit = reference_analysis = None
    if use_python:
        if config.fit:
            reference_fit = load_reference_coordinates(
                structure, atom_indices=fit_idx, fallback_trajectory=trajectories[0]
            )
        reference_analysis = load_reference_coordinates(
            structure, atom_indices=analysis_idx, fallback_trajectory=trajectories[0]
        )
        if config.fit:
            # Express the analysis reference in the fitted frame of reference.
            from conformatlas.superpose import superpose_frame

            ref_fit_raw = load_reference_coordinates(
                structure, atom_indices=fit_idx, fallback_trajectory=trajectories[0]
            )
            reference_analysis = superpose_frame(ref_fit_raw, reference_analysis, reference_fit)
        for rep_id, t_path in traj_map.items():
            coords, times, fit_rmsd = stream_trajectory_coordinates(
                structure,
                t_path,
                atom_indices=analysis_idx,
                stride=config.stride,
                fit=config.fit,
                fit_atom_indices=fit_idx,
                reference_fit_coords=reference_fit,
                mass_weighted=config.mass_weighted_fit,
                begin_ps=config.begin_ps,
                end_ps=config.end_ps,
                return_fit_rmsd=True,
            )
            coords_dict[rep_id] = coords
            times_dict[rep_id] = times
            fit_rmsd_dict[rep_id] = fit_rmsd
            echo(f"  • {rep_id}: {len(times)} frames kept ({times[0]:.0f}-{times[-1]:.0f} ps)")

    # -------------------------------------------------------- equilibration
    equil_rows = []
    if coords_dict:
        for rep_id in rep_ids:
            series = (
                fit_rmsd_dict[rep_id]
                if config.fit
                else _rg_series(coords_dict[rep_id], protein_mask)
            )
            res = detect_equilibration(series)
            t = times_dict[rep_id]
            t0_ps = float(t[res.t0_index]) if res.t0_index < len(t) else float(t[-1])
            equil_rows.append(
                {
                    "Replicate": rep_id,
                    "observable": "fit-group RMSD" if config.fit else "Rg",
                    "t0_index": res.t0_index,
                    "t0_ps": t0_ps,
                    "frames_discarded": res.t0_index if config.equilibration == "auto" else 0,
                    "g_production": res.g,
                    "N_eff_production": res.n_eff,
                    "N_eff_full": res.n_eff_full,
                    "applied": config.equilibration == "auto",
                }
            )
            if config.equilibration == "auto" and res.t0_index > 0:
                coords_dict[rep_id] = coords_dict[rep_id][res.t0_index :]
                times_dict[rep_id] = t[res.t0_index :]
                fit_rmsd_dict[rep_id] = fit_rmsd_dict[rep_id][res.t0_index :]
                echo(f"  • {rep_id}: discarded first {res.t0_index} frames (t0 = {t0_ps:.0f} ps)")
            elif res.t0_index > 0:
                frac = res.t0_index / max(1, res.n_total)
                msg = (
                    f"{rep_id}: equilibration detection suggests discarding the first {res.t0_index} frames "
                    f"({frac * 100:.0f}%, t0 = {t0_ps:.0f} ps); run with --equilibration auto or -b {t0_ps:.0f} to apply."
                )
                warnings_.append(msg)
                echo(f"  NOTE: {msg}")
    equil_df = pd.DataFrame(equil_rows) if equil_rows else None

    # ------------------------------------------------------- decomposition
    echo(
        f"Step 2: {config.pca_type.upper() if config.pca_type != 'cartesian' else 'Cartesian'} decomposition"
    )
    feature_dict = coords_dict
    feature_labels = None
    mode_prefix = "PC"
    dt_ps = _median_dt(times_dict) if times_dict else 1.0
    if use_python and config.pca_type == "dihedral":
        feature_dict = {}
        for rep_id, t_path in traj_map.items():
            feats, ftimes, feature_labels = stream_backbone_dihedrals(
                structure,
                t_path,
                stride=config.stride,
                begin_ps=config.begin_ps,
                end_ps=config.end_ps,
            )
            keep = np.isin(np.round(ftimes, 3), np.round(times_dict[rep_id], 3))
            feature_dict[rep_id] = feats[keep]
            if feature_dict[rep_id].shape[0] != len(times_dict[rep_id]):
                raise RuntimeError(
                    f"Dihedral features and coordinates differ in frame count for {rep_id}."
                )
        mode_prefix = "dPC"

    if use_python:
        if config.pca_type == "tica":
            evals, evecs, exp_var, cum_var, mean_struct = compute_tica(
                feature_dict,
                lag=config.tica_lag,
                n_components=config.n_components,
                weighting=config.pca_weighting,
            )
            mode_prefix = "IC"
            extras["tica_timescales_ps"] = implied_timescales(evals, config.tica_lag, dt_ps)
        else:
            evals, evecs, exp_var, cum_var, mean_struct = compute_cartesian_pca(
                feature_dict, weighting=config.pca_weighting, n_components=config.n_components
            )
        proj_dfs = []
        for rep_id, feats in feature_dict.items():
            proj = project_coordinates(feats, mean_struct, evecs)
            df = pd.DataFrame(proj, columns=[f"PC{i + 1}" for i in range(proj.shape[1])])
            df.insert(0, "system", config.system_name)
            df.insert(1, "replicate", rep_id)
            df.insert(2, "frame", np.arange(len(times_dict[rep_id])))
            df.insert(3, "time_ps", times_dict[rep_id])
            proj_dfs.append(df)
        projections = pd.concat(proj_dfs, ignore_index=True)
        pca_results = PCAResults(
            eigenvalues=evals,
            eigenvectors=evecs,
            explained_variance=exp_var,
            cumulative_variance=cum_var,
            projections=projections,
            mean_coords=mean_struct,
        )
    else:
        pca_results = run_gromacs_pca(
            gmx_runner,
            Path(structure),
            [Path(p) for p in trajectories],
            lsq_group=config.lsq_group,
            cov_group=config.cov_group,
            output_dir=out_dir,
            n_components=config.n_components,
        )
        projections = pca_results.projections
        for rep_id, df in projections.groupby("replicate", sort=False):
            times_dict[str(rep_id)] = df["time_ps"].to_numpy()
    save_pca_results(pca_results, out_dir)
    if config.pca_type == "tica":
        pd.DataFrame(
            {
                "Mode": [f"IC{i + 1}" for i in range(len(evals))],
                "Eigenvalue": evals,
                "Implied_timescale_ps": extras["tica_timescales_ps"],
            }
        ).to_csv(out_dir / "pca" / "tica_timescales.csv", index=False)

    n_modes_avail = len(pca_results.explained_variance)
    top2 = (
        float(pca_results.cumulative_variance[min(1, n_modes_avail - 1)]) if n_modes_avail else 0.0
    )
    kind = "kinetic variance" if config.pca_type == "tica" else "variance"
    echo(f"  • Leading two modes explain {top2:.1f}% of the {kind}")
    if config.pca_type != "tica" and top2 < 50.0:
        warnings_.append(
            f"PC1+PC2 explain only {top2:.1f}% of the variance; a 2-D landscape on them captures a minority "
            "of the motion. Inspect the scree plot and consider dihedral PCA, TICA or user CVs."
        )
        echo(f"  WARNING: {warnings_[-1]}")

    # ----------------------------------------------- collective variables
    echo("Step 3: Collective variables and free energy landscape")
    cv_df = projections.copy()
    descriptors = None
    if coords_dict and reference_analysis is not None:
        descriptors = compute_descriptors(
            coords_dict,
            times_dict,
            reference_analysis,
            atom_info,
            ligand_mask,
            masses=atom_info["mass"].to_numpy()
            if atom_info is not None and np.isfinite(atom_info["mass"]).all()
            else None,
        )
        save_descriptors(descriptors, out_dir)
        extras["descriptors"] = descriptors

    i1, i2 = config.fel_pcs
    if config.cv == "pc":
        c1, c2 = f"PC{i1}", f"PC{i2}"
        if c1 not in cv_df.columns or c2 not in cv_df.columns:
            raise ValueError(
                f"Requested components {config.fel_pcs} but only {n_modes_avail} were retained."
            )
        cv_df["CV1"], cv_df["CV2"] = cv_df[c1], cv_df[c2]
        cv_names = (f"{mode_prefix}{i1}", f"{mode_prefix}{i2}")
        if config.pca_type == "dihedral":
            cv_unit = ""
        elif not use_python:
            cv_unit = "nm"  # gmx anaeig projections are in nm
        else:
            cv_unit = "Å"
    elif config.cv == "rmsd-rg":
        if descriptors is None:
            raise RuntimeError("--cv rmsd-rg requires the MDAnalysis backend.")
        ts = descriptors["timeseries"]
        cv_df["CV1"], cv_df["CV2"] = ts["RMSD"].to_numpy(), ts["Rg"].to_numpy()
        cv_names = ("RMSD", "Rg")
        cv_unit = "Å"
    else:
        if len(config.cv_files) != len(trajectories):
            raise ValueError("--cv-file must be given once per trajectory, in the same order.")
        parts = []
        for rep_id, f in zip(rep_ids, config.cv_files):
            cv, t_file = read_cv_file(f, columns=config.cv_columns)
            t_rep = times_dict[rep_id]
            if len(cv) != len(t_rep):
                if t_file is None:
                    raise ValueError(
                        f"{f}: {len(cv)} rows but {len(t_rep)} analysed frames for {rep_id}, and no time column to align on."
                    )
                sel = np.searchsorted(t_file, t_rep)
                sel = np.clip(sel, 0, len(t_file) - 1)
                if np.max(np.abs(t_file[sel] - t_rep)) > 0.5 * dt_ps:
                    raise ValueError(f"{f}: times do not match the analysed frames of {rep_id}.")
                cv = cv[sel]
            parts.append(cv)
        allcv = np.vstack(parts)
        cv_df["CV1"], cv_df["CV2"] = allcv[:, 0], allcv[:, 1]
        cv_names = config.cv_labels or ("CV1", "CV2")
        cv_unit = ""
    set_cv_labels(cv_names[0], cv_names[1], cv_unit)
    cv1 = cv_df["CV1"].to_numpy()
    cv2 = cv_df["CV2"].to_numpy()

    weights = None
    if config.weights_files:
        if len(config.weights_files) != len(trajectories):
            raise ValueError("--weights-file must be given once per trajectory.")
        parts = []
        for rep_id, f in zip(rep_ids, config.weights_files):
            w = read_weights_file(f)
            n_rep = len(times_dict[rep_id])
            if len(w) != n_rep:
                raise ValueError(f"{f}: {len(w)} weights for {n_rep} analysed frames of {rep_id}.")
            parts.append(w)
        weights = np.concatenate(parts)
        echo(
            "  • Per-frame weights supplied: populations and ΔG are reweighted; kinetics are skipped"
        )

    bins = config.bins if config.bins is not None else choose_bins(cv1, cv2)
    echo(f"  • Grid: {bins} x {bins} bins" + (" (automatic)" if config.bins is None else ""))
    fel_grid = calculate_fel(
        cv1,
        cv2,
        temperature=config.temperature,
        bins=bins,
        smooth_sigma=config.smooth_sigma,
        weights=weights,
    )
    save_fel_grid(fel_grid, out_dir)

    # --------------------------------------------------------------- basins
    echo("Step 4: Basin detection")
    states, unassigned_pop, frame_assignments, basin_map = detect_energy_basins(
        fel_grid,
        cv_df,
        min_state_population=config.min_state_population,
        max_states=config.max_states,
        min_barrier=config.min_barrier,
        weights=weights,
        cv_columns=("CV1", "CV2"),
    )
    save_basin_results(states, frame_assignments, basin_map, out_dir)
    barriers = compute_state_barriers(basin_map, fel_grid.free_energy, states)
    barriers.to_csv(out_dir / "states" / "barriers.csv", index=False)
    extras["barriers"] = barriers
    for st in states:
        echo(
            f"  • {st.label}: {st.population * 100:.1f}% | minimum at {cv_names[0]}={st.min_pc1:.2f}, "
            f"{cv_names[1]}={st.min_pc2:.2f} (ΔG = {st.min_free_energy:.2f} kJ/mol)"
        )
    if unassigned_pop > 0:
        echo(f"  • Minor / transitional: {unassigned_pop * 100:.1f}%")

    # -------------------------------------------------------- convergence
    echo("Step 5: Sampling and convergence diagnostics")
    conv_input = feature_dict if feature_dict else {"Rep_1": np.zeros((len(projections), 1, 3))}
    conv = compute_convergence_diagnostics(
        conv_input, projections, frame_assignments, n_modes=config.convergence_modes
    )
    conv.equilibration = equil_df
    save_convergence_results(conv, out_dir)
    echo(
        f"  • Half-vs-half RMSIP: {conv.rmsip_halves:.2f} | PC1 cosine content: {conv.cosine_content.get(1, 0.0):.2f}"
    )
    echo(f"  • Assessment: {conv.sampling_assessment}")

    g_by_rep = {
        rep: max(v.values()) if v else 1.0 for rep, v in conv.statistical_inefficiency.items()
    }
    block_lengths = {
        rep: block_length_from_g(g_by_rep.get(rep, 1.0), len(times_dict[rep])) for rep in times_dict
    }

    # ---------------------------------------------------------- uncertainty
    echo("Step 6: Uncertainty")
    g_single = max(g_by_rep.values()) if g_by_rep else None
    summary_pop_df, detailed_pop_df, unc_type = compute_replicate_statistics(
        frame_assignments, statistical_inefficiency=g_single, weights=weights
    )
    boot = None
    if config.n_bootstrap > 0 and states:
        boot = bootstrap_landscape(
            cv1,
            cv2,
            frame_assignments["replicate"].to_numpy(),
            fel_grid,
            basin_map,
            states,
            block_lengths,
            n_boot=config.n_bootstrap,
            weights=weights,
            seed=config.seed,
        )
        extras["bootstrap"] = boot
        np.save(out_dir / "fel" / "free_energy_bootstrap_sd.npy", boot.fe_std)
        np.save(out_dir / "fel" / "free_energy_bootstrap_ci_low.npy", boot.fe_ci_low)
        np.save(out_dir / "fel" / "free_energy_bootstrap_ci_high.npy", boot.fe_ci_high)
        boot.state_table.to_csv(out_dir / "states" / "state_free_energies.csv", index=False)
        boot.barrier_table.to_csv(out_dir / "states" / "barriers.csv", index=False)
        extras["barriers"] = boot.barrier_table
        # merge bootstrap CIs into the population summary
        if len(summary_pop_df):
            ci = boot.state_table.set_index("state")
            summary_pop_df["Bootstrap_CI_95_Lower"] = summary_pop_df["State"].map(
                ci["population_ci95_low"] * 100.0
            )
            summary_pop_df["Bootstrap_CI_95_Upper"] = summary_pop_df["State"].map(
                ci["population_ci95_high"] * 100.0
            )
        for _, r in boot.state_table.iterrows():
            echo(
                f"  • {r['state']}: population {r['population'] * 100:.1f}% "
                f"[{r['population_ci95_low'] * 100:.1f}, {r['population_ci95_high'] * 100:.1f}] | "
                f"ΔG vs {r['reference_state']} = {r['dG_state_kJ_mol']:.2f} "
                f"[{r['dG_state_ci95_low']:.2f}, {r['dG_state_ci95_high']:.2f}] kJ/mol"
            )
    save_uncertainty_results(summary_pop_df, detailed_pop_df, out_dir)
    echo(
        f"  • Population uncertainty: {unc_type}"
        + (f"; bootstrap x{config.n_bootstrap}" if boot else "")
    )

    # ------------------------------------------------------------ kinetics
    kinetics = None
    if weights is None and states:
        kinetics = compute_state_kinetics(frame_assignments, [s.label for s in states], dt_ps=dt_ps)
        save_kinetics(kinetics, out_dir)
        extras["kinetics"] = kinetics
        ds = kinetics["dwell_summary"]
        for _, r in ds.iterrows():
            life = r["lifetime_from_rate_ps"]
            life_s = f"{life / 1000:.2f} ns" if np.isfinite(life) else "no exits observed"
            echo(
                f"  • {r['state']}: {int(r['visits'])} visits, {int(r['transitions_out'])} exits, lifetime {life_s}"
            )

    # --------------------------------------------------------- structures
    echo("Step 7: Representative structures and state ensembles")
    extract_representative_structures(
        states,
        {config.system_name: Path(structure)},
        traj_map,
        output_dir=out_dir,
        gmx_runner=gmx_runner,
        system_name=config.system_name,
        frame_assignments=frame_assignments,
        n_representatives=config.n_representatives,
        method=config.representative_method,
        cv_columns=("CV1", "CV2"),
        cv_names=cv_names,
    )
    state_summary = None
    if coords_dict and atom_info is not None:
        state_summary = state_ensemble_summaries(
            states,
            frame_assignments,
            coords_dict,
            reference_analysis,
            atom_info,
            out_dir,
            topology=structure,
            atom_indices=analysis_idx,
            protein_mask=protein_mask,
        )
        extras["state_summary"] = state_summary
    mode_files = {}
    if coords_dict and atom_info is not None and config.pca_type != "dihedral":
        mode_files = save_mode_interpretation(
            out_dir,
            structure,
            analysis_idx,
            atom_info,
            pca_results.mean_coords,
            pca_results.eigenvectors,
            pca_results.eigenvalues,
            projections,
            n_modes=3,
            trajectory=trajectories[0],
        )
    elif config.pca_type == "dihedral" and feature_labels is not None:
        lab = feature_labels.copy()
        for j in range(min(3, pca_results.eigenvectors.shape[1])):
            lab[f"PC{j + 1}"] = np.abs(pca_results.eigenvectors[:, j])
        lab.to_csv(out_dir / "pca" / "mode_contributions_features.csv", index=False)

    # -------------------------------------------------------------- figures
    echo("Step 8: Figures")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "pca_variance": plot_eigenvalues_and_variance(
            pca_results, fig_dir / "explained_variance.png"
        ),
        "pca_scatter": plot_pca_scatter(projections, fig_dir / "pca_scatter.png"),
        "pca_density": plot_pca_density(projections, fig_dir / "pca_density.png"),
        "fel_2d": plot_fel_2d(fel_grid, states=states, out_path=fig_dir / "fel_2d.png"),
        "fel_3d": plot_fel_3d(fel_grid, out_path=fig_dir / "fel_3d.png"),
        "basin_map": plot_basin_map(fel_grid, basin_map, states, fig_dir / "basin_map.png"),
        "populations": plot_state_populations(
            summary_pop_df, detailed_pop_df, fig_dir / "state_populations.png"
        ),
        "convergence_progressive": plot_convergence_progressive(
            conv, fig_dir / "convergence_progressive.png"
        ),
        "convergence_cosine": plot_convergence_cosine(conv, fig_dir / "convergence_cosine.png"),
        "convergence_stability": plot_convergence_stability(
            conv, fig_dir / "convergence_stability.png"
        ),
    }
    plot_fel(fel_grid, states=states, out_path=fig_dir / "fel_2d_3d.png")
    plot_pca_projections(projections, fig_dir / "pca.png")
    figures["convergence"] = plot_convergence_diagnostics(conv, fig_dir / "convergence.png")
    reps_arr = frame_assignments["replicate"].to_numpy()
    figures["fel_1d"] = plot_fel_1d(
        cv1,
        cv2,
        reps_arr,
        config.temperature,
        fig_dir / "fel_1d.png",
        bins=max(20, bins),
        block_lengths=block_lengths,
        weights=weights,
        n_boot=min(config.n_bootstrap, 200) or 50,
        seed=config.seed,
    )
    if boot is not None:
        figures["fel_uncertainty"] = plot_fel_uncertainty(
            fel_grid, boot.fe_std, fig_dir / "fel_uncertainty.png", states
        )
    if descriptors is not None:
        figures["descriptors"] = plot_descriptors(
            descriptors, fig_dir / "descriptors.png", equil_df
        )
    if kinetics is not None:
        figures["state_timeseries"] = plot_state_timeseries(
            frame_assignments, states, fig_dir / "state_timeseries.png", kinetics["dwell_times"]
        )
    if "contributions_residues" in mode_files:
        res_df = pd.read_csv(mode_files["contributions_residues"])
        figures["mode_contributions"] = plot_mode_contributions(
            res_df, fig_dir / "mode_contributions.png"
        )

    # ------------------------------------------------------------ metadata
    echo("Step 9: Report")
    run_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "conformatlas_version": conformatlas.__version__,
        "temperature_K": config.temperature,
        "bins": bins,
        "bins_mode": "auto (Freedman-Diaconis)" if config.bins is None else "user",
        "smooth_sigma": config.smooth_sigma,
        "stride": config.stride,
        "time_window_ps": [config.begin_ps, config.end_ps],
        "equilibration": config.equilibration,
        "pca_type": config.pca_type,
        "tica_lag_frames": config.tica_lag if config.pca_type == "tica" else None,
        "pca_weighting": config.pca_weighting,
        "pca_backend": "python" if use_python else "gromacs",
        "collective_variables": {
            "mode": config.cv,
            "names": list(cv_names),
            "fel_pcs": list(config.fel_pcs),
        },
        "weights_files": [str(p) for p in config.weights_files],
        "lsq_group": config.lsq_group,
        "cov_group": config.cov_group,
        "lig_group": config.lig_group,
        "index_file": str(config.index_file) if config.index_file else None,
        "min_barrier_kJ_mol": config.min_barrier
        if config.min_barrier is not None
        else R_GAS_CONSTANT * config.temperature,
        "superposition": (
            {
                "enabled": True,
                "fit_group": config.lsq_group,
                "reference": str(Path(structure).resolve()),
                "mass_weighted": config.mass_weighted_fit,
            }
            if (config.fit and use_python)
            else {"enabled": False, "note": "gmx covar -fit" if not use_python else "--no-fit"}
        ),
        "bootstrap": {
            "n": config.n_bootstrap,
            "block_lengths_frames": block_lengths,
            "seed": config.seed,
        },
        "representatives": {"n": config.n_representatives, "method": config.representative_method},
        "topology": str(Path(structure).resolve()),
        "trajectories": [str(Path(p).resolve()) for p in trajectories],
        "uncertainty_type": unc_type,
        "gromacs_version": gmx_version,
        "warnings": warnings_,
    }
    save_json(run_meta, out_dir / "run_metadata.json")

    results = AnalysisResults(
        config=config,
        trajectories_info=traj_infos,
        pca=pca_results,
        fel=fel_grid,
        states=states,
        unassigned_population=unassigned_pop,
        frame_assignments=frame_assignments,
        convergence=conv,
        replicate_statistics=summary_pop_df,
        output_dir=out_dir,
        run_metadata=run_meta,
        extras=extras,
        warnings=warnings_,
    )

    summary = {
        "system": config.system_name,
        "temperature_K": config.temperature,
        "total_frames": int(len(frame_assignments)),
        "collective_variables": list(cv_names),
        "states": [
            {
                "label": s.label,
                "population_percent": s.population * 100.0,
                "min_cv1": s.min_pc1,
                "min_cv2": s.min_pc2,
                "min_free_energy_kJ_mol": s.min_free_energy,
                "representative_replicate": s.representative_replicate,
                "representative_time_ps": s.representative_time_ps,
            }
            for s in states
        ],
        "unassigned_population_percent": unassigned_pop * 100.0,
        "convergence": {
            "rmsip_halves": conv.rmsip_halves,
            "pc1_cosine_content": conv.cosine_content.get(1, 0.0),
            "n_eff_total": float(sum(conv.n_eff.values())) if conv.n_eff else None,
            "assessment": conv.sampling_assessment,
        },
        "warnings": warnings_,
    }
    if boot is not None:
        summary["state_free_energies"] = boot.state_table.to_dict("records")
        summary["barriers"] = boot.barrier_table.to_dict("records")
    save_json(summary, out_dir / "summary.json")

    sections = _report_sections(
        results, figures, cv_names, equil_df, boot, kinetics, descriptors, state_summary, mode_files
    )
    generate_html_report(
        results, figures, gromacs_version=gmx_version, extra_sections=sections, pdf=config.pdf
    )
    echo(f"Report: {out_dir / 'report.html'}")
    return results


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _rg_series(coords: np.ndarray, protein_mask: np.ndarray | None) -> np.ndarray:
    from conformatlas.descriptors import radius_of_gyration

    return radius_of_gyration(coords, protein_mask)


def _median_dt(times_dict: dict[str, np.ndarray]) -> float:
    diffs = (
        np.concatenate([np.diff(t) for t in times_dict.values() if len(t) > 1])
        if times_dict
        else np.array([])
    )
    return float(np.median(diffs)) if len(diffs) else 1.0


def _report_sections(
    results, figures, cv_names, equil_df, boot, kinetics, descriptors, state_summary, mode_files
):
    sections = []
    if descriptors is not None:
        sections.append(
            {
                "title": "Ensemble descriptors",
                "paragraphs": [
                    "RMSD of the analysis group to the reference after superposition, radius of gyration and "
                    "per-residue RMSF. Dashed vertical lines mark the equilibration point suggested by the "
                    "Chodera maximum-effective-samples criterion on the fit-group RMSD."
                ],
                "figure": figures.get("descriptors"),
                "table": descriptors["summary"],
            }
        )
    if equil_df is not None:
        sections.append(
            {
                "title": "Equilibration and effective sample size",
                "paragraphs": [
                    "t0 is the production start that maximises the number of effectively independent frames "
                    "(N_eff = N / g, g = statistical inefficiency). 'applied' states whether the transient was discarded."
                ],
                "table": equil_df,
            }
        )
    sections.append(
        {
            "title": f"One-dimensional profiles and landscape uncertainty ({cv_names[0]}, {cv_names[1]})",
            "paragraphs": [
                "Free energy along each collective variable with 95% block-bootstrap bands; the right-hand map "
                "shows the bootstrap standard deviation of ΔG(x, y) on the 2-D grid."
                if boot
                else "Free energy along each collective variable with 95% block-bootstrap bands."
            ],
            "figure": figures.get("fel_1d"),
        }
    )
    if boot is not None:
        sections.append(
            {
                "title": "Landscape uncertainty map",
                "paragraphs": boot.notes,
                "figure": figures.get("fel_uncertainty"),
            }
        )
        sections.append(
            {
                "title": "State free energies and barriers (bootstrap 95% CI)",
                "paragraphs": [
                    "ΔG_state = −RT ln(P_state / P_reference) relative to the most populated state; min ΔG is the "
                    "basin minimum on the landscape. Barriers are measured from each state's minimum to the lowest "
                    "saddle on the shared boundary of two adjacent basins."
                ],
                "table": boot.state_table,
            }
        )
        if len(boot.barrier_table):
            sections.append(
                {
                    "title": "Barriers between adjacent states",
                    "paragraphs": [],
                    "table": boot.barrier_table,
                }
            )
    if kinetics is not None:
        sections.append(
            {
                "title": "State kinetics",
                "paragraphs": [
                    "Transitions are counted between consecutive major-state frames of the same replicate "
                    "(unassigned frames skipped). Dwell times exclude the censored first and last visit of each "
                    "replicate; the lifetime from the exit rate uses all visits. Mean first-passage times assume "
                    "a Markov chain at lag one frame and are indicative only."
                ],
                "figure": figures.get("state_timeseries"),
                "table": kinetics["dwell_summary"],
            }
        )
    if state_summary is not None and len(state_summary):
        sections.append(
            {
                "title": "Per-state ensemble summary",
                "paragraphs": [
                    "Average structure, RMSF and compactness of the frames assigned to each state."
                ],
                "table": state_summary,
            }
        )
    if "contributions_residues" in mode_files:
        sections.append(
            {
                "title": "What the leading modes are",
                "paragraphs": [
                    "Per-residue weight of each eigenvector. Mode vectors are exported as pca/modes.nmd (VMD NMWiz / "
                    "ProDy) and as interpolated extreme structures pca/PC*_extremes.pdb; pca/average.pdb is the mean structure."
                ],
                "figure": figures.get("mode_contributions"),
            }
        )
    if results.warnings:
        sections.append(
            {"title": "Warnings raised during the run", "paragraphs": list(results.warnings)}
        )
    return sections
