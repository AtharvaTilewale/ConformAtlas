"""The single-system analysis pipeline behind ``conformatlas analyze``.

``run_analysis(config)`` performs every step from trajectory streaming to the HTML
report and returns an :class:`AnalysisResults`. The CLI and the YAML
configuration both construct an :class:`AnalysisConfig` and call this function.

Extensions: a chunked *streaming* PCA engine (``config.engine``), equilibration
detection on the landscape CVs (``equilibration_on='cv'``), the re-segmentation
bootstrap, a Markov-state-model layer (``config.msm``), residue-level analyses
(``config.residue_analysis``), side-chain / nucleic dihedral sets, an HDF5 results
container (``save_hdf5``) and the ``from_results`` route that re-analyses an
existing output directory (rebin / resegment / MSM) without reading trajectories.
"""

from __future__ import annotations

import datetime
import logging
import shutil
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
from conformatlas.errors import MissingDependencyError
from conformatlas.features import read_cv_file, read_weights_file, stream_dihedrals
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
    get_figure_style,
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
    set_figure_style,
)
from conformatlas.plotting_extra import (
    plot_descriptors,
    plot_fel_1d,
    plot_fel_uncertainty,
    plot_mode_contributions,
    plot_state_timeseries,
)
from conformatlas.progress import track
from conformatlas.provenance import collect_environment
from conformatlas.report import generate_html_report, set_report_options
from conformatlas.representatives import (
    extract_representative_structures,
    state_ensemble_summaries,
)
from conformatlas.timeseries import (
    block_length_from_g,
    detect_equilibration,
    detect_equilibration_multi,
)
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


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------


def _apply_figure_options(config: AnalysisConfig) -> None:
    """Push the figure / report options of ``config`` into the module-level style.

    Fields left at ``None`` keep the active style (so API users may wrap
    ``run_analysis`` in :func:`conformatlas.plotting.figure_style`).
    """
    set_figure_style(
        fig_format=config.fig_format,
        colormap=config.colormap,
        font_scale=config.font_scale,
        dpi=config.fig_dpi,
        path_overlay=config.path_overlay,
    )
    opts = {
        k: v
        for k, v in {
            "interactive": config.interactive,
            "plotly_js": config.plotly_js,
            "path_overlay": config.path_overlay,
        }.items()
        if v is not None
    }
    if opts:
        set_report_options(**opts)


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


def _equilibration_row(rep_id, observable, res, t, applied, deciding=None):
    t0_ps = float(t[res.t0_index]) if res.t0_index < len(t) else float(t[-1])
    row = {
        "Replicate": rep_id,
        "observable": observable,
        "t0_index": int(res.t0_index),
        "t0_ps": t0_ps,
        "frames_discarded": int(res.t0_index) if applied else 0,
        "g_production": res.g,
        "N_eff_production": res.n_eff,
        "N_eff_full": res.n_eff_full,
        "applied": bool(applied),
    }
    if deciding is not None:
        row["deciding_observable"] = deciding
    return row


def _detect_cv_transients(
    config: AnalysisConfig,
    rep_ids: list[str],
    cv_df: pd.DataFrame,
    cv_names: tuple[str, str],
    times_dict: dict[str, np.ndarray],
) -> tuple[dict[str, int], list[dict]]:
    """Chodera criterion on both landscape CVs per replicate (latest t0 adopted)."""
    t0: dict[str, int] = {}
    rows: list[dict] = []
    reps = cv_df["replicate"].to_numpy()
    for rep_id in rep_ids:
        m = reps == rep_id
        if m.sum() < 4:
            continue
        res_multi = detect_equilibration_multi(
            {
                cv_names[0]: cv_df.loc[m, "CV1"].to_numpy(dtype=float),
                cv_names[1]: cv_df.loc[m, "CV2"].to_numpy(dtype=float),
            }
        )
        res = res_multi.per_observable[res_multi.deciding_observable]
        t0[rep_id] = int(res_multi.t0_index)
        rows.append(
            _equilibration_row(
                rep_id,
                f"{cv_names[0]}+{cv_names[1]} (latest t0)",
                res,
                times_dict[rep_id],
                config.equilibration == "auto",
                deciding=res_multi.deciding_observable,
            )
        )
    return t0, rows


def _decompose(
    config: AnalysisConfig,
    structure: Path,
    traj_map: dict[str, Path],
    coords_dict: dict[str, np.ndarray],
    times_dict: dict[str, np.ndarray],
    extras: dict[str, Any],
    dt_ps: float,
) -> tuple[PCAResults, pd.DataFrame, str, pd.DataFrame | None, dict[str, np.ndarray]]:
    """Cartesian / dihedral PCA or TICA on the in-memory feature arrays."""
    feature_dict = coords_dict
    feature_labels = None
    mode_prefix = "PC"
    if config.pca_type == "dihedral":
        feature_dict = {}
        selection = "nucleic" if config.dihedral_set == "nucleic" else "protein"
        for rep_id, t_path in traj_map.items():
            feats, ftimes, feature_labels, _angles = stream_dihedrals(
                structure,
                t_path,
                selection=selection,
                dihedral_set=config.dihedral_set,
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
    return pca_results, projections, mode_prefix, feature_labels, feature_dict


def _assemble_cvs(
    config: AnalysisConfig,
    projections: pd.DataFrame,
    times_dict: dict[str, np.ndarray],
    rep_ids: list[str],
    mode_prefix: str,
    use_python: bool,
    descriptors: dict | None,
    dt_ps: float,
) -> tuple[pd.DataFrame, tuple[str, str], str]:
    """Add the ``CV1``/``CV2`` columns to a copy of the projection table."""
    cv_df = projections.copy()
    n_modes_avail = sum(c.startswith("PC") for c in projections.columns)
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
            raise RuntimeError(
                "--cv rmsd-rg requires the MDAnalysis backend with the in-memory engine."
            )
        ts = descriptors["timeseries"]
        cv_df["CV1"], cv_df["CV2"] = ts["RMSD"].to_numpy(), ts["Rg"].to_numpy()
        cv_names = ("RMSD", "Rg")
        cv_unit = "Å"
    else:
        if len(config.cv_files) != len(rep_ids):
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
    return cv_df, tuple(cv_names), cv_unit


def _read_weights(config: AnalysisConfig, rep_ids, times_dict) -> np.ndarray | None:
    if not config.weights_files:
        return None
    if len(config.weights_files) != len(rep_ids):
        raise ValueError("--weights-file must be given once per trajectory.")
    parts = []
    for rep_id, f in zip(rep_ids, config.weights_files):
        w = read_weights_file(f)
        n_rep = len(times_dict[rep_id])
        if len(w) != n_rep:
            raise ValueError(f"{f}: {len(w)} weights for {n_rep} analysed frames of {rep_id}.")
        parts.append(w)
    return np.concatenate(parts)


def _ligand_selection(config: AnalysisConfig, lig_idx: np.ndarray | None) -> str | None:
    """MDAnalysis selection of the ligand for the contact analysis."""
    if config.ligand_selection:
        return config.ligand_selection
    if lig_idx is not None and len(lig_idx):
        return "index " + " ".join(str(int(i)) for i in np.asarray(lig_idx))
    return None


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------


def run_analysis(config: AnalysisConfig, echo: Echo = _noop) -> AnalysisResults:
    """Run the complete single-system analysis described by ``config``."""
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_figure_options(config)
    if config.from_results is not None:
        return _run_from_results(config, echo)

    warnings_: list[str] = []
    extras: dict[str, Any] = {}
    structure = config.topology
    trajectories = list(config.trajectories)
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    rep_ids = [f"Rep_{i + 1}" for i in range(len(trajectories))]
    traj_map = dict(zip(rep_ids, trajectories))
    quiet_progress = not config.progress

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
        raise MissingDependencyError(
            "MDAnalysis",
            "pip install MDAnalysis",
            "the Python PCA backend (--pca-backend python)",
            optional=False,
        )

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
    lig_idx = None
    mem_gb = 0.0
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

    # ------------------------------------------------------------ engine
    use_streaming = False
    if use_python:
        if config.engine == "streaming":
            if config.pca_type != "cartesian":
                raise ValueError(
                    "The streaming engine supports Cartesian PCA only (--pca-type cartesian)."
                )
            if config.cv == "rmsd-rg":
                raise ValueError("--cv rmsd-rg needs the in-memory engine (--engine memory).")
            use_streaming = True
        elif (
            config.engine == "auto"
            and config.pca_type == "cartesian"
            and config.cv != "rmsd-rg"
            and not config.residue_analysis
            and mem_gb > config.streaming_threshold_gb
        ):
            use_streaming = True
            msg = (
                f"Holding {n_frames_total} frames x {len(analysis_idx)} atoms needs about {mem_gb:.1f} GB; "
                "switching to the streaming engine (identical PCA; descriptors, state ensembles and mode "
                "structures are skipped). Use --engine memory to force the in-memory path."
            )
            warnings_.append(msg)
            echo(f"  NOTE: {msg}")
        elif mem_gb > 8:
            warnings_.append(
                f"Holding {n_frames_total} frames x {len(analysis_idx)} atoms needs about {mem_gb:.1f} GB; "
                "consider --stride, a smaller --cov-group or --engine streaming."
            )
            echo(f"  WARNING: {warnings_[-1]}")
    elif config.engine == "streaming":
        raise ValueError("--engine streaming requires the MDAnalysis (python) PCA backend.")

    # ----------------------------------------------------------- streaming
    echo(
        "Step 1: Streaming trajectories"
        + (" with superposition" if config.fit else "")
        + (" (streaming engine)" if use_streaming else "")
    )
    coords_dict: dict[str, np.ndarray] = {}
    times_dict: dict[str, np.ndarray] = {}
    fit_rmsd_dict: dict[str, np.ndarray] = {}
    reference_fit = reference_analysis = None
    spca = None
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
        if use_streaming:
            from conformatlas.streaming import streaming_pca

            def _stream(begin):
                return streaming_pca(
                    structure,
                    traj_map,
                    analysis_idx,
                    fit=config.fit,
                    fit_atom_indices=fit_idx,
                    reference_fit_coords=reference_fit,
                    mass_weighted=config.mass_weighted_fit,
                    stride=config.stride,
                    begin_ps=begin,
                    end_ps=config.end_ps,
                    weighting=config.pca_weighting,
                    n_components=config.n_components,
                    chunk_size=config.chunk_size,
                    n_workers=config.n_workers,
                    precision=config.precision,
                    projection_store=out_dir / "pca" / "projections.npy",
                    system_name=config.system_name,
                    quiet=quiet_progress,
                )

            spca = _stream(config.begin_ps)
            times_dict = dict(spca.times)
            fit_rmsd_dict = dict(spca.fit_rmsd)
        else:
            for rep_id, t_path in track(
                list(traj_map.items()), description="Streaming trajectories", quiet=quiet_progress
            ):
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
        for rep_id in rep_ids:
            t = times_dict[rep_id]
            echo(f"  • {rep_id}: {len(t)} frames kept ({t[0]:.0f}-{t[-1]:.0f} ps)")

    # -------------------------------------------------------- equilibration
    # Stage A: fit-group RMSD (or Rg without superposition), (default observable).
    equil_rows: list[dict] = []
    t0_stage_a: dict[str, int] = {}
    if times_dict and use_python:
        for rep_id in rep_ids:
            if config.fit:
                series = fit_rmsd_dict[rep_id]
                observable = "fit-group RMSD"
            elif coords_dict:
                series = _rg_series(coords_dict[rep_id], protein_mask)
                observable = "Rg"
            else:
                series = None
            if series is None or len(series) < 4:
                continue
            res = detect_equilibration(series)
            t = times_dict[rep_id]
            t0_stage_a[rep_id] = int(res.t0_index)
            equil_rows.append(
                _equilibration_row(rep_id, observable, res, t, config.equilibration == "auto")
            )
            t0_ps = equil_rows[-1]["t0_ps"]
            if config.equilibration == "auto" and res.t0_index > 0:
                if coords_dict:
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
        if use_streaming and config.equilibration == "auto" and any(t0_stage_a.values()):
            # Re-read with a per-replicate production start so the covariance excludes the transient.
            begin_map = {
                rep: float(times_dict[rep][t0]) for rep, t0 in t0_stage_a.items() if t0 > 0
            }
            echo("  • Streaming engine: re-reading from the production start of each replicate")
            spca = _stream(begin_map)
            times_dict = dict(spca.times)
            fit_rmsd_dict = dict(spca.fit_rmsd)
    if use_streaming and not config.fit:
        warnings_.append(
            "Equilibration detection needs the fit-group RMSD with the streaming engine; "
            "run with --fit or --engine memory to detect the transient."
        )

    # ------------------------------------------------------- decomposition
    echo(
        f"Step 2: {config.pca_type.upper() if config.pca_type != 'cartesian' else 'Cartesian'} decomposition"
    )
    feature_dict: dict[str, np.ndarray] = coords_dict
    feature_labels = None
    mode_prefix = "PC"
    dt_ps = _median_dt(times_dict) if times_dict else 1.0
    descriptors = None

    def _descriptors(cdict, tdict):
        if not cdict or reference_analysis is None:
            return None
        return compute_descriptors(
            cdict,
            tdict,
            reference_analysis,
            atom_info,
            ligand_mask,
            masses=atom_info["mass"].to_numpy()
            if atom_info is not None and np.isfinite(atom_info["mass"]).all()
            else None,
        )

    if use_python and use_streaming:
        pca_results = spca.to_pca_results()
        projections = pca_results.projections
    elif use_python:
        pca_results, projections, mode_prefix, feature_labels, feature_dict = _decompose(
            config, structure, traj_map, coords_dict, times_dict, extras, dt_ps
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
        dt_ps = _median_dt(times_dict)

    # Stage B: equilibration on the landscape CVs (``--equilibration-on cv``).
    if config.equilibration_on == "cv":
        descriptors = _descriptors(coords_dict, times_dict)
        cv_df, cv_names, _unit = _assemble_cvs(
            config, projections, times_dict, rep_ids, mode_prefix, use_python, descriptors, dt_ps
        )
        t0_cv, rows_cv = _detect_cv_transients(config, rep_ids, cv_df, cv_names, times_dict)
        equil_rows.extend(rows_cv)
        extra = {rep: t0 for rep, t0 in t0_cv.items() if t0 > 0}
        if extra and config.equilibration == "auto" and use_python:
            for rep, t0 in extra.items():
                echo(
                    f"  • {rep}: CV-based criterion discards {t0} further frames "
                    f"(t0 = {times_dict[rep][t0]:.0f} ps); refitting the decomposition"
                )
            if use_streaming:
                begin_map = {rep: float(times_dict[rep][t0]) for rep, t0 in extra.items()}
                spca = _stream(begin_map)
                times_dict = dict(spca.times)
                fit_rmsd_dict = dict(spca.fit_rmsd)
                pca_results = spca.to_pca_results()
                projections = pca_results.projections
            else:
                for rep, t0 in extra.items():
                    coords_dict[rep] = coords_dict[rep][t0:]
                    times_dict[rep] = times_dict[rep][t0:]
                    fit_rmsd_dict[rep] = fit_rmsd_dict[rep][t0:]
                pca_results, projections, mode_prefix, feature_labels, feature_dict = _decompose(
                    config, structure, traj_map, coords_dict, times_dict, extras, dt_ps
                )
            descriptors = None  # recomputed below on the production window
        elif extra and config.equilibration == "auto":
            warnings_.append(
                "--equilibration auto with --equilibration-on cv cannot discard frames with the "
                "GROMACS PCA backend; the CV-based t0 is reported only."
            )
        elif extra:
            for rep, t0 in extra.items():
                msg = (
                    f"{rep}: the landscape CVs are still relaxing at the RMSD-based t0; the CV criterion "
                    f"suggests discarding {t0} further frames (t0 = {times_dict[rep][t0]:.0f} ps)."
                )
                warnings_.append(msg)
                echo(f"  NOTE: {msg}")
    equil_df = pd.DataFrame(equil_rows) if equil_rows else None

    save_pca_results(pca_results, out_dir)
    if config.pca_type == "tica":
        evals = pca_results.eigenvalues
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
    if descriptors is None:
        descriptors = _descriptors(coords_dict, times_dict)
    if descriptors is not None:
        save_descriptors(descriptors, out_dir)
        extras["descriptors"] = descriptors
    cv_df, cv_names, cv_unit = _assemble_cvs(
        config, projections, times_dict, rep_ids, mode_prefix, use_python, descriptors, dt_ps
    )
    set_cv_labels(cv_names[0], cv_names[1], cv_unit)
    cv1 = cv_df["CV1"].to_numpy()
    cv2 = cv_df["CV2"].to_numpy()

    weights = _read_weights(config, rep_ids, times_dict)
    if weights is not None:
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
    extras["basin_map"] = basin_map
    for st in states:
        echo(
            f"  • {st.label}: {st.population * 100:.1f}% | minimum at {cv_names[0]}={st.min_pc1:.2f}, "
            f"{cv_names[1]}={st.min_pc2:.2f} (ΔG = {st.min_free_energy:.2f} kJ/mol)"
        )
    if unassigned_pop > 0:
        echo(f"  • Minor / transitional: {unassigned_pop * 100:.1f}%")

    # -------------------------------------------------------- convergence
    echo("Step 5: Sampling and convergence diagnostics")
    conv = compute_convergence_diagnostics(
        feature_dict or None, projections, frame_assignments, n_modes=config.convergence_modes
    )
    conv.equilibration = equil_df
    save_convergence_results(conv, out_dir)
    echo(
        f"  • Half-vs-half RMSIP: {conv.rmsip_halves:.2f} | PC1 cosine content: {conv.cosine_content.get(1, 0.0):.2f}"
    )
    echo(f"  • Assessment: {conv.sampling_assessment}")
    if use_streaming:
        warnings_.append(
            "Streaming engine: no coordinates are held, so the subspace-overlap (RMSIP) diagnostics, "
            "descriptors, state-ensemble structures and mode structures are not produced."
        )

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
    boot = _run_bootstrap(
        config,
        cv1,
        cv2,
        frame_assignments,
        fel_grid,
        basin_map,
        states,
        block_lengths,
        weights,
        out_dir,
        summary_pop_df,
        extras,
        echo,
    )
    save_uncertainty_results(summary_pop_df, detailed_pop_df, out_dir)
    echo(
        f"  • Population uncertainty: {unc_type}"
        + (f"; bootstrap x{config.n_bootstrap} ({config.bootstrap_mode})" if boot else "")
    )

    # ------------------------------------------------------------ kinetics
    kinetics = _run_kinetics(frame_assignments, states, weights, dt_ps, out_dir, extras, echo)

    # ------------------------------------------------------------- MSM
    msm_res = None
    if config.msm and states:
        msm_res = _run_msm(config, frame_assignments, states, dt_ps, out_dir, extras, echo)

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

    # ---------------------------------------------------- residue analyses
    residue_files: dict[str, Path] = {}
    if config.residue_analysis:
        if not use_python:
            warnings_.append("Residue-level analyses need the MDAnalysis backend; skipped.")
        else:
            from conformatlas.residue_analysis import analyze_residues

            echo("Step 7b: Residue-level analyses (DCCM, state differences, contacts)")
            lig_sel = _ligand_selection(config, lig_idx)
            residue_files = analyze_residues(
                structure,
                traj_map,
                frame_assignments,
                out_dir / "residues",
                analysis_idx,
                coords_by_replicate=coords_dict or None,
                ligand_selection=lig_sel,
                cutoff=config.contact_cutoff,
                stride=config.stride,
                begin_ps=config.begin_ps,
                end_ps=config.end_ps,
                fit=config.fit,
                fit_atom_indices=fit_idx,
                reference_fit_coords=reference_fit,
                mass_weighted=config.mass_weighted_fit,
                weights=weights,
                figure_dir=out_dir / "figures",
            )
            extras["residue_files"] = residue_files
            echo(f"  • {len(residue_files)} residue-level output file(s) in residues/")

    # -------------------------------------------------------------- figures
    echo("Step 8: Figures")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = _standard_figures(
        config,
        pca_results,
        projections,
        fel_grid,
        states,
        basin_map,
        summary_pop_df,
        detailed_pop_df,
        conv,
        cv1,
        cv2,
        frame_assignments,
        bins,
        block_lengths,
        weights,
        boot,
        descriptors,
        equil_df,
        kinetics,
        mode_files,
        fig_dir,
        cv_names,
    )
    if msm_res is not None:
        figures.update(
            _msm_figures(config, msm_res, fel_grid, frame_assignments, cv_names, fig_dir)
        )
    for key in ("fig_dccm", "fig_state_residue_differences", "fig_contact_persistence"):
        p = residue_files.get(key)
        if p is not None and Path(p).exists():
            figures["residue_" + key[4:]] = Path(p)

    # ------------------------------------------------------------ metadata
    echo("Step 9: Report")
    style = get_figure_style()
    run_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "conformatlas_version": conformatlas.__version__,
        "system": config.system_name,
        "temperature_K": config.temperature,
        "bins": bins,
        "bins_mode": "auto (Freedman-Diaconis)" if config.bins is None else "user",
        "smooth_sigma": config.smooth_sigma,
        "min_state_population": config.min_state_population,
        "max_states": config.max_states,
        "stride": config.stride,
        "time_window_ps": [config.begin_ps, config.end_ps],
        "equilibration": config.equilibration,
        "equilibration_on": config.equilibration_on,
        "pca_type": config.pca_type,
        "dihedral_set": config.dihedral_set if config.pca_type == "dihedral" else None,
        "tica_lag_frames": config.tica_lag if config.pca_type == "tica" else None,
        "pca_weighting": config.pca_weighting,
        "pca_backend": "python" if use_python else "gromacs",
        "engine": "streaming" if use_streaming else ("memory" if use_python else "gromacs"),
        "precision": config.precision if use_streaming else "float64",
        "n_workers": config.n_workers if use_streaming else 1,
        "memory_estimate_gb": float(mem_gb),
        "collective_variables": {
            "mode": config.cv,
            "names": list(cv_names),
            "unit": cv_unit,
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
            "mode": config.bootstrap_mode,
            "block_lengths_frames": block_lengths,
            "seed": config.seed,
        },
        "representatives": {"n": config.n_representatives, "method": config.representative_method},
        "residue_analysis": (
            {
                "enabled": True,
                "ligand_selection": _ligand_selection(config, lig_idx),
                "contact_cutoff_A": config.contact_cutoff,
            }
            if config.residue_analysis
            else {"enabled": False}
        ),
        "figures": {
            "format": list(style.formats),
            "colormap": style.colormap,
            "font_scale": style.font_scale,
            "dpi": style.dpi,
            "path_overlay": style.path_overlay,
        },
        "topology": str(Path(structure).resolve()),
        "trajectories": [str(Path(p).resolve()) for p in trajectories],
        "uncertainty_type": unc_type,
        "gromacs_version": gmx_version,
        "environment": collect_environment(
            config.gmx_command,
            gromacs_version=gmx_version if gmx_runner is not None else None,
            include_gromacs=gmx_runner is not None,
        ),
        "warnings": warnings_,
    }
    if msm_res is not None:
        run_meta["msm"] = _msm_metadata(config, msm_res)
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

    _write_summary(results, cv_names, boot, msm_res, out_dir)
    sections = _report_sections(
        results,
        figures,
        cv_names,
        equil_df,
        boot,
        kinetics,
        descriptors,
        state_summary,
        mode_files,
        msm_res=msm_res,
        residue_files=residue_files,
    )
    generate_html_report(
        results, figures, gromacs_version=gmx_version, extra_sections=sections, pdf=config.pdf
    )
    echo(f"Report: {out_dir / 'report.html'}")
    _maybe_save_hdf5(config, results, echo, warnings_)
    return results


# ----------------------------------------------------------------------------
# step helpers shared by the trajectory route and the ``from_results`` route
# ----------------------------------------------------------------------------


def _run_bootstrap(
    config,
    cv1,
    cv2,
    frame_assignments,
    fel_grid,
    basin_map,
    states,
    block_lengths,
    weights,
    out_dir,
    summary_pop_df,
    extras,
    echo,
):
    if not (config.n_bootstrap > 0 and states):
        return None
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
        mode=config.bootstrap_mode,
        resegment_kwargs={
            "min_state_population": config.min_state_population,
            "max_states": config.max_states,
            "min_barrier": config.min_barrier,
        },
    )
    extras["bootstrap"] = boot
    (out_dir / "fel").mkdir(parents=True, exist_ok=True)
    (out_dir / "states").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "fel" / "free_energy_bootstrap_sd.npy", boot.fe_std)
    np.save(out_dir / "fel" / "free_energy_bootstrap_ci_low.npy", boot.fe_ci_low)
    np.save(out_dir / "fel" / "free_energy_bootstrap_ci_high.npy", boot.fe_ci_high)
    boot.state_table.to_csv(out_dir / "states" / "state_free_energies.csv", index=False)
    boot.barrier_table.to_csv(out_dir / "states" / "barriers.csv", index=False)
    extras["barriers"] = boot.barrier_table
    if len(summary_pop_df):
        ci = boot.state_table.set_index("state")
        summary_pop_df["Bootstrap_CI_95_Lower"] = summary_pop_df["State"].map(
            ci["population_ci95_low"] * 100.0
        )
        summary_pop_df["Bootstrap_CI_95_Upper"] = summary_pop_df["State"].map(
            ci["population_ci95_high"] * 100.0
        )
    if boot.mode == "resegment" and boot.resegment_state_table is not None:
        boot.state_count_distribution.to_csv(
            out_dir / "states" / "state_count_distribution.csv", index=False
        )
        boot.resegment_state_table.to_csv(
            out_dir / "states" / "state_populations_resegmented.csv", index=False
        )
        if len(summary_pop_df):
            rs = boot.resegment_state_table.set_index("state")
            summary_pop_df["Resegment_CI_95_Lower"] = summary_pop_df["State"].map(
                rs["population_ci95_low_resegmented"] * 100.0
            )
            summary_pop_df["Resegment_CI_95_Upper"] = summary_pop_df["State"].map(
                rs["population_ci95_high_resegmented"] * 100.0
            )
            summary_pop_df["Resegment_match_rate"] = summary_pop_df["State"].map(rs["match_rate"])
        dist = boot.state_count_distribution
        mode_n = int(dist.loc[dist["count"].idxmax(), "n_states"]) if len(dist) else len(states)
        frac = (
            float(dist.loc[dist["n_states"] == len(states), "fraction"].sum()) if len(dist) else 0.0
        )
        echo(
            f"  • Re-segmentation: {len(states)} states recovered in {frac * 100:.0f}% of resamples "
            f"(most frequent count {mode_n})"
        )
    for _, r in boot.state_table.iterrows():
        echo(
            f"  • {r['state']}: population {r['population'] * 100:.1f}% "
            f"[{r['population_ci95_low'] * 100:.1f}, {r['population_ci95_high'] * 100:.1f}] | "
            f"ΔG vs {r['reference_state']} = {r['dG_state_kJ_mol']:.2f} "
            f"[{r['dG_state_ci95_low']:.2f}, {r['dG_state_ci95_high']:.2f}] kJ/mol"
        )
    return boot


def _run_kinetics(frame_assignments, states, weights, dt_ps, out_dir, extras, echo):
    if weights is not None or not states:
        return None
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
    return kinetics


def _msm_columns(config: AnalysisConfig, frame_assignments: pd.DataFrame) -> list[str]:
    """Clustering components: the two landscape PCs first, then the next leading PCs."""
    pcs = [c for c in frame_assignments.columns if c.startswith("PC") and c[2:].isdigit()]
    pcs = sorted(pcs, key=lambda c: int(c[2:]))
    n = max(1, min(config.msm_n_components, len(pcs)))
    if config.cv == "pc":
        first = [f"PC{config.fel_pcs[0]}", f"PC{config.fel_pcs[1]}"]
        rest = [c for c in pcs if c not in first]
        cols = (first + rest)[:n] if n >= 2 else first[:1]
    else:
        cols = pcs[:n]
    return cols


def _run_msm(config, frame_assignments, states, dt_ps, out_dir, extras, echo):
    from conformatlas.msm import build_msm, save_msm

    echo("Step 6b: Markov state model")
    cols = _msm_columns(config, frame_assignments)
    n_clusters = int(min(config.msm_n_clusters, max(2, len(frame_assignments) // 10)))
    m = config.msm_n_metastable or max(2, len(states))
    msm_res = build_msm(
        frame_assignments,
        cols,
        lag=config.msm_lag,
        lags=config.msm_lags,
        n_clusters=n_clusters,
        clustering=config.msm_clustering,
        dmin=config.msm_dmin,
        n_metastable=m,
        reversible=config.msm_reversible,
        dt_ps=dt_ps,
        ck_k_max=config.msm_ck_k,
        seed=config.seed,
        basin_states=frame_assignments["state"].to_numpy(),
    )
    save_msm(msm_res, out_dir)
    extras["msm"] = msm_res
    est = msm_res.msm
    echo(
        f"  • {n_clusters} microstates in {', '.join(cols)}; {len(est.active_set)} active; "
        f"lag {config.msm_lag} frames ({config.msm_lag * dt_ps:.1f} ps); {m} PCCA+ sets"
    )
    ts = est.timescales_ps
    if len(ts):
        shown = ", ".join(f"{t:.1f}" for t in ts[: min(3, len(ts))])
        echo(f"  • Implied timescales at the model lag (ps): {shown}")
    ck = msm_res.ck_summary
    echo(
        f"  • Chapman-Kolmogorov: {'passed' if ck.get('passed') else 'not passed'} "
        f"(max |deviation| {ck.get('max_abs_deviation', float('nan')):.3f})"
    )
    for w in msm_res.notes:
        echo("  • " + w)
    return msm_res


def _msm_metadata(config: AnalysisConfig, msm_res) -> dict[str, Any]:
    est = msm_res.msm
    return {
        "lag_frames": int(msm_res.lag),
        "lag_ps": float(msm_res.lag * msm_res.dt_ps),
        "components": list(msm_res.columns),
        "clustering": msm_res.clustering.method,
        "n_microstates": int(msm_res.clustering.n_clusters),
        "n_active": int(len(est.active_set)),
        "reversible": bool(msm_res.reversible),
        "n_metastable": int(msm_res.n_metastable),
        "suggested_lag_frames": msm_res.lag_report.get("suggested_lag_frames"),
        "plateau_found": msm_res.lag_report.get("plateau_found"),
        "ck_passed": msm_res.ck_summary.get("passed"),
        "ck_max_abs_deviation": msm_res.ck_summary.get("max_abs_deviation"),
        "implied_timescales_ps": [float(t) for t in est.timescales_ps],
        "notes": list(msm_res.notes),
    }


def _msm_figures(config, msm_res, fel_grid, frame_assignments, cv_names, fig_dir) -> dict:
    from conformatlas.plotting_msm import (
        plot_ck_test,
        plot_implied_timescales,
        plot_metastable_timeseries,
        plot_pcca_landscape,
    )

    figs = {
        "msm_implied_timescales": plot_implied_timescales(
            msm_res.implied_timescales,
            fig_dir / "msm_implied_timescales.png",
            msm_res.lag_report,
            config.msm_lag,
        ),
        "msm_ck_test": plot_ck_test(msm_res.ck_test, fig_dir / "msm_ck_test.png"),
        "msm_metastable_timeseries": plot_metastable_timeseries(
            msm_res.frame_table, fig_dir / "msm_metastable_timeseries.png"
        ),
    }
    if config.cv == "pc" and len(msm_res.columns) >= 2:
        # The first two clustering components are the landscape CVs (see _msm_columns).
        figs["msm_pcca_landscape"] = plot_pcca_landscape(
            fel_grid,
            frame_assignments,
            msm_res.frame_table,
            msm_res.clustering.centers,
            msm_res.memberships,
            fig_dir / "msm_pcca_landscape.png",
            cv_columns=("CV1", "CV2"),
        )
    return figs


def _per_replicate_figure(config, frame_assignments, fel_grid, bins, weights, cv_names, fig_dir):
    from conformatlas.plotting_msm import plot_per_replicate_landscapes

    path, _grids = plot_per_replicate_landscapes(
        frame_assignments,
        {
            "temperature": config.temperature,
            "bins": bins,
            "smooth_sigma": config.smooth_sigma,
            "x_range": (float(fel_grid.x_edges[0]), float(fel_grid.x_edges[-1])),
            "y_range": (float(fel_grid.y_edges[0]), float(fel_grid.y_edges[-1])),
        },
        fig_dir / "per_replicate_landscapes.png",
        cv_columns=("CV1", "CV2"),
        weights=weights,
    )
    return path


def _standard_figures(
    config,
    pca_results,
    projections,
    fel_grid,
    states,
    basin_map,
    summary_pop_df,
    detailed_pop_df,
    conv,
    cv1,
    cv2,
    frame_assignments,
    bins,
    block_lengths,
    weights,
    boot,
    descriptors,
    equil_df,
    kinetics,
    mode_files,
    fig_dir,
    cv_names,
) -> dict[str, Path]:
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
    if frame_assignments["replicate"].nunique() > 1:
        figures["per_replicate_landscapes"] = _per_replicate_figure(
            config, frame_assignments, fel_grid, bins, weights, cv_names, fig_dir
        )
    return figures


def _write_summary(results: AnalysisResults, cv_names, boot, msm_res, out_dir: Path) -> dict:
    config, conv, states = results.config, results.convergence, results.states
    summary = {
        "system": config.system_name,
        "temperature_K": config.temperature,
        "total_frames": int(len(results.frame_assignments)),
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
        "unassigned_population_percent": results.unassigned_population * 100.0,
        "convergence": {
            "rmsip_halves": conv.rmsip_halves,
            "pc1_cosine_content": conv.cosine_content.get(1, 0.0),
            "n_eff_total": float(sum(conv.n_eff.values())) if conv.n_eff else None,
            "assessment": conv.sampling_assessment,
        },
        "warnings": results.warnings,
    }
    if boot is not None:
        summary["state_free_energies"] = boot.state_table.to_dict("records")
        summary["barriers"] = boot.barrier_table.to_dict("records")
        summary["bootstrap_mode"] = boot.mode
        if boot.mode == "resegment" and boot.state_count_distribution is not None:
            summary["state_count_distribution"] = boot.state_count_distribution.to_dict("records")
            summary["state_populations_resegmented"] = boot.resegment_state_table.to_dict("records")
    if msm_res is not None:
        summary["msm"] = _msm_metadata(config, msm_res)
        summary["msm"]["coarse_stationary"] = msm_res.coarse_stationary.to_dict("records")
        summary["msm"]["mfpt_ps"] = msm_res.mfpt_ps.to_dict()
    save_json(summary, out_dir / "summary.json")
    return summary


def _maybe_save_hdf5(config, results, echo, warnings_):
    if not config.save_hdf5:
        return None
    from conformatlas.results_io import HAS_H5PY, save_results

    if not HAS_H5PY:
        msg = "--save-hdf5 requested but h5py is not installed (pip install h5py); results.h5 skipped."
        warnings_.append(msg)
        echo(f"  WARNING: {msg}")
        return None
    path = save_results(results, results.output_dir / "results.h5")
    echo(f"Results container: {path}")
    return path


# ----------------------------------------------------------------------------
# report sections
# ----------------------------------------------------------------------------


def _report_sections(
    results,
    figures,
    cv_names,
    equil_df,
    boot,
    kinetics,
    descriptors,
    state_summary,
    mode_files,
    msm_res=None,
    residue_files=None,
):
    residue_files = residue_files or {}
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
        paragraphs = [
            "t0 is the production start that maximises the number of effectively independent frames "
            "(N_eff = N / g, g = statistical inefficiency). 'applied' states whether the transient was discarded."
        ]
        if results.config.equilibration_on == "cv":
            paragraphs.append(
                "Rows labelled with the two collective variables apply the same criterion to each landscape "
                "CV and adopt the later production start ('deciding_observable'), so that neither CV is still "
                "relaxing at t0. When the transient is discarded, the decomposition is refitted on the "
                "production window."
            )
        sections.append(
            {
                "title": "Equilibration and effective sample size",
                "paragraphs": paragraphs,
                "table": equil_df,
            }
        )
    if "per_replicate_landscapes" in figures:
        sections.append(
            {
                "title": "Per-replicate landscapes",
                "paragraphs": [
                    "Each replicate's landscape on the shared grid, referenced to its own minimum. The panels "
                    "compare the shape of the landscapes sampled by the replicates; they carry no quantitative "
                    "convergence statistic (see the RMSIP and population-stability diagnostics for that)."
                ],
                "figure": figures["per_replicate_landscapes"],
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
        if boot.mode == "resegment" and boot.state_count_distribution is not None:
            sections.append(
                {
                    "title": "Re-segmentation bootstrap: how stable is the number of states?",
                    "paragraphs": [
                        "The watershed segmentation was re-run on every block-bootstrap resample (same grid, "
                        "same population threshold, maximum number of states and minimum barrier). The first "
                        "table gives the distribution of the number of major states found; the second matches "
                        "the resampled states to the reference states (≥ 50 % frame overlap or minimum inside "
                        "the reference basin; a definition, not a validated quantity) and reports how often "
                        "each state was recovered ('match_rate') together with its population interval under "
                        "re-segmentation."
                    ],
                    "table": boot.state_count_distribution,
                }
            )
            sections.append(
                {
                    "title": "State populations under re-segmentation",
                    "paragraphs": [],
                    "table": boot.resegment_state_table,
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
    if msm_res is not None:
        sections.extend(_msm_sections(results.config, msm_res, figures))
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
    if residue_files:
        sections.extend(_residue_sections(results, residue_files, figures))
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


def _msm_sections(config, msm_res, figures) -> list[dict]:
    est = msm_res.msm
    ck = msm_res.ck_summary
    lag_note = str(msm_res.lag_report.get("note", ""))
    intro = (
        f"Frames were clustered into {msm_res.clustering.n_clusters} microstates "
        f"({msm_res.clustering.method}) in {', '.join(msm_res.columns)}; transition counts at a lag of "
        f"{msm_res.lag} frames ({msm_res.lag * msm_res.dt_ps:.1f} ps) were accumulated within each replicate "
        f"and the {'reversible maximum-likelihood' if msm_res.reversible else 'row-normalised'} transition "
        f"matrix was estimated on the largest connected set ({len(est.active_set)} microstates). "
        f"{msm_res.n_metastable} metastable sets were obtained with PCCA+. The lag is the user's choice; "
        "the implied-timescale scan only reports whether a plateau was found."
    )
    ck_line = (
        f"Chapman–Kolmogorov test: {'passed' if ck.get('passed') else 'not passed'} "
        f"(max |deviation| {ck.get('max_abs_deviation', float('nan')):.3f}; a set passes when every absolute "
        "deviation is below max(0.05, 3 SE) — a convention, the deviations themselves are exact)."
    )
    sections = [
        {
            "title": "Markov state model: implied timescales",
            "paragraphs": [intro, lag_note] if lag_note else [intro],
            "figure": figures.get("msm_implied_timescales"),
            "table": msm_res.implied_timescales,
        },
        {
            "title": "Markov state model: Chapman–Kolmogorov test",
            "paragraphs": [ck_line],
            "figure": figures.get("msm_ck_test"),
        },
    ]
    if "msm_pcca_landscape" in figures:
        sections.append(
            {
                "title": "Markov state model: PCCA+ sets on the landscape",
                "paragraphs": [
                    "Microstate centres coloured by their crisp PCCA+ set, drawn over the free-energy landscape. "
                    "The overlap table cross-tabulates frames by metastable set and watershed state; "
                    "row_fraction_max near 1 means a kinetic set coincides with one landscape basin."
                ],
                "figure": figures["msm_pcca_landscape"],
                "table": msm_res.basin_overlap,
            }
        )
    elif msm_res.basin_overlap is not None:
        sections.append(
            {
                "title": "Markov state model: metastable sets vs landscape states",
                "paragraphs": [
                    "Frames cross-tabulated by PCCA+ set and watershed state (row_fraction_max near 1: a kinetic "
                    "set coincides with one landscape basin)."
                ],
                "table": msm_res.basin_overlap,
            }
        )
    sections.append(
        {
            "title": "Markov state model: metastable-set populations and mean first-passage times",
            "paragraphs": [
                "Stationary probability of each PCCA+ set (from the microstate stationary distribution) and "
                "mean first-passage times between sets in ps (rows = origin). MFPTs are in units of the lag and "
                "inherit the Markov assumption at that lag."
            ],
            "figure": figures.get("msm_metastable_timeseries"),
            "table": msm_res.coarse_stationary,
        }
    )
    mfpt = msm_res.mfpt_ps.copy()
    mfpt.insert(0, "from \\ to", [str(i) for i in mfpt.index])
    sections.append(
        {
            "title": "Mean first-passage times between metastable sets (ps)",
            "paragraphs": [],
            "table": mfpt,
        }
    )
    if msm_res.notes:
        sections.append({"title": "Markov state model: notes", "paragraphs": list(msm_res.notes)})
    return sections


def _residue_sections(results, residue_files, figures) -> list[dict]:
    sections = []
    paragraphs = [
        "Dynamic cross-correlation map C_ij = ⟨Δr_i·Δr_j⟩ / sqrt(⟨Δr_i²⟩⟨Δr_j²⟩) of the analysis group, "
        "computed from the globally superposed coordinates (no re-fit per state), averaged per residue."
    ]
    sections.append(
        {
            "title": "Residue-level analysis: dynamic cross-correlation",
            "paragraphs": paragraphs,
            "figure": figures.get("residue_dccm"),
        }
    )
    diff_path = residue_files.get("state_residue_differences")
    if diff_path is not None and Path(diff_path).exists():
        diff = pd.read_csv(diff_path)
        if len(diff) and "mean_rmsd" in diff.columns:
            top = diff.sort_values("mean_rmsd", ascending=False).head(10)
            sections.append(
                {
                    "title": "Residue-level analysis: differences between states",
                    "paragraphs": [
                        "Per residue and state pair: RMSF within each state, their difference and the RMSD "
                        "between the state-average positions (mean_rmsd). The ten residues with the largest "
                        "displacement of the state means are listed."
                    ],
                    "figure": figures.get("residue_state_residue_differences"),
                    "table": top,
                }
            )
    contacts_path = residue_files.get("ligand_contacts_per_state")
    if contacts_path is not None and Path(contacts_path).exists():
        contacts = pd.read_csv(contacts_path)
        if "max_persistence" in contacts.columns:
            site = contacts[contacts["max_persistence"] >= 0.5]
        else:
            site = contacts
        cutoff = results.run_metadata.get("residue_analysis", {}).get("contact_cutoff_A", 4.5)
        sections.append(
            {
                "title": "Residue-level analysis: ligand contacts",
                "paragraphs": [
                    f"Residues with any heavy-atom pair within {cutoff} Å of the ligand; persistence is the "
                    "fraction of each state's frames in contact. Residues with persistence ≥ 0.5 in at least one "
                    "state define the binding site (a conventional threshold)."
                ],
                "figure": figures.get("residue_contact_persistence"),
                "table": site,
            }
        )
    return sections


# ----------------------------------------------------------------------------
# ``--from <results_dir>``: re-analysis without trajectories
# ----------------------------------------------------------------------------


def _run_from_results(config: AnalysisConfig, echo: Echo) -> AnalysisResults:
    """Rebin / re-segment / build an MSM from a stored ``conformatlas analyze`` directory.

    Trajectories are not read: the PCA tables of the source directory are reused
    (copied when the output directory differs). Steps that need coordinates
    (descriptors, state-ensemble structures, mode structures, subspace-overlap
    diagnostics, residue analyses) are not recomputed; representative structures
    are extracted when the recorded topology and trajectories still exist.
    """
    from conformatlas.results_io import load_results_dir, rebin

    src = Path(config.from_results)
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings_: list[str] = []
    echo(f"Step 1: Loading stored results from {src}")
    base = load_results_dir(src)
    meta0 = dict(base.run_metadata)
    proj = base.pca.projections
    rep_ids = [str(r) for r in proj["replicate"].drop_duplicates()]
    times_dict = {
        rep: proj.loc[proj["replicate"] == rep, "time_ps"].to_numpy(dtype=float) for rep in rep_ids
    }
    dt_ps = _median_dt(times_dict)
    echo(f"  • {len(proj)} frames in {len(rep_ids)} replicate(s); Δt = {dt_ps:.1f} ps")

    cv_meta = meta0.get("collective_variables", {})
    cv_names = tuple(cv_meta.get("names", ("PC1", "PC2")))
    cv_unit = cv_meta.get("unit")
    if cv_unit is None:
        cv_unit = (
            ""
            if (cv_meta.get("mode") == "file" or meta0.get("pca_type") == "dihedral")
            else ("nm" if meta0.get("pca_backend") == "gromacs" else "Å")
        )
    set_cv_labels(cv_names[0], cv_names[1], cv_unit)

    weights = _read_weights(config, rep_ids, times_dict)
    if weights is None and meta0.get("weights_files"):
        msg = (
            "The stored run used per-frame weights; pass the same --weights-file arguments to --from, "
            "otherwise populations are unweighted."
        )
        warnings_.append(msg)
        echo(f"  WARNING: {msg}")

    # ---- rebin -----------------------------------------------------------
    echo("Step 2: Landscape and basin detection on the stored collective variables")
    fa0 = base.frame_assignments
    cv1_all = fa0["CV1"].to_numpy(dtype=float) if "CV1" in fa0.columns else None
    if cv1_all is None:
        i1, i2 = base.config.fel_pcs
        cv1_all, cv2_all = proj[f"PC{i1}"].to_numpy(), proj[f"PC{i2}"].to_numpy()
    else:
        cv2_all = fa0["CV2"].to_numpy(dtype=float)
    if config.bins is not None:
        bins = int(config.bins)
        bins_mode = "user"
    elif meta0.get("bins") is not None:
        bins = int(meta0["bins"])
        bins_mode = "stored"
    else:
        bins = choose_bins(cv1_all, cv2_all)
        bins_mode = "auto (Freedman-Diaconis)"
    echo(f"  • Grid: {bins} x {bins} bins ({bins_mode}); smoothing σ = {config.smooth_sigma}")
    res = rebin(
        base,
        bins,
        config.smooth_sigma,
        min_state_population=config.min_state_population,
        max_states=config.max_states,
        min_barrier=config.min_barrier,
        weights=weights,
        temperature=config.temperature,
    )
    fel_grid, states, frame_assignments = res.fel, res.states, res.frame_assignments
    basin_map = res.extras["basin_map"]
    cv1 = frame_assignments["CV1"].to_numpy(dtype=float)
    cv2 = frame_assignments["CV2"].to_numpy(dtype=float)
    extras: dict[str, Any] = dict(res.extras)
    for st in states:
        echo(
            f"  • {st.label}: {st.population * 100:.1f}% | minimum at {cv_names[0]}={st.min_pc1:.2f}, "
            f"{cv_names[1]}={st.min_pc2:.2f} (ΔG = {st.min_free_energy:.2f} kJ/mol)"
        )
    if res.unassigned_population > 0:
        echo(f"  • Minor / transitional: {res.unassigned_population * 100:.1f}%")

    # ---- write tables ----------------------------------------------------
    if src.resolve() != out_dir.resolve():
        for sub in ("pca", "descriptors"):
            if (src / sub).is_dir():
                shutil.copytree(src / sub, out_dir / sub, dirs_exist_ok=True)
    save_fel_grid(fel_grid, out_dir)
    save_basin_results(states, frame_assignments, basin_map, out_dir)
    extras["barriers"].to_csv(out_dir / "states" / "barriers.csv", index=False)

    # ---- convergence (projection-based parts recomputed; RMSIP needs coordinates) -----
    echo("Step 3: Sampling diagnostics from the stored projections")
    conv = compute_convergence_diagnostics(
        None, proj, frame_assignments, n_modes=config.convergence_modes
    )
    conv.equilibration = base.convergence.equilibration
    # keep the coordinate-based numbers of the original run, which binning does not change
    b = base.convergence
    if np.isfinite(b.rmsip_halves):
        conv.rmsip_halves = b.rmsip_halves
        conv.progressive_overlap = dict(b.progressive_overlap)
        conv.pairwise_rmsip = b.pairwise_rmsip
        conv.rmsip_halves_per_replicate = dict(b.rmsip_halves_per_replicate)
        conv.assessment_notes = [
            n for n in conv.assessment_notes if "Subspace-overlap diagnostics" not in n
        ] + [
            "Subspace-overlap (RMSIP) values are those of the original run; they do not depend on the grid."
        ]
        pc1_cos = conv.cosine_content.get(1, 0.0)
        n_eff_total = float(sum(conv.n_eff.values())) if conv.n_eff else float(len(proj))
        if conv.rmsip_halves >= 0.70 and pc1_cos < 0.50:
            conv.sampling_assessment = "Good evidence of stability"
        elif conv.rmsip_halves >= 0.50 and pc1_cos < 0.70:
            conv.sampling_assessment = "Mixed evidence"
        else:
            conv.sampling_assessment = "Insufficient evidence"
        if n_eff_total < 50:
            conv.sampling_assessment = "Insufficient evidence"
    save_convergence_results(conv, out_dir)
    echo(f"  • Assessment: {conv.sampling_assessment}")
    g_by_rep = {
        rep: max(v.values()) if v else 1.0 for rep, v in conv.statistical_inefficiency.items()
    }
    block_lengths = {
        rep: block_length_from_g(g_by_rep.get(rep, 1.0), len(times_dict[rep])) for rep in times_dict
    }

    # ---- uncertainty, kinetics, MSM ---------------------------------------
    echo("Step 4: Uncertainty")
    g_single = max(g_by_rep.values()) if g_by_rep else None
    summary_pop_df, detailed_pop_df, unc_type = compute_replicate_statistics(
        frame_assignments, statistical_inefficiency=g_single, weights=weights
    )
    boot = _run_bootstrap(
        config,
        cv1,
        cv2,
        frame_assignments,
        fel_grid,
        basin_map,
        states,
        block_lengths,
        weights,
        out_dir,
        summary_pop_df,
        extras,
        echo,
    )
    save_uncertainty_results(summary_pop_df, detailed_pop_df, out_dir)
    kinetics = _run_kinetics(frame_assignments, states, weights, dt_ps, out_dir, extras, echo)
    msm_res = None
    if config.msm and states:
        msm_res = _run_msm(config, frame_assignments, states, dt_ps, out_dir, extras, echo)

    # ---- representative structures (only when the inputs still exist) ------
    echo("Step 5: Representative structures")
    topology = Path(meta0.get("topology", str(config.topology)))
    traj_paths = [Path(p) for p in meta0.get("trajectories", [])]
    traj_map = dict(zip(rep_ids, traj_paths))
    gmx_runner = None
    gmx_version = "None"
    try:
        gmx_runner = GromacsRunner(config.gmx_command)
        gmx_version = gmx_runner.version
    except Exception:
        pass
    if topology.exists() and traj_paths and all(p.exists() for p in traj_paths) and HAS_MDANALYSIS:
        extract_representative_structures(
            states,
            {config.system_name: topology},
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
    else:
        msg = (
            "Representative structures were not extracted: the topology/trajectories recorded in "
            "run_metadata.json are not available."
        )
        warnings_.append(msg)
        echo(f"  NOTE: {msg}")

    # ---- figures -----------------------------------------------------------
    echo("Step 6: Figures")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pca_for_plot = base.pca
    figures = _standard_figures(
        config,
        pca_for_plot,
        proj,
        fel_grid,
        states,
        basin_map,
        summary_pop_df,
        detailed_pop_df,
        conv,
        cv1,
        cv2,
        frame_assignments,
        bins,
        block_lengths,
        weights,
        boot,
        None,
        conv.equilibration,
        kinetics,
        {},
        fig_dir,
        cv_names,
    )
    if msm_res is not None:
        figures.update(
            _msm_figures(config, msm_res, fel_grid, frame_assignments, cv_names, fig_dir)
        )

    # ---- metadata, summary, report -----------------------------------------
    echo("Step 7: Report")
    style = get_figure_style()
    run_meta = dict(meta0)
    run_meta.update(
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "command_line": " ".join(sys.argv),
            "conformatlas_version": conformatlas.__version__,
            "from_results": str(src.resolve()),
            "temperature_K": config.temperature,
            "bins": bins,
            "bins_mode": bins_mode,
            "smooth_sigma": config.smooth_sigma,
            "min_barrier_kJ_mol": config.min_barrier
            if config.min_barrier is not None
            else R_GAS_CONSTANT * config.temperature,
            "min_state_population": config.min_state_population,
            "max_states": config.max_states,
            "weights_files": [str(p) for p in config.weights_files],
            "bootstrap": {
                "n": config.n_bootstrap,
                "mode": config.bootstrap_mode,
                "block_lengths_frames": block_lengths,
                "seed": config.seed,
            },
            "representatives": {
                "n": config.n_representatives,
                "method": config.representative_method,
            },
            "figures": {
                "format": list(style.formats),
                "colormap": style.colormap,
                "font_scale": style.font_scale,
                "dpi": style.dpi,
                "path_overlay": style.path_overlay,
            },
            "uncertainty_type": unc_type,
            "environment": collect_environment(
                config.gmx_command,
                gromacs_version=gmx_version if gmx_runner is not None else None,
                include_gromacs=gmx_runner is not None,
            ),
            "warnings": warnings_,
        }
    )
    run_meta.pop("msm", None)
    if msm_res is not None:
        run_meta["msm"] = _msm_metadata(config, msm_res)
    save_json(run_meta, out_dir / "run_metadata.json")

    cfg = config
    cfg.topology = topology
    cfg.trajectories = traj_paths or list(config.trajectories)
    cfg.bins = bins
    results = AnalysisResults(
        config=cfg,
        trajectories_info=base.trajectories_info,
        pca=base.pca,
        fel=fel_grid,
        states=states,
        unassigned_population=res.unassigned_population,
        frame_assignments=frame_assignments,
        convergence=conv,
        replicate_statistics=summary_pop_df,
        output_dir=out_dir,
        run_metadata=run_meta,
        extras=extras,
        warnings=warnings_,
    )
    _write_summary(results, cv_names, boot, msm_res, out_dir)
    sections = _report_sections(
        results,
        figures,
        cv_names,
        conv.equilibration,
        boot,
        kinetics,
        None,
        None,
        {},
        msm_res=msm_res,
    )
    generate_html_report(
        results, figures, gromacs_version=gmx_version, extra_sections=sections, pdf=config.pdf
    )
    echo(f"Report: {out_dir / 'report.html'}")
    _maybe_save_hdf5(config, results, echo, warnings_)
    return results
