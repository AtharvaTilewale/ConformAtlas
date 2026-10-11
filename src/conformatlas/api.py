"""High-level Python API returning in-memory objects.

The command line (``conformatlas analyze``) runs the whole pipeline and writes a
results directory. This module exposes the same building blocks as plain
functions for scripts and notebooks::

    from conformatlas import api

    ens = api.load_trajectories("topol.pdb", ["rep1.xtc", "rep2.xtc"], group="C-alpha")
    pcs = api.pca(ens, n_components=10)                 # PCAResults
    land = api.landscape(pcs, temperature=300, bins=32)  # Landscape (FELGrid + CV table)
    st = api.states(land, min_population=0.05)          # StateAssignment
    kin = api.kinetics(st)                               # dict of DataFrames
    conv = api.convergence(ens, pcs, st)                 # ConvergenceResults
    fig, ax = api.plot_landscape(land, states=st.states)

    results = api.analyze("topol.pdb", ["rep1.xtc"], bins=32, output_dir="out")  # full pipeline

Every function is a thin, documented wrapper around the validated engines in
:mod:`conformatlas.pca`, :mod:`conformatlas.fel`, :mod:`conformatlas.basins`,
:mod:`conformatlas.kinetics` and :mod:`conformatlas.convergence`; no numerics
are re-implemented here. Units: Å, ps, kJ/mol; R = 0.008314462618 kJ/(mol K).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conformatlas.basins import compute_state_barriers, detect_energy_basins
from conformatlas.convergence import compute_convergence_diagnostics
from conformatlas.fel import calculate_fel, choose_bins
from conformatlas.kinetics import compute_state_kinetics
from conformatlas.models import (
    AnalysisConfig,
    AnalysisResults,
    BasinState,
    ConvergenceResults,
    FELGrid,
    PCAResults,
)
from conformatlas.ndx import parse_ndx
from conformatlas.pca import compute_cartesian_pca, compute_tica, project_coordinates
from conformatlas.superpose import superpose_frame
from conformatlas.trajectory import (
    atom_info_for_indices,
    load_reference_coordinates,
    resolve_selection_indices,
    stream_trajectory_coordinates,
)

logger = logging.getLogger("conformatlas.api")

__all__ = [
    "Ensemble",
    "Landscape",
    "StateAssignment",
    "load_trajectories",
    "pca",
    "landscape",
    "states",
    "kinetics",
    "convergence",
    "analyze",
    "plot_landscape",
]


# =============================================================================
# Containers
# =============================================================================


@dataclass
class Ensemble:
    """Superposed coordinates of one or more replicates (output of :func:`load_trajectories`).

    Attributes
    ----------
    coords : dict replicate_id -> (n_frames, n_atoms, 3) float32, Å, superposed
    times : dict replicate_id -> (n_frames,) ps
    fit_rmsd : dict replicate_id -> (n_frames,) Å, RMSD of the fit group after superposition
    atom_indices, fit_indices : 0-based atom indices of the analysis / fit groups
    atom_info : table (atom_index, chain, residue_number, residue_name, atom_name, mass)
    reference : (n_atoms, 3) analysis-group reference coordinates in the fitted frame
    ligand_mask : boolean mask over analysis atoms (None without a ligand group)
    """

    topology: Path
    trajectories: dict[str, Path]
    coords: dict[str, np.ndarray]
    times: dict[str, np.ndarray]
    fit_rmsd: dict[str, np.ndarray]
    atom_indices: np.ndarray
    fit_indices: np.ndarray
    atom_info: pd.DataFrame
    reference: np.ndarray
    ligand_mask: np.ndarray | None = None
    group: str = "Backbone"
    fit_group: str = "Backbone"
    system_name: str = "System"

    @property
    def replicate_ids(self) -> list[str]:
        return list(self.coords)

    @property
    def n_frames(self) -> int:
        return int(sum(len(t) for t in self.times.values()))

    @property
    def n_atoms(self) -> int:
        return int(len(self.atom_indices))

    @property
    def dt_ps(self) -> float:
        diffs = [np.diff(t) for t in self.times.values() if len(t) > 1]
        return float(np.median(np.concatenate(diffs))) if diffs else 1.0


@dataclass
class Landscape:
    """Free-energy landscape plus the collective-variable table it was built from."""

    fel: FELGrid
    cv: pd.DataFrame  # projections table plus CV1, CV2 columns
    cv_names: tuple[str, str]
    cv_unit: str = "Å"
    bins: int = 0
    weights: np.ndarray | None = None


@dataclass
class StateAssignment:
    """Basins, per-frame state labels and barriers (output of :func:`states`)."""

    states: list[BasinState]
    unassigned_population: float
    frame_assignments: pd.DataFrame
    basin_map: np.ndarray
    barriers: pd.DataFrame
    landscape: Landscape = field(repr=False)

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.states]

    def populations(self) -> pd.DataFrame:
        """State populations (fraction) and minimum free energies as a table."""
        return pd.DataFrame(
            {
                "state": [s.label for s in self.states],
                "population": [s.population for s in self.states],
                "min_cv1": [s.min_pc1 for s in self.states],
                "min_cv2": [s.min_pc2 for s in self.states],
                "min_free_energy_kJ_mol": [s.min_free_energy for s in self.states],
            }
        )


# =============================================================================
# 1. Trajectories
# =============================================================================


def load_trajectories(
    topology: str | Path,
    trajectories: Sequence[str | Path] | Mapping[str, str | Path],
    *,
    group: str = "Backbone",
    fit_group: str | None = None,
    lig_group: str | None = None,
    index_file: str | Path | None = None,
    stride: int = 1,
    begin_ps: float | None = None,
    end_ps: float | None = None,
    fit: bool = True,
    mass_weighted_fit: bool = False,
    system_name: str = "System",
) -> Ensemble:
    """Read, superpose and hold the analysis-group coordinates of every replicate.

    Parameters
    ----------
    topology : structure file (.pdb/.gro; also the superposition reference) or .tpr
    trajectories : trajectory paths; a mapping gives explicit replicate ids,
        a sequence is labelled ``Rep_1, Rep_2, …``
    group : analysis group (GROMACS group name, MDAnalysis selection or index group)
    fit_group : superposition group (defaults to ``group``)
    lig_group : extra group (e.g. ligand residue name) appended to the analysis group
    index_file : GROMACS ``.ndx`` file providing named groups
    stride, begin_ps, end_ps : frame selection
    fit : least-squares superposition onto the reference before storing frames
    mass_weighted_fit : mass-weight the superposition
    system_name : label used in result tables

    Returns
    -------
    Ensemble
        Coordinates are float32 (n_frames, n_atoms, 3) in Å, as in the pipeline.
    """
    topology = Path(topology)
    if isinstance(trajectories, Mapping):
        traj_map = {str(k): Path(v) for k, v in trajectories.items()}
    else:
        traj_map = {f"Rep_{i + 1}": Path(p) for i, p in enumerate(trajectories)}
    if not traj_map:
        raise ValueError("At least one trajectory is required.")
    fit_group = fit_group or group
    ndx = parse_ndx(index_file) if index_file else None
    first = next(iter(traj_map.values()))
    analysis_idx = resolve_selection_indices(topology, group, ndx, extra_group=lig_group)
    fit_idx = resolve_selection_indices(topology, fit_group, ndx)
    atom_info = atom_info_for_indices(topology, analysis_idx)
    ligand_mask = None
    if lig_group:
        lig_idx = resolve_selection_indices(topology, lig_group, ndx)
        ligand_mask = np.isin(analysis_idx, lig_idx)

    reference_fit = load_reference_coordinates(
        topology, atom_indices=fit_idx, fallback_trajectory=first
    )
    reference = load_reference_coordinates(
        topology, atom_indices=analysis_idx, fallback_trajectory=first
    )
    if fit:
        reference = superpose_frame(reference_fit, reference, reference_fit)

    coords: dict[str, np.ndarray] = {}
    times: dict[str, np.ndarray] = {}
    fit_rmsd: dict[str, np.ndarray] = {}
    for rep_id, path in traj_map.items():
        c, t, r = stream_trajectory_coordinates(
            topology,
            path,
            atom_indices=analysis_idx,
            stride=stride,
            fit=fit,
            fit_atom_indices=fit_idx,
            reference_fit_coords=reference_fit if fit else None,
            mass_weighted=mass_weighted_fit,
            begin_ps=begin_ps,
            end_ps=end_ps,
            return_fit_rmsd=True,
        )
        coords[rep_id], times[rep_id], fit_rmsd[rep_id] = c, t, r
    return Ensemble(
        topology=topology,
        trajectories=traj_map,
        coords=coords,
        times=times,
        fit_rmsd=fit_rmsd,
        atom_indices=np.asarray(analysis_idx),
        fit_indices=np.asarray(fit_idx),
        atom_info=atom_info,
        reference=reference,
        ligand_mask=ligand_mask,
        group=group,
        fit_group=fit_group,
        system_name=system_name,
    )


# =============================================================================
# 2. Decomposition
# =============================================================================


def _coords_and_times(
    data: Ensemble | Mapping[str, np.ndarray],
    times: Mapping[str, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    if isinstance(data, Ensemble):
        return dict(data.coords), dict(data.times), data.system_name
    coords = {str(k): np.asarray(v) for k, v in data.items()}
    if times is None:
        times = {k: np.arange(v.shape[0], dtype=float) for k, v in coords.items()}
    return coords, {str(k): np.asarray(v, dtype=float) for k, v in times.items()}, "System"


def pca(
    data: Ensemble | Mapping[str, np.ndarray],
    *,
    n_components: int = 10,
    weighting: str = "equal-replicate",
    method: str = "cartesian",
    lag: int = 10,
    times: Mapping[str, np.ndarray] | None = None,
    system_name: str | None = None,
) -> PCAResults:
    """Principal component (or TICA) decomposition of an ensemble.

    Parameters
    ----------
    data : :class:`Ensemble` or mapping replicate_id -> (n_frames, n_atoms, 3) / (n_frames, n_features)
    n_components : components retained
    weighting : ``'equal-replicate'`` (each replicate total weight 1/K) or ``'frames'`` (1/N)
    method : ``'cartesian'`` (covariance PCA) or ``'tica'`` (time-lagged ICA at ``lag`` frames)
    lag : TICA lag in frames
    times : per-replicate times (ps) when ``data`` is a plain mapping (defaults to frame index)
    system_name : label in the projection table (defaults to the ensemble's)

    Returns
    -------
    PCAResults
        ``eigenvalues`` (Å² for Cartesian PCA; autocorrelations for TICA),
        ``eigenvectors`` (3·n_atoms, k), ``explained_variance`` (%),
        ``projections`` table with columns system, replicate, frame, time_ps, PC1…PCk,
        ``mean_coords`` (3·n_atoms,).
    """
    coords, times_d, default_name = _coords_and_times(data, times)
    name = system_name or default_name
    if method == "cartesian":
        evals, evecs, exp_var, cum_var, mean = compute_cartesian_pca(
            coords, weighting=weighting, n_components=n_components
        )
    elif method == "tica":
        evals, evecs, exp_var, cum_var, mean = compute_tica(
            coords, lag=lag, n_components=n_components, weighting=weighting
        )
    else:
        raise ValueError("method must be 'cartesian' or 'tica'")
    dfs = []
    for rep, c in coords.items():
        proj = project_coordinates(c, mean, evecs)
        df = pd.DataFrame(proj, columns=[f"PC{i + 1}" for i in range(proj.shape[1])])
        df.insert(0, "system", name)
        df.insert(1, "replicate", rep)
        df.insert(2, "frame", np.arange(proj.shape[0]))
        df.insert(3, "time_ps", times_d[rep])
        dfs.append(df)
    return PCAResults(
        eigenvalues=evals,
        eigenvectors=evecs,
        explained_variance=exp_var,
        cumulative_variance=cum_var,
        projections=pd.concat(dfs, ignore_index=True),
        mean_coords=mean,
    )


# =============================================================================
# 3. Landscape
# =============================================================================


def landscape(
    source: PCAResults | pd.DataFrame | AnalysisResults,
    *,
    temperature: float = 300.0,
    bins: int | None = None,
    smooth_sigma: float = 0.0,
    pcs: tuple[int, int] = (1, 2),
    cv_columns: tuple[str, str] | None = None,
    cv_names: tuple[str, str] | None = None,
    cv_unit: str = "Å",
    weights: np.ndarray | None = None,
) -> Landscape:
    """Boltzmann-inverted 2-D free-energy landscape ΔG = −RT ln(P/P_max).

    Parameters
    ----------
    source : :class:`PCAResults`, a projection table, or an :class:`AnalysisResults`
    temperature : K
    bins : square grid size; ``None`` applies the Freedman–Diaconis rule
        (:func:`conformatlas.fel.choose_bins`)
    smooth_sigma : Gaussian smoothing of the probability (grid units); 0 = none
    pcs : 1-based component indices used as CV1/CV2 (ignored when ``cv_columns`` is given)
    cv_columns : explicit column names to use as CV1 / CV2
    cv_names, cv_unit : axis labels for plots
    weights : per-frame statistical weights (reweighted ensembles)

    Returns
    -------
    Landscape
        ``fel`` is a :class:`FELGrid` (kJ/mol, NaN where unsampled); ``cv`` is the
        projection table with CV1/CV2 columns appended (input to :func:`states`).
    """
    if isinstance(source, AnalysisResults):
        table = source.pca.projections.copy()
    elif isinstance(source, PCAResults):
        table = source.projections.copy()
    else:
        table = pd.DataFrame(source).copy()
    if cv_columns is None:
        c1, c2 = f"PC{pcs[0]}", f"PC{pcs[1]}"
        names = cv_names or (c1, c2)
    else:
        c1, c2 = cv_columns
        names = cv_names or (str(c1), str(c2))
    for c in (c1, c2):
        if c not in table.columns:
            raise ValueError(f"column {c!r} not found in the projection table")
    table["CV1"], table["CV2"] = table[c1].to_numpy(), table[c2].to_numpy()
    cv1, cv2 = table["CV1"].to_numpy(), table["CV2"].to_numpy()
    nbins = int(bins) if bins is not None else choose_bins(cv1, cv2)
    fel = calculate_fel(
        cv1, cv2, temperature=temperature, bins=nbins, smooth_sigma=smooth_sigma, weights=weights
    )
    return Landscape(
        fel=fel, cv=table, cv_names=tuple(names), cv_unit=cv_unit, bins=nbins, weights=weights
    )


# =============================================================================
# 4. States
# =============================================================================


def states(
    land: Landscape,
    *,
    min_population: float = 0.05,
    max_states: int = 10,
    min_barrier: float | None = None,
    min_peak_distance: int = 3,
) -> StateAssignment:
    """Watershed segmentation of the landscape into basins and per-frame state labels.

    Parameters
    ----------
    land : output of :func:`landscape`
    min_population : minimum fraction for a basin to be reported as a major state
    max_states : maximum number of major states
    min_barrier : kJ/mol; adjacent basins separated by a lower saddle are merged
        (default 1 kT at the landscape temperature)
    min_peak_distance : minimum grid distance between seed minima

    Returns
    -------
    StateAssignment
        ``states`` (:class:`BasinState` list, populations as fractions),
        ``frame_assignments`` (CV table + ``basin_id`` + ``state``), ``basin_map``
        (grid labels) and ``barriers`` (saddles between adjacent major states).
    """
    st, unassigned, assignments, basin_map = detect_energy_basins(
        land.fel,
        land.cv,
        min_state_population=min_population,
        max_states=max_states,
        min_peak_distance=min_peak_distance,
        min_barrier=min_barrier,
        weights=land.weights,
        cv_columns=("CV1", "CV2"),
    )
    barriers = compute_state_barriers(basin_map, land.fel.free_energy, st)
    return StateAssignment(
        states=st,
        unassigned_population=unassigned,
        frame_assignments=assignments,
        basin_map=basin_map,
        barriers=barriers,
        landscape=land,
    )


# =============================================================================
# 5. Kinetics
# =============================================================================


def kinetics(assignment: StateAssignment, dt_ps: float | None = None) -> dict[str, pd.DataFrame]:
    """Transition counts, lag-1 transition matrix, dwell times and Markov MFPTs.

    Assumptions: transitions are counted between consecutive major-state frames of
    the same replicate (unassigned frames are skipped); the mean first-passage
    times assume a Markov chain at lag one frame and are indicative only.

    Returns the dictionary of tables described in
    :func:`conformatlas.kinetics.compute_state_kinetics`.
    """
    if assignment.landscape.weights is not None:
        raise ValueError("kinetics are not defined for reweighted (weighted) ensembles")
    return compute_state_kinetics(assignment.frame_assignments, assignment.labels, dt_ps=dt_ps)


# =============================================================================
# 6. Convergence
# =============================================================================


def convergence(
    data: Ensemble | Mapping[str, np.ndarray],
    pcs: PCAResults,
    assignment: StateAssignment | None = None,
    *,
    n_modes: int = 10,
) -> ConvergenceResults:
    """Sampling diagnostics: half-vs-half and progressive RMSIP, cosine content,
    statistical inefficiency / N_eff and state-population stability.

    Parameters
    ----------
    data : :class:`Ensemble` or mapping replicate_id -> coordinates used for the PCA
    pcs : the :class:`PCAResults` of that ensemble
    assignment : optional :class:`StateAssignment` (population stability needs it)
    n_modes : size of the essential subspace compared
    """
    coords = (
        dict(data.coords)
        if isinstance(data, Ensemble)
        else {str(k): np.asarray(v) for k, v in data.items()}
    )
    if assignment is not None:
        fa = assignment.frame_assignments
    else:
        fa = pcs.projections.copy()
        fa["state"] = "Unassigned"
    return compute_convergence_diagnostics(coords, pcs.projections, fa, n_modes=n_modes)


# =============================================================================
# 7. Full pipeline
# =============================================================================


def analyze(
    topology: str | Path,
    trajectories: Sequence[str | Path],
    *,
    output_dir: str | Path = "Outputs",
    echo: Callable[[str], None] | None = None,
    **options: Any,
) -> AnalysisResults:
    """Run the complete ``conformatlas analyze`` pipeline and return its results.

    Keyword options are the fields of :class:`~conformatlas.models.AnalysisConfig`
    (``temperature``, ``lsq_group``, ``cov_group``, ``lig_group``, ``bins``,
    ``smooth_sigma``, ``pca_weighting``, ``stride``, ``min_state_population``,
    ``max_states``, ``n_components``, ``pca_type``, ``tica_lag``, ``cv``,
    ``fel_pcs``, ``index_file``, ``n_bootstrap``, ``seed``, ``begin_ps``,
    ``end_ps``, ``equilibration``, ``fit``, ``mass_weighted_fit``,
    ``n_representatives``, ``representative_method``, ``system_name``, …).
    Unknown names raise ``TypeError``. ``echo`` receives progress lines.
    """
    from conformatlas.pipeline import run_analysis

    valid = {f.name for f in dataclasses.fields(AnalysisConfig)}
    unknown = sorted(set(options) - valid)
    if unknown:
        raise TypeError(f"Unknown analyze() options: {unknown}. Valid: {sorted(valid)}")
    cfg = AnalysisConfig(
        topology=Path(topology),
        trajectories=[Path(p) for p in trajectories],
        output_dir=Path(output_dir),
        **options,
    )
    return run_analysis(cfg, echo=echo) if echo is not None else run_analysis(cfg)


# =============================================================================
# 8. Plotting for notebooks
# =============================================================================


def plot_landscape(
    source: Landscape | FELGrid | AnalysisResults,
    ax=None,
    *,
    states: Sequence[BasinState] | None = None,
    vmax: float | None = None,
    cmap: str = "turbo",
    levels: int = 40,
    contour: bool = True,
    annotate: bool = True,
    cv_names: tuple[str, str] | None = None,
    cv_unit: str | None = None,
):
    """Draw the 2-D free-energy landscape on a matplotlib axis.

    Parameters
    ----------
    source : :class:`Landscape`, :class:`FELGrid` or :class:`AnalysisResults`
    ax : existing axis (a new figure is created when None)
    states : basin minima to mark (defaults to ``source.states`` for AnalysisResults)
    vmax : colour-scale ceiling in kJ/mol (default: 98th percentile, at most 35)
    cmap, levels, contour, annotate : appearance
    cv_names, cv_unit : axis labels (default from the source)

    Returns
    -------
    (fig, ax, mappable)
        The figure, the axis and the filled-contour mappable (for colour bars).
    """
    import matplotlib.pyplot as plt

    if isinstance(source, AnalysisResults):
        fel = source.fel
        names = cv_names or tuple(
            source.run_metadata.get("collective_variables", {}).get("names", ("PC1", "PC2"))
        )
        unit = cv_unit if cv_unit is not None else "Å"
        if states is None:
            states = source.states
    elif isinstance(source, Landscape):
        fel = source.fel
        names = cv_names or source.cv_names
        unit = cv_unit if cv_unit is not None else source.cv_unit
    else:
        fel = source
        names = cv_names or ("PC1", "PC2")
        unit = cv_unit if cv_unit is not None else "Å"

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
    else:
        fig = ax.figure
    X, Y = np.meshgrid(fel.x_centers, fel.y_centers)
    Z = fel.free_energy
    finite = Z[np.isfinite(Z)]
    top = (
        vmax
        if vmax is not None
        else (min(float(np.percentile(finite, 98)), 35.0) if finite.size else 25.0)
    )
    top = max(top, 1e-6)
    lv = np.linspace(0.0, top, int(levels))
    ax.set_facecolor("#f2f2f2")
    cf = ax.contourf(X, Y, Z, levels=lv, cmap=cmap, extend="max")
    if contour:
        ax.contour(X, Y, Z, levels=lv[::4], colors="black", linewidths=0.4, alpha=0.5)
    if states:
        for s in states:
            ax.plot(s.min_pc1, s.min_pc2, marker="*", ms=12, mec="black", mfc="white", ls="none")
            if annotate:
                ax.annotate(
                    f"{s.label} ({s.population * 100:.0f}%)",
                    (s.min_pc1, s.min_pc2),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8, "lw": 0.5},
                )
    suffix = f" ({unit})" if unit else ""
    ax.set_xlabel(f"{names[0]}{suffix}")
    ax.set_ylabel(f"{names[1]}{suffix}")
    ax.set_title(f"Free energy landscape (T = {fel.temperature:.0f} K)")
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("ΔG (kJ/mol)")
    return fig, ax, cf
