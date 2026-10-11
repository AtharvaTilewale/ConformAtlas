"""Persistence of :class:`~conformatlas.models.AnalysisResults`.

Three entry points:

* :func:`save_results` / :func:`load_results` — one HDF5 container (``h5py``)
  holding projections, eigen-data, FEL grids, basin labels, state tables,
  convergence / uncertainty tables, kinetics, bootstrap, metadata and warnings.
  Every array is stored as a dataset, every table as a group of column
  datasets, every dataclass as a group; the structure is self-describing
  (``__type__`` attributes), so the round trip restores the same Python objects.
  The CSV / NPY files written by the pipeline are unchanged — the container is
  an addition for programmatic reuse.
* :func:`load_results_dir` — rebuild an ``AnalysisResults`` from an existing
  results directory (``pca/``, ``fel/``, ``states/``, ``statistics/``,
  ``convergence/``, ``kinetics/``, ``run_metadata.json``). Eigenvectors are not
  written by the pipeline, so ``pca.eigenvectors`` is empty after this route;
  everything needed to re-bin or re-segment the landscape is recovered.
* :func:`rebin` — recompute the free-energy landscape, basins, barriers and
  population statistics from the stored collective variables with new ``bins``
  / ``smooth_sigma`` without touching the PCA. Validated to equal a fresh run
  with the same settings (``validation/checks_performance.py``).

``h5py`` is an optional dependency: the module imports without it and the
HDF5 functions raise a clear error.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import conformatlas
from conformatlas.basins import compute_state_barriers, detect_energy_basins
from conformatlas.fel import calculate_fel
from conformatlas.models import (
    AnalysisConfig,
    AnalysisResults,
    BasinState,
    ConvergenceResults,
    FELGrid,
    PCAResults,
    TrajectoryInfo,
)
from conformatlas.uncertainty import compute_replicate_statistics
from conformatlas.utils import NumpyEncoder

try:
    import h5py

    HAS_H5PY = True
except ImportError:  # pragma: no cover
    HAS_H5PY = False

logger = logging.getLogger("conformatlas.results_io")

__all__ = [
    "HAS_H5PY",
    "FORMAT_VERSION",
    "save_results",
    "load_results",
    "load_results_dir",
    "rebin",
]

FORMAT_VERSION = 1

#: Dataclasses that may be reconstructed from the container (module allow-list).
_ALLOWED_MODULES = ("conformatlas.models", "conformatlas.bootstrap")


def _require_h5py() -> None:
    if not HAS_H5PY:
        raise RuntimeError("h5py is required for the HDF5 results container (pip install h5py).")


# =============================================================================
# Generic object <-> HDF5 encoding
# =============================================================================


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str, np.generic))


def _json_safe(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, tuple):
        return list(v)
    return v


class _Encoder(NumpyEncoder):
    """NumpyEncoder plus Path support."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _write_json(group, name: str, value: Any, kind: str = "json") -> None:
    text = json.dumps(_json_safe(value), cls=_Encoder)
    ds = group.create_dataset(name, data=np.bytes_(text.encode("utf-8")))
    ds.attrs["__type__"] = kind


def _write_dataframe(group, df: pd.DataFrame) -> None:
    group.attrs["__type__"] = "dataframe"
    group.attrs["columns"] = json.dumps([str(c) for c in df.columns])
    group.attrs["column_names_are_str"] = bool(all(isinstance(c, str) for c in df.columns))
    cols = group.create_group("columns")
    for i, col in enumerate(df.columns):
        _write_series(cols, f"c{i}", df[col])
    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0 or df.index.step != 1:
        _write_series(group, "index", pd.Series(df.index.to_numpy()))
        group.attrs["index_name"] = json.dumps(df.index.name)


def _write_series(group, name: str, s: pd.Series) -> None:
    values = s.to_numpy()
    if values.dtype.kind in "OUS":
        # strings (and None) as variable-length UTF-8
        text = np.array(["" if v is None else str(v) for v in values], dtype=object)
        ds = group.create_dataset(name, data=text, dtype=h5py.string_dtype("utf-8"))
        ds.attrs["kind"] = "str"
        if values.dtype.kind == "O" and any(v is None for v in values):
            ds.attrs["none_mask"] = np.array([v is None for v in values])
    elif values.dtype.kind == "b":
        ds = group.create_dataset(name, data=values.astype(np.uint8))
        ds.attrs["kind"] = "bool"
    elif values.dtype.kind == "M":
        ds = group.create_dataset(name, data=values.astype("datetime64[ns]").astype(np.int64))
        ds.attrs["kind"] = "datetime64[ns]"
    else:
        ds = group.create_dataset(name, data=values)
        ds.attrs["kind"] = str(values.dtype)


def _read_series(ds) -> np.ndarray:
    kind = ds.attrs.get("kind", "")
    data = ds[()]
    if kind == "str":
        out = np.array(
            [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in data], dtype=object
        )
        if "none_mask" in ds.attrs:
            mask = np.asarray(ds.attrs["none_mask"], dtype=bool)
            out[mask] = None
        return out
    if kind == "bool":
        return np.asarray(data, dtype=bool)
    if kind == "datetime64[ns]":
        return np.asarray(data, dtype=np.int64).astype("datetime64[ns]")
    return np.asarray(data)


def _read_dataframe(group) -> pd.DataFrame:
    columns = json.loads(group.attrs["columns"])
    cols = group["columns"]
    data = {}
    for i, col in enumerate(columns):
        data[col] = _read_series(cols[f"c{i}"])
    df = pd.DataFrame(data, columns=columns)
    if not bool(group.attrs.get("column_names_are_str", True)):
        df.columns = [_maybe_int(c) for c in df.columns]
    if "index" in group:
        idx = _read_series(group["index"])
        name = json.loads(group.attrs.get("index_name", "null"))
        df.index = pd.Index(idx, name=name)
    return df


def _maybe_int(c: str):
    try:
        return int(c)
    except (TypeError, ValueError):
        return c


def _write_object(group, name: str, value: Any) -> None:
    """Encode ``value`` under ``group[name]``."""
    if isinstance(value, pd.DataFrame):
        _write_dataframe(group.create_group(name), value)
    elif isinstance(value, pd.Series):
        _write_dataframe(group.create_group(name), value.to_frame())
        group[name].attrs["__type__"] = "series"
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in "OUS":
            _write_json(group, name, value.tolist(), kind="json_array")
        else:
            ds = group.create_dataset(name, data=value)
            ds.attrs["__type__"] = "ndarray"
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        g = group.create_group(name)
        g.attrs["__type__"] = "dataclass"
        g.attrs["__class__"] = f"{type(value).__module__}.{type(value).__qualname__}"
        for f in dataclasses.fields(value):
            _write_object(g, f.name, getattr(value, f.name))
    elif isinstance(value, dict):
        if _json_like(value):
            _write_json(group, name, {str(k): v for k, v in value.items()}, kind="json_dict")
            group[name].attrs["key_types"] = json.dumps([type(k).__name__ for k in value])
        else:
            g = group.create_group(name)
            g.attrs["__type__"] = "dict"
            g.attrs["keys"] = json.dumps([[str(k), type(k).__name__] for k in value])
            for i, (k, v) in enumerate(value.items()):
                _write_object(g, f"k{i}", v)
    elif isinstance(value, (list, tuple)):
        if _json_like(value):
            _write_json(
                group,
                name,
                list(value),
                kind="json_list" if isinstance(value, list) else "json_tuple",
            )
        else:
            g = group.create_group(name)
            g.attrs["__type__"] = "list"
            g.attrs["n"] = len(value)
            for i, v in enumerate(value):
                _write_object(g, f"i{i}", v)
    elif isinstance(value, Path):
        _write_json(group, name, str(value), kind="path")
    elif _is_scalar(value):
        _write_json(group, name, value, kind="json")
    else:
        raise TypeError(f"Cannot serialise {name!r} of type {type(value).__name__} to HDF5.")


def _json_like(v: Any) -> bool:
    if _is_scalar(v) or isinstance(v, Path):
        return True
    if isinstance(v, (list, tuple)):
        return all(_json_like(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, (str, int, float, bool)) for k in v) and all(
            _json_like(x) for x in v.values()
        )
    return False


def _restore_keys(keys: list[str], types: list[str]) -> list:
    out = []
    for k, t in zip(keys, types):
        if t == "int":
            out.append(int(k))
        elif t == "float":
            out.append(float(k))
        elif t == "bool":
            out.append(k in ("True", "true", "1"))
        else:
            out.append(k)
    return out


def _import_dataclass(path: str):
    module, _, qual = path.rpartition(".")
    if module not in _ALLOWED_MODULES:
        raise ValueError(f"Refusing to import {path!r} from an HDF5 file (not an allowed module).")
    return getattr(importlib.import_module(module), qual)


def _read_object(node) -> Any:
    kind = node.attrs.get("__type__", None)
    if isinstance(kind, bytes):
        kind = kind.decode()
    if isinstance(node, h5py.Dataset):
        if kind in ("json", "json_list", "json_tuple", "json_dict", "json_array", "path"):
            raw = node[()]
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            value = json.loads(text)
            if kind == "json_tuple":
                return tuple(value)
            if kind == "json_array":
                return np.array(value, dtype=object)
            if kind == "path":
                return Path(value)
            if kind == "json_dict":
                types = json.loads(node.attrs.get("key_types", "[]"))
                keys = list(value.keys())
                if types and len(types) == len(keys):
                    return dict(zip(_restore_keys(keys, types), value.values()))
                return value
            return value
        return node[()]
    # groups
    if kind == "dataframe":
        return _read_dataframe(node)
    if kind == "series":
        df = _read_dataframe(node)
        return df.iloc[:, 0]
    if kind == "dataclass":
        cls = _import_dataclass(node.attrs["__class__"])
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: _read_object(node[k]) for k in node.keys() if k in field_names}
        return cls(**kwargs)
    if kind == "dict":
        entries = json.loads(node.attrs["keys"])
        keys = _restore_keys([e[0] for e in entries], [e[1] for e in entries])
        return {k: _read_object(node[f"k{i}"]) for i, k in enumerate(keys)}
    if kind == "list":
        n = int(node.attrs["n"])
        return [_read_object(node[f"i{i}"]) for i in range(n)]
    raise ValueError(f"Unknown node type {kind!r} at {node.name}")


# =============================================================================
# Public API: HDF5 container
# =============================================================================


def save_results(results: AnalysisResults, path: str | Path) -> Path:
    """Write ``results`` to one HDF5 file and return the path.

    Layout (top-level groups): ``config``, ``trajectories_info``, ``pca``,
    ``fel``, ``states``, ``frame_assignments``, ``convergence``,
    ``replicate_statistics``, ``run_metadata``, ``extras``, ``warnings`` plus
    scalar attributes ``format_version``, ``conformatlas_version``,
    ``unassigned_population`` and ``output_dir``.
    """
    _require_h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["format_version"] = FORMAT_VERSION
        h5.attrs["conformatlas_version"] = conformatlas.__version__
        h5.attrs["unassigned_population"] = float(results.unassigned_population)
        h5.attrs["output_dir"] = str(results.output_dir)
        for name in (
            "config",
            "trajectories_info",
            "pca",
            "fel",
            "states",
            "frame_assignments",
            "convergence",
            "replicate_statistics",
            "run_metadata",
            "extras",
            "warnings",
        ):
            _write_object(h5, name, getattr(results, name))
    return path


def load_results(path: str | Path) -> AnalysisResults:
    """Read an HDF5 container written by :func:`save_results`."""
    _require_h5py()
    path = Path(path)
    with h5py.File(path, "r") as h5:
        version = int(h5.attrs.get("format_version", 0))
        if version > FORMAT_VERSION:
            raise ValueError(
                f"{path} uses container format {version}; this version reads <= {FORMAT_VERSION}."
            )
        kwargs = {name: _read_object(h5[name]) for name in h5.keys()}
        kwargs["unassigned_population"] = float(h5.attrs["unassigned_population"])
        kwargs["output_dir"] = Path(str(h5.attrs["output_dir"]))
    return AnalysisResults(**kwargs)


# =============================================================================
# Public API: results directory
# =============================================================================


def _read_csv(path: Path, **kw) -> pd.DataFrame | None:
    # pandas' default float parser is not exact (it can differ from the written
    # float64 by 1-2 ULP for roughly a third of values). Re-binning reloaded
    # frames against the saved grid edges needs the exact values: a frame on the
    # outermost edge would otherwise fall one ULP outside the grid.
    kw.setdefault("float_precision", "round_trip")
    return pd.read_csv(path, **kw) if path.exists() else None


def load_results_dir(path: str | Path) -> AnalysisResults:
    """Reconstruct an :class:`AnalysisResults` from a ``conformatlas analyze`` output directory.

    Reads ``run_metadata.json``, ``pca/explained_variance.csv``,
    ``pca/projections.csv``, ``fel/*.npy``, ``states/basins.csv``,
    ``states/frame_assignments.csv``, ``states/basin_map.npy``,
    ``states/barriers.csv``, ``statistics/*.csv``, ``convergence/*.csv`` and
    ``kinetics/*.csv`` when present. Fields the pipeline never writes
    (eigenvectors, mean structure, bootstrap arrays other than the saved
    free-energy SD/CI maps) are left empty or ``None``.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a results directory")
    meta_path = root / "run_metadata.json"
    meta: dict[str, Any] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )

    # ---- configuration (as far as it was recorded) --------------------------
    cv_meta = meta.get("collective_variables", {})
    config = AnalysisConfig(
        topology=meta.get("topology", "unknown.pdb"),
        trajectories=meta.get("trajectories", []),
        temperature=float(meta.get("temperature_K", 300.0)),
        lsq_group=meta.get("lsq_group", "Backbone"),
        cov_group=meta.get("cov_group", "Backbone"),
        lig_group=meta.get("lig_group"),
        output_dir=root,
        bins=int(meta["bins"])
        if meta.get("bins_mode", "user") == "user" and "bins" in meta
        else None,
        smooth_sigma=float(meta.get("smooth_sigma", 0.0)),
        pca_weighting=meta.get("pca_weighting", "equal-replicate"),
        stride=int(meta.get("stride", 1)),
        pca_type=meta.get("pca_type", "cartesian"),
        tica_lag=int(meta.get("tica_lag_frames") or 10),
        cv=cv_meta.get("mode", "pc"),
        fel_pcs=tuple(cv_meta.get("fel_pcs", (1, 2))),
        index_file=meta.get("index_file"),
        equilibration=meta.get("equilibration", "none"),
        min_barrier=None,
        n_bootstrap=int(meta.get("bootstrap", {}).get("n", 200)),
        seed=int(meta.get("bootstrap", {}).get("seed", 0)),
        n_representatives=int(meta.get("representatives", {}).get("n", 1)),
        representative_method=meta.get("representatives", {}).get("method", "minimum"),
        fit=bool(meta.get("superposition", {}).get("enabled", True)),
        mass_weighted_fit=bool(meta.get("superposition", {}).get("mass_weighted", False)),
    )
    system_name = meta.get("system")
    if system_name:
        config.system_name = system_name

    # ---- PCA ----------------------------------------------------------------
    proj = _read_csv(root / "pca" / "projections.csv")
    if proj is None:
        raise FileNotFoundError(f"{root / 'pca' / 'projections.csv'} is missing")
    if "system" in proj.columns and len(proj):
        config.system_name = str(proj["system"].iloc[0])
    ev = _read_csv(root / "pca" / "explained_variance.csv")
    if ev is not None:
        eigenvalues = ev["Eigenvalue"].to_numpy(dtype=float)
        explained = ev["ExplainedVariance_Percent"].to_numpy(dtype=float)
        cumulative = ev["CumulativeVariance_Percent"].to_numpy(dtype=float)
    else:
        eigenvalues = explained = cumulative = np.array([])
    pca = PCAResults(
        eigenvalues=eigenvalues,
        eigenvectors=np.empty((0, 0)),
        explained_variance=explained,
        cumulative_variance=cumulative,
        projections=proj,
        mean_coords=None,
    )

    # ---- FEL ----------------------------------------------------------------
    fel_dir = root / "fel"
    prob = np.load(fel_dir / "probability.npy")
    fe = np.load(fel_dir / "free_energy.npy")
    x_edges = np.load(fel_dir / "x_edges.npy")
    y_edges = np.load(fel_dir / "y_edges.npy")
    smooth = float(meta.get("smooth_sigma", 0.0))
    # occupancy: bins with finite free energy when unsmoothed; when smoothed the
    # pipeline's raw-count mask is not written, so it is approximated by the
    # frames' bin occupancy from the frame assignments below.
    fel = FELGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=0.5 * (x_edges[:-1] + x_edges[1:]),
        y_centers=0.5 * (y_edges[:-1] + y_edges[1:]),
        probability=prob,
        free_energy=fe,
        temperature=config.temperature,
        smoothed=smooth > 0,
        smooth_sigma=smooth,
        occupancy_mask=None,
    )

    # ---- states -------------------------------------------------------------
    assignments = _read_csv(root / "states" / "frame_assignments.csv")
    if assignments is None:
        raise FileNotFoundError(f"{root / 'states' / 'frame_assignments.csv'} is missing")
    basin_map_path = root / "states" / "basin_map.npy"
    basin_map = np.load(basin_map_path) if basin_map_path.exists() else None
    if "CV1" in assignments.columns and "CV2" in assignments.columns:
        from conformatlas.basins import frame_bin_indices

        xi, yi, valid = frame_bin_indices(
            fel, assignments["CV1"].to_numpy(), assignments["CV2"].to_numpy()
        )
        occ = np.zeros_like(prob, dtype=bool)
        occ[yi[valid], xi[valid]] = True
        fel.occupancy_mask = occ
    basins_df = _read_csv(root / "states" / "basins.csv")
    states: list[BasinState] = []
    if basins_df is not None:
        label_to_basin = {}
        if basin_map is not None and "basin_id" in assignments.columns:
            for lab, grp in assignments.groupby("state"):
                if lab != "Unassigned":
                    label_to_basin[lab] = int(grp["basin_id"].mode().iloc[0])
        for i, r in basins_df.iterrows():
            states.append(
                BasinState(
                    state_id=i + 1,
                    label=str(r["State"]),
                    population=float(r["Population_Percent"]) / 100.0,
                    min_pc1=float(r["Min_PC1"]),
                    min_pc2=float(r["Min_PC2"]),
                    min_free_energy=float(r["Min_FreeEnergy_kJ_mol"]),
                    representative_replicate=str(r["Representative_Replicate"]),
                    representative_frame=int(r["Representative_Frame"]),
                    representative_time_ps=float(r["Representative_Time_ps"]),
                    representative_distance=float(r["Representative_Distance"]),
                    basin_id=label_to_basin.get(str(r["State"]), -1),
                )
            )
    unassigned = max(0.0, 1.0 - sum(s.population for s in states))

    # ---- convergence --------------------------------------------------------
    conv_dir = root / "convergence"
    rmsip = _read_csv(conv_dir / "rmsip.csv")
    rmsip_halves = float("nan")
    per_rep: dict[str, float] = {}
    assessment = "Insufficient evidence"
    if rmsip is not None and len(rmsip):
        pooled = rmsip[rmsip["Scope"] == "all_replicates_pooled"]
        if len(pooled):
            rmsip_halves = float(pooled["Value"].iloc[0])
            assessment = str(pooled["Sampling_Assessment"].iloc[0])
        for _, r in rmsip[rmsip["Scope"] != "all_replicates_pooled"].iterrows():
            per_rep[str(r["Scope"])] = float(r["Value"])
    cos = _read_csv(conv_dir / "cosine_content.csv")
    cosine = (
        {int(str(r["PC"]).replace("PC", "")): float(r["Cosine_Content"]) for _, r in cos.iterrows()}
        if cos is not None
        else {}
    )
    prog = _read_csv(conv_dir / "progressive_overlap.csv")
    progressive = (
        {int(r["Fraction_Percent"]): float(r["Subspace_RMSIP"]) for _, r in prog.iterrows()}
        if prog is not None
        else {}
    )
    eff = _read_csv(conv_dir / "effective_samples.csv")
    ineff: dict[str, dict[str, float]] = {}
    n_eff: dict[str, float] = {}
    if eff is not None:
        for _, r in eff.iterrows():
            rep = str(r["Replicate"])
            ineff[rep] = {c[2:]: float(r[c]) for c in eff.columns if c.startswith("g_")}
            n_eff[rep] = float(r["N_eff"])
    convergence = ConvergenceResults(
        rmsip_halves=rmsip_halves,
        progressive_overlap=progressive,
        cosine_content=cosine,
        population_stability=_read_csv(conv_dir / "population_stability.csv"),
        pairwise_rmsip=_read_csv(conv_dir / "pairwise_rmsip.csv", index_col=0),
        sampling_assessment=assessment,
        rmsip_halves_per_replicate=per_rep,
        statistical_inefficiency=ineff,
        n_eff=n_eff,
        equilibration=_read_csv(conv_dir / "equilibration.csv"),
    )

    # ---- statistics / extras ------------------------------------------------
    replicate_statistics = _read_csv(root / "statistics" / "state_populations.csv")
    extras: dict[str, Any] = {}
    barriers = _read_csv(root / "states" / "barriers.csv")
    if barriers is not None:
        extras["barriers"] = barriers
    if basin_map is not None:
        extras["basin_map"] = basin_map
    kin_dir = root / "kinetics"
    if kin_dir.is_dir():
        kin = {}
        for key, fname, idx in (
            ("transition_counts", "transition_counts.csv", 0),
            ("transition_counts_per_replicate", "transition_counts_per_replicate.csv", None),
            ("transition_matrix", "transition_matrix.csv", 0),
            ("dwell_times", "dwell_times.csv", None),
            ("dwell_summary", "dwell_summary.csv", None),
            ("mfpt", "mfpt_markov_ps.csv", 0),
            ("summary", "replicate_summary.csv", None),
        ):
            df = _read_csv(kin_dir / fname, index_col=idx)
            if df is not None:
                kin[key] = df
        if kin:
            extras["kinetics"] = kin
    for name in (
        "free_energy_bootstrap_sd",
        "free_energy_bootstrap_ci_low",
        "free_energy_bootstrap_ci_high",
    ):
        p = fel_dir / f"{name}.npy"
        if p.exists():
            extras[name] = np.load(p)
    desc = root / "descriptors"
    if desc.is_dir():
        d = {
            k: _read_csv(desc / f"{k}.csv")
            for k in ("timeseries", "rmsf_atoms", "rmsf_residues", "summary")
        }
        d = {k: v for k, v in d.items() if v is not None}
        if d:
            extras["descriptors"] = d

    traj_infos = [
        TrajectoryInfo(path=Path(p), replicate_id=f"Rep_{i + 1}", system_name=config.system_name)
        for i, p in enumerate(meta.get("trajectories", []))
    ]
    for info in traj_infos:
        sub = proj[proj["replicate"] == info.replicate_id] if "replicate" in proj.columns else proj
        if len(sub):
            info.n_frames = int(len(sub))
            info.start_time_ps = float(sub["time_ps"].iloc[0])
            info.end_time_ps = float(sub["time_ps"].iloc[-1])
            if len(sub) > 1:
                info.timestep_ps = float(np.median(np.diff(sub["time_ps"].to_numpy())))

    return AnalysisResults(
        config=config,
        trajectories_info=traj_infos,
        pca=pca,
        fel=fel,
        states=states,
        unassigned_population=unassigned,
        frame_assignments=assignments,
        convergence=convergence,
        replicate_statistics=replicate_statistics,
        output_dir=root,
        run_metadata=meta,
        extras=extras,
        warnings=list(meta.get("warnings", [])),
    )


# =============================================================================
# Public API: re-binning
# =============================================================================


def rebin(
    results: AnalysisResults,
    bins: int,
    smooth_sigma: float = 0.0,
    *,
    min_state_population: float | None = None,
    max_states: int | None = None,
    min_barrier: float | None = None,
    weights: np.ndarray | None = None,
    temperature: float | None = None,
) -> AnalysisResults:
    """Recompute landscape, basins, barriers and populations on a new grid.

    Uses the stored collective variables (``CV1``/``CV2`` columns of
    ``results.frame_assignments``, falling back to the ``fel_pcs`` columns of
    the projections) and leaves the PCA untouched. Returns a new
    :class:`AnalysisResults`; ``results`` is not modified. Bootstrap tables,
    kinetics and representative structures of the input are *not* recomputed
    and are dropped from ``extras`` (``barriers`` is recomputed).

    ``weights`` must be passed again for reweighted ensembles (per-frame
    weights are not stored in the results); if the original run used weight
    files and none are given, a warning is logged and populations are unweighted.
    """
    cfg = results.config
    fa = results.frame_assignments
    if "CV1" in fa.columns and "CV2" in fa.columns:
        cv_df = fa.drop(columns=[c for c in ("basin_id", "state") if c in fa.columns]).copy()
    else:
        i1, i2 = cfg.fel_pcs
        cv_df = results.pca.projections.copy()
        cv_df["CV1"], cv_df["CV2"] = cv_df[f"PC{i1}"], cv_df[f"PC{i2}"]
    if weights is None and results.run_metadata.get("weights_files"):
        logger.warning(
            "Original run used per-frame weights; rebin() without weights is unweighted."
        )
    T = float(temperature if temperature is not None else cfg.temperature)
    fel = calculate_fel(
        cv_df["CV1"].to_numpy(),
        cv_df["CV2"].to_numpy(),
        temperature=T,
        bins=int(bins),
        smooth_sigma=float(smooth_sigma),
        weights=weights,
    )
    states, unassigned, assignments, basin_map = detect_energy_basins(
        fel,
        cv_df,
        min_state_population=cfg.min_state_population
        if min_state_population is None
        else min_state_population,
        max_states=cfg.max_states if max_states is None else max_states,
        min_barrier=cfg.min_barrier if min_barrier is None else min_barrier,
        weights=weights,
        cv_columns=("CV1", "CV2"),
    )
    barriers = compute_state_barriers(basin_map, fel.free_energy, states)
    g_single = None
    if results.convergence.statistical_inefficiency:
        g_single = max(
            max(v.values()) if v else 1.0
            for v in results.convergence.statistical_inefficiency.values()
        )
    summary_pop, _detailed, _unc = compute_replicate_statistics(
        assignments, statistical_inefficiency=g_single, weights=weights
    )
    new_cfg = dataclasses.replace(
        cfg, bins=int(bins), smooth_sigma=float(smooth_sigma), temperature=T
    )
    extras = {k: v for k, v in results.extras.items() if k in ("descriptors", "tica_timescales_ps")}
    extras["barriers"] = barriers
    extras["basin_map"] = basin_map
    meta = dict(results.run_metadata)
    meta.update(
        {"bins": int(bins), "bins_mode": "user (rebin)", "smooth_sigma": float(smooth_sigma)}
    )
    return AnalysisResults(
        config=new_cfg,
        trajectories_info=list(results.trajectories_info),
        pca=results.pca,
        fel=fel,
        states=states,
        unassigned_population=unassigned,
        frame_assignments=assignments,
        convergence=results.convergence,
        replicate_statistics=summary_pop,
        output_dir=results.output_dir,
        run_metadata=meta,
        extras=extras,
        warnings=list(results.warnings),
    )
