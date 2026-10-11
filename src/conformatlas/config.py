"""Configuration loading, validation, and generation for ConformAtlas."""

from pathlib import Path
from typing import Any

import yaml

from conformatlas.models import ComparisonConfig, SystemConfig


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file safely."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config in {path}: expected dictionary root.")
    return data


def parse_comparison_config(config_path: str | Path) -> ComparisonConfig:
    """Parse a YAML comparison configuration into a typed ComparisonConfig."""
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    data = load_yaml_config(config_file)
    project = data.get("project", "Comparison_Project")
    temperature = float(data.get("temperature", 300.0))
    raw_output = data.get("output_dir", f"Outputs/{project}")
    output_dir = Path(raw_output)
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()

    pca_dict = data.get("pca", {})
    fit_group = pca_dict.get("fit_group", "Backbone")
    analysis_group = pca_dict.get("analysis_group", "Backbone")
    reference = pca_dict.get("reference", "combined")
    reference_system = pca_dict.get("reference_system", None)
    components = int(pca_dict.get("components", 10))
    fit = bool(pca_dict.get("fit", True))
    mass_weighted_fit = bool(pca_dict.get("mass_weighted_fit", False))

    bins_raw = data.get("bins", "auto")
    bins = None if bins_raw in (None, "auto") else int(bins_raw)
    smooth_sigma = float(data.get("smooth_sigma", 0.0))
    begin_ps = data.get("begin_ps", None)
    end_ps = data.get("end_ps", None)
    index_file = data.get("index_file", None)
    n_bootstrap = int(data.get("n_bootstrap", 200))
    seed = int(data.get("seed", 0))
    min_state_population = float(data.get("min_state_population", 0.05))
    max_states = int(data.get("max_states", 10))
    min_barrier_raw = data.get("min_barrier", None)
    min_barrier = float(min_barrier_raw) if min_barrier_raw is not None else None
    stride = int(data.get("stride", 1))
    bootstrap_mode = str(data.get("bootstrap_mode", "fixed"))
    fig_format = data.get("fig_format", None)
    colormap = data.get("colormap", None)
    font_scale = data.get("font_scale", None)
    fig_dpi = data.get("fig_dpi", None)
    interactive = data.get("interactive", None)
    plotly_js = data.get("plotly_js", None)

    systems_raw = data.get("systems", [])
    if not systems_raw:
        raise ValueError("Comparison config must contain at least one system under 'systems'.")

    def _resolve_file(p: str | Path) -> Path:
        cand = Path(p)
        if cand.is_absolute() or cand.exists():
            return cand.resolve()
        rel_to_config = config_dir / cand
        if rel_to_config.exists():
            return rel_to_config.resolve()
        return cand

    systems = []
    for s_raw in systems_raw:
        s_name = s_raw.get("name")
        if not s_name:
            raise ValueError("Each system must have a 'name' field.")
        s_topo = _resolve_file(s_raw.get("topology"))
        s_trajs = [_resolve_file(t) for t in s_raw.get("trajectories", [])]
        if not s_trajs:
            raise ValueError(
                f"System '{s_name}' must have at least one trajectory in 'trajectories'."
            )
        systems.append(SystemConfig(name=s_name, topology=s_topo, trajectories=s_trajs))

    return ComparisonConfig(
        project=project,
        temperature=temperature,
        systems=systems,
        output_dir=output_dir,
        fit_group=fit_group,
        analysis_group=analysis_group,
        reference=reference,
        reference_system=reference_system,
        components=components,
        bins=bins,
        smooth_sigma=smooth_sigma,
        min_state_population=min_state_population,
        max_states=max_states,
        stride=stride,
        fit=fit,
        mass_weighted_fit=mass_weighted_fit,
        min_barrier=min_barrier,
        begin_ps=float(begin_ps) if begin_ps is not None else None,
        end_ps=float(end_ps) if end_ps is not None else None,
        index_file=_resolve_file(index_file) if index_file else None,
        n_bootstrap=n_bootstrap,
        seed=seed,
        bootstrap_mode=bootstrap_mode,
        fig_format=fig_format,
        colormap=colormap,
        font_scale=None if font_scale is None else float(font_scale),
        fig_dpi=None if fig_dpi is None else int(fig_dpi),
        interactive=None if interactive is None else bool(interactive),
        plotly_js=plotly_js,
    )


def generate_example_config(config_type: str = "compare") -> str:
    """Generate starter YAML configuration text."""
    if config_type == "compare":
        return """# ConformAtlas WT vs Mutant Comparison Configuration
project: WT_vs_Mutant

temperature: 300.0  # Kelvin (mandatory, > 0)
output_dir: Outputs/comparison

pca:
  fit_group: Backbone
  analysis_group: Backbone
  reference: combined       # Options: 'combined' or 'reference-system'
  reference_system: WT      # Required if reference is 'reference-system'
  components: 10
  fit: true                 # superpose frames on fit_group before PCA (false only for pre-fitted trajectories)
  mass_weighted_fit: false

bins: auto                  # integer or 'auto' (Freedman-Diaconis)
smooth_sigma: 1.0           # Gaussian smoothing sigma (0.0 for raw)
min_state_population: 0.05  # Minimum fraction for major state (5%)
max_states: 10
# min_barrier: 2.5          # kJ/mol; basins separated by less are merged (default: 1 kT)
begin_ps: null              # time window applied to every trajectory
end_ps: null
index_file: null            # GROMACS .ndx for group names
n_bootstrap: 200
bootstrap_mode: fixed       # fixed | resegment (re-run the watershed on every resample)
stride: 1                   # Frame stride

# Figures and report (optional; defaults keep png / turbo / 1.0 / 300 dpi)
# fig_format: png           # png | svg | pdf | all | png,svg
# colormap: turbo
# font_scale: 1.0
# fig_dpi: 300
# interactive: null         # comparison_interactive.html when plotly is installed
# plotly_js: inline         # inline | cdn

systems:
  - name: WT
    topology: WT/topology.tpr
    trajectories:
      - WT/rep1.xtc
      - WT/rep2.xtc
      - WT/rep3.xtc

  - name: Mutant
    topology: Mutant/topology.tpr
    trajectories:
      - Mutant/rep1.xtc
      - Mutant/rep2.xtc
      - Mutant/rep3.xtc
"""
    elif config_type == "analyze":
        return """# ConformAtlas single-system / multi-replicate analysis configuration
# Every key corresponds to a `conformatlas analyze` option; flags on the command
# line override values given here.
system_name: WT
topology: topology.gro        # .gro/.pdb also serves as superposition reference; .tpr uses frame 0
temperature: 300.0            # Kelvin (mandatory, > 0)
output_dir: Outputs/WT

trajectories:
  - rep1.xtc
  - rep2.xtc
  - rep3.xtc

# Time window and equilibration
begin_ps: null                # e.g. 5000 to skip the first 5 ns
end_ps: null
equilibration: none           # 'auto' discards the transient detected per replicate (reported either way)
stride: 1

# Selections (GROMACS group names, MDAnalysis selection strings, or .ndx groups)
index_file: null              # index.ndx
lsq_group: Backbone           # fit group
cov_group: Backbone           # analysis group
lig_group: null               # e.g. ZMR / ATP: added to the analysis group, own RMSD reported
fit: true                     # superpose on lsq_group before PCA (false only for pre-fitted trajectories)
mass_weighted_fit: false

# Decomposition
pca_type: cartesian           # cartesian | dihedral | tica
tica_lag: 10                  # frames (tica only)
n_components: 10
pca_weighting: equal-replicate  # equal-replicate | frames
pca_backend: auto             # auto | python | gromacs

# Collective variables of the landscape
cv: pc                        # pc | rmsd-rg | file
fel_pcs: [1, 2]               # which components (cv: pc)
cv_files: []                  # one file per trajectory (cv: file)
cv_columns: [1, 2]            # 0-based columns in cv_files
cv_labels: null               # ["d1 (nm)", "d2 (nm)"]
weights_files: []             # per-frame weights for biased sampling (one per trajectory)

# Landscape and states
bins: auto                    # integer or 'auto' (Freedman-Diaconis)
smooth_sigma: 0.0             # 1.0 for mild Gaussian smoothing
min_state_population: 0.05
max_states: 10
min_barrier: null             # kJ/mol; null = 1 kT
n_bootstrap: 200              # block-bootstrap resamples (0 disables)
seed: 0

bootstrap_mode: fixed         # fixed | resegment (re-run the watershed on every resample)
equilibration_on: rmsd        # rmsd | cv (Chodera criterion on both landscape CVs, latest t0)

# Markov state model (after basin detection)
msm: false
msm_lag: 10                   # frames
msm_lags: null                # e.g. [1, 2, 5, 10, 20, 50] for the implied-timescale scan
msm_n_clusters: 100
msm_clustering: kmeans        # kmeans | regular-space (needs msm_dmin)
msm_dmin: null
msm_n_components: 3           # leading components used for clustering
msm_n_metastable: null        # PCCA+ sets; null = number of landscape states
msm_reversible: true
msm_ck_k: 5                   # Chapman-Kolmogorov multiples of the lag

# Residue-level analyses
residue_analysis: false       # DCCM, per-residue state differences, ligand contacts (residues/)
ligand_selection: null        # MDAnalysis selection, e.g. "resname ZMR"; null = from lig_group
contact_cutoff: 4.5           # Å, heavy atoms
dihedral_set: backbone        # backbone | backbone+chi | chi | nucleic (pca_type: dihedral)

# Engine and performance
engine: auto                  # auto | memory | streaming (chunked two-pass PCA, O(D^2) memory)
precision: float64            # float64 | float32 (stored projections, streaming engine)
n_workers: 1                  # worker processes for the streaming engine (one per trajectory)
chunk_size: 256               # frames per chunk (streaming engine)
save_hdf5: false              # also write results.h5 (requires h5py)
from_results: null            # existing output directory: rebin / resegment / MSM without trajectories

# Structures and output
n_representatives: 1
representative_method: minimum  # minimum | medoid
pdf: false
fel_backend: python

# Figures and report (null keeps the defaults: png, turbo, 1.0, 300 dpi)
fig_format: null              # png | svg | pdf | all | png,svg
colormap: null
font_scale: null
fig_dpi: null
path_overlay: false           # time-coloured replicate paths on the static landscape
interactive: null             # Plotly landscape + NGL viewer in report.html when plotly is installed
plotly_js: null               # inline | cdn
"""
    else:
        raise ValueError(f"Unknown config type: {config_type}. Supported: 'compare', 'analyze'")


# ----------------------------------------------------------------------------
# analyze: YAML + command-line merge
# ----------------------------------------------------------------------------

_ANALYZE_DEFAULTS = {
    "system_name": "System",
    "temperature": 300.0,
    "output": "Outputs",
    "lsq_group": "Backbone",
    "cov_group": "Backbone",
    "lig_group": None,
    "bins": "auto",
    "smooth_sigma": 0.0,
    "pca_weighting": "equal-replicate",
    "stride": 1,
    "min_state_population": 0.05,
    "max_states": 10,
    "fel_backend": "python",
    "gmx_command": None,
    "fit": True,
    "mass_weighted_fit": False,
    "min_barrier": None,
    "begin_ps": None,
    "end_ps": None,
    "equilibration": "none",
    "pca_type": "cartesian",
    "tica_lag": 10,
    "n_components": 10,
    "pca_backend": "auto",
    "cv": "pc",
    "fel_pcs": (1, 2),
    "cv_files": (),
    "cv_columns": (1, 2),
    "cv_labels": None,
    "weights_files": (),
    "index_file": None,
    "n_bootstrap": 200,
    "seed": 0,
    "n_representatives": 1,
    "representative_method": "minimum",
    "pdf": False,
    # analysis extensions
    "msm": False,
    "msm_lag": 10,
    "msm_lags": None,
    "msm_n_clusters": 100,
    "msm_clustering": "kmeans",
    "msm_dmin": None,
    "msm_n_components": 3,
    "msm_n_metastable": None,
    "msm_reversible": True,
    "msm_ck_k": 5,
    "bootstrap_mode": "fixed",
    "equilibration_on": "rmsd",
    "engine": "auto",
    "precision": "float64",
    "n_workers": 1,
    "chunk_size": 256,
    "save_hdf5": False,
    "from_results": None,
    "progress": True,
    "residue_analysis": False,
    "ligand_selection": None,
    "contact_cutoff": 4.5,
    "dihedral_set": "backbone",
    "fig_format": None,
    "colormap": None,
    "font_scale": None,
    "fig_dpi": None,
    "path_overlay": None,
    "interactive": None,
    "plotly_js": None,
}

_YAML_ALIASES = {
    "output_dir": "output",
    "trajectory": "trajectories",
    "fel_pcs": "fel_pcs",
    "cv_cols": "cv_columns",
    "msm_clusters": "msm_n_clusters",
    "msm_components": "msm_n_components",
    "msm_states": "msm_n_metastable",
    "pca_engine": "engine",
    "from": "from_results",
}


def _int_list(value, name: str) -> list[int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [v for v in value.replace(";", ",").split(",") if v.strip()]
    else:
        parts = list(value)
    out = [int(v) for v in parts]
    if not out:
        raise ValueError(f"{name} needs at least one integer")
    return out


def _pair(value, name: str) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [v for v in value.replace(";", ",").split(",") if v.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"{name} needs exactly two values, got {value!r}")
    return int(parts[0]), int(parts[1])


def build_analysis_config(options: dict, config_file: str | Path | None = None):
    """Merge YAML (if any), command-line options and defaults into an AnalysisConfig.

    ``options`` is the dict of click parameters; ``None`` means "not given" and
    falls back to the YAML value, then to the built-in default. Relative paths in
    the YAML are resolved against the YAML's directory.
    """
    from conformatlas.models import AnalysisConfig

    merged: dict = dict(_ANALYZE_DEFAULTS)
    given: set[str] = set()
    trajectories: list = []
    structure = None
    base_dir = Path.cwd()
    if config_file:
        cfg_path = Path(config_file).resolve()
        base_dir = cfg_path.parent
        data = load_yaml_config(cfg_path) or {}
        for key, value in data.items():
            key = _YAML_ALIASES.get(key, key)
            if key == "topology" or key == "structure":
                structure = value
            elif key == "trajectories":
                trajectories = list(value or [])
            elif key in merged:
                merged[key] = value
                given.add(key)
            else:
                raise ValueError(f"Unknown key '{key}' in {cfg_path.name}")

    def _resolve(p):
        if p is None:
            return None
        cand = Path(p)
        if cand.is_absolute() or cand.exists():
            return cand
        rel = base_dir / cand
        return rel if rel.exists() else cand

    # command line overrides
    for key, value in options.items():
        if key == "structure":
            if value is not None:
                structure = value
        elif key == "trajectory":
            if value:
                trajectories = list(value)
        elif value is None or (isinstance(value, tuple) and len(value) == 0):
            continue
        else:
            merged[key] = value
            given.add(key)

    from_results = _resolve(merged["from_results"])
    if from_results is not None:
        from_results = Path(from_results)
        if not from_results.is_dir():
            raise FileNotFoundError(f"--from: {from_results} is not a results directory")
        # topology / trajectories default to those recorded by the previous run
        import json

        meta_path = from_results / "run_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        # settings not given on the command line / YAML default to the stored run
        stored = {
            "temperature": meta.get("temperature_K"),
            "bins": meta.get("bins")
            if meta.get("bins_mode", "user") != "auto (Freedman-Diaconis)"
            else None,
            "smooth_sigma": meta.get("smooth_sigma"),
            "min_barrier": meta.get("min_barrier_kJ_mol"),
            "min_state_population": meta.get("min_state_population"),
            "max_states": meta.get("max_states"),
            "n_bootstrap": (meta.get("bootstrap") or {}).get("n"),
            "seed": (meta.get("bootstrap") or {}).get("seed"),
            "system_name": meta.get("system"),
            "weights_files": meta.get("weights_files") or None,
        }
        for key, value in stored.items():
            if key not in given and value is not None:
                merged[key] = value
        if structure is None:
            structure = meta.get("topology", "unknown.pdb")
        if not trajectories:
            trajectories = list(meta.get("trajectories", []))
        if not trajectories:
            trajectories = ["<from results>"]
        structure = _resolve(structure)
        trajectories = [_resolve(t) for t in trajectories]
    else:
        if structure is None:
            raise ValueError(
                "A structure/topology file is required (-s or 'topology' in the config)."
            )
        if not trajectories:
            raise ValueError(
                "At least one trajectory is required (-f or 'trajectories' in the config)."
            )
        structure = _resolve(structure)
        trajectories = [_resolve(t) for t in trajectories]
        for pth in [structure, *trajectories]:
            if not Path(pth).exists():
                raise FileNotFoundError(f"File not found: {pth}")

    bins = merged["bins"]
    bins = None if bins in (None, "auto", "") else int(bins)
    cv_labels = merged["cv_labels"]
    if isinstance(cv_labels, str):
        cv_labels = tuple(v.strip() for v in cv_labels.split(","))
        if len(cv_labels) != 2:
            raise ValueError("--cv-labels needs two comma-separated names")
    elif cv_labels is not None:
        cv_labels = tuple(cv_labels)

    return AnalysisConfig(
        topology=structure,
        trajectories=trajectories,
        temperature=float(merged["temperature"]),
        lsq_group=merged["lsq_group"],
        cov_group=merged["cov_group"],
        lig_group=merged["lig_group"] or None,
        output_dir=_resolve(merged["output"]) or Path("Outputs"),
        bins=bins,
        smooth_sigma=float(merged["smooth_sigma"]),
        pca_weighting=merged["pca_weighting"],
        stride=int(merged["stride"]),
        min_state_population=float(merged["min_state_population"]),
        max_states=int(merged["max_states"]),
        fel_backend=merged["fel_backend"],
        gmx_command=merged["gmx_command"],
        system_name=merged["system_name"],
        fit=bool(merged["fit"]),
        mass_weighted_fit=bool(merged["mass_weighted_fit"]),
        min_barrier=None if merged["min_barrier"] is None else float(merged["min_barrier"]),
        begin_ps=None if merged["begin_ps"] is None else float(merged["begin_ps"]),
        end_ps=None if merged["end_ps"] is None else float(merged["end_ps"]),
        equilibration=merged["equilibration"],
        pca_type=merged["pca_type"],
        tica_lag=int(merged["tica_lag"]),
        n_components=int(merged["n_components"]),
        pca_backend=merged["pca_backend"],
        cv=merged["cv"],
        fel_pcs=_pair(merged["fel_pcs"], "fel_pcs"),
        cv_files=[_resolve(f) for f in (merged["cv_files"] or [])],
        cv_columns=_pair(merged["cv_columns"], "cv_columns"),
        cv_labels=cv_labels,
        weights_files=[_resolve(f) for f in (merged["weights_files"] or [])],
        index_file=_resolve(merged["index_file"]),
        n_bootstrap=int(merged["n_bootstrap"]),
        seed=int(merged["seed"]),
        n_representatives=int(merged["n_representatives"]),
        representative_method=merged["representative_method"],
        pdf=bool(merged["pdf"]),
        msm=bool(merged["msm"]),
        msm_lag=int(merged["msm_lag"]),
        msm_lags=_int_list(merged["msm_lags"], "msm_lags"),
        msm_n_clusters=int(merged["msm_n_clusters"]),
        msm_clustering=merged["msm_clustering"],
        msm_dmin=None if merged["msm_dmin"] is None else float(merged["msm_dmin"]),
        msm_n_components=int(merged["msm_n_components"]),
        msm_n_metastable=None
        if merged["msm_n_metastable"] is None
        else int(merged["msm_n_metastable"]),
        msm_reversible=bool(merged["msm_reversible"]),
        msm_ck_k=int(merged["msm_ck_k"]),
        bootstrap_mode=merged["bootstrap_mode"],
        equilibration_on=merged["equilibration_on"],
        engine=merged["engine"],
        precision=merged["precision"],
        n_workers=int(merged["n_workers"]),
        chunk_size=int(merged["chunk_size"]),
        save_hdf5=bool(merged["save_hdf5"]),
        from_results=from_results,
        progress=bool(merged["progress"]),
        residue_analysis=bool(merged["residue_analysis"]),
        ligand_selection=merged["ligand_selection"] or None,
        contact_cutoff=float(merged["contact_cutoff"]),
        dihedral_set=merged["dihedral_set"],
        fig_format=merged["fig_format"],
        colormap=merged["colormap"],
        font_scale=None if merged["font_scale"] is None else float(merged["font_scale"]),
        fig_dpi=None if merged["fig_dpi"] is None else int(merged["fig_dpi"]),
        path_overlay=None if merged["path_overlay"] is None else bool(merged["path_overlay"]),
        interactive=None if merged["interactive"] is None else bool(merged["interactive"]),
        plotly_js=merged["plotly_js"],
    )
