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

    bins = int(data.get("bins", 64))
    smooth_sigma = float(data.get("smooth_sigma", 0.0))
    min_state_population = float(data.get("min_state_population", 0.05))
    max_states = int(data.get("max_states", 10))
    stride = int(data.get("stride", 1))

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
            raise ValueError(f"System '{s_name}' must have at least one trajectory in 'trajectories'.")
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

bins: 64                    # Number of 2D FEL bins along PC1 and PC2
smooth_sigma: 1.0           # Gaussian smoothing sigma (0.0 for raw)
min_state_population: 0.05  # Minimum fraction for major state (5%)
max_states: 10
stride: 1                   # Frame stride

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
        return """# ConformAtlas Single System / Multi-Replicate Analysis Configuration
system_name: WT
topology: topology.tpr
temperature: 300.0          # Kelvin (mandatory, > 0)
output_dir: Outputs/WT

trajectories:
  - rep1.xtc
  - rep2.xtc
  - rep3.xtc

lsq_group: Backbone
cov_group: Backbone
lig_group: null             # Optional ligand group name (e.g. LIG)

bins: 64
smooth_sigma: 0.0           # Set to 1.0 for mild Gaussian smoothing
pca_weighting: equal-replicate  # Options: 'equal-replicate' or 'frames'
stride: 1
min_state_population: 0.05
max_states: 10
fel_backend: python         # Options: 'python' or 'gromacs'
"""
    else:
        raise ValueError(f"Unknown config type: {config_type}. Supported: 'compare', 'analyze'")
