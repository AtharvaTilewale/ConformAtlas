"""Data models and typed structures for ConformAtlas."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrajectoryInfo:
    """Metadata for a single trajectory file."""

    path: Path
    replicate_id: str
    system_name: str
    n_frames: int = 0
    start_time_ps: float = 0.0
    end_time_ps: float = 0.0
    timestep_ps: float = 0.0


@dataclass
class SystemConfig:
    """Configuration for a single molecular system (e.g. WT or Mutant)."""

    name: str
    topology: Path
    trajectories: list[Path] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """Configuration for single or multi-replicate analysis."""

    topology: Path
    trajectories: list[Path] = field(default_factory=list)
    temperature: float = 300.0  # Kelvin
    lsq_group: str = "Backbone"
    cov_group: str = "Backbone"
    lig_group: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("Outputs"))
    bins: int = 64
    smooth_sigma: float = 0.0  # 0.0 means no Gaussian smoothing
    pca_weighting: str = "equal-replicate"  # 'equal-replicate' or 'frames'
    stride: int = 1
    min_state_population: float = 0.05  # 5% threshold for major states
    max_states: int = 10
    convergence_modes: int = 10
    fel_backend: str = "python"  # 'python' or 'gromacs'
    gmx_command: str | None = None
    system_name: str = "System"

    def __post_init__(self):
        self.topology = Path(self.topology)
        self.trajectories = [Path(p) for p in self.trajectories]
        self.output_dir = Path(self.output_dir)
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive Kelvin (got {self.temperature})")
        if self.bins < 5:
            raise ValueError(f"Number of bins must be at least 5 (got {self.bins})")
        if self.stride < 1:
            raise ValueError(f"Stride must be at least 1 (got {self.stride})")
        if self.min_state_population < 0 or self.min_state_population > 1:
            raise ValueError(
                f"min_state_population must be between 0 and 1 (got {self.min_state_population})"
            )


@dataclass
class ComparisonConfig:
    """Configuration for WT vs Mutant condition comparisons in a shared PCA space."""

    project: str
    temperature: float = 300.0
    systems: list[SystemConfig] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("FEL_comparison"))
    fit_group: str = "Backbone"
    analysis_group: str = "Backbone"
    reference: str = "combined"  # 'combined' or 'reference-system'
    reference_system: str | None = None
    components: int = 10
    bins: int = 64
    smooth_sigma: float = 0.0
    min_state_population: float = 0.05
    max_states: int = 10
    pca_weighting: str = "equal-replicate"
    stride: int = 1
    gmx_command: str | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive Kelvin (got {self.temperature})")


@dataclass
class PCAResults:
    """Results of Principal Component Analysis."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance: np.ndarray  # percentage (0 - 100)
    cumulative_variance: np.ndarray  # percentage (0 - 100)
    projections: pd.DataFrame  # columns: system, replicate, frame, time_ps, PC1, PC2, ...
    mean_coords: np.ndarray | None = None
    atom_names: list[str] | None = None
    residue_numbers: list[int] | None = None


@dataclass
class FELGrid:
    """Free Energy Landscape grid calculation results."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    probability: np.ndarray  # 2D normalized probability density
    free_energy: np.ndarray  # 2D free energy in kJ/mol (masked / NaN where P == 0)
    temperature: float
    smoothed: bool = False
    smooth_sigma: float = 0.0
    occupancy_mask: np.ndarray | None = None


@dataclass
class BasinState:
    """Identified conformational basin / energy minimum state."""

    state_id: int
    label: str  # e.g., 'State A'
    population: float  # fraction (0.0 to 1.0)
    min_pc1: float
    min_pc2: float
    min_free_energy: float
    representative_replicate: str
    representative_frame: int
    representative_time_ps: float
    representative_distance: float
    representative_pdb: Path | None = None


@dataclass
class ConvergenceResults:
    """PCA and trajectory sampling convergence diagnostics."""

    rmsip_halves: float  # Top modes overlap between 1st half and 2nd half
    progressive_overlap: dict[int, float]  # fraction (e.g. 25, 50, 75, 100) -> RMSIP
    cosine_content: dict[int, float]  # PC index -> cosine content (0.0 to 1.0)
    population_stability: pd.DataFrame | None = None
    pairwise_rmsip: pd.DataFrame | None = None
    sampling_assessment: str = "Insufficient evidence"
    assessment_notes: list[str] = field(default_factory=list)


@dataclass
class AnalysisResults:
    """Comprehensive container for all outputs of an analysis run."""

    config: AnalysisConfig
    trajectories_info: list[TrajectoryInfo]
    pca: PCAResults
    fel: FELGrid
    states: list[BasinState]
    unassigned_population: float
    frame_assignments: pd.DataFrame
    convergence: ConvergenceResults
    replicate_statistics: pd.DataFrame | None = None
    output_dir: Path = field(default_factory=lambda: Path("Outputs"))
    run_metadata: dict[str, Any] = field(default_factory=dict)
