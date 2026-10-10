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
    bins: int | None = None  # None -> Freedman-Diaconis automatic choice
    smooth_sigma: float = 0.0  # 0.0 means no Gaussian smoothing
    pca_weighting: str = "equal-replicate"  # 'equal-replicate' or 'frames'
    stride: int = 1
    min_state_population: float = 0.05  # 5% threshold for major states
    max_states: int = 10
    convergence_modes: int = 10
    fel_backend: str = "python"  # 'python' or 'gromacs'
    gmx_command: str | None = None
    system_name: str = "System"
    fit: bool = True  # least-squares superposition on lsq_group before PCA
    mass_weighted_fit: bool = False
    min_barrier: float | None = None  # kJ/mol; None -> 1 kT
    # --- time window and equilibration ---
    begin_ps: float | None = None
    end_ps: float | None = None
    equilibration: str = "none"  # 'none' (report only) or 'auto' (discard detected transient)
    # --- decomposition ---
    pca_type: str = "cartesian"  # 'cartesian' | 'dihedral' | 'tica'
    tica_lag: int = 10  # frames
    n_components: int = 10
    pca_backend: str = "auto"  # 'auto' | 'python' | 'gromacs'
    # --- collective variables for the landscape ---
    cv: str = "pc"  # 'pc' | 'rmsd-rg' | 'file'
    fel_pcs: tuple[int, int] = (1, 2)
    cv_files: list[Path] = field(default_factory=list)
    cv_columns: tuple[int, int] = (1, 2)
    cv_labels: tuple[str, str] | None = None
    weights_files: list[Path] = field(default_factory=list)
    # --- selections ---
    index_file: Path | None = None
    # --- statistics ---
    n_bootstrap: int = 200
    seed: int = 0
    # --- structures ---
    n_representatives: int = 1
    representative_method: str = "minimum"  # 'minimum' | 'medoid'
    # --- output ---
    pdf: bool = False

    def __post_init__(self):
        self.topology = Path(self.topology)
        self.trajectories = [Path(p) for p in self.trajectories]
        self.output_dir = Path(self.output_dir)
        self.cv_files = [Path(p) for p in self.cv_files]
        self.weights_files = [Path(p) for p in self.weights_files]
        self.index_file = Path(self.index_file) if self.index_file else None
        self.fel_pcs = tuple(int(v) for v in self.fel_pcs)
        self.cv_columns = tuple(int(v) for v in self.cv_columns)
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive Kelvin (got {self.temperature})")
        if self.bins is not None and self.bins < 5:
            raise ValueError(f"Number of bins must be at least 5 (got {self.bins})")
        if self.equilibration not in ("none", "auto"):
            raise ValueError("equilibration must be 'none' or 'auto'")
        if self.pca_type not in ("cartesian", "dihedral", "tica"):
            raise ValueError("pca_type must be 'cartesian', 'dihedral' or 'tica'")
        if self.cv not in ("pc", "rmsd-rg", "file"):
            raise ValueError("cv must be 'pc', 'rmsd-rg' or 'file'")
        if self.representative_method not in ("minimum", "medoid"):
            raise ValueError("representative_method must be 'minimum' or 'medoid'")
        if self.pca_backend not in ("auto", "python", "gromacs"):
            raise ValueError("pca_backend must be 'auto', 'python' or 'gromacs'")
        if len(self.fel_pcs) != 2 or min(self.fel_pcs) < 1 or self.fel_pcs[0] == self.fel_pcs[1]:
            raise ValueError("fel_pcs must be two different 1-based component indices")
        if self.begin_ps is not None and self.end_ps is not None and self.end_ps <= self.begin_ps:
            raise ValueError("end_ps must be larger than begin_ps")
        if self.n_representatives < 1:
            raise ValueError("n_representatives must be >= 1")
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
    bins: int | None = None  # None -> automatic
    smooth_sigma: float = 0.0
    min_state_population: float = 0.05
    max_states: int = 10
    pca_weighting: str = "equal-replicate"
    stride: int = 1
    gmx_command: str | None = None
    fit: bool = True  # least-squares superposition on fit_group before PCA
    mass_weighted_fit: bool = False
    min_barrier: float | None = None  # kJ/mol; None -> 1 kT
    begin_ps: float | None = None
    end_ps: float | None = None
    index_file: Path | None = None
    n_bootstrap: int = 200
    seed: int = 0

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.index_file = Path(self.index_file) if self.index_file else None
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
    basin_id: int = -1  # label in the watershed basin map


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
    rmsip_halves_per_replicate: dict[str, float] = field(default_factory=dict)
    statistical_inefficiency: dict[str, dict[str, float]] = field(default_factory=dict)
    n_eff: dict[str, float] = field(default_factory=dict)
    equilibration: pd.DataFrame | None = None


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
    extras: dict[str, Any] = field(default_factory=dict)  # descriptors, kinetics, bootstrap, ...
    warnings: list[str] = field(default_factory=list)
