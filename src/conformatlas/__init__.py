"""ConformAtlas: mapping conformational landscapes from molecular simulations."""

__version__ = "1.0.2"
__author__ = "Atharva Tilewale"

from conformatlas import api  # noqa: E402,F401  (imports pipeline lazily)
from conformatlas.models import (
    AnalysisConfig,
    BasinState,
    ComparisonConfig,
    ConvergenceResults,
    FELGrid,
    PCAResults,
)
from conformatlas.msm import build_msm, save_msm  # noqa: E402
from conformatlas.results_io import load_results, load_results_dir, save_results  # noqa: E402

__all__ = [
    "AnalysisConfig",
    "BasinState",
    "ComparisonConfig",
    "ConvergenceResults",
    "FELGrid",
    "PCAResults",
    "__version__",
    "api",
    "build_msm",
    "save_msm",
    "load_results",
    "load_results_dir",
    "save_results",
]
