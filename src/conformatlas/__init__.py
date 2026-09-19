"""ConformAtlas: MD ensemble, PCA, and Free Energy Landscape analysis."""

__version__ = "0.1.1"
__author__ = "Atharva Tilewale"

from conformatlas.models import (
    AnalysisConfig,
    BasinState,
    ComparisonConfig,
    ConvergenceResults,
    FELGrid,
    PCAResults,
)

__all__ = [
    "AnalysisConfig",
    "BasinState",
    "ComparisonConfig",
    "ConvergenceResults",
    "FELGrid",
    "PCAResults",
    "__version__",
]
