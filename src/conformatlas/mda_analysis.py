"""MDAnalysis ``AnalysisBase`` interface to ConformAtlas Cartesian PCA.

:class:`ConformAtlasPCA` plugs the streaming covariance accumulator of
:mod:`conformatlas.streaming` into the standard MDAnalysis analysis protocol, so
it can be combined with ``AnalysisBase`` features (``start/stop/step``,
``verbose`` progress, ``frames``/``times`` bookkeeping) and with other
MDAnalysis analyses.

Example::

    import MDAnalysis as mda
    from conformatlas.mda_analysis import ConformAtlasPCA

    u = mda.Universe("topol.pdb", "traj.xtc")
    pca = ConformAtlasPCA(u.select_atoms("name CA")).run()
    pca.results.eigenvalues          # Å², descending
    pca.results.projections          # (n_frames, n_components) Å
    pca.results.explained_variance   # percent

Equivalence with the in-memory path (``stream_trajectory_coordinates`` +
``compute_cartesian_pca`` with ``weighting='frames'``) is validated in
``validation/checks_performance.py``. A single trajectory has K = 1 replicate,
so ``'equal-replicate'`` and ``'frames'`` weighting coincide here.
"""

from __future__ import annotations

import numpy as np

try:
    from MDAnalysis.analysis.base import AnalysisBase

    HAS_MDANALYSIS = True
except ImportError:  # pragma: no cover
    HAS_MDANALYSIS = False

    class AnalysisBase:  # type: ignore[no-redef]
        """Placeholder so the module imports without MDAnalysis; the class
        constructor raises a clear error."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("MDAnalysis is required for conformatlas.mda_analysis.")


from conformatlas.streaming import CovarianceAccumulator, eigendecompose_covariance
from conformatlas.superpose import superpose_frame

__all__ = ["ConformAtlasPCA"]


class ConformAtlasPCA(AnalysisBase):
    """Cartesian PCA of an AtomGroup with per-frame least-squares superposition.

    Parameters
    ----------
    atomgroup : MDAnalysis.AtomGroup
        Analysis group (the covariance is built from these atoms).
    fit_atomgroup : AtomGroup, optional
        Fit group for the superposition; defaults to ``atomgroup``.
    reference : (n_fit, 3) array, optional
        Reference coordinates of the fit group; defaults to the fit group's
        coordinates in the first frame of the run.
    fit : bool
        Superpose every frame before accumulating (default True).
    mass_weighted : bool
        Mass-weight the superposition.
    n_components : int
        Components retained.
    precision : {'float64', 'float32'}
        Storage dtype of the projections; the covariance is always float64.
    chunk_size : int
        Frames buffered between accumulator updates.

    Attributes (after ``run()``)
    ----------------------------
    results.eigenvalues : (k,) ndarray, Å²
    results.eigenvectors : (3·n_atoms, k) ndarray
    results.explained_variance, results.cumulative_variance : (k,) percent
    results.mean : (3·n_atoms,) mean structure of the superposed frames
    results.projections : (n_frames, k) ndarray
    results.covariance : (3·n_atoms, 3·n_atoms) covariance (1/N normalisation)
    results.n_frames : int

    Notes
    -----
    The second pass needed for the projections re-reads the frames selected by
    ``start/stop/step`` inside ``_conclude``; coordinates are never stored.
    Frames are quantised to float32 before accumulation, as in the in-memory path.
    """

    def __init__(
        self,
        atomgroup,
        fit_atomgroup=None,
        reference: np.ndarray | None = None,
        fit: bool = True,
        mass_weighted: bool = False,
        n_components: int = 10,
        precision: str = "float64",
        chunk_size: int = 256,
        **kwargs,
    ) -> None:
        if not HAS_MDANALYSIS:  # pragma: no cover
            raise RuntimeError("MDAnalysis is required for conformatlas.mda_analysis.")
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        if precision not in ("float64", "float32"):
            raise ValueError("precision must be 'float64' or 'float32'")
        self.atomgroup = atomgroup
        self.fit_atomgroup = fit_atomgroup if fit_atomgroup is not None else atomgroup
        self.fit = bool(fit)
        self.mass_weighted = bool(mass_weighted)
        self.n_components = int(n_components)
        self.precision = precision
        self.chunk_size = max(1, int(chunk_size))
        self._reference = None if reference is None else np.asarray(reference, dtype=np.float64)
        if self.fit and len(self.fit_atomgroup) < 3:
            raise ValueError("fit group needs at least 3 atoms for a rigid superposition")

    # ----------------------------------------------------------- protocol
    def _prepare(self) -> None:
        self._acc = CovarianceAccumulator(3 * len(self.atomgroup))
        self._buffer: list[np.ndarray] = []
        self._weights = None
        if self.fit and self.mass_weighted:
            w = np.asarray(self.fit_atomgroup.masses, dtype=np.float64)
            if np.all(np.isfinite(w)) and w.sum() > 0:
                self._weights = w
        if self.fit and self._reference is None:
            # the first frame of the run defines the reference
            self._trajectory[self.start if self.start is not None else 0]
            self._reference = np.array(self.fit_atomgroup.positions, dtype=np.float64)
        if self.fit and self._reference.shape != (len(self.fit_atomgroup), 3):
            raise ValueError(
                f"reference has shape {self._reference.shape}, expected {(len(self.fit_atomgroup), 3)}"
            )

    def _frame_coordinates(self) -> np.ndarray:
        if self.fit:
            pos = superpose_frame(
                self.fit_atomgroup.positions,
                self.atomgroup.positions,
                self._reference,
                self._weights,
            )
        else:
            pos = self.atomgroup.positions
        return np.array(pos, dtype=np.float32).reshape(-1)

    def _flush(self) -> None:
        if self._buffer:
            self._acc.update(np.array(self._buffer, dtype=np.float64))
            self._buffer = []

    def _single_frame(self) -> None:
        self._buffer.append(self._frame_coordinates())
        if len(self._buffer) >= self.chunk_size:
            self._flush()

    def _conclude(self) -> None:
        self._flush()
        if self._acc.n == 0:
            raise ValueError("no frames were analysed")
        cov = self._acc.covariance(ddof=0)
        evals, evecs, explained, cumulative = eigendecompose_covariance(cov, self.n_components)
        self.results.eigenvalues = evals
        self.results.eigenvectors = evecs
        self.results.explained_variance = explained
        self.results.cumulative_variance = cumulative
        self.results.mean = self._acc.mean
        self.results.covariance = cov
        self.results.n_frames = int(self._acc.n)

        # Second pass: projections. The frame range of the run is replayed.
        proj = np.empty((self._acc.n, evecs.shape[1]), dtype=self.precision)
        for i, _ts in enumerate(self._sliced_trajectory):
            x = self._frame_coordinates().astype(np.float64)
            proj[i] = (x - self._acc.mean) @ evecs
        self.results.projections = proj
