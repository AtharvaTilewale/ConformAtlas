"""Chunked, memory-lean Cartesian PCA that never holds a full trajectory in memory.

The in-memory path (:func:`conformatlas.trajectory.stream_trajectory_coordinates`
followed by :func:`conformatlas.pca.compute_cartesian_pca`) stores every fitted
frame before forming the covariance. This module computes the same quantities
in two passes over each trajectory:

1. **Pass 1 – moments.** Frames are read in chunks of ``chunk_size`` frames,
   superposed on the reference with :func:`conformatlas.superpose.superpose_frame`
   (exactly as the in-memory path does) and folded into a per-replicate
   :class:`CovarianceAccumulator` with Chan's parallel update formula
   (Chan, Golub & LeVeque 1979), which is the chunked generalisation of
   Welford's algorithm. Replicates are processed independently, optionally in
   parallel worker processes, and merged afterwards with the same weighting
   semantics as ``compute_cartesian_pca``:

   * ``equal-replicate``: mean = (1/K) Σ_k mean_k and
     C = Σ_k 1/(K n_k) Σ_{frames i in k} (x_i − mean)(x_i − mean)ᵀ
   * ``frames``: mean = (1/N) Σ_i x_i and C = (1/N) Σ_i (x_i − mean)(x_i − mean)ᵀ

   The per-replicate sums about the replicate mean are shifted to the pooled
   mean with the identity Σ_i (x_i − μ)(x_i − μ)ᵀ = M2_k + n_k (μ_k − μ)(μ_k − μ)ᵀ.
2. **Pass 2 – projections.** The eigenvectors of C are computed with
   ``numpy.linalg.eigh``; every trajectory is read again in chunks and the
   centred frames are projected onto the leading components. Projections can
   be written to a memory-mapped ``.npy`` file so that very long trajectories
   never require an in-memory projection table either.

Memory: the accumulator holds one ``D × D`` float64 matrix per replicate being
processed (``D = 3 × n_atoms``), i.e. 8·D² bytes, plus one chunk of frames. With
``n_workers > 1`` each worker holds its own matrix. The eigendecomposition of the
``D × D`` covariance is O(D³); for selections above roughly 7,000 atoms
(D ≈ 20,000; 3.2 GB per matrix) use a coarser selection or the in-memory path,
which switches to the dual (SVD) route when frames are fewer than coordinates.

Numerical equivalence with the in-memory path is validated in
``validation/checks_performance.py`` (eigenvalues, eigenvectors and projections
agree to rtol 1e-8 on the bundled demo and the neuraminidase fixture for both
weightings). Frames are quantised to float32 before accumulation, exactly as the
in-memory path stores them, so both paths see identical input numbers.

Units follow the MDAnalysis backend: Å for coordinates, ps for time.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from conformatlas.models import PCAResults
from conformatlas.progress import ProgressReporter
from conformatlas.superpose import rmsd, superpose_frame

logger = logging.getLogger("conformatlas.streaming")

try:
    import MDAnalysis as mda

    HAS_MDANALYSIS = True
except ImportError:  # pragma: no cover
    HAS_MDANALYSIS = False

__all__ = [
    "CovarianceAccumulator",
    "ReplicateMoments",
    "StreamingPCAResult",
    "accumulate_replicate",
    "combine_replicate_moments",
    "eigendecompose_covariance",
    "iter_frame_chunks",
    "project_replicate",
    "streaming_pca",
]

PRECISIONS = ("float64", "float32")


# =============================================================================
# Covariance accumulator (Welford / Chan)
# =============================================================================


class CovarianceAccumulator:
    """Running mean and scatter matrix of D-dimensional samples, updated in chunks.

    Maintains ``n``, ``mean`` (D,) and ``m2`` (D, D) = Σ_i (x_i − mean)(x_i − mean)ᵀ
    in float64. Chunks are combined with Chan's formula, so the result does not
    depend on the chunk size, and two accumulators can be merged with
    :meth:`merge` (parallel reduction).
    """

    def __init__(self, n_dims: int) -> None:
        if n_dims < 1:
            raise ValueError("n_dims must be positive")
        self.n_dims = int(n_dims)
        self.n = 0
        self.mean = np.zeros(self.n_dims, dtype=np.float64)
        self.m2 = np.zeros((self.n_dims, self.n_dims), dtype=np.float64)

    def update(self, chunk: np.ndarray) -> None:
        """Fold a (m, D) array of samples into the running moments."""
        x = np.asarray(chunk, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_dims:
            raise ValueError(f"chunk must have shape (m, {self.n_dims}), got {x.shape}")
        m = x.shape[0]
        if m == 0:
            return
        mean_b = x.mean(axis=0)
        centred = x - mean_b
        m2_b = centred.T @ centred
        self._merge_moments(m, mean_b, m2_b)

    def merge(self, other: CovarianceAccumulator) -> None:
        """Merge another accumulator over the same dimension into this one."""
        if other.n_dims != self.n_dims:
            raise ValueError("cannot merge accumulators of different dimension")
        if other.n == 0:
            return
        self._merge_moments(other.n, other.mean, other.m2)

    def _merge_moments(self, n_b: int, mean_b: np.ndarray, m2_b: np.ndarray) -> None:
        n_a = self.n
        if n_a == 0:
            self.n = int(n_b)
            self.mean = np.array(mean_b, dtype=np.float64, copy=True)
            self.m2 = np.array(m2_b, dtype=np.float64, copy=True)
            return
        n = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n)
        self.m2 = self.m2 + m2_b + np.outer(delta, delta) * (n_a * n_b / n)
        self.n = int(n)

    def covariance(self, ddof: int = 0) -> np.ndarray:
        """Covariance matrix M2 / (n − ddof)."""
        if self.n - ddof <= 0:
            raise ValueError(f"need more than {ddof} samples, have {self.n}")
        return self.m2 / (self.n - ddof)


@dataclass
class ReplicateMoments:
    """Pass-1 output for one replicate."""

    key: str
    n: int
    mean: np.ndarray  # (D,)
    m2: np.ndarray  # (D, D) scatter about the replicate mean
    times: np.ndarray  # (n,) ps
    fit_rmsd: np.ndarray  # (n,) Å; zeros when fit=False


def combine_replicate_moments(
    moments: Sequence[ReplicateMoments], weighting: str = "equal-replicate"
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled mean and covariance with the weighting of ``compute_cartesian_pca``.

    Returns ``(mean (D,), covariance (D, D))``. Replicates with zero frames are
    ignored. Weighting ``'equal-replicate'`` gives every replicate total weight
    1/K; ``'frames'`` gives every frame weight 1/N.
    """
    if weighting not in ("equal-replicate", "frames"):
        raise ValueError("weighting must be 'equal-replicate' or 'frames'")
    live = [m for m in moments if m.n > 0]
    if not live:
        raise ValueError("no frames to combine")
    n_dims = live[0].mean.shape[0]
    for m in live:
        if m.mean.shape[0] != n_dims or m.m2.shape != (n_dims, n_dims):
            raise ValueError(f"dimension mismatch in replicate {m.key}")
    k = len(live)
    n_total = sum(m.n for m in live)
    if weighting == "equal-replicate":
        mean = sum(m.mean for m in live) / k
    else:
        mean = sum(m.n * m.mean for m in live) / n_total
    cov = np.zeros((n_dims, n_dims), dtype=np.float64)
    for m in live:
        w = 1.0 / (k * m.n) if weighting == "equal-replicate" else 1.0 / n_total
        delta = m.mean - mean
        cov += w * (m.m2 + m.n * np.outer(delta, delta))
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def eigendecompose_covariance(
    cov: np.ndarray, n_components: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Descending eigenpairs of a symmetric covariance plus variance percentages.

    Mirrors the covariance branch of ``compute_cartesian_pca``: eigenvalues are
    clipped at zero, explained variance is relative to the trace, and the leading
    ``n_components`` are returned.
    """
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals[::-1], 0.0)
    evecs = evecs[:, ::-1]
    total = float(np.sum(evals))
    explained = 100.0 * evals / total if total > 0 else np.zeros_like(evals)
    cumulative = np.cumsum(explained)
    k = min(int(n_components), len(evals))
    return evals[:k], evecs[:, :k], explained[:k], cumulative[:k]


# =============================================================================
# Chunked frame reading (MDAnalysis)
# =============================================================================


def _open_source(
    topology: str | Path,
    trajectory: str | Path,
    atom_indices: Sequence[int],
    fit: bool,
    fit_atom_indices: Sequence[int] | None,
    reference_fit_coords: np.ndarray | None,
    mass_weighted: bool,
):
    if not HAS_MDANALYSIS:
        raise RuntimeError("MDAnalysis is required for streaming trajectory processing.")
    u = mda.Universe(str(topology), str(trajectory))
    ag = u.atoms[np.asarray(atom_indices, dtype=int)]
    if len(ag) == 0:
        raise ValueError("Analysis selection matched 0 atoms.")
    fit_ag = None
    weights = None
    ref = None
    if fit:
        fit_ag = (
            u.atoms[np.asarray(fit_atom_indices, dtype=int)] if fit_atom_indices is not None else ag
        )
        if len(fit_ag) < 3:
            raise ValueError(
                f"Fit group has only {len(fit_ag)} atoms; at least 3 are needed for a rigid fit."
            )
        if reference_fit_coords is None:
            u.trajectory[0]
            ref = np.array(fit_ag.positions, dtype=np.float64)
        else:
            ref = np.asarray(reference_fit_coords, dtype=np.float64)
            if ref.shape != (len(fit_ag), 3):
                raise ValueError(
                    f"reference_fit_coords has shape {ref.shape}, expected {(len(fit_ag), 3)}."
                )
        if mass_weighted:
            try:
                weights = np.asarray(fit_ag.masses, dtype=np.float64)
                if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
                    raise ValueError("non-positive or missing masses")
            except Exception as e:  # pragma: no cover - topology dependent
                logger.warning(f"Mass-weighted fit requested but masses unavailable ({e}).")
                weights = None
    return u, ag, fit_ag, ref, weights


def iter_frame_chunks(
    topology: str | Path,
    trajectory: str | Path,
    atom_indices: Sequence[int],
    *,
    fit: bool = True,
    fit_atom_indices: Sequence[int] | None = None,
    reference_fit_coords: np.ndarray | None = None,
    mass_weighted: bool = False,
    stride: int = 1,
    begin_ps: float | None = None,
    end_ps: float | None = None,
    chunk_size: int = 256,
    compute_fit_rmsd: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(coords, times, fit_rmsd)`` chunks of at most ``chunk_size`` frames.

    ``coords`` has shape (m, n_atoms, 3) in float32 — the same per-frame
    treatment (superposition, float32 storage) as
    :func:`conformatlas.trajectory.stream_trajectory_coordinates`, so the
    concatenation of all chunks equals that function's output exactly.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    u, ag, fit_ag, ref, weights = _open_source(
        topology,
        trajectory,
        atom_indices,
        fit,
        fit_atom_indices,
        reference_fit_coords,
        mass_weighted,
    )
    buf_c: list[np.ndarray] = []
    buf_t: list[float] = []
    buf_r: list[float] = []
    for ts in u.trajectory[:: max(1, int(stride))]:
        t = float(ts.time)
        if begin_ps is not None and t < begin_ps:
            continue
        if end_ps is not None and t > end_ps:
            break
        if fit:
            fit_pos = fit_ag.positions
            frame = superpose_frame(fit_pos, ag.positions, ref, weights)
            if compute_fit_rmsd:
                fitted = superpose_frame(fit_pos, fit_pos, ref, weights)
                buf_r.append(rmsd(fitted, ref))
            else:
                buf_r.append(0.0)
        else:
            frame = ag.positions
            buf_r.append(0.0)
        buf_c.append(np.array(frame, dtype=np.float32))
        buf_t.append(t)
        if len(buf_c) >= chunk_size:
            yield np.array(buf_c, dtype=np.float32), np.array(buf_t), np.array(buf_r)
            buf_c, buf_t, buf_r = [], [], []
    if buf_c:
        yield np.array(buf_c, dtype=np.float32), np.array(buf_t), np.array(buf_r)


# =============================================================================
# Pass 1 / pass 2 per replicate
# =============================================================================


def accumulate_replicate(
    key: str,
    topology: str | Path,
    trajectory: str | Path,
    atom_indices: Sequence[int],
    *,
    fit: bool = True,
    fit_atom_indices: Sequence[int] | None = None,
    reference_fit_coords: np.ndarray | None = None,
    mass_weighted: bool = False,
    stride: int = 1,
    begin_ps: float | None = None,
    end_ps: float | None = None,
    chunk_size: int = 256,
    quiet: bool = True,
) -> ReplicateMoments:
    """Pass 1: stream one trajectory into a :class:`CovarianceAccumulator`."""
    acc: CovarianceAccumulator | None = None
    times: list[np.ndarray] = []
    rms: list[np.ndarray] = []
    with ProgressReporter(description=f"{key}: moments", quiet=quiet) as bar:
        for coords, t, r in iter_frame_chunks(
            topology,
            trajectory,
            atom_indices,
            fit=fit,
            fit_atom_indices=fit_atom_indices,
            reference_fit_coords=reference_fit_coords,
            mass_weighted=mass_weighted,
            stride=stride,
            begin_ps=begin_ps,
            end_ps=end_ps,
            chunk_size=chunk_size,
        ):
            flat = coords.reshape(coords.shape[0], -1).astype(np.float64)
            if acc is None:
                acc = CovarianceAccumulator(flat.shape[1])
            acc.update(flat)
            times.append(t)
            rms.append(r)
            bar.update(len(t))
    if acc is None or acc.n == 0:
        raise ValueError(
            f"No frames of {trajectory} fall in the window [{begin_ps}, {end_ps}] ps with stride {stride}."
        )
    return ReplicateMoments(
        key=key,
        n=acc.n,
        mean=acc.mean,
        m2=acc.m2,
        times=np.concatenate(times),
        fit_rmsd=np.concatenate(rms),
    )


def project_replicate(
    key: str,
    topology: str | Path,
    trajectory: str | Path,
    atom_indices: Sequence[int],
    mean: np.ndarray,
    eigenvectors: np.ndarray,
    out: np.ndarray | None = None,
    offset: int = 0,
    *,
    fit: bool = True,
    fit_atom_indices: Sequence[int] | None = None,
    reference_fit_coords: np.ndarray | None = None,
    mass_weighted: bool = False,
    stride: int = 1,
    begin_ps: float | None = None,
    end_ps: float | None = None,
    chunk_size: int = 256,
    dtype: str = "float64",
    quiet: bool = True,
) -> np.ndarray:
    """Pass 2: project one trajectory onto ``eigenvectors`` chunk by chunk.

    Rows are written into ``out[offset:offset+n]`` when ``out`` is given (an
    array or an opened ``.npy`` memmap); otherwise a new array is returned.
    """
    pieces: list[np.ndarray] = []
    row = int(offset)
    mean = np.asarray(mean, dtype=np.float64)
    evecs = np.asarray(eigenvectors, dtype=np.float64)
    with ProgressReporter(description=f"{key}: projections", quiet=quiet) as bar:
        for coords, t, _ in iter_frame_chunks(
            topology,
            trajectory,
            atom_indices,
            fit=fit,
            fit_atom_indices=fit_atom_indices,
            reference_fit_coords=reference_fit_coords,
            mass_weighted=mass_weighted,
            stride=stride,
            begin_ps=begin_ps,
            end_ps=end_ps,
            chunk_size=chunk_size,
            compute_fit_rmsd=False,
        ):
            flat = coords.reshape(coords.shape[0], -1).astype(np.float64)
            proj = ((flat - mean) @ evecs).astype(dtype)
            if out is not None:
                out[row : row + proj.shape[0]] = proj
            else:
                pieces.append(proj)
            row += proj.shape[0]
            bar.update(len(t))
    if out is not None:
        if hasattr(out, "flush"):
            out.flush()
        return out[offset:row]
    return np.concatenate(pieces) if pieces else np.empty((0, evecs.shape[1]), dtype=dtype)


# worker entry points (must be importable for process pools)
def _accumulate_worker(kwargs: dict) -> ReplicateMoments:
    return accumulate_replicate(**kwargs)


def _project_worker(kwargs: dict) -> np.ndarray | None:
    store = kwargs.pop("store_path", None)
    if store is not None:
        mm = np.load(store, mmap_mode="r+")
        project_replicate(out=mm, **kwargs)
        mm.flush()
        del mm
        return None
    return project_replicate(**kwargs)


def _mp_context():
    """Multiprocessing start method for the worker pool.

    ``fork`` (Linux default) starts instantly and does not re-import the main
    module, so it also works from ``python -c`` and interactive sessions; the
    workers open their own MDAnalysis readers, so no file handles are shared.
    Override with ``CONFORMATLAS_MP_START=spawn|forkserver|fork``.
    """
    wanted = os.environ.get("CONFORMATLAS_MP_START", "").strip()
    order = [wanted] if wanted else ["fork", "spawn"]
    for name in order:
        try:
            return mp.get_context(name)
        except ValueError:
            continue
    return mp.get_context()


def _pool_map(func, jobs: list[dict], n_workers: int) -> list:
    """Map ``func`` over ``jobs`` in a process pool; fall back to serial execution
    (with a warning) when the platform refuses to start worker processes."""
    if n_workers <= 1 or len(jobs) <= 1:
        return [func(j) for j in jobs]
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_mp_context()) as pool:
            return list(pool.map(func, jobs))
    except (OSError, BrokenProcessPool) as e:
        logger.warning(f"Worker pool unavailable ({e!r}); processing replicates serially.")
        return [func(j) for j in jobs]


# =============================================================================
# Driver
# =============================================================================


@dataclass
class StreamingPCAResult:
    """Output of :func:`streaming_pca`."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray  # (D, k)
    explained_variance: np.ndarray
    cumulative_variance: np.ndarray
    mean_coords: np.ndarray  # (D,)
    projection_array: np.ndarray  # (N_total, k) in-memory or np.memmap
    replicate_keys: list[str]
    n_frames: dict[str, int]
    times: dict[str, np.ndarray]
    fit_rmsd: dict[str, np.ndarray]
    weighting: str
    precision: str
    projection_path: Path | None = None
    system_name: str = "System"
    covariance: np.ndarray | None = field(default=None, repr=False)

    def projections_for(self, key: str) -> np.ndarray:
        """Projection block (n_k, k) of one replicate."""
        start = 0
        for k in self.replicate_keys:
            n = self.n_frames[k]
            if k == key:
                return np.asarray(self.projection_array[start : start + n])
            start += n
        raise KeyError(key)

    def projections_dataframe(self) -> pd.DataFrame:
        """Projection table with the pipeline's layout (system, replicate, frame, time_ps, PC…)."""
        k = self.eigenvectors.shape[1]
        dfs = []
        for key in self.replicate_keys:
            block = self.projections_for(key)
            df = pd.DataFrame(block, columns=[f"PC{i + 1}" for i in range(k)])
            df.insert(0, "system", self.system_name)
            df.insert(1, "replicate", key)
            df.insert(2, "frame", np.arange(len(block)))
            df.insert(3, "time_ps", self.times[key])
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def to_pca_results(self) -> PCAResults:
        """Convert to the :class:`~conformatlas.models.PCAResults` used by the pipeline."""
        return PCAResults(
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            explained_variance=self.explained_variance,
            cumulative_variance=self.cumulative_variance,
            projections=self.projections_dataframe(),
            mean_coords=self.mean_coords,
        )


def streaming_pca(
    topology: str | Path,
    trajectories: Mapping[str, str | Path] | Sequence[str | Path],
    atom_indices: Sequence[int],
    *,
    fit: bool = True,
    fit_atom_indices: Sequence[int] | None = None,
    reference_fit_coords: np.ndarray | None = None,
    mass_weighted: bool = False,
    stride: int = 1,
    begin_ps: float | Mapping[str, float | None] | None = None,
    end_ps: float | None = None,
    weighting: str = "equal-replicate",
    n_components: int = 10,
    chunk_size: int = 256,
    n_workers: int = 1,
    precision: str = "float64",
    projection_store: str | Path | None = None,
    system_name: str = "System",
    quiet: bool = False,
    keep_covariance: bool = False,
) -> StreamingPCAResult:
    """Two-pass, chunked Cartesian PCA over one or more trajectories.

    Parameters
    ----------
    topology, trajectories
        Structure file and trajectories (a mapping ``replicate_id -> path`` or a
        sequence, which is labelled ``Rep_1, Rep_2, …``).
    atom_indices
        0-based indices of the analysis group (see
        :func:`conformatlas.trajectory.resolve_selection_indices`).
    fit, fit_atom_indices, reference_fit_coords, mass_weighted
        Superposition settings, identical to ``stream_trajectory_coordinates``.
        With ``reference_fit_coords=None`` the first frame of *each* trajectory
        is its own reference; pass the shared reference from
        :func:`conformatlas.trajectory.load_reference_coordinates` for
        multi-replicate work.
    stride, begin_ps, end_ps
        Frame selection. ``begin_ps`` may also be a mapping ``replicate_id ->
        first time`` (per-replicate production start, e.g. from equilibration
        detection); replicates missing from the mapping start at the first frame.
    weighting
        ``'equal-replicate'`` or ``'frames'`` (see :func:`compute_cartesian_pca`).
    n_components
        Number of components retained.
    chunk_size
        Frames per chunk (memory of one chunk = chunk_size × n_atoms × 12 bytes
        plus the float64 copy used for the update).
    n_workers
        Worker processes (one trajectory per worker, at most one per trajectory).
    precision
        ``'float64'`` or ``'float32'`` — storage dtype of the projections. The
        covariance is always accumulated in float64.
    projection_store
        Optional path of a ``.npy`` file; projections are written there as a
        memory map (shape (N_total, n_components)) instead of being held in RAM.
    system_name
        Value of the ``system`` column in the projection table.
    quiet
        Suppress progress bars (they are silent anyway when stderr is not a TTY).
    keep_covariance
        Keep the pooled covariance matrix on the result (``D × D`` float64).

    Returns
    -------
    StreamingPCAResult
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if weighting not in ("equal-replicate", "frames"):
        raise ValueError("weighting must be 'equal-replicate' or 'frames'")
    if isinstance(trajectories, Mapping):
        traj_map = {str(k): Path(v) for k, v in trajectories.items()}
    else:
        traj_map = {f"Rep_{i + 1}": Path(p) for i, p in enumerate(trajectories)}
    if not traj_map:
        raise ValueError("At least one trajectory is required.")
    atom_indices = np.asarray(atom_indices, dtype=int)
    n_workers = max(1, min(int(n_workers), len(traj_map), os.cpu_count() or 1))

    common = dict(
        topology=str(topology),
        atom_indices=atom_indices,
        fit=fit,
        fit_atom_indices=None
        if fit_atom_indices is None
        else np.asarray(fit_atom_indices, dtype=int),
        reference_fit_coords=None
        if reference_fit_coords is None
        else np.asarray(reference_fit_coords, dtype=np.float64),
        mass_weighted=mass_weighted,
        stride=stride,
        begin_ps=None,
        end_ps=end_ps,
        chunk_size=chunk_size,
    )

    def _begin(key: str) -> float | None:
        if isinstance(begin_ps, Mapping):
            return begin_ps.get(key)
        return begin_ps

    # ---------------------------------------------------------------- pass 1
    jobs = [
        dict(common, key=key, trajectory=str(path), quiet=quiet, begin_ps=_begin(key))
        for key, path in traj_map.items()
    ]
    moments = _pool_map(_accumulate_worker, jobs, n_workers)

    mean, cov = combine_replicate_moments(moments, weighting=weighting)
    evals, evecs, explained, cumulative = eigendecompose_covariance(cov, n_components)
    k = evecs.shape[1]

    # ---------------------------------------------------------------- pass 2
    keys = list(traj_map)
    n_frames = {m.key: int(m.n) for m in moments}
    n_total = sum(n_frames.values())
    offsets = dict(zip(keys, np.concatenate([[0], np.cumsum([n_frames[kk] for kk in keys])[:-1]])))
    store_path = None
    if projection_store is not None:
        store_path = Path(projection_store)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(store_path, mode="w+", dtype=precision, shape=(n_total, k))
        mm.flush()
        del mm
        proj_array = np.load(store_path, mmap_mode="r+")
    else:
        proj_array = np.empty((n_total, k), dtype=precision)

    pjobs = [
        dict(
            common,
            key=key,
            trajectory=str(traj_map[key]),
            begin_ps=_begin(key),
            mean=mean,
            eigenvectors=evecs,
            offset=int(offsets[key]),
            dtype=precision,
            quiet=quiet,
            store_path=None if store_path is None else str(store_path),
        )
        for key in keys
    ]
    if n_workers > 1 and len(keys) > 1 and store_path is not None:
        # Workers write their own rows straight into the memory map.
        del proj_array
        _pool_map(_project_worker, pjobs, n_workers)
        proj_array = np.load(store_path, mmap_mode="r")
    elif n_workers > 1 and len(keys) > 1:
        blocks = _pool_map(_project_worker, [dict(j, store_path=None) for j in pjobs], n_workers)
        for key, block in zip(keys, blocks):
            proj_array[offsets[key] : offsets[key] + n_frames[key]] = block
    else:
        for job in pjobs:
            job.pop("store_path", None)
            project_replicate(out=proj_array, **job)
        if store_path is not None:
            proj_array.flush()
            proj_array = np.load(store_path, mmap_mode="r")

    return StreamingPCAResult(
        eigenvalues=evals,
        eigenvectors=evecs,
        explained_variance=explained,
        cumulative_variance=cumulative,
        mean_coords=mean,
        projection_array=proj_array,
        replicate_keys=keys,
        n_frames=n_frames,
        times={m.key: m.times for m in moments},
        fit_rmsd={m.key: m.fit_rmsd for m in moments},
        weighting=weighting,
        precision=precision,
        projection_path=store_path,
        system_name=system_name,
        covariance=cov if keep_covariance else None,
    )
