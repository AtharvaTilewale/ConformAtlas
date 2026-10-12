"""Streaming covariance / PCA equals the in-memory path; progress reporting is silent off-TTY."""

from __future__ import annotations

import importlib.util
import io
import multiprocessing as mp
import operator
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from conformatlas.pca import compute_cartesian_pca, project_coordinates
from conformatlas.progress import ProgressReporter, is_interactive, track
from conformatlas.streaming import (
    CovarianceAccumulator,
    ReplicateMoments,
    combine_replicate_moments,
    eigendecompose_covariance,
)

HAS_MDA = importlib.util.find_spec("MDAnalysis") is not None
ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "demo_dataset" / "WT"
H1N1 = ROOT / "tests" / "data" / "h1n1_ca"


# --------------------------------------------------------------------------- accumulator
def test_accumulator_matches_numpy_cov_regardless_of_chunking():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(503, 7)) * np.array([1, 2, 3, 4, 5, 6, 7]) + 10.0
    ref_mean = x.mean(axis=0)
    ref_cov = np.cov(x, rowvar=False, ddof=0)
    for chunk in (1, 7, 100, 503):
        acc = CovarianceAccumulator(7)
        for i in range(0, len(x), chunk):
            acc.update(x[i : i + chunk])
        assert acc.n == 503
        np.testing.assert_allclose(acc.mean, ref_mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(acc.covariance(0), ref_cov, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(acc.covariance(1), np.cov(x, rowvar=False), rtol=1e-12)


def test_accumulator_merge_equals_single_pass():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 5))
    a, b = CovarianceAccumulator(5), CovarianceAccumulator(5)
    a.update(x[:120])
    b.update(x[120:])
    a.merge(b)
    full = CovarianceAccumulator(5)
    full.update(x)
    np.testing.assert_allclose(a.mean, full.mean, atol=1e-13)
    np.testing.assert_allclose(a.m2, full.m2, rtol=1e-12, atol=1e-12)


def test_accumulator_rejects_wrong_shape():
    acc = CovarianceAccumulator(3)
    with pytest.raises(ValueError):
        acc.update(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        acc.covariance(ddof=1)  # no samples


def _moments(key, x):
    acc = CovarianceAccumulator(x.shape[1])
    acc.update(x)
    return ReplicateMoments(
        key=key,
        n=acc.n,
        mean=acc.mean,
        m2=acc.m2,
        times=np.arange(len(x)),
        fit_rmsd=np.zeros(len(x)),
    )


@pytest.mark.parametrize("weighting", ["equal-replicate", "frames"])
def test_combined_moments_match_compute_cartesian_pca(weighting):
    rng = np.random.default_rng(3)
    reps = {"a": rng.normal(size=(60, 4, 3)), "b": rng.normal(size=(140, 4, 3)) + 1.0}
    ev, vec, exp, cum, mean = compute_cartesian_pca(reps, weighting=weighting, n_components=12)
    mean_s, cov_s = combine_replicate_moments(
        [_moments(k, v.reshape(v.shape[0], -1)) for k, v in reps.items()], weighting=weighting
    )
    ev_s, vec_s, exp_s, cum_s = eigendecompose_covariance(cov_s, 12)
    np.testing.assert_allclose(mean_s, mean, atol=1e-13)
    np.testing.assert_allclose(ev_s, ev, rtol=1e-10)
    np.testing.assert_allclose(exp_s, exp, rtol=1e-10)
    np.testing.assert_allclose(np.abs(np.sum(vec_s * vec, axis=0)), 1.0, atol=1e-9)


def test_combine_rejects_bad_weighting():
    with pytest.raises(ValueError):
        combine_replicate_moments([_moments("a", np.zeros((3, 2)))], weighting="nope")


# --------------------------------------------------------------------------- progress
def test_progress_is_silent_when_not_a_tty():
    stream = io.StringIO()
    with ProgressReporter(total=5, description="x", stream=stream) as bar:
        for _ in range(5):
            bar.update()
    assert stream.getvalue() == ""
    assert bar.backend == "silent"
    assert list(track([1, 2, 3], stream=stream)) == [1, 2, 3]
    assert stream.getvalue() == ""


def test_progress_env_override_and_quiet(monkeypatch):
    class FakeTTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("CONFORMATLAS_NO_PROGRESS", "1")
    assert not is_interactive(FakeTTY())
    monkeypatch.delenv("CONFORMATLAS_NO_PROGRESS", raising=False)
    assert is_interactive(FakeTTY())
    s = FakeTTY()
    bar = ProgressReporter(total=2, quiet=True, stream=s)
    bar.start()
    bar.update(2)
    bar.close()
    assert s.getvalue() == ""


def test_progress_plain_renderer_writes_on_tty(monkeypatch):
    import conformatlas.progress as prog

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("CONFORMATLAS_NO_PROGRESS", raising=False)
    monkeypatch.setattr(prog, "HAS_RICH", False)
    s = FakeTTY()
    with ProgressReporter(total=3, description="demo", stream=s, min_interval=0.0) as bar:
        for _ in range(3):
            bar.update()
    out = s.getvalue()
    assert "demo" in out and "3/3" in out and out.endswith("\n")


# --------------------------------------------------------------------------- trajectories
@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis is required")
@pytest.mark.parametrize("weighting", ["equal-replicate", "frames"])
def test_streaming_pca_equals_in_memory_on_demo(weighting, tmp_path):
    from conformatlas.streaming import streaming_pca
    from conformatlas.trajectory import (
        load_reference_coordinates,
        resolve_selection_indices,
        stream_trajectory_coordinates,
    )

    os.environ["CONFORMATLAS_NO_PROGRESS"] = "1"
    top = DEMO / "topology.pdb"
    trajs = [DEMO / "rep1.xtc", DEMO / "rep2.xtc"]
    idx = resolve_selection_indices(top, "Backbone")
    ref = load_reference_coordinates(top, atom_indices=idx, fallback_trajectory=trajs[0])
    coords = {}
    for i, t in enumerate(trajs):
        coords[f"Rep_{i + 1}"], _ = stream_trajectory_coordinates(
            top, t, atom_indices=idx, fit_atom_indices=idx, reference_fit_coords=ref
        )
    ev, vec, _, _, mean = compute_cartesian_pca(coords, weighting=weighting, n_components=8)
    proj = np.vstack([project_coordinates(c, mean, vec) for c in coords.values()])

    for n_workers, precision, store in [(1, "float64", None), (2, "float32", tmp_path / "p.npy")]:
        r = streaming_pca(
            top,
            trajs,
            idx,
            fit_atom_indices=idx,
            reference_fit_coords=ref,
            weighting=weighting,
            n_components=8,
            chunk_size=13,
            n_workers=n_workers,
            precision=precision,
            projection_store=store,
        )
        np.testing.assert_allclose(r.eigenvalues, ev, rtol=1e-8)
        np.testing.assert_allclose(np.abs(np.sum(r.eigenvectors * vec, axis=0)), 1.0, atol=1e-8)
        sign = np.sign(np.sum(r.eigenvectors * vec, axis=0))
        tol = 1e-4 if precision == "float32" else 1e-8
        np.testing.assert_allclose(
            np.asarray(r.projection_array) * sign, proj, rtol=tol, atol=tol * np.abs(proj).max()
        )
        assert r.projection_array.dtype == np.dtype(precision)
        if store is not None:
            assert store.exists()
            assert np.load(store, mmap_mode="r").shape == proj.shape
        # projection table layout
        df = r.projections_dataframe()
        assert list(df.columns[:4]) == ["system", "replicate", "frame", "time_ps"]
        assert len(df) == proj.shape[0]
        assert set(df["replicate"]) == {"Rep_1", "Rep_2"}
        assert r.to_pca_results().eigenvalues.shape == (8,)


@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis is required")
def test_frame_chunks_concatenate_to_stream_trajectory_coordinates():
    from conformatlas.streaming import iter_frame_chunks
    from conformatlas.trajectory import resolve_selection_indices, stream_trajectory_coordinates

    top, traj = H1N1 / "neuraminidase_ca.pdb", H1N1 / "neuraminidase_ca.xtc"
    idx = resolve_selection_indices(top, "name CA")
    ref_c, ref_t, ref_r = stream_trajectory_coordinates(
        top, traj, atom_indices=idx, stride=2, begin_ps=1000, end_ps=15000, return_fit_rmsd=True
    )
    chunks = list(
        iter_frame_chunks(top, traj, idx, stride=2, begin_ps=1000, end_ps=15000, chunk_size=5)
    )
    c = np.concatenate([x[0] for x in chunks])
    t = np.concatenate([x[1] for x in chunks])
    r = np.concatenate([x[2] for x in chunks])
    assert c.dtype == np.float32
    np.testing.assert_array_equal(c, ref_c)
    np.testing.assert_array_equal(t, ref_t)
    np.testing.assert_allclose(r, ref_r, rtol=1e-12)
    assert max(len(x[1]) for x in chunks) == 5


@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis is required")
def test_mda_analysis_class_matches_in_memory():
    import MDAnalysis as mda

    from conformatlas.mda_analysis import ConformAtlasPCA
    from conformatlas.trajectory import stream_trajectory_coordinates

    top, traj = H1N1 / "neuraminidase_ca.pdb", H1N1 / "neuraminidase_ca.xtc"
    u = mda.Universe(str(top), str(traj))
    ca = u.select_atoms("name CA")
    u.trajectory[0]
    ref = ca.positions.copy()
    res = ConformAtlasPCA(ca, reference=ref, n_components=6, chunk_size=10).run(
        start=5, stop=75, step=2
    )
    c, _ = stream_trajectory_coordinates(
        top, traj, atom_indices=ca.indices, fit_atom_indices=ca.indices, reference_fit_coords=ref
    )
    c = c[5:75:2]
    ev, vec, exp, _, mean = compute_cartesian_pca({"r": c}, weighting="frames", n_components=6)
    proj = project_coordinates(c, mean, vec)
    np.testing.assert_allclose(res.results.eigenvalues, ev, rtol=1e-8)
    np.testing.assert_allclose(res.results.explained_variance, exp, rtol=1e-8)
    sign = np.sign(np.sum(res.results.eigenvectors * vec, axis=0))
    np.testing.assert_allclose(res.results.projections * sign, proj, rtol=1e-8, atol=1e-8)
    assert res.results.n_frames == len(c)


# --------------------------------------------------------------------------- worker pool start method
def test_start_method_is_spawn_off_linux(monkeypatch):
    """fork crashed a worker on macOS; only Linux may fork by default."""
    from conformatlas import streaming

    monkeypatch.delenv("CONFORMATLAS_MP_START", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert streaming._mp_context().get_start_method() == "spawn"
    monkeypatch.setattr(sys, "platform", "win32")
    assert streaming._mp_context().get_start_method() == "spawn"


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="fork is not available")
def test_start_method_is_fork_on_linux(monkeypatch):
    from conformatlas import streaming

    monkeypatch.delenv("CONFORMATLAS_MP_START", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert streaming._mp_context().get_start_method() == "fork"


def test_start_method_override(monkeypatch):
    from conformatlas import streaming

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("CONFORMATLAS_MP_START", "spawn")
    assert streaming._mp_context().get_start_method() == "spawn"


def test_pool_map_works_with_spawned_workers(monkeypatch):
    """The pool must produce the serial result when its workers are spawned."""
    from conformatlas import streaming

    monkeypatch.setenv("CONFORMATLAS_MP_START", "spawn")
    jobs = [{"x": 1}, {"x": 2}, {"x": 3}]
    assert streaming._pool_map(operator.itemgetter("x"), jobs, 2) == [1, 2, 3]
