"""Leading-eigenpair (Lanczos) PCA route versus the dense decompositions.

The truncated solver must reproduce the full eigendecomposition: same eigenvalues,
same eigenvectors up to sign, same explained-variance percentages, for both
weightings, for fewer frames than coordinates and the reverse, and through the
streaming covariance path.
"""

import numpy as np
import pytest

import conformatlas.pca as pca_mod
import conformatlas.streaming as streaming
from conformatlas.pca import (
    compute_cartesian_pca,
    leading_eigenpairs,
    leading_eigenpairs_of_scaled_data,
    use_truncated_solver,
)
from conformatlas.streaming import (
    ReplicateMoments,
    combine_replicate_moments,
    eigendecompose_covariance,
)

K = 8


def _ensemble(n_frames: int, n_dims: int, seed: int, rank: int = 30) -> np.ndarray:
    """Low-rank signal with a decaying spectrum plus noise, so eigenvalues are separated."""
    rng = np.random.default_rng(seed)
    scales = np.geomspace(5.0, 0.3, rank)
    signal = (rng.standard_normal((n_frames, rank)) * scales) @ rng.standard_normal((rank, n_dims))
    return signal / np.sqrt(n_dims) + 0.05 * rng.standard_normal((n_frames, n_dims))


def _both(reps, weighting, monkeypatch):
    """Run the truncated route and, with it disabled, the dense route."""
    truncated = compute_cartesian_pca(reps, weighting=weighting, n_components=K)
    with monkeypatch.context() as m:
        m.setattr(pca_mod, "TRUNCATED_MIN_DIM", 10**9)
        dense = compute_cartesian_pca(reps, weighting=weighting, n_components=K)
    return truncated, dense


@pytest.mark.parametrize("weighting", ["frames", "equal-replicate"])
@pytest.mark.parametrize(
    ("n_frames", "n_dims"), [(1100, 1800), (1500, 1200)], ids=["frames<coords", "frames>coords"]
)
def test_truncated_matches_dense(n_frames, n_dims, weighting, monkeypatch):
    x = _ensemble(n_frames, n_dims, seed=7)
    # unequal replicate sizes exercise the weighting
    reps = {
        "a": x[: n_frames // 3].reshape(-1, n_dims // 3, 3),
        "b": x[n_frames // 3 :].reshape(-1, n_dims // 3, 3),
    }
    (ev_t, vec_t, ex_t, cum_t, mean_t), (ev_d, vec_d, ex_d, cum_d, mean_d) = _both(
        reps, weighting, monkeypatch
    )
    assert ev_t.shape == (K,) and vec_t.shape == (n_dims, K)
    np.testing.assert_allclose(ev_t, ev_d, rtol=1e-9)
    np.testing.assert_allclose(ex_t, ex_d, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(cum_t, cum_d, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(mean_t, mean_d, rtol=0, atol=1e-12)
    # eigenvectors agree up to sign
    np.testing.assert_allclose(np.abs(np.sum(vec_t * vec_d, axis=0)), 1.0, atol=1e-8)


def test_truncated_pairs_satisfy_eigen_equation():
    """Independent of the dense route: C v = lambda v, orthonormal, explained variance sums."""
    n_frames, n_dims = 1200, 1500
    x = _ensemble(n_frames, n_dims, seed=11)
    ev, vec, explained, _, mean = compute_cartesian_pca(
        {"r": x.reshape(n_frames, -1, 3)}, weighting="frames", n_components=K
    )
    centred = x - mean
    cov = centred.T @ centred / n_frames
    residual = np.linalg.norm(cov @ vec - vec * ev, axis=0) / ev
    assert residual.max() < 1e-9
    np.testing.assert_allclose(vec.T @ vec, np.eye(K), atol=1e-10)
    # explained variance is relative to the full trace, not to the retained eigenvalues
    np.testing.assert_allclose(explained, 100.0 * ev / np.trace(cov), rtol=1e-10)
    assert explained.sum() < 100.0


def test_truncated_route_is_deterministic_and_sign_fixed():
    x = _ensemble(1100, 1401, seed=5).reshape(1100, -1, 3)
    first = compute_cartesian_pca({"r": x}, weighting="frames", n_components=K)
    second = compute_cartesian_pca({"r": x}, weighting="frames", n_components=K)
    np.testing.assert_array_equal(first[1], second[1])
    top = first[1][np.argmax(np.abs(first[1]), axis=0), np.arange(K)]
    assert np.all(top > 0)


def test_selector_thresholds():
    assert not use_truncated_solver(999, 5)  # small problem: dense is as fast
    assert use_truncated_solver(1000, 5)
    assert not use_truncated_solver(1000, 200)  # many components: dense is better
    assert use_truncated_solver(5000, 500)


def test_many_components_use_dense_route(monkeypatch):
    """Asking for a large share of the spectrum must not change results."""
    x = _ensemble(1100, 1500, seed=2).reshape(1100, -1, 3)
    called = []
    real = pca_mod.leading_eigenpairs_of_scaled_data

    def spy(*args, **kwargs):
        called.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pca_mod, "leading_eigenpairs_of_scaled_data", spy)
    ev, *_ = compute_cartesian_pca({"r": x}, weighting="frames", n_components=400)
    assert not called
    assert ev.shape == (400,)


def test_falls_back_when_spectrum_is_rank_deficient():
    """Numerically zero requested eigenvalues return None so the caller uses the dense route."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1100, 3)) @ rng.standard_normal((3, 1300))
    assert leading_eigenpairs_of_scaled_data(data - data.mean(axis=0), K) is None


def test_leading_eigenpairs_rejects_degenerate_request():
    assert leading_eigenpairs(np.eye(5), 5) is None
    assert leading_eigenpairs(np.eye(5), 0) is None


def test_streaming_eigendecomposition_matches_dense(monkeypatch):
    x = _ensemble(1500, 1300, seed=13)
    cov = np.cov(x.T, bias=True)
    ev_t, vec_t, ex_t, cum_t = eigendecompose_covariance(cov, K)
    with monkeypatch.context() as m:
        m.setattr(streaming, "use_truncated_solver", lambda *a: False)
        ev_d, vec_d, ex_d, cum_d = streaming.eigendecompose_covariance(cov, K)
    np.testing.assert_allclose(ev_t, ev_d, rtol=1e-9)
    np.testing.assert_allclose(ex_t, ex_d, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(cum_t, cum_d, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.abs(np.sum(vec_t * vec_d, axis=0)), 1.0, atol=1e-8)


@pytest.mark.parametrize("weighting", ["frames", "equal-replicate"])
def test_combined_moments_match_direct_weighted_covariance(weighting):
    """The in-place covariance merge keeps the weighting of compute_cartesian_pca."""
    rng = np.random.default_rng(4)
    dims = 40
    blocks = [rng.standard_normal((n, dims)) + off for n, off in [(30, 0.0), (70, 0.5), (50, -0.3)]]
    moments = []
    for i, b in enumerate(blocks):
        mean_b = b.mean(axis=0)
        scatter = (b - mean_b).T @ (b - mean_b)
        moments.append(
            ReplicateMoments(f"r{i}", len(b), mean_b, scatter, np.arange(len(b)), np.zeros(len(b)))
        )
    mean, cov = combine_replicate_moments(moments, weighting=weighting)

    n_total = sum(len(b) for b in blocks)
    if weighting == "equal-replicate":
        ref_mean = sum(b.mean(axis=0) for b in blocks) / len(blocks)
        ref = sum((b - ref_mean).T @ (b - ref_mean) / (len(blocks) * len(b)) for b in blocks)
    else:
        allx = np.vstack(blocks)
        ref_mean = allx.mean(axis=0)
        ref = (allx - ref_mean).T @ (allx - ref_mean) / n_total
    np.testing.assert_allclose(mean, ref_mean, atol=1e-12)
    np.testing.assert_allclose(cov, ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cov, cov.T, atol=0)
