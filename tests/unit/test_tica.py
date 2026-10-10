"""TICA should recover a slow direction hidden under fast, high-variance noise."""

import numpy as np

from conformatlas.pca import compute_cartesian_pca, compute_tica, implied_timescales


def _slow_fast_data(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    slow = np.empty(n)
    slow[0] = 0.0
    phi = 0.995  # correlation time ~200 frames
    for i in range(1, n):
        slow[i] = phi * slow[i - 1] + np.sqrt(1 - phi**2) * rng.normal()
    fast = rng.normal(size=(n, 5)) * 3.0  # high variance, zero autocorrelation
    data = np.column_stack([slow, fast])
    # mix so the slow mode is not aligned with an axis
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    return data @ q.T, q[:, 0]


def test_tica_finds_slow_direction_pca_does_not():
    x, slow_dir = _slow_fast_data()
    coords = {"Rep_1": x.reshape(len(x), 2, 3)}
    evals, evecs, kin, cum, mean = compute_tica(coords, lag=20, n_components=3)
    assert evals[0] > evals[1]
    overlap_tica = abs(np.dot(evecs[:, 0] / np.linalg.norm(evecs[:, 0]), slow_dir))
    assert overlap_tica > 0.95
    _, pca_vecs, _, _, _ = compute_cartesian_pca(coords, n_components=3)
    overlap_pca = abs(np.dot(pca_vecs[:, 0], slow_dir))
    assert overlap_pca < 0.6
    ts = implied_timescales(evals, lag_frames=20, dt_ps=1.0)
    # -1/ln(0.995) ~ 200 frames; sampling noise and finite lag allow a generous window
    assert 100 < ts[0] < 400
    assert np.all(np.diff(cum) >= -1e-9)
