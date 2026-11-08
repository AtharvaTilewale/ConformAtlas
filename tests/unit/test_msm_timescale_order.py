"""Implied timescales are ranked by |lambda| (deeptime / PyEMMA convention) and negative
eigenvalues among the slow timescales are reported."""

import numpy as np
import pytest

from conformatlas import msm as M

# 4-state symmetric (hence reversible, uniform pi) chain with eigenvalues 1, 0.5, -0.3, 0.1,
# built from an orthonormal Hadamard basis; smallest entry (1 - 0.5 - 0.3 - 0.1) / 4 = 0.025 > 0.
_H = 0.5 * np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]], dtype=float).T
EIG = np.array([1.0, 0.5, -0.3, 0.1])
T_NEG = _H @ np.diag(EIG) @ _H.T


def _sample(T, n, seed=0):
    rng = np.random.default_rng(seed)
    cum = np.cumsum(T, axis=1)
    s = np.empty(n, dtype=int)
    s[0] = 0
    u = rng.random(n)
    for t in range(1, n):
        s[t] = min(int(np.searchsorted(cum[s[t - 1]], u[t])), T.shape[0] - 1)
    return s


def test_construction():
    assert T_NEG.min() > 0
    np.testing.assert_allclose(T_NEG.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(T_NEG))[::-1], np.sort(EIG)[::-1], atol=1e-12
    )


def test_default_order_unchanged_for_reversible():
    vals, _, _ = M.eigen_decomposition(T_NEG)
    np.testing.assert_allclose(vals, [1.0, 0.5, 0.1, -0.3], atol=1e-12)


def test_magnitude_order_and_eigenvector_consistency():
    vals, R, L = M.eigen_decomposition(T_NEG, order="magnitude")
    np.testing.assert_allclose(vals, [1.0, 0.5, -0.3, 0.1], atol=1e-12)
    np.testing.assert_allclose(T_NEG @ R, R * vals, atol=1e-12)  # columns stay paired with values
    np.testing.assert_allclose(L.T @ R, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(R[:, 0], 1.0)


def test_magnitude_order_breaks_ties_towards_positive():
    T = np.array([[0.0, 1.0], [1.0, 0.0]])  # eigenvalues +1 and -1 (periodic chain)
    vals, R, _ = M.eigen_decomposition(T, order="magnitude")
    np.testing.assert_allclose(vals, [1.0, -1.0])
    np.testing.assert_allclose(R[:, 0], 1.0)


def test_bad_order_rejected():
    with pytest.raises(ValueError):
        M.eigen_decomposition(T_NEG, order="abs")


def test_implied_timescales_magnitude():
    lag, dt = 3, 2.0
    ts = M.implied_timescales([1.0, 0.5, -0.3, 0.1], lag, dt, magnitude=True)
    np.testing.assert_allclose(ts, -lag * dt / np.log([0.5, 0.3, 0.1]))
    ts_signed = M.implied_timescales([1.0, 0.5, -0.3, 0.1], lag, dt)
    assert np.isnan(ts_signed[1])  # legacy behaviour kept when magnitude=False


def test_negative_slow_eigenvalues():
    np.testing.assert_allclose(M.negative_slow_eigenvalues([1.0, 0.5, -0.3, 0.1], 2), [-0.3])
    assert len(M.negative_slow_eigenvalues([1.0, 0.5, -0.3, 0.1], 1)) == 0


def test_estimate_msm_ranks_by_magnitude_and_records_negatives():
    est = M.estimate_msm([_sample(T_NEG, 200_000)], lag=1, reversible=True)
    assert est.eigenvalues[2] < 0  # -0.3 ranked before 0.1
    assert est.estimator_info["negative_eigenvalues"]
    np.testing.assert_allclose(est.eigenvalues[1:], [0.5, -0.3, 0.1], atol=0.01)
    np.testing.assert_allclose(est.timescales_ps, -1.0 / np.log(np.abs(est.eigenvalues[1:])))


def test_scan_reports_negative_count():
    its = M.implied_timescales_scan([_sample(T_NEG, 50_000)], [1, 2], n_timescales=3)
    assert "n_negative_eigenvalues_reported" in its.columns
    assert its.loc[its.lag_frames == 1, "n_negative_eigenvalues_reported"].item() == 1
    # at lag 2 every eigenvalue is squared, so none is negative
    assert its.loc[its.lag_frames == 2, "n_negative_eigenvalues_reported"].item() == 0


def test_matches_deeptime_timescales():
    deeptime = pytest.importorskip("deeptime")
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM

    dtraj = _sample(T_NEG, 200_000, seed=3)
    est = M.estimate_msm([dtraj], lag=1, reversible=True)
    dm = MaximumLikelihoodMSM(reversible=True).fit_fetch(
        TransitionCountEstimator(lagtime=1, count_mode="sliding").fit_fetch(dtraj)
    )
    np.testing.assert_allclose(est.timescales_ps, dm.timescales(), rtol=1e-6)
    assert deeptime.__version__
