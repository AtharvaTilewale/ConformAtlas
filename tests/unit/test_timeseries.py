"""Statistical inefficiency, equilibration detection and block lengths."""

import numpy as np

from conformatlas.timeseries import (
    block_length_from_g,
    detect_equilibration,
    effective_sample_size,
    statistical_inefficiency,
)


def _ar1(n, phi, rng, burn=0, offset=0.0):
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * rng.normal()
    if burn:
        x[:burn] += offset * np.linspace(1, 0, burn)
    return x


def test_white_noise_has_g_near_one():
    rng = np.random.default_rng(1)
    g = statistical_inefficiency(rng.normal(size=20000))
    assert 0.9 <= g <= 1.3


def test_ar1_recovers_known_inefficiency():
    rng = np.random.default_rng(2)
    phi = 0.9
    x = _ar1(200_000, phi, rng)
    # For AR(1): g = (1 + phi) / (1 - phi) = 19
    g = statistical_inefficiency(x)
    assert abs(g - 19.0) / 19.0 < 0.2
    assert abs(effective_sample_size(x) - len(x) / g) < 1e-6


def test_constant_series_is_finite():
    g = statistical_inefficiency(np.ones(100))
    assert np.isfinite(g) and g >= 1.0


def test_equilibration_detects_initial_drift():
    rng = np.random.default_rng(3)
    x = _ar1(4000, 0.5, rng, burn=800, offset=25.0)
    res = detect_equilibration(x, nskip=10)
    assert 500 <= res.t0_index <= 1100
    assert res.n_eff > res.n_eff_full
    assert res.n_total == 4000


def test_equilibration_keeps_everything_when_stationary():
    rng = np.random.default_rng(4)
    res = detect_equilibration(rng.normal(size=3000), nskip=20)
    assert res.t0_index <= 300  # noise can move it a little, but not far


def test_block_length_respects_min_blocks():
    assert block_length_from_g(1.0, 1000) >= 1
    assert block_length_from_g(50.0, 100, min_blocks=5) <= 20
    assert block_length_from_g(3.0, 10_000) >= 3
