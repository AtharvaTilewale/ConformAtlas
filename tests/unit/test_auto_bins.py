"""Automatic grid selection and weighted landscapes."""

import numpy as np

from conformatlas.fel import calculate_fel, choose_bins


def test_choose_bins_grows_with_sample_size():
    rng = np.random.default_rng(0)
    small = rng.normal(size=(300, 2))
    large = rng.normal(size=(60_000, 2))
    b_small = choose_bins(small[:, 0], small[:, 1])
    b_large = choose_bins(large[:, 0], large[:, 1])
    assert 16 <= b_small <= 128 and 16 <= b_large <= 128
    assert b_large > b_small


def test_choose_bins_clips_for_degenerate_data():
    x = np.zeros(100)
    assert choose_bins(x, x) == 16


def test_weighted_fel_shifts_minimum():
    rng = np.random.default_rng(1)
    a = rng.normal(-2, 0.3, size=(500, 2))
    b = rng.normal(2, 0.3, size=(500, 2))
    xy = np.vstack([a, b])
    # Weights favouring the second well should move the global minimum there.
    w = np.concatenate([np.ones(500), np.full(500, 20.0)])
    fel = calculate_fel(xy[:, 0], xy[:, 1], temperature=300.0, bins=20, weights=w)
    iy, ix = np.unravel_index(np.nanargmin(fel.free_energy), fel.free_energy.shape)
    assert fel.x_centers[ix] > 0 and fel.y_centers[iy] > 0
    fel0 = calculate_fel(xy[:, 0], xy[:, 1], temperature=300.0, bins=20)
    # unweighted: the two wells are (nearly) degenerate; weighted: well A sits ~kT ln 20 higher
    kT = 0.0083144626 * 300.0
    ia = np.nanmin(
        fel.free_energy[: fel.free_energy.shape[0] // 2, : fel.free_energy.shape[1] // 2]
    )
    assert abs(ia - kT * np.log(20.0)) < 1.5
    assert abs(np.nanmin(fel0.free_energy[:10, :10])) < 1.5
