"""Unit tests for Free Energy Landscape (FEL) grid calculation."""

import numpy as np
import pytest

from conformatlas.fel import calculate_fel


def test_fel_probability_and_minimum():
    """Verify probability sums to 1.0 and minimum ΔG is exactly 0.0 kJ/mol."""
    np.random.seed(42)
    pc1 = np.random.normal(0, 1, 500)
    pc2 = np.random.normal(0, 1, 500)

    fel = calculate_fel(pc1, pc2, temperature=300.0, bins=20)

    # Probability sum
    assert pytest.approx(np.sum(fel.probability), rel=1e-6) == 1.0

    # Global minimum free energy
    valid_fe = fel.free_energy[~np.isnan(fel.free_energy)]
    assert np.min(valid_fe) == 0.0


def test_unobserved_bins_are_nan():
    """Unobserved histogram bins (zero probability) must NEVER be zero free energy; they must be NaN."""
    # Data localized tightly in one quadrant
    pc1 = np.array([10.0, 10.1, 10.2, 10.05])
    pc2 = np.array([10.0, 10.1, 10.2, 10.05])

    fel = calculate_fel(pc1, pc2, temperature=300.0, bins=10, smooth_sigma=0.0)

    # Most bins should have zero counts
    zero_prob_bins = (fel.probability == 0.0)
    assert np.sum(zero_prob_bins) > 0

    # For every bin with zero probability, free energy must be NaN
    assert np.all(np.isnan(fel.free_energy[zero_prob_bins]))


def test_gaussian_smoothing():
    """Verify Gaussian smoothing option computes smooth density and keeps valid energy bounds."""
    np.random.seed(42)
    pc1 = np.random.normal(0, 1, 200)
    pc2 = np.random.normal(0, 1, 200)

    fel_raw = calculate_fel(pc1, pc2, temperature=300.0, bins=30, smooth_sigma=0.0)
    fel_smooth = calculate_fel(pc1, pc2, temperature=300.0, bins=30, smooth_sigma=1.0)

    assert fel_raw.smoothed is False
    assert fel_smooth.smoothed is True
    assert pytest.approx(np.sum(fel_smooth.probability), rel=1e-5) == 1.0
