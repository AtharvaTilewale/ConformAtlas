"""Unit tests for temperature scaling of free energy landscape."""

import numpy as np
import pytest

from conformatlas.fel import calculate_fel


def test_temperature_scaling_ratio():
    """Verify that ΔG scales strictly linearly with temperature: ΔG(T2) / ΔG(T1) == T2 / T1."""
    np.random.seed(42)
    # Generate 1000 points from a 2D Gaussian distribution
    pc1 = np.random.normal(0.0, 1.0, 1000)
    pc2 = np.random.normal(0.0, 1.0, 1000)

    t1 = 300.0
    t2 = 330.0

    fel1 = calculate_fel(pc1, pc2, temperature=t1, bins=32)
    fel2 = calculate_fel(pc1, pc2, temperature=t2, bins=32)

    # Probabilities should be identical for same data and bins
    np.testing.assert_allclose(fel1.probability, fel2.probability, atol=1e-10)

    # Valid mask where both are non-zero and finite
    valid = (
        (~np.isnan(fel1.free_energy)) & (~np.isnan(fel2.free_energy)) & (fel1.free_energy > 0.05)
    )

    fe1_vals = fel1.free_energy[valid]
    fe2_vals = fel2.free_energy[valid]

    ratio = fe2_vals / fe1_vals
    expected_ratio = t2 / t1  # 330 / 300 = 1.10

    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-5)


def test_temperature_validation():
    """Ensure invalid temperatures (<= 0 K) raise ValueError."""
    pc1 = np.array([1.0, 2.0, 3.0])
    pc2 = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Temperature must be positive"):
        calculate_fel(pc1, pc2, temperature=0.0)

    with pytest.raises(ValueError, match="Temperature must be positive"):
        calculate_fel(pc1, pc2, temperature=-100.0)
