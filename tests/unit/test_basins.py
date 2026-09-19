"""Unit tests for topological energy basin detection and state population recovery."""

import numpy as np
import pandas as pd
import pytest

from conformatlas.basins import detect_energy_basins
from conformatlas.fel import calculate_fel


def test_basin_detection_recovers_known_gaussian_mixture():
    """Verify that basin detection recovers 3 known clusters (65%, 25%, 10%) within tolerance."""
    np.random.seed(101)
    n_total = 2000

    # Cluster A: 65% at (-3.0, -3.0)
    # Cluster B: 25% at (3.0, 3.0)
    # Cluster C: 10% at (3.0, -3.0)
    n_a = int(0.65 * n_total)
    n_b = int(0.25 * n_total)
    n_c = n_total - n_a - n_b

    a_x = np.random.normal(-3.0, 0.5, n_a)
    a_y = np.random.normal(-3.0, 0.5, n_a)

    b_x = np.random.normal(3.0, 0.5, n_b)
    b_y = np.random.normal(3.0, 0.5, n_b)

    c_x = np.random.normal(3.0, 0.5, n_c)
    c_y = np.random.normal(-3.0, 0.5, n_c)

    pc1 = np.concatenate([a_x, b_x, c_x])
    pc2 = np.concatenate([a_y, b_y, c_y])

    projections = pd.DataFrame({
        "system": "TestSystem",
        "replicate": "Rep1",
        "frame": np.arange(n_total),
        "time_ps": np.arange(n_total) * 10.0,
        "PC1": pc1,
        "PC2": pc2,
    })

    fel = calculate_fel(pc1, pc2, temperature=300.0, bins=40, smooth_sigma=1.0)
    states, unassigned_pop, frame_assignments, basin_map = detect_energy_basins(
        fel, projections, min_state_population=0.05, max_states=5, min_peak_distance=3
    )

    # Should detect 3 major states
    assert len(states) >= 3

    # States ordered by population descending
    assert states[0].label == "State A"
    assert states[1].label == "State B"
    assert states[2].label == "State C"

    # Verify population proportions within 5% tolerance
    assert pytest.approx(states[0].population, abs=0.07) == 0.65
    assert pytest.approx(states[1].population, abs=0.06) == 0.25
    assert pytest.approx(states[2].population, abs=0.05) == 0.10

    # Total population check: sum(major) + unassigned ≈ 1.0 (100%)
    total_pop_sum = sum(s.population for s in states) + unassigned_pop
    assert pytest.approx(total_pop_sum, abs=1e-5) == 1.0


def test_state_minimum_coordinates():
    """Verify that basin minimum coordinates are placed close to cluster centers."""
    np.random.seed(42)
    # Single well centered at (2.5, -1.5)
    pc1 = np.random.normal(2.5, 0.2, 500)
    pc2 = np.random.normal(-1.5, 0.2, 500)

    projections = pd.DataFrame({
        "system": "Test",
        "replicate": "Rep1",
        "frame": np.arange(500),
        "time_ps": np.arange(500),
        "PC1": pc1,
        "PC2": pc2,
    })

    fel = calculate_fel(pc1, pc2, temperature=300.0, bins=30, smooth_sigma=1.0)
    states, unassigned_pop, _, _ = detect_energy_basins(fel, projections, min_state_population=0.5)

    assert len(states) == 1
    assert pytest.approx(states[0].min_pc1, abs=0.3) == 2.5
    assert pytest.approx(states[0].min_pc2, abs=0.3) == -1.5
    assert states[0].min_free_energy == 0.0
