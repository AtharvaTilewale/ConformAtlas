"""Overlap metrics, subspace RMSIP and bootstrap p-values used in comparisons."""

import numpy as np
import pandas as pd

from conformatlas.comparison_stats import (
    _bootstrap_p_value,
    landscape_overlap,
    landscape_overlap_table,
    probability_grid,
    subspace_overlap_matrix,
)
from conformatlas.fel import calculate_fel


def test_overlap_identical_and_disjoint():
    p = np.zeros((4, 4))
    p[0, 0] = p[1, 1] = 0.5
    q = np.zeros((4, 4))
    q[3, 3] = 1.0
    same = landscape_overlap(p, p)
    assert abs(same["Bhattacharyya_Coefficient"] - 1.0) < 1e-12
    assert abs(same["JS_Divergence_bits"]) < 1e-12
    assert abs(same["Sampled_Bin_Jaccard"] - 1.0) < 1e-12
    diff = landscape_overlap(p, q)
    assert abs(diff["Bhattacharyya_Coefficient"]) < 1e-12
    assert abs(diff["JS_Divergence_bits"] - 1.0) < 1e-12
    assert diff["Sampled_Bin_Jaccard"] == 0.0


def test_probability_grid_from_fel_roundtrip():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(5000, 2))
    fel = calculate_fel(xy[:, 0], xy[:, 1], temperature=300.0, bins=10)
    p = probability_grid(fel)
    assert abs(p.sum() - 1.0) < 1e-9
    # agrees with the stored probability where sampled
    mask = np.isfinite(fel.free_energy)
    stored = fel.probability / fel.probability.sum()
    np.testing.assert_allclose(p[mask], stored[mask], rtol=1e-6, atol=1e-9)
    tbl = landscape_overlap_table({"A": fel, "B": fel})
    assert len(tbl) == 1 and abs(tbl["Bhattacharyya_Coefficient"][0] - 1) < 1e-9


def test_subspace_matrix_symmetric_unit_diagonal():
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(30, 30)))
    a = q[:, :10]
    b = q[:, 5:15]  # shares 5 of 10 directions -> RMSIP = sqrt(5/10)
    df = subspace_overlap_matrix({"A": a, "B": b, "C": q[:, 10:20]}, n_modes=10)
    assert isinstance(df, pd.DataFrame)
    np.testing.assert_allclose(np.diag(df.to_numpy()), 1.0)
    assert abs(df.loc["A", "B"] - np.sqrt(0.5)) < 1e-9
    assert abs(df.loc["A", "C"]) < 1e-9
    assert df.loc["A", "B"] == df.loc["B", "A"]


def test_bootstrap_p_value():
    assert _bootstrap_p_value(np.array([1.0, 2.0, 3.0, 4.0])) == 0.25  # floor 1/B
    assert _bootstrap_p_value(np.array([-1.0, 1.0, -1.0, 1.0])) == 1.0
    assert (
        abs(_bootstrap_p_value(np.concatenate([np.full(95, 1.0), np.full(5, -1.0)])) - 0.10) < 1e-12
    )
