"""Reloaded results must reproduce the written float64 values bit-for-bit.

pandas' default CSV float parser can differ from the written value by 1-2 ULP.
That is harmless for plotting but not for re-binning reloaded frames against
grid edges that were derived from the exact values: the frame sitting on the
outermost edge ends up one ULP outside the grid and is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conformatlas.basins import frame_bin_indices
from conformatlas.fel import calculate_fel
from conformatlas.results_io import _read_csv


def test_read_csv_is_bit_exact(tmp_path):
    x = np.random.default_rng(0).normal(size=5000)
    path = tmp_path / "x.csv"
    pd.DataFrame({"a": x}).to_csv(path, index=False)
    assert np.array_equal(_read_csv(path)["a"].to_numpy(), x)


def test_reloaded_frames_stay_inside_saved_grid(tmp_path):
    """Every frame (including the extremes that define the grid edges) re-bins validly."""
    for seed in range(25):
        rng = np.random.default_rng(seed)
        cv1, cv2 = rng.normal(size=200), rng.normal(size=200)
        grid = calculate_fel(cv1, cv2, temperature=300.0, bins=10)
        path = tmp_path / f"cv_{seed}.csv"
        pd.DataFrame({"CV1": cv1, "CV2": cv2}).to_csv(path, index=False)
        back = _read_csv(path)
        _, _, valid = frame_bin_indices(grid, back["CV1"].to_numpy(), back["CV2"].to_numpy())
        assert valid.all(), f"seed {seed}: {int((~valid).sum())} frame(s) fell outside the grid"
