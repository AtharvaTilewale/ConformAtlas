"""Minimum-distance vs medoid representative selection."""

import numpy as np
import pandas as pd

from conformatlas.models import BasinState
from conformatlas.representatives import select_representative_frames


def _state(label="State A", min_pc1=0.0, min_pc2=0.0):
    return BasinState(
        state_id=1,
        label=label,
        population=1.0,
        min_pc1=min_pc1,
        min_pc2=min_pc2,
        min_free_energy=0.0,
        representative_replicate="Rep_1",
        representative_frame=0,
        representative_time_ps=0.0,
        representative_distance=0.0,
        basin_id=1,
    )


def test_minimum_and_medoid_pick_expected_frames():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 2))
    pts[7] = [5.0, 5.0]  # outlier far from everybody
    pts[3] = [0.01, 0.0]  # closest to the (0,0) minimum
    df = pd.DataFrame(
        {
            "replicate": "Rep_1",
            "frame": np.arange(200),
            "time_ps": 10.0 * np.arange(200),
            "PC1": pts[:, 0],
            "PC2": pts[:, 1],
            "state": "State A",
        }
    )
    df.loc[7, "PC1"], df.loc[7, "PC2"] = 5.0, 5.0
    st = _state()
    mn = select_representative_frames(st, df, n_representatives=1, method="minimum")
    assert mn["frame"].iloc[0] == 3
    med = select_representative_frames(st, df, n_representatives=3, method="medoid")
    assert len(med) == 3
    assert 7 not in med["frame"].tolist()
    assert med["selection_score"].is_monotonic_increasing
    # medoid is near the cloud centre
    assert np.hypot(med["PC1"].iloc[0], med["PC2"].iloc[0]) < 0.5


def test_empty_state_returns_empty():
    df = pd.DataFrame(
        {"replicate": [], "frame": [], "time_ps": [], "PC1": [], "PC2": [], "state": []}
    )
    out = select_representative_frames(_state(), df)
    assert out.empty
