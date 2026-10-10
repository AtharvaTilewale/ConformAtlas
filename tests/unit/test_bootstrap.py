"""Block bootstrap of the landscape and derived state statistics."""

import numpy as np
import pandas as pd

from conformatlas.basins import detect_energy_basins
from conformatlas.bootstrap import bootstrap_landscape, circular_block_indices
from conformatlas.fel import calculate_fel


def _two_state_projection(n=4000, p_a=0.7, seed=0):
    rng = np.random.default_rng(seed)
    # Markov switching so the series is autocorrelated
    state = np.empty(n, dtype=int)
    state[0] = 0
    stay = 0.97
    for i in range(1, n):
        if rng.random() < stay:
            state[i] = state[i - 1]
        else:
            state[i] = 1 - state[i - 1]
    # bias to target population by relabelling fractions of visits is messy; instead use
    # asymmetric switching: p(0->1) = (1-p_a)/p_a * p(1->0)
    p10 = 0.03
    p01 = p10 * (1 - p_a) / p_a
    state[0] = 0
    for i in range(1, n):
        if state[i - 1] == 0:
            state[i] = 1 if rng.random() < p01 else 0
        else:
            state[i] = 0 if rng.random() < p10 else 1
    centres = np.array([[-3.0, 0.0], [3.0, 0.0]])
    xy = centres[state] + rng.normal(0, 0.4, size=(n, 2))
    df = pd.DataFrame(
        {
            "system": "S",
            "replicate": "Rep_1",
            "frame": np.arange(n),
            "time_ps": 10.0 * np.arange(n),
            "PC1": xy[:, 0],
            "PC2": xy[:, 1],
        }
    )
    return df, state


def test_circular_block_indices_cover_series():
    rng = np.random.default_rng(0)
    idx = circular_block_indices(100, 10, rng)
    assert idx.shape == (100,)
    assert idx.min() >= 0 and idx.max() < 100
    # consecutive entries inside a block increase by one (mod n)
    assert np.all((np.diff(idx[:10]) % 100) == 1)


def test_bootstrap_ci_contains_true_population():
    df, state = _two_state_projection()
    fel = calculate_fel(df["PC1"].to_numpy(), df["PC2"].to_numpy(), temperature=300.0, bins=24)
    states, _, assignments, basin_map = detect_energy_basins(fel, df, min_state_population=0.05)
    assert len(states) == 2
    labels = assignments["state"].to_numpy()
    res = bootstrap_landscape(
        df["PC1"].to_numpy(),
        df["PC2"].to_numpy(),
        df["replicate"].to_numpy(),
        fel,
        basin_map,
        states,
        block_lengths={"Rep_1": 60},
        n_boot=100,
        seed=1,
    )
    assert res.fe_std.shape == fel.free_energy.shape
    table = res.state_table.set_index("state")
    # Which detected state is the x<0 well?
    for st in states:
        truth = np.mean(state == (0 if st.min_pc1 < 0 else 1))
        row = table.loc[st.label]
        assert row["population_ci95_low"] <= truth <= row["population_ci95_high"], (
            st.label,
            truth,
            row,
        )
        assert row["population_sd"] > 0.0
    # Point estimates match the frame assignments
    for st in states:
        assert abs(table.loc[st.label, "population"] - np.mean(labels == st.label)) < 1e-9
    # The free-energy SD is finite inside the wells and larger at the barrier/edges
    inside = np.isfinite(res.fe_std)
    assert inside.sum() > 10
    assert np.nanmedian(res.fe_std[inside]) < np.nanmax(res.fe_std[inside])
