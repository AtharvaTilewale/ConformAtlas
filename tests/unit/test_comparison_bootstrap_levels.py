"""Replicate-level (hierarchical) bootstrap, Welch interval and replicate
heterogeneity in ``conformatlas compare`` statistics."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from conformatlas.basins import detect_energy_basins, frame_bin_indices
from conformatlas.bootstrap import _free_energy_from_counts, circular_block_indices
from conformatlas.comparison_stats import (
    bootstrap_system,
    comparison_warnings,
    heterogeneity_ratio,
    population_difference_table,
    system_block_lengths,
    welch_interval,
)
from conformatlas.fel import calculate_fel
from conformatlas.models import ComparisonConfig

CENTRES = np.array([[-3.0, 0.0], [3.0, 0.0]])


def _frames(system, rep, states, rng):
    xy = CENTRES[states] + rng.normal(0, 0.5, size=(len(states), 2))
    n = len(states)
    return pd.DataFrame(
        {
            "system": system,
            "replicate": f"{system}_{rep}",
            "frame": np.arange(n),
            "time_ps": np.arange(n, dtype=float),
            "PC1": xy[:, 0],
            "PC2": xy[:, 1],
        }
    )


def _iid(n, p_b, rng):
    return (rng.random(n) < p_b).astype(int)


def _setup(rep_state_lists, seed=0):
    """rep_state_lists: {system: [state array per replicate]} -> df, fel, states, assign, labels"""
    rng = np.random.default_rng(seed)
    rows = [
        _frames(sys_name, i, st, rng)
        for sys_name, reps in rep_state_lists.items()
        for i, st in enumerate(reps)
    ]
    df = pd.concat(rows, ignore_index=True)
    fel = calculate_fel(df["PC1"].to_numpy(), df["PC2"].to_numpy(), temperature=300.0, bins=24)
    states, _, assign, labels = detect_energy_basins(fel, df, min_state_population=0.05)
    assert len(states) == 2
    return df, fel, states, assign, labels


def _legacy_bootstrap_populations(projections, fel_grid, basin_labels, states, n_boot, rng):
    """The 1.0.x within-replicate block bootstrap, populations only."""
    cv1 = projections["PC1"].to_numpy(dtype=np.float64)
    cv2 = projections["PC2"].to_numpy(dtype=np.float64)
    reps = projections["replicate"].to_numpy()
    n = len(cv1)
    block_lengths = system_block_lengths(projections)
    x_idx, y_idx, valid = frame_bin_indices(fel_grid, cv1, cv2)
    frame_basin = np.zeros(n, dtype=np.int64)
    frame_basin[valid] = basin_labels[y_idx[valid], x_idx[valid]]
    basin_ids = [s.basin_id for s in states if s.basin_id > 0]
    rep_index = {rep: np.where(reps == rep)[0] for rep in dict.fromkeys(reps)}
    sigma = fel_grid.smooth_sigma if fel_grid.smoothed else 0.0
    pops = np.full((n_boot, len(basin_ids)), np.nan)
    for b in range(n_boot):
        sel = np.concatenate(
            [
                idx[circular_block_indices(len(idx), block_lengths.get(str(rep), 1), rng)]
                for rep, idx in rep_index.items()
            ]
        )
        hist, _, _ = np.histogram2d(cv1[sel], cv2[sel], bins=[fel_grid.x_edges, fel_grid.y_edges])
        _free_energy_from_counts(hist.T, fel_grid.temperature, sigma)
        for k, bid in enumerate(basin_ids):
            pops[b, k] = np.mean(frame_basin[sel] == bid)
    return pops


def test_frame_level_reproduces_legacy_stream():
    rng = np.random.default_rng(3)
    reps = {"A": [_iid(600, 0.3, rng), _iid(500, 0.35, rng)], "B": [_iid(600, 0.3, rng)]}
    df, fel, states, _, labels = _setup(reps)
    sub = df[df["system"] == "A"]
    new = bootstrap_system(
        "A", sub, fel, labels, states, 30, np.random.default_rng(9), level="frame"
    )
    old = _legacy_bootstrap_populations(sub, fel, labels, states, 30, np.random.default_rng(9))
    np.testing.assert_array_equal(new.populations, old)
    assert new.level == "frame" and new.n_replicates == 2


def test_single_replicate_falls_back_to_frame_level_identically():
    rng = np.random.default_rng(4)
    df, fel, states, _, labels = _setup({"A": [_iid(800, 0.3, rng)], "B": [_iid(800, 0.3, rng)]})
    sub = df[df["system"] == "A"]
    r = bootstrap_system("A", sub, fel, labels, states, 25, np.random.default_rng(1))
    f = bootstrap_system("A", sub, fel, labels, states, 25, np.random.default_rng(1), level="frame")
    assert r.level == "frame"
    np.testing.assert_array_equal(r.populations, f.populations)
    np.testing.assert_array_equal(r.fe_grids, f.fe_grids)


def test_replicate_level_includes_between_replicate_spread():
    """Replicates trapped in different states: the frame-level interval is (near)
    zero-width, the replicate-level one spans the replicate populations."""
    a = [np.zeros(400, int), np.ones(400, int), np.zeros(400, int)]
    b = [np.zeros(400, int), np.ones(400, int), np.ones(400, int)]
    df, fel, states, assign, labels = _setup({"A": a, "B": b})
    boots_f = {
        n: bootstrap_system(
            n, df[df.system == n], fel, labels, states, 300, np.random.default_rng(2), level="frame"
        )
        for n in "AB"
    }
    boots_r = {
        n: bootstrap_system(
            n, df[df.system == n], fel, labels, states, 300, np.random.default_rng(2)
        )
        for n in "AB"
    }
    assert boots_r["A"].level == "replicate" and boots_r["A"].n_replicates == 3
    sd_f = np.std(boots_f["A"].populations[:, 0])
    sd_r = np.std(boots_r["A"].populations[:, 0])
    assert sd_f < 0.01 < 0.15 < sd_r
    # every replicate-level resample is a mixture of ~0 % and ~100 % replicates, so
    # its population sits near a multiple of 1/3 (a few noisy frames fall outside
    # both basins), and at least three of the four mixtures occur
    v = boots_r["A"].populations[:, 0] * 3
    assert np.all(np.abs(v - np.round(v)) < 0.05)
    assert len(np.unique(np.round(v))) >= 3
    tf = population_difference_table(states, assign, ["A", "B"], "A", boots_f)
    tr = population_difference_table(states, assign, ["A", "B"], "A", boots_r)
    width_f = tf["B_minus_A_CI95_High"][0] - tf["B_minus_A_CI95_Low"][0]
    width_r = tr["B_minus_A_CI95_High"][0] - tr["B_minus_A_CI95_Low"][0]
    assert width_f < 2.0 and width_r > 30.0
    assert tr["B_minus_A_CI95_Low"][0] <= 0 <= tr["B_minus_A_CI95_High"][0]
    assert tf["A_Bootstrap_Level"][0] == "frame" and tr["A_Bootstrap_Level"][0] == "replicate"
    # heterogeneity: replicates differ by ~100 points while each barely fluctuates
    assert tr["A_Replicate_Heterogeneity_Ratio"][0] > 50.0
    assert tr["A_Replicate_Min_Percent"][0] < 2.0 and tr["A_Replicate_Max_Percent"][0] > 98.0
    notes = comparison_warnings(tr, ["A", "B"], "A", boots_r)
    assert any("replicates disagree" in n and n.startswith("A,") for n in notes)
    notes_f = comparison_warnings(tf, ["A", "B"], "A", boots_f)
    assert any("bootstrap_level is 'frame'" in n for n in notes_f)


def test_homogeneous_replicates_have_ratio_near_one_and_no_warning():
    rng = np.random.default_rng(5)
    reps = {
        "A": [_iid(2000, 0.3, rng) for _ in range(4)],
        "B": [_iid(2000, 0.3, rng) for _ in range(4)],
    }
    df, fel, states, assign, labels = _setup(reps)
    boots = {
        n: bootstrap_system(
            n, df[df.system == n], fel, labels, states, 200, np.random.default_rng(6)
        )
        for n in "AB"
    }
    t = population_difference_table(states, assign, ["A", "B"], "A", boots)
    r = t["A_Replicate_Heterogeneity_Ratio"].to_numpy()
    assert np.all((r > 0.2) & (r < 2.0))
    assert not any("disagree" in n for n in comparison_warnings(t, ["A", "B"], "A", boots))


def test_single_replicate_warning():
    rng = np.random.default_rng(7)
    df, fel, states, assign, labels = _setup(
        {"A": [_iid(800, 0.3, rng)], "B": [_iid(800, 0.3, rng)]}
    )
    boots = {
        n: bootstrap_system(
            n, df[df.system == n], fel, labels, states, 20, np.random.default_rng(0)
        )
        for n in "AB"
    }
    t = population_difference_table(states, assign, ["A", "B"], "A", boots)
    assert np.isnan(t["B_minus_A_Welch_CI95_Low"][0])
    notes = comparison_warnings(t, ["A", "B"], "A", boots)
    assert sum("single replicate" in n for n in notes) == 2


def test_welch_interval_matches_scipy():
    a = np.array([65.0, 70.0, 61.0])
    b = np.array([40.0, 46.0, 55.0, 49.0])
    d, lo, hi, df = welch_interval(b, a)
    tt = sps.ttest_ind(b, a, equal_var=False)
    se = np.sqrt(a.var(ddof=1) / 3 + b.var(ddof=1) / 4)
    assert d == pytest.approx(b.mean() - a.mean())
    assert d / se == pytest.approx(tt.statistic, rel=1e-12)
    half = sps.t.ppf(0.975, df) * se
    assert lo == pytest.approx(d - half) and hi == pytest.approx(d + half)
    if hasattr(tt, "confidence_interval"):
        ci = tt.confidence_interval(0.95)
        assert lo == pytest.approx(ci.low, abs=1e-10) and hi == pytest.approx(ci.high, abs=1e-10)
        assert df == pytest.approx(tt.df, abs=1e-10)
    assert all(np.isnan(v) for v in welch_interval(b, a[:1]))
    assert welch_interval(np.array([5.0, 5.0]), np.array([3.0, 3.0]))[:3] == (2.0, 2.0, 2.0)


def test_heterogeneity_ratio_edge_cases():
    assert np.isnan(heterogeneity_ratio(np.array([10.0]), [np.array([9.0, 11.0])]))
    assert np.isnan(heterogeneity_ratio(np.array([0.0, 0.0]), [np.zeros(5), np.zeros(5)]))
    assert np.isinf(heterogeneity_ratio(np.array([0.0, 100.0]), [np.zeros(5), np.full(5, 100.0)]))
    w = [np.array([9.0, 11.0, 10.0, 10.0]), np.array([19.0, 21.0, 20.0, 20.0])]
    expected = np.std([10.0, 20.0], ddof=1) / np.sqrt(np.mean([np.var(x, ddof=1) for x in w]))
    assert heterogeneity_ratio(np.array([10.0, 20.0]), w) == pytest.approx(expected)


def test_invalid_level_rejected(tmp_path):
    rng = np.random.default_rng(8)
    df, fel, states, _, labels = _setup({"A": [_iid(300, 0.3, rng)], "B": [_iid(300, 0.3, rng)]})
    with pytest.raises(ValueError, match="level"):
        bootstrap_system("A", df, fel, labels, states, 2, rng, level="block")
    with pytest.raises(ValueError, match="bootstrap_level"):
        ComparisonConfig(
            project="x",
            systems=[],
            reference_system="A",
            output_dir=tmp_path,
            bootstrap_level="block",
        )
