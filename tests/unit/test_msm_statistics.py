"""Segmentation-aware bootstrap, multi-observable equilibration and MSM figures."""

import numpy as np
import pandas as pd
import pytest

from conformatlas import msm as M
from conformatlas import plotting_msm as PM
from conformatlas.basins import detect_energy_basins
from conformatlas.bootstrap import bootstrap_landscape
from conformatlas.fel import calculate_fel
from conformatlas.timeseries import (
    block_length_from_g,
    detect_equilibration,
    detect_equilibration_multi,
    statistical_inefficiency,
)


def _markov_chain(T, n, rng, s0=0):
    cum = np.cumsum(T, axis=1)
    s = np.empty(n, dtype=np.int64)
    s[0] = s0
    u = rng.random(n)
    for i in range(1, n):
        s[i] = np.searchsorted(cum[s[i - 1]], u[i])
    return s


def _two_well(n=3000, sep=3.0, sd=0.5, seed=0, n_rep=1):
    rng = np.random.default_rng(seed)
    P = np.array([[0.985, 0.015], [0.035, 0.965]])
    dfs = []
    for r in range(n_rep):
        s = _markov_chain(P, n, rng)
        xy = np.array([[-sep, 0.0], [sep, 0.0]])[s] + rng.normal(0, sd, size=(n, 2))
        dfs.append(
            pd.DataFrame(
                {
                    "system": "S",
                    "replicate": f"R{r + 1}",
                    "frame": np.arange(n),
                    "time_ps": np.arange(n, dtype=float),
                    "PC1": xy[:, 0],
                    "PC2": xy[:, 1],
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


# --------------------------------------------------------------------- bootstrap


def test_resegment_mode_two_well():
    df = _two_well()
    x, y = df["PC1"].to_numpy(), df["PC2"].to_numpy()
    fel = calculate_fel(x, y, 300.0, bins=24)
    states, _, _, labels = detect_energy_basins(fel, df, min_state_population=0.05)
    assert len(states) == 2
    g = max(statistical_inefficiency(x), statistical_inefficiency(y))
    L = block_length_from_g(g, len(df))
    res = bootstrap_landscape(
        x,
        y,
        df["replicate"].to_numpy(),
        fel,
        labels,
        states,
        {"R1": L},
        n_boot=60,
        seed=1,
        mode="resegment",
        resegment_kwargs={"min_state_population": 0.05},
    )
    assert res.mode == "resegment"
    assert res.n_states_per_resample.shape == (60,)
    dist = res.state_count_distribution
    assert np.isclose(dist["fraction"].sum(), 1.0)
    assert dist.loc[dist["n_states"] == 2, "fraction"].iloc[0] > 0.9
    tbl = res.resegment_state_table.set_index("state")
    assert (tbl["match_rate"] > 0.9).all()
    for s in states:
        row = tbl.loc[s.label]
        assert row["population_ci95_low_resegmented"] <= s.population
        assert s.population <= row["population_ci95_high_resegmented"]
    # fixed-label tables still present and identical to fixed mode with same seed
    fixed = bootstrap_landscape(
        x, y, df["replicate"].to_numpy(), fel, labels, states, {"R1": L}, n_boot=60, seed=1
    )
    assert fixed.mode == "fixed" and fixed.state_count_distribution is None
    pd.testing.assert_frame_equal(fixed.state_table, res.state_table)
    assert any("Re-segmentation" in n for n in res.notes)


def test_bootstrap_mode_validation():
    df = _two_well(n=500)
    fel = calculate_fel(df["PC1"].to_numpy(), df["PC2"].to_numpy(), 300.0, bins=16)
    states, _, _, labels = detect_energy_basins(fel, df, min_state_population=0.05)
    with pytest.raises(ValueError):
        bootstrap_landscape(
            df["PC1"].to_numpy(),
            df["PC2"].to_numpy(),
            df["replicate"].to_numpy(),
            fel,
            labels,
            states,
            {"R1": 5},
            n_boot=3,
            mode="bogus",
        )


# --------------------------------------------------------------------- equilibration


def _ar1(n, phi, rng):
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * rng.normal()
    return x


def test_detect_equilibration_multi_latest_t0_and_joint():
    rng = np.random.default_rng(5)
    n = 8000
    a, b = _ar1(n, 0.9, rng), _ar1(n, 0.9, rng)
    a[:600] += 6.0 * np.linspace(1, 0, 600) ** 2
    b[:2000] += 6.0 * np.linspace(1, 0, 2000) ** 2
    res = detect_equilibration_multi({"CV1": a, "CV2": b}, nskip=20)
    ta = detect_equilibration(a, nskip=20).t0_index
    tb = detect_equilibration(b, nskip=20).t0_index
    assert res.per_observable["CV1"].t0_index == ta
    assert res.per_observable["CV2"].t0_index == tb
    assert res.t0_index == max(ta, tb) and res.deciding_observable == "CV2"
    assert res.n_total == n
    joint = detect_equilibration_multi({"CV1": a, "CV2": b}, method="joint", nskip=20)
    # the joint optimum cannot discard less than the slower observable needs by much
    assert joint.t0_index >= 0.5 * tb
    with pytest.raises(ValueError):
        detect_equilibration_multi({"a": a, "b": b[:-1]})
    with pytest.raises(ValueError):
        detect_equilibration_multi({"a": a}, method="nope")
    with pytest.raises(ValueError):
        detect_equilibration_multi({})


def test_detect_equilibration_multi_against_pymbar():
    pts = pytest.importorskip("pymbar.timeseries")
    rng = np.random.default_rng(11)
    n = 10000
    a, b = _ar1(n, 0.95, rng), _ar1(n, 0.95, rng)
    a[:800] += 8.0 * np.linspace(1, 0, 800) ** 2
    b[:2500] += 8.0 * np.linspace(1, 0, 2500) ** 2
    res = detect_equilibration_multi({"a": a, "b": b}, nskip=25)
    for name, series in (("a", a), ("b", b)):
        t_pm = pts.detect_equilibration(series, nskip=25, fast=False)[0]
        assert abs(res.per_observable[name].t0_index - t_pm) <= 25


# --------------------------------------------------------------------- figures


def test_msm_figures_render(tmp_path):
    df = _two_well(n=2500, n_rep=2, seed=3)
    fel = calculate_fel(df["PC1"].to_numpy(), df["PC2"].to_numpy(), 300.0, bins=24)
    res = M.build_msm(df, ["PC1", "PC2"], lag=3, lags=[1, 2, 3, 6], n_clusters=20, n_metastable=2)
    p1 = PM.plot_implied_timescales(
        res.implied_timescales, tmp_path / "its.png", res.lag_report, chosen_lag=3
    )
    p2 = PM.plot_ck_test(res.ck_test, tmp_path / "ck.png")
    p3 = PM.plot_pcca_landscape(
        fel,
        df,
        res.frame_table,
        res.clustering.centers,
        res.memberships,
        tmp_path / "pcca.png",
    )
    p4 = PM.plot_metastable_timeseries(res.frame_table, tmp_path / "ts.png")
    p5, grids = PM.plot_per_replicate_landscapes(
        df, {"temperature": 300.0, "bins": 24}, tmp_path / "per_rep.png"
    )
    for p in (p1, p2, p3, p4, p5):
        assert p.exists() and p.stat().st_size > 1000
    assert set(grids) == {"R1", "R2"}
    g1, g2 = grids["R1"], grids["R2"]
    assert np.array_equal(g1.x_edges, g2.x_edges) and np.array_equal(g1.y_edges, g2.y_edges)
    assert np.nanmin(g1.free_energy) == 0.0
    with pytest.raises(ValueError):
        PM.plot_ck_test(res.ck_test.iloc[0:0], tmp_path / "empty.png")
