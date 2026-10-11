"""Plotly landscape / profile / comparison figures and their embedded data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformatlas.models import BasinState, FELGrid
from conformatlas.utils import R_GAS_CONSTANT

plotly = pytest.importorskip("plotly")

from conformatlas.plotting_interactive import (  # noqa: E402
    build_comparison_figure,
    build_landscape_figure,
    build_projection_figure,
    decode_plotly_array,
    extract_plotly_data,
    marginal_profiles,
    mpl_to_plotly_colorscale,
    plotly_js,
    write_html,
)

T = 300.0


def _samples(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([0.0, 0.0], [0.6, 0.4], size=(int(0.7 * n), 2))
    b = rng.normal([1.8, -1.0], [0.4, 0.4], size=(n - len(a), 2))
    return np.vstack([a, b])


def _grid_from_samples(xy, n=20) -> tuple[FELGrid, np.ndarray]:
    H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=n)
    p = (H / H.sum()).T  # [y, x]
    g = np.full(p.shape, np.nan)
    g[p > 0] = -R_GAS_CONSTANT * T * np.log(p[p > 0] / p.max())
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    basin = np.where(np.meshgrid(xc, yc)[0] < 1.0, 1, 2)
    return FELGrid(xe, ye, xc, yc, p, g, T), basin


def _frames(xy):
    n = len(xy)
    rep = np.where(np.arange(n) < n // 2, "Rep_1", "Rep_2")
    t = np.concatenate([np.arange(n // 2) * 2.0, np.arange(n - n // 2) * 2.0])
    return pd.DataFrame({"replicate": rep, "time_ps": t, "CV1": xy[:, 0], "CV2": xy[:, 1]})


@pytest.fixture(scope="module")
def setup():
    xy = _samples()
    fel, basin = _grid_from_samples(xy)
    states = [
        BasinState(1, "State A", 0.7, 0.0, 0.0, 0.0, "Rep_1", 0, 0.0, 0.0, basin_id=1),
        BasinState(2, "State B", 0.3, 1.8, -1.0, 2.0, "Rep_1", 0, 0.0, 0.0, basin_id=2),
    ]
    return xy, fel, basin, states, _frames(xy)


def test_colorscale_matches_matplotlib():
    import matplotlib

    scale = mpl_to_plotly_colorscale("viridis", n=5)
    assert scale[0][0] == 0.0 and scale[-1][0] == 1.0 and len(scale) == 5
    r, g, b, _ = matplotlib.colormaps["viridis"](1.0)
    assert scale[-1][1] == f"rgb({round(r * 255)},{round(g * 255)},{round(b * 255)})"


def test_marginal_profiles_equal_direct_histogram(setup):
    """Marginal of the 2-D grid == 1-D histogram on the same edges (exact identity)."""
    xy, fel, _, _, _ = setup
    prof = marginal_profiles(fel)
    hx, _ = np.histogram(xy[:, 0], bins=fel.x_edges)
    hy, _ = np.histogram(xy[:, 1], bins=fel.y_edges)
    px, py = hx / hx.sum(), hy / hy.sum()
    assert np.allclose(prof["p_x"], px, atol=1e-12)
    assert np.allclose(prof["p_y"], py, atol=1e-12)
    gx = -R_GAS_CONSTANT * T * np.log(px[px > 0] / px.max())
    assert np.allclose(prof["g_x"][px > 0], gx, atol=1e-9)
    assert np.isnan(prof["g_x"][px == 0]).all()
    assert np.nanmin(prof["g_x"]) == 0.0


def test_landscape_figure_contents(setup):
    xy, fel, basin, states, fa = setup
    fig = build_landscape_figure(
        fel, states=states, frame_assignments=fa, basin_map=basin, max_slider_steps=50
    )
    types = [tr.type for tr in fig.data]
    assert types[0] == "heatmap"
    hm = fig.data[0]
    assert np.array_equal(
        np.nan_to_num(np.asarray(hm.z, float), nan=-1), np.nan_to_num(fel.free_energy, nan=-1)
    )
    assert np.allclose(np.asarray(hm.customdata)[:, :, 0].astype(float), fel.probability)
    labels = np.asarray(hm.customdata)[:, :, 1]
    assert {"State A", "State B"} <= set(np.unique(labels[np.isfinite(fel.free_energy)]))
    assert "unsampled" in labels[~np.isfinite(fel.free_energy)]
    assert types.count("contour") == 2  # one basin outline per state
    paths = [tr for tr in fig.data if tr.name and tr.name.startswith("path ")]
    assert len(paths) == 2 and all(len(tr.x) == len(fa) // 2 for tr in paths)
    assert fig.data[-1].name == "state minima" and list(fig.data[-1].x) == [0.0, 1.8]
    sliders = fig.layout.sliders
    assert len(sliders) == 1 and len(sliders[0].steps) == 50
    step = sliders[0].steps[-1]
    assert step.method == "restyle" and len(step.args[0]["x"]) == 2
    # last slider position = last frame of each replicate
    for rep_i, rep in enumerate(("Rep_1", "Rep_2")):
        last = fa[fa["replicate"] == rep].iloc[-1]
        assert step.args[0]["x"][rep_i][0] == pytest.approx(last["CV1"])


def test_landscape_hover_template_mentions_all_fields(setup):
    _, fel, basin, states, fa = setup
    fig = build_landscape_figure(fel, states=states, frame_assignments=fa, basin_map=basin)
    ht = fig.data[0].hovertemplate
    for token in ("%{x", "%{y", "%{z", "customdata[0]", "customdata[1]", "kJ/mol"):
        assert token in ht


def test_projection_figure(setup):
    _, fel, _, states, _ = setup
    fig = build_projection_figure(fel, states=states)
    assert len(fig.data) == 2
    prof = marginal_profiles(fel)
    got = np.asarray(fig.data[0].y, float)
    assert np.allclose(np.nan_to_num(got, nan=-1), np.nan_to_num(prof["g_x"], nan=-1))


def test_comparison_figure_linked_axes(setup):
    _, fel, _, states, _ = setup
    fels = {"WT": fel, "MUT": fel}
    fig = build_comparison_figure(fels, states_by_system={"WT": states, "MUT": states})
    heat = [tr for tr in fig.data if tr.type == "heatmap"]
    assert len(heat) == 2 and all(tr.coloraxis == "coloraxis" for tr in heat)
    assert fig.layout.xaxis2.matches == "x" and fig.layout.yaxis2.matches == "y"
    assert fig.layout.coloraxis.cmin == 0.0


def test_html_roundtrip_and_js_modes(setup, tmp_path):
    _, fel, basin, states, fa = setup
    fig = build_landscape_figure(fel, states=states, frame_assignments=fa, basin_map=basin)
    p = write_html(fig, tmp_path / "x.html", "t", plotly_js_mode="inline")
    html = p.read_text()
    assert '<script src="https://cdn.plot.ly' not in html and "Plotly.newPlot" in html
    data = extract_plotly_data(html)
    assert len(data) == 1
    (traces,) = data.values()
    z = decode_plotly_array(traces[0]["z"])
    assert np.array_equal(np.nan_to_num(z, nan=-1), np.nan_to_num(fel.free_energy, nan=-1))
    cdn = plotly_js("cdn")
    assert cdn.startswith('<script src="https://cdn.plot.ly/plotly-') and plotly.__version__ in cdn
    assert "</script>" not in plotly_js("inline")[len('<script type="text/javascript">') : -9]
    with pytest.raises(ValueError):
        plotly_js("bogus")


def test_path_and_slider_subsampling(setup):
    _, fel, _, _, fa = setup
    fig = build_landscape_figure(
        fel, frame_assignments=fa, max_path_points=100, max_slider_steps=10
    )
    paths = [tr for tr in fig.data if tr.name and tr.name.startswith("path ")]
    assert all(len(tr.x) == 100 for tr in paths)
    assert len(fig.layout.sliders[0].steps) == 10
