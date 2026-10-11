"""Figure style context: formats, vector SVG, dpi, colormap, font scale, path overlay."""

from __future__ import annotations

import re

import matplotlib
import numpy as np
import pandas as pd
import pytest

from conformatlas.models import BasinState, FELGrid
from conformatlas.plotting import (
    FigureStyle,
    figure_style,
    get_figure_style,
    parse_fig_format,
    plot_fel_2d,
    reset_figure_style,
    save_figure,
    set_figure_style,
)
from conformatlas.utils import R_GAS_CONSTANT


def _grid(n=24, T=300.0) -> FELGrid:
    x = np.linspace(-3, 3, n + 1)
    y = np.linspace(-2, 2, n + 1)
    xc, yc = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    X, Y = np.meshgrid(xc, yc)
    p = np.exp(-0.5 * (X**2 + 2 * Y**2)) + 0.6 * np.exp(
        -0.5 * ((X - 1.5) ** 2 + 2 * (Y + 0.8) ** 2)
    )
    p /= p.sum()
    g = -R_GAS_CONSTANT * T * np.log(p / p.max())
    g[p < 1e-6] = np.nan
    return FELGrid(x, y, xc, yc, p, g, T)


def _states():
    return [
        BasinState(1, "State A", 0.7, 0.0, 0.0, 0.0, "Rep_1", 3, 30.0, 0.1),
        BasinState(2, "State B", 0.3, 1.5, -0.8, 1.2, "Rep_1", 7, 70.0, 0.1),
    ]


def _path(n=60):
    rng = np.random.default_rng(1)
    rows = []
    for rep in ("Rep_1", "Rep_2"):
        xy = np.cumsum(rng.normal(0, 0.15, size=(n, 2)), axis=0)
        for i in range(n):
            rows.append({"replicate": rep, "time_ps": 10.0 * i, "CV1": xy[i, 0], "CV2": xy[i, 1]})
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _reset_style():
    reset_figure_style()
    yield
    reset_figure_style()


def test_parse_fig_format():
    assert parse_fig_format(None) == ("png",)
    assert parse_fig_format("all") == ("png", "svg", "pdf")
    assert parse_fig_format("svg,png") == ("svg", "png")
    assert parse_fig_format(".PDF") == ("pdf",)
    with pytest.raises(ValueError, match="Unknown figure format"):
        parse_fig_format("tiff")


def test_defaults_unchanged():
    st = get_figure_style()
    assert st == FigureStyle()
    assert st.formats == ("png",) and st.colormap == "turbo" and st.dpi == 300
    assert st.font_scale == 1.0 and st.path_overlay is False


def test_set_style_validation():
    with pytest.raises(ValueError, match="Unknown colormap"):
        set_figure_style(colormap="not-a-colormap")
    with pytest.raises(ValueError, match="font_scale"):
        set_figure_style(font_scale=10)
    with pytest.raises(ValueError, match="dpi"):
        set_figure_style(dpi=5)


def test_context_manager_restores():
    with figure_style(fig_format="svg", colormap="viridis", font_scale=1.5, dpi=72):
        st = get_figure_style()
        assert st.formats == ("svg",) and st.colormap == "viridis"
        assert matplotlib.rcParams["font.size"] == pytest.approx(11 * 1.5)
    assert get_figure_style() == FigureStyle()
    assert matplotlib.rcParams["font.size"] == pytest.approx(11)


def test_all_formats_written_and_primary_path(tmp_path):
    with figure_style(fig_format="all"):
        p = plot_fel_2d(_grid(), states=_states(), out_path=tmp_path / "fel.png")
    assert p == tmp_path / "fel.png"
    for ext in ("png", "svg", "pdf"):
        assert (tmp_path / f"fel.{ext}").stat().st_size > 1000
        assert (tmp_path / f"fel_unannotated.{ext}").exists()
    with figure_style(fig_format="svg"):
        p = plot_fel_2d(_grid(), out_path=tmp_path / "only.png")
    assert p == tmp_path / "only.svg" and not (tmp_path / "only.png").exists()


def test_svg_is_vector(tmp_path):
    with figure_style(fig_format="svg"):
        p = plot_fel_2d(_grid(), states=_states(), out_path=tmp_path / "fel.png")
    svg = p.read_text(encoding="utf-8")
    assert svg.lstrip().startswith("<?xml") or "<svg" in svg[:200]
    assert svg.count("<path") > 50
    assert "<image" not in svg  # no embedded raster


def test_png_dpi_metadata(tmp_path):
    from PIL import Image

    for dpi in (72, 150):
        with figure_style(dpi=dpi):
            p = plot_fel_2d(_grid(), out_path=tmp_path / f"d{dpi}.png", save_unannotated=False)
        with Image.open(p) as im:
            assert im.info["dpi"][0] == pytest.approx(dpi, abs=1)


def test_colormap_colours_in_svg(tmp_path):
    """Every contour fill colour must be a colour of the requested colormap."""
    name = "viridis"
    cmap = matplotlib.colormaps[name]
    palette = {matplotlib.colors.to_hex(cmap(i / 255.0)) for i in range(256)}
    with figure_style(fig_format="svg", colormap=name):
        p = plot_fel_2d(_grid(), out_path=tmp_path / "cm.png", save_unannotated=False)
    svg = p.read_text(encoding="utf-8")
    fills = set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)) | set(
        re.findall(r"fill:\s*(#[0-9a-fA-F]{6})", svg)
    )
    contour_fills = {f.lower() for f in fills} - {"#ffffff", "#000000", "#f2f2f2"}
    assert len(contour_fills) > 5
    missing = contour_fills - palette
    assert not missing, f"fill colours not in {name}: {sorted(missing)[:5]}"


def test_font_scale_doubles_text_size(tmp_path):
    sizes = {}
    for scale in (1.0, 2.0):
        with (
            matplotlib.rc_context({"svg.fonttype": "none"}),
            figure_style(fig_format="svg", font_scale=scale),
        ):
            p = plot_fel_2d(_grid(), out_path=tmp_path / f"f{scale}.png", save_unannotated=False)
        svg = p.read_text(encoding="utf-8")
        found = [float(v) for v in re.findall(r"font(?:-size)?:[^;\"]*?([0-9.]+)px", svg)]
        sizes[scale] = max(found) if found else None
    assert sizes[1.0] and sizes[2.0]
    assert sizes[2.0] / sizes[1.0] == pytest.approx(2.0, rel=0.02)


def test_path_overlay_draws_lines(tmp_path):
    with figure_style(fig_format="svg"):
        p_plain = plot_fel_2d(_grid(), out_path=tmp_path / "plain.png", save_unannotated=False)
        p_path = plot_fel_2d(
            _grid(), out_path=tmp_path / "path.png", save_unannotated=False, path=_path()
        )
    assert p_path.stat().st_size > p_plain.stat().st_size
    # the LineCollection adds one <path> per segment: 2 replicates x 59 segments
    n_plain = p_plain.read_text(encoding="utf-8").count("<path")
    n_path = p_path.read_text(encoding="utf-8").count("<path")
    assert n_path - n_plain >= 2 * 59


def test_save_figure_closes_and_returns(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with figure_style(fig_format="pdf,png"):
        out = save_figure(fig, tmp_path / "x.png")
    assert out == tmp_path / "x.pdf" and (tmp_path / "x.png").exists()
    assert not plt.fignum_exists(fig.number)
