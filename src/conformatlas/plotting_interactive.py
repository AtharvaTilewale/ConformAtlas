"""Interactive (Plotly) figures for the HTML report.

Everything here is optional: ``plotly`` is imported lazily, and callers should
use :func:`plotly_available` (or catch :class:`MissingDependencyError`) to fall
back to the static matplotlib figures.

All quantities are taken verbatim from the pipeline objects: the heatmap ``z``
is ``FELGrid.free_energy`` (kJ/mol), hover probabilities are
``FELGrid.probability`` and the 1-D profiles are the Boltzmann inversion of the
marginals of ``FELGrid.probability`` (``-RT ln(P_i / max P)``). No smoothing or
re-binning is applied in this module.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conformatlas.errors import MissingDependencyError
from conformatlas.models import BasinState, FELGrid
from conformatlas.plotting import CV_LABELS, get_figure_style
from conformatlas.utils import R_GAS_CONSTANT

PLOTLY_PIP = "pip install plotly"


def plotly_available() -> bool:
    """True when plotly can be imported."""
    try:
        import plotly  # noqa: F401
    except Exception:
        return False
    return True


def _go():
    try:
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover - exercised through plotly_available
        raise MissingDependencyError(
            "plotly", PLOTLY_PIP, "interactive landscape figures in the HTML report"
        ) from e
    return go


def mpl_to_plotly_colorscale(name: str | None = None, n: int = 32) -> list[list[Any]]:
    """Sample a matplotlib colormap into a Plotly colorscale (``n`` evenly spaced stops)."""
    import matplotlib

    cmap = matplotlib.colormaps[name or get_figure_style().colormap]
    scale = []
    for i in range(n):
        pos = i / (n - 1)
        r, g, b, _ = cmap(pos)
        scale.append(
            [pos, f"rgb({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))})"]
        )
    return scale


def plotly_js(mode: str = "inline") -> str:
    """Return the ``<script>`` tag that provides Plotly.

    ``mode='inline'`` embeds the full plotly.min.js shipped with the Python
    package (self-contained report, ~4.8 MB); ``mode='cdn'`` references the
    official CDN (small file, needs network when opened).
    """
    if mode == "cdn":
        import plotly

        ver = plotly.__version__.split("+")[0]
        return f'<script src="https://cdn.plot.ly/plotly-{ver}.min.js" charset="utf-8"></script>'
    if mode != "inline":
        raise ValueError("plotly_js mode must be 'inline' or 'cdn'")
    from plotly.offline import get_plotlyjs

    return f'<script type="text/javascript">{get_plotlyjs()}</script>'


def figure_to_div(fig, div_id: str | None = None, height: int | None = None) -> str:
    """HTML fragment (``<div>`` + ``Plotly.newPlot`` script) without the library."""
    kw: dict[str, Any] = {
        "full_html": False,
        "include_plotlyjs": False,
        "div_id": div_id,
        "config": {"responsive": True, "displaylogo": False},
    }
    if height is not None:
        kw["default_height"] = height
    return fig.to_html(**kw)


def write_html(fig_or_divs, path: str | Path, title: str, plotly_js_mode: str = "inline") -> Path:
    """Write one or several figures to a standalone HTML page."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(fig_or_divs, (list, tuple)):
        fig_or_divs = [fig_or_divs]
    divs = [f if isinstance(f, str) else figure_to_div(f) for f in fig_or_divs]
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>{plotly_js(plotly_js_mode)}"
        "<style>body{font-family:sans-serif;margin:20px}</style></head>"
        f"<body><h1>{title}</h1>{''.join(divs)}</body></html>"
    )
    path.write_text(html, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _state_label_grid(
    fel_grid: FELGrid, basin_map: np.ndarray | None, states: list[BasinState] | None
) -> np.ndarray:
    """Per-bin state label ('State A', 'minor basin', 'unsampled')."""
    ny, nx = fel_grid.free_energy.shape
    labels = np.full((ny, nx), "unsampled", dtype=object)
    sampled = np.isfinite(fel_grid.free_energy)
    labels[sampled] = "minor / transitional"
    if basin_map is not None and states:
        by_id = {int(s.basin_id): s.label for s in states if s.basin_id is not None}
        bm = np.asarray(basin_map)
        if bm.shape == labels.shape:
            for bid, lab in by_id.items():
                labels[(bm == bid) & sampled] = lab
    return labels


def _basin_outlines(fel_grid: FELGrid, basin_map: np.ndarray, states: list[BasinState]):
    """One constraint-contour trace per major state outlining its watershed basin."""
    go = _go()
    traces = []
    bm = np.asarray(basin_map)
    if bm.shape != fel_grid.free_energy.shape:
        return traces
    for st in states:
        mask = (bm == int(st.basin_id)).astype(float)
        if mask.sum() == 0:
            continue
        traces.append(
            go.Contour(
                x=fel_grid.x_centers,
                y=fel_grid.y_centers,
                z=mask,
                contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                line=dict(color="black", width=1.5),
                showscale=False,
                hoverinfo="skip",
                name=f"{st.label} basin outline",
                legendgroup="outlines",
                showlegend=False,
            )
        )
    return traces


def marginal_profiles(fel_grid: FELGrid) -> dict[str, np.ndarray]:
    """1-D free-energy profiles along each axis from the marginals of the 2-D probability.

    ``P_x[i] = sum_j P[j, i]`` (``probability`` is indexed ``[y, x]``), ``G = -RT ln(P / max P)``.
    Unsampled bins are NaN. This is an exact identity of the 2-D histogram, not a
    new estimate: with an unsmoothed grid it equals the 1-D histogram of the CV on
    the same edges (validated in validation/checks_report.py).
    """
    P = np.nan_to_num(np.asarray(fel_grid.probability, dtype=float), nan=0.0)
    T = fel_grid.temperature
    out: dict[str, np.ndarray] = {}
    for key, p in (("x", P.sum(axis=0)), ("y", P.sum(axis=1))):
        g = np.full(p.shape, np.nan)
        pos = p > 0
        if pos.any():
            g[pos] = -R_GAS_CONSTANT * T * np.log(p[pos] / p[pos].max())
        out[f"p_{key}"] = p
        out[f"g_{key}"] = g
    out["x"] = np.asarray(fel_grid.x_centers)
    out["y"] = np.asarray(fel_grid.y_centers)
    return out


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------


def build_landscape_figure(
    fel_grid: FELGrid,
    states: list[BasinState] | None = None,
    frame_assignments: pd.DataFrame | None = None,
    basin_map: np.ndarray | None = None,
    cv_labels: tuple[str, str] | None = None,
    colormap: str | None = None,
    cv_columns: tuple[str, str] = ("CV1", "CV2"),
    max_slider_steps: int = 200,
    max_path_points: int = 5000,
    title: str | None = None,
):
    """Interactive 2-D landscape: heatmap of ΔG with hover details, basin outlines,
    state minima, per-replicate time-coloured paths (legend-toggleable) and a time
    slider moving a marker along every replicate.

    Hover text shows the bin centre, ΔG (kJ/mol), probability and the state label
    of the bin (from ``basin_map`` when given). ``frame_assignments`` must hold the
    columns ``replicate``, ``time_ps`` and ``cv_columns``; paths longer than
    ``max_path_points`` and sliders with more than ``max_slider_steps`` positions are
    subsampled evenly (the landscape itself never is).
    """
    go = _go()
    cv1, cv2 = cv_labels or (CV_LABELS[0], CV_LABELS[1])
    z = np.asarray(fel_grid.free_energy, dtype=float)
    prob = np.asarray(fel_grid.probability, dtype=float)
    labels = _state_label_grid(fel_grid, basin_map, states)
    custom = np.dstack([prob, labels.astype(object)])
    finite = z[np.isfinite(z)]
    zmax = float(min(np.percentile(finite, 98), 35.0)) if finite.size else 25.0

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=fel_grid.x_centers,
            y=fel_grid.y_centers,
            z=z,
            customdata=custom,
            colorscale=mpl_to_plotly_colorscale(colormap),
            zmin=0.0,
            zmax=zmax,
            colorbar=dict(title="ΔG (kJ/mol)"),
            hoverongaps=False,
            hovertemplate=(
                f"{cv1}: %{{x:.3f}}<br>{cv2}: %{{y:.3f}}<br>"
                "ΔG: %{z:.3f} kJ/mol<br>P: %{customdata[0]:.3e}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
            name="ΔG",
        )
    )
    if basin_map is not None and states:
        for tr in _basin_outlines(fel_grid, basin_map, states):
            fig.add_trace(tr)

    # trajectory paths
    slider_traces: list[int] = []
    time_grid = None
    rep_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if frame_assignments is not None and len(frame_assignments):
        fa = frame_assignments
        reps = list(dict.fromkeys(fa["replicate"].tolist()))
        for rep in reps:
            sub = fa[fa["replicate"] == rep].sort_values("time_ps", kind="stable")
            t = sub["time_ps"].to_numpy(dtype=float)
            x = sub[cv_columns[0]].to_numpy(dtype=float)
            y = sub[cv_columns[1]].to_numpy(dtype=float)
            if len(x) > max_path_points:
                keep = np.linspace(0, len(x) - 1, max_path_points).astype(int)
                t, x, y = t[keep], x[keep], y[keep]
            rep_series[rep] = (t, x, y)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    line=dict(color="rgba(255,255,255,0.55)", width=1),
                    marker=dict(
                        size=4,
                        color=t,
                        colorscale="Greys",
                        cmin=float(t.min()),
                        cmax=float(t.max()),
                        showscale=False,
                        line=dict(width=0),
                    ),
                    customdata=t,
                    hovertemplate=(
                        f"{rep}<br>t = %{{customdata:.1f}} ps<br>%{{x:.3f}}, %{{y:.3f}}<extra></extra>"
                    ),
                    name=f"path {rep}",
                    legendgroup=f"path-{rep}",
                    visible=True,
                )
            )
        # slider marker traces (one per replicate), initial position = first frame
        all_t = np.concatenate([v[0] for v in rep_series.values()])
        t_lo, t_hi = float(all_t.min()), float(all_t.max())
        n_steps = int(min(max_slider_steps, len(np.unique(all_t))))
        time_grid = np.linspace(t_lo, t_hi, max(n_steps, 1))
        for rep, (t, x, y) in rep_series.items():
            slider_traces.append(len(fig.data))
            fig.add_trace(
                go.Scatter(
                    x=[x[0]],
                    y=[y[0]],
                    mode="markers",
                    marker=dict(
                        size=13,
                        symbol="circle",
                        color="#ff2d55",
                        line=dict(color="black", width=1.2),
                    ),
                    name=f"current frame {rep}",
                    hovertemplate=f"{rep} at t = %{{text}} ps<extra></extra>",
                    text=[f"{t[0]:.1f}"],
                    legendgroup=f"path-{rep}",
                    showlegend=False,
                )
            )

    # state minima
    if states:
        fig.add_trace(
            go.Scatter(
                x=[s.min_pc1 for s in states],
                y=[s.min_pc2 for s in states],
                mode="markers+text",
                marker=dict(
                    size=16, symbol="star", color="white", line=dict(color="black", width=1.5)
                ),
                text=[f"{s.label} ({s.population * 100:.1f}%)" for s in states],
                textposition="top center",
                textfont=dict(color="white", size=12),
                customdata=[[s.min_free_energy, s.population * 100] for s in states],
                hovertemplate="%{text}<br>min ΔG = %{customdata[0]:.2f} kJ/mol<extra></extra>",
                name="state minima",
            )
        )

    layout = dict(
        title=title or f"Free energy landscape (T = {fel_grid.temperature:.1f} K)",
        xaxis=dict(title=cv1, constrain="domain"),
        yaxis=dict(title=cv2),
        plot_bgcolor="#f2f2f2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=90, b=60),
        height=620,
    )
    if time_grid is not None and slider_traces:
        steps = []
        for tv in time_grid:
            xs, ys, txt = [], [], []
            for rep, (t, x, y) in rep_series.items():
                k = int(np.argmin(np.abs(t - tv)))
                xs.append([float(x[k])])
                ys.append([float(y[k])])
                txt.append([f"{t[k]:.1f}"])
            steps.append(
                dict(
                    method="restyle",
                    args=[{"x": xs, "y": ys, "text": txt}, slider_traces],
                    label=f"{tv:.0f}",
                )
            )
        layout["sliders"] = [
            dict(
                active=0,
                currentvalue=dict(prefix="time (ps): "),
                pad=dict(t=40),
                steps=steps,
            )
        ]
        layout["margin"]["b"] = 110
        layout["height"] = 700
    fig.update_layout(**layout)
    return fig


def build_projection_figure(
    fel_grid: FELGrid,
    cv_labels: tuple[str, str] | None = None,
    states: list[BasinState] | None = None,
):
    """Two-panel 1-D free-energy profiles along each collective variable.

    Profiles come from :func:`marginal_profiles` (marginals of the 2-D grid); the
    dotted vertical lines mark state minima.
    """
    go = _go()
    from plotly.subplots import make_subplots

    cv1, cv2 = cv_labels or (CV_LABELS[0], CV_LABELS[1])
    prof = marginal_profiles(fel_grid)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"ΔG along {cv1}", f"ΔG along {cv2}"))
    for col, (key, lab) in enumerate((("x", cv1), ("y", cv2)), start=1):
        fig.add_trace(
            go.Scatter(
                x=prof[key],
                y=prof[f"g_{key}"],
                mode="lines+markers",
                name=f"ΔG({lab})",
                customdata=prof[f"p_{key}"],
                hovertemplate=f"{lab} = %{{x:.3f}}<br>ΔG = %{{y:.3f}} kJ/mol<br>P = %{{customdata:.3e}}<extra></extra>",
                line=dict(color="#2b6cb0", width=2),
                connectgaps=False,
            ),
            row=1,
            col=col,
        )
        if states:
            for s in states:
                xv = s.min_pc1 if key == "x" else s.min_pc2
                fig.add_vline(x=xv, line=dict(color="grey", dash="dot", width=1), row=1, col=col)
        fig.update_xaxes(title_text=lab, row=1, col=col)
        fig.update_yaxes(title_text="ΔG (kJ/mol)", row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=60, r=20, t=50, b=50), showlegend=False)
    return fig


def build_comparison_figure(
    fels: dict[str, FELGrid],
    states_by_system: dict[str, list[BasinState]] | None = None,
    cv_labels: tuple[str, str] | None = None,
    colormap: str | None = None,
    zmax: float | None = None,
):
    """Side-by-side heatmaps of several systems' landscapes on a shared colour scale
    with linked x/y axes (zooming one panel zooms all)."""
    go = _go()
    from plotly.subplots import make_subplots

    cv1, cv2 = cv_labels or (CV_LABELS[0], CV_LABELS[1])
    names = list(fels)
    if zmax is None:
        vals = np.concatenate(
            [np.asarray(f.free_energy)[np.isfinite(f.free_energy)].ravel() for f in fels.values()]
        )
        zmax = float(min(np.percentile(vals, 98), 35.0)) if vals.size else 25.0
    fig = make_subplots(
        rows=1,
        cols=len(names),
        shared_xaxes="all",
        shared_yaxes="all",
        subplot_titles=names,
        horizontal_spacing=0.04,
    )
    scale = mpl_to_plotly_colorscale(colormap)
    for col, name in enumerate(names, start=1):
        f = fels[name]
        fig.add_trace(
            go.Heatmap(
                x=f.x_centers,
                y=f.y_centers,
                z=f.free_energy,
                customdata=np.asarray(f.probability),
                coloraxis="coloraxis",
                hoverongaps=False,
                hovertemplate=(
                    f"{name}<br>{cv1}: %{{x:.3f}}<br>{cv2}: %{{y:.3f}}<br>"
                    "ΔG: %{z:.3f} kJ/mol<br>P: %{customdata:.3e}<extra></extra>"
                ),
                name=name,
            ),
            row=1,
            col=col,
        )
        sts = (states_by_system or {}).get(name)
        if sts:
            fig.add_trace(
                go.Scatter(
                    x=[s.min_pc1 for s in sts],
                    y=[s.min_pc2 for s in sts],
                    mode="markers+text",
                    marker=dict(
                        size=14, symbol="star", color="white", line=dict(color="black", width=1.2)
                    ),
                    text=[f"{s.label} ({s.population * 100:.1f}%)" for s in sts],
                    textposition="top center",
                    textfont=dict(color="white", size=11),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                    name=f"{name} minima",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text=cv1, row=1, col=col)
    fig.update_yaxes(title_text=cv2, row=1, col=1)
    fig.update_layout(
        coloraxis=dict(colorscale=scale, cmin=0.0, cmax=zmax, colorbar=dict(title="ΔG (kJ/mol)")),
        height=520,
        plot_bgcolor="#f2f2f2",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ----------------------------------------------------------------------------
# parsing helpers (used by the validation suite and tests)
# ----------------------------------------------------------------------------

_NEWPLOT_RE = re.compile(r"Plotly\.newPlot\(\s*\"([^\"]+)\"\s*,")


def extract_plotly_data(html: str) -> dict[str, list[dict]]:
    """Parse every ``Plotly.newPlot(id, data, ...)`` call in ``html`` → ``{div_id: data}``.

    The data array is the JSON that Plotly embeds; this reads it back without a
    browser so the numbers in the report can be checked against the arrays on disk.
    """
    dec = json.JSONDecoder()
    out: dict[str, list[dict]] = {}
    for m in _NEWPLOT_RE.finditer(html):
        start = html.index("[", m.end())
        data, _ = dec.raw_decode(html, start)
        out[m.group(1)] = data
    return out


def decode_plotly_array(obj: Any) -> np.ndarray:
    """Decode a Plotly JSON array (plain list or base64 typed-array dict) to numpy."""
    if isinstance(obj, dict) and "bdata" in obj:
        import base64

        arr = np.frombuffer(base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"]))
        shape = obj.get("shape")
        if shape:
            arr = arr.reshape([int(v) for v in str(shape).split(",")])
        return arr
    return np.asarray(obj, dtype=float)
