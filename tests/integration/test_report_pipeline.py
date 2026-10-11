"""End-to-end: interactive report, structure viewer, figure formats and PDF through the CLI."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from conformatlas.cli import main
from conformatlas.plotting import reset_figure_style
from conformatlas.report import external_resource_urls, set_report_options, weasyprint_available

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "demo_dataset"
HAS_MDA = importlib.util.find_spec("MDAnalysis") is not None
HAS_PLOTLY = importlib.util.find_spec("plotly") is not None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis is required"),
]


@pytest.fixture(autouse=True)
def _reset():
    reset_figure_style()
    set_report_options(interactive=True, plotly_js="inline", path_overlay=False)
    yield
    reset_figure_style()
    set_report_options(interactive=True, plotly_js="inline", path_overlay=False)


def _analyze(out, *extra):
    args = [
        "analyze",
        "-s",
        str(DEMO / "WT" / "topology.pdb"),
        "-f",
        str(DEMO / "WT" / "rep1.xtc"),
        "-f",
        str(DEMO / "WT" / "rep2.xtc"),
        "-T",
        "300",
        "--bootstrap",
        "20",
        "-o",
        str(out),
        *extra,
    ]
    return CliRunner().invoke(main, args)


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly is required for the interactive report")
def test_full_report_with_interactive_viewer_and_formats(tmp_path):
    out = tmp_path / "wt"
    extra = ["--fig-format", "png,svg", "--path-overlay", "--colormap", "cividis", "--quiet"]
    if weasyprint_available():
        extra.append("--pdf")
    res = _analyze(out, *extra)
    assert res.exit_code == 0, res.output
    assert "Step" not in res.output  # quiet

    html = (out / "report.html").read_text(encoding="utf-8")
    # interactive landscape + profiles, data equal to the arrays on disk
    from conformatlas.plotting_interactive import decode_plotly_array, extract_plotly_data

    data = extract_plotly_data(html)
    assert {"interactive-landscape", "interactive-profiles"} <= set(data)
    z = decode_plotly_array(data["interactive-landscape"][0]["z"])
    fe = np.load(out / "fel" / "free_energy.npy")
    assert z.shape == fe.shape
    assert np.array_equal(np.nan_to_num(z, nan=-1), np.nan_to_num(fe, nan=-1))
    assert '"sliders"' in html
    # self-contained
    assert external_resource_urls(html) == []
    assert "<script src" not in html and "NGL.Stage" in html and "Plotly.newPlot" in html
    # structure viewer with one entry per state + average
    n_states = len(json.loads((out / "summary.json").read_text(encoding="utf-8"))["states"])
    assert html.count('"id": "struct-') == n_states + 1
    # interpretive text and limits paragraph
    interp = json.loads((out / "report_interpretation.json").read_text(encoding="utf-8"))
    for s in interp["populations"] + interp["n_eff"] + interp["convergence"]:
        assert s in html or s.replace("'", "&#39;") in html
    assert "does not show" in html and 'class="not-shown"' in html
    assert "warning(s) were raised" in html
    # figure formats and path overlay
    figs = out / "figures"
    assert (figs / "fel_2d.svg").exists() and (figs / "fel_2d_path.png").exists()
    assert not list(figs.glob("*.pdf"))
    assert (figs / "fel_2d.svg").read_text(encoding="utf-8").count("<path") > 50
    assert "Figure 1b" in html
    if weasyprint_available():
        assert (out / "report.pdf").read_bytes()[:5] == b"%PDF-"


def test_report_without_interactive(tmp_path):
    out = tmp_path / "wt"
    res = _analyze(out, "--no-interactive", "--quiet")
    assert res.exit_code == 0, res.output
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" not in html and "--no-interactive" in html
    assert "NGL.Stage" in html  # the viewer does not depend on plotly
    assert external_resource_urls(html) == []
    assert re.search(r"The population of State \w+ is [0-9.]+ %", html)


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly is required")
def test_compare_writes_interactive_comparison(tmp_path):
    out = tmp_path / "cmp"
    res = CliRunner().invoke(
        main, ["compare", "-c", str(DEMO / "comparison.yaml"), "-o", str(out), "--quiet"]
    )
    assert res.exit_code == 0, res.output
    html = (out / "comparison_interactive.html").read_text(encoding="utf-8")
    from conformatlas.plotting_interactive import extract_plotly_data

    (traces,) = extract_plotly_data(html).values()
    heat = [t for t in traces if t["type"] == "heatmap"]
    assert len(heat) == 2 and external_resource_urls(html) == []
