"""CLI polish: banner/TTY behaviour, --quiet, friendly errors and exit codes."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import conformatlas.cli as cli
from conformatlas.cli import main
from conformatlas.plotting import reset_figure_style
from conformatlas.utils import BANNER_RAW

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "demo_dataset" / "WT"
BANNER_LINE = BANNER_RAW.splitlines()[0].strip()[:12]


@pytest.fixture(autouse=True)
def _reset():
    cli._QUIET = False
    reset_figure_style()
    from conformatlas.report import set_report_options

    set_report_options(interactive=True, plotly_js="inline", path_overlay=False)
    yield
    cli._QUIET = False
    reset_figure_style()
    set_report_options(interactive=True, plotly_js="inline", path_overlay=False)


def _base(out):
    return [
        "analyze",
        "-s",
        str(DEMO / "topology.pdb"),
        "-f",
        str(DEMO / "rep1.xtc"),
        "-o",
        str(out),
    ]


def _fake_run(monkeypatch, exc=None):
    """Replace the pipeline so CLI tests stay fast; records the config it received."""
    seen = {}

    def run_analysis(cfg, echo=None):
        seen["cfg"] = cfg
        if exc is not None:
            raise exc
        raise SystemExit(0)

    import conformatlas.pipeline

    monkeypatch.setattr(conformatlas.pipeline, "run_analysis", run_analysis)
    return seen


# --------------------------------------------------------------------------- banner


def test_no_banner_for_help_and_version():
    r = CliRunner()
    for args in (["--help"], ["--version"], ["analyze", "--help"], ["compare", "-h"]):
        res = r.invoke(main, args)
        assert res.exit_code == 0, res.output
        assert BANNER_LINE not in res.output
        assert "Usage:" in res.output or "version" in res.output


def test_no_banner_when_stdout_is_not_a_tty(monkeypatch, tmp_path):
    _fake_run(monkeypatch)
    res = CliRunner().invoke(main, _base(tmp_path / "o"))
    assert BANNER_LINE not in res.output
    assert "analysis pipeline" in res.output


def test_banner_only_on_tty(monkeypatch, capsys):
    printed = []
    monkeypatch.setattr(cli, "print_banner", lambda ctx=None: printed.append(1))
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False, raising=False)
    cli.maybe_banner()
    assert printed == []
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    cli.maybe_banner()
    assert printed == [1]
    cli._QUIET = True
    cli.maybe_banner()
    assert printed == [1]


def test_quiet_suppresses_progress(monkeypatch, tmp_path):
    _fake_run(monkeypatch)
    res = CliRunner().invoke(main, _base(tmp_path / "o") + ["--quiet"])
    assert res.exit_code == 0
    assert "analysis pipeline" not in res.output and "Topology" not in res.output
    assert cli.is_quiet()


# --------------------------------------------------------------------------- usage errors (exit 2)


def test_malformed_yaml_reports_line_and_column(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "topology: top.pdb\ntrajectories: [a.xtc, b.xtc\ntemperature: 300\n", encoding="utf-8"
    )
    res = CliRunner().invoke(main, ["analyze", "--config", str(bad)])
    assert res.exit_code == 2
    assert "Malformed YAML" in res.output and "line" in res.output and "column" in res.output
    assert "Traceback" not in res.output
    res = CliRunner().invoke(main, ["compare", "-c", str(bad)])
    assert res.exit_code == 2 and "Malformed YAML" in res.output


def test_missing_trajectory_file(tmp_path):
    res = CliRunner().invoke(
        main, ["analyze", "-s", str(DEMO / "topology.pdb"), "-f", str(tmp_path / "nope.xtc")]
    )
    assert res.exit_code == 2
    assert "does not exist" in res.output and "Traceback" not in res.output


def test_missing_file_in_yaml(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"topology: {DEMO / 'topology.pdb'}\ntrajectories: [missing.xtc]\n", encoding="utf-8"
    )
    res = CliRunner().invoke(main, ["analyze", "--config", str(cfg), "-o", str(tmp_path / "o")])
    assert res.exit_code == 2
    assert "File not found" in res.output and "Traceback" not in res.output


def test_unknown_group_lists_available(tmp_path):
    pytest.importorskip("MDAnalysis")
    ndx = tmp_path / "index.ndx"
    ndx.write_text("[ Peptide_CA ]\n2 7 12 17 22\n", encoding="utf-8")
    res = CliRunner().invoke(main, _base(tmp_path / "o") + ["-n", str(ndx), "-c", "NoSuchGroup"])
    assert res.exit_code == 2, res.output
    assert "Group 'NoSuchGroup' matched no atoms" in res.output
    assert "Peptide_CA" in res.output and "Backbone" in res.output
    assert "Traceback" not in res.output


def test_unknown_group_raised_inside_pipeline(monkeypatch, tmp_path):
    _fake_run(monkeypatch, ValueError("Selection 'Zzz' matched 0 atoms in top.pdb."))
    monkeypatch.setattr(cli, "check_groups", lambda cfg: None)
    res = CliRunner().invoke(main, _base(tmp_path / "o"))
    assert res.exit_code == 2 and "Group 'Zzz' matched no atoms" in res.output


def test_missing_weasyprint(monkeypatch, tmp_path):
    import conformatlas.report as report

    monkeypatch.setattr(report, "weasyprint_available", lambda: False)
    _fake_run(monkeypatch)
    res = CliRunner().invoke(main, _base(tmp_path / "o") + ["--pdf"])
    assert res.exit_code == 2
    assert "weasyprint" in res.output and "pip install weasyprint" in res.output
    assert "Traceback" not in res.output


def test_missing_plotly_explicit_and_implicit(monkeypatch, tmp_path):
    import conformatlas.plotting_interactive as pi
    from conformatlas.report import get_report_options

    monkeypatch.setattr(pi, "plotly_available", lambda: False)
    _fake_run(monkeypatch)
    res = CliRunner().invoke(main, _base(tmp_path / "o") + ["--interactive"])
    assert res.exit_code == 2 and "pip install plotly" in res.output
    res = CliRunner().invoke(main, _base(tmp_path / "o"))
    assert res.exit_code == 0, res.output
    assert "plotly is not installed" in res.output
    assert get_report_options().interactive is False


@pytest.mark.parametrize(
    "args,needle",
    [
        (["--colormap", "nope"], "Unknown colormap"),
        (["--fig-format", "tiff"], "Unknown figure format"),
        (["--font-scale", "9"], "font_scale"),
        (["--fig-dpi", "1"], "dpi"),
        (["--plotly-js", "local"], "Invalid value"),
    ],
)
def test_invalid_style_options(monkeypatch, tmp_path, args, needle):
    _fake_run(monkeypatch)
    res = CliRunner().invoke(main, _base(tmp_path / "o") + args)
    assert res.exit_code == 2, res.output
    assert needle in res.output and "Traceback" not in res.output


def test_style_options_are_applied_and_recorded_in_config(monkeypatch, tmp_path):
    from conformatlas.plotting import get_figure_style
    from conformatlas.report import get_report_options

    seen = _fake_run(monkeypatch)
    res = CliRunner().invoke(
        main,
        _base(tmp_path / "o")
        + [
            "--fig-format",
            "svg,png",
            "--colormap",
            "viridis",
            "--font-scale",
            "1.3",
            "--fig-dpi",
            "150",
            "--path-overlay",
            "--plotly-js",
            "cdn",
            "-T",
            "310",
        ],
    )
    assert res.exit_code == 0, res.output
    st = get_figure_style()
    assert st.formats == ("svg", "png") and st.colormap == "viridis"
    assert st.font_scale == pytest.approx(1.3) and st.dpi == 150 and st.path_overlay
    assert get_report_options().plotly_js == "cdn" and get_report_options().path_overlay
    assert seen["cfg"].temperature == 310.0
    # the options are also recorded in the AnalysisConfig (YAML keys), None = not given
    cfg = seen["cfg"]
    assert cfg.fig_format == "svg,png" and cfg.colormap == "viridis"
    assert cfg.font_scale == pytest.approx(1.3) and cfg.fig_dpi == 150 and cfg.path_overlay
    assert cfg.plotly_js == "cdn" and cfg.interactive is None
    assert not hasattr(cfg, "quiet")


# --------------------------------------------------------------------------- analysis errors (exit 1)


def test_analysis_error_exit_code_1(monkeypatch, tmp_path):
    _fake_run(monkeypatch, RuntimeError("matrix is singular"))
    res = CliRunner().invoke(main, _base(tmp_path / "o"))
    assert res.exit_code == 1
    assert "matrix is singular" in res.output and "Traceback" not in res.output
    assert "log file" in res.output


def test_memory_error_message(monkeypatch, tmp_path):
    _fake_run(monkeypatch, MemoryError())
    res = CliRunner().invoke(main, _base(tmp_path / "o"))
    assert res.exit_code == 1 and "--stride" in res.output


def test_doctor_lists_plotly():
    res = CliRunner().invoke(main, ["doctor"])
    assert res.exit_code == 0 and "plotly" in res.output


def test_quiet_flag_does_not_leak_between_invocations(tmp_path):
    """--quiet must apply to one invocation only (module state is reset each call)."""
    import conformatlas.cli as cli_mod

    cli_mod._set_quiet(True)
    assert cli_mod.is_quiet()
    cli_mod._set_quiet(None)
    assert not cli_mod.is_quiet()
