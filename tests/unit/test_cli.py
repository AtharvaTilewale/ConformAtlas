"""Unit tests for the Click CLI subcommands."""

from click.testing import CliRunner

import conformatlas
from conformatlas.cli import main


def test_cli_version():
    """Verify --version returns package version."""
    runner = CliRunner()
    res = runner.invoke(main, ["--version"])
    assert res.exit_code == 0
    assert conformatlas.__version__ in res.output


def test_cli_help():
    """Verify --help displays available subcommands."""
    runner = CliRunner()
    res = runner.invoke(main, ["--help"])
    assert res.exit_code == 0
    assert "analyze" in res.output
    assert "compare" in res.output
    assert "doctor" in res.output
    assert "init-config" in res.output
    assert "example" in res.output


def test_cli_doctor():
    """Verify doctor subcommand runs diagnostics."""
    runner = CliRunner()
    res = runner.invoke(main, ["doctor"])
    assert res.exit_code == 0
    assert "ConformAtlas System Diagnostics" in res.output
    assert "Python Version" in res.output
    assert "numpy" in res.output


def test_cli_init_config(tmp_path):
    """Verify init-config creates valid YAML templates."""
    runner = CliRunner()
    out_yaml = tmp_path / "test_config.yaml"

    res = runner.invoke(main, ["init-config", "--type", "compare", "-o", str(out_yaml)])
    assert res.exit_code == 0
    assert out_yaml.exists()
    content = out_yaml.read_text()
    assert "WT_vs_Mutant" in content
    assert "temperature:" in content


def test_cli_example_generation(tmp_path):
    """Verify example subcommand generates demo project."""
    runner = CliRunner()
    out_dir = tmp_path / "demo_project"

    res = runner.invoke(main, ["example", "-o", str(out_dir)])
    assert res.exit_code == 0
    assert out_dir.exists()
    assert (out_dir / "WT" / "topology.pdb").exists()
    assert (out_dir / "WT" / "rep1.xtc").exists()
    assert (out_dir / "comparison.yaml").exists()
