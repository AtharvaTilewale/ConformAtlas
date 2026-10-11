"""`conformatlas example` must fail loudly, writing nothing, when MDAnalysis is missing."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import conformatlas.examples_gen as examples_gen
from conformatlas.cli import main
from conformatlas.errors import MissingDependencyError


def test_generator_raises_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(examples_gen, "HAS_MDA", False)
    target = tmp_path / "ex"
    with pytest.raises(MissingDependencyError) as info:
        examples_gen.generate_mini_example_dataset(target)
    assert "pip install MDAnalysis" in str(info.value)
    assert "optional" not in str(info.value)  # MDAnalysis is a core dependency
    assert not target.exists()


def test_cli_exits_with_usage_code(tmp_path, monkeypatch):
    monkeypatch.setattr(examples_gen, "HAS_MDA", False)
    result = CliRunner().invoke(main, ["example", "-o", str(tmp_path / "ex")])
    assert result.exit_code == 2
    assert "MDAnalysis" in result.output and "pip install MDAnalysis" in result.output
    assert "Example dataset created" not in result.output
