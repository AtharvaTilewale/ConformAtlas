"""Tests for conformatlas.provenance (software-environment record)."""

import json
import shutil
import sys
from importlib import metadata

import numpy
import pytest

import conformatlas
from conformatlas.provenance import (
    TRACKED_DISTRIBUTIONS,
    collect_environment,
    format_environment,
    gromacs_info,
    package_version,
)


def test_package_version_matches_importlib_metadata_and_module_attribute():
    assert package_version("numpy") == metadata.version("numpy") == numpy.__version__
    assert package_version("conformatlas") == conformatlas.__version__


def test_package_version_of_unknown_distribution_is_none():
    assert package_version("this-distribution-does-not-exist-0xdeadbeef") is None


def test_collect_environment_structure_and_json_serialisable():
    env = collect_environment(include_gromacs=False)
    assert set(env) == {"collected_at", "python", "platform", "packages", "gromacs"}
    assert env["python"]["version"] == sys.version.split()[0]
    assert env["python"]["executable"] == sys.executable
    assert set(TRACKED_DISTRIBUTIONS) <= set(env["packages"])
    for required in ("numpy", "scipy", "pandas", "matplotlib", "scikit-image", "click"):
        assert env["packages"][required] is not None
    # optional dependencies are present as keys even when absent
    assert "MDAnalysis" in env["packages"] and "weasyprint" in env["packages"]
    assert env["gromacs"] == {"path": None, "version": None}
    # must round-trip through the standard json module (no numpy / Path objects)
    assert json.loads(json.dumps(env)) == env


def test_collect_environment_extra_distributions():
    env = collect_environment(include_gromacs=False, extra_distributions=("pytest",))
    assert env["packages"]["pytest"] == metadata.version("pytest")


def test_collect_environment_accepts_precomputed_gromacs_version(monkeypatch):
    called = []

    def _boom(*a, **k):  # pragma: no cover - must not be called
        called.append(1)
        raise AssertionError("GromacsRunner must not be constructed")

    monkeypatch.setattr("conformatlas.gromacs.GromacsRunner", _boom)
    env = collect_environment(gromacs_version="2026.3-test")
    assert env["gromacs"]["version"] == "2026.3-test"
    assert not called


def test_gromacs_info_missing_executable_is_none_values():
    info = gromacs_info("/nonexistent/path/to/gmx_binary")
    assert info == {"path": None, "version": None}


@pytest.mark.skipif(shutil.which("gmx") is None, reason="gmx not on PATH")
def test_gromacs_info_detects_installed_version():
    info = gromacs_info()
    assert info["path"] and info["version"]
    assert info["version"][0].isdigit()  # e.g. '2026.3-conda_forge'


def test_format_environment_lists_every_package():
    env = collect_environment(include_gromacs=False)
    text = format_environment(env)
    for name in TRACKED_DISTRIBUTIONS:
        assert name in text
    assert "GROMACS not found" in text
