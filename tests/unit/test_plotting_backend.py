"""Importing the plotting modules selects Agg for the CLI but keeps a notebook backend."""

import os
import subprocess
import sys

import pytest

MODULES = "conformatlas.plotting, conformatlas.plotting_extra, conformatlas.plotting_msm, conformatlas.plotting_residue"


def _backend_after_import(prelude: str, env=None) -> str:
    code = f"{prelude}; import matplotlib, {MODULES}; print(matplotlib.get_backend())"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-500:]
    return r.stdout.strip().splitlines()[-1]


def test_default_and_gui_backend_become_agg():
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    assert _backend_after_import("pass", env).lower() == "agg"
    assert _backend_after_import("pass", {**env, "MPLBACKEND": "QtAgg"}).lower() == "agg"


def test_notebook_backend_is_kept():
    pytest.importorskip("matplotlib_inline")
    backend = "module://matplotlib_inline.backend_inline"
    got = _backend_after_import(f"import matplotlib; matplotlib.use({backend!r})")
    assert got == backend
