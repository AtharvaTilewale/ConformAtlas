"""Software-environment provenance for ``run_metadata.json``.

:func:`collect_environment` returns a JSON-serialisable dictionary with the
versions of the interpreter, the scientific Python stack ConformAtlas depends
on, GROMACS (when an executable is available) and the platform. The pipeline
stores it under the ``environment`` key of ``run_metadata.json`` so that every
result directory records the software that produced it.

Versions are read from the installed distribution metadata
(:mod:`importlib.metadata`), not from ``__version__`` attributes, so no package is
imported as a side effect of collecting provenance. Optional dependencies that are
not installed are reported as ``None`` rather than omitted, so the set of keys is
stable across machines.
"""

from __future__ import annotations

import datetime
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

#: Distributions whose versions are recorded, in the order they appear in the
#: output. Required dependencies are listed in ``pyproject.toml``; the
#: optional ones are those a result can depend on (trajectory backend, PDF export).
TRACKED_DISTRIBUTIONS: tuple[str, ...] = (
    "conformatlas",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-image",
    "click",
    "jinja2",
    "pyyaml",
    "MDAnalysis",
    "weasyprint",
)


def package_version(distribution: str) -> str | None:
    """Installed version of ``distribution`` or ``None`` when it is not installed.

    Parameters
    ----------
    distribution : str
        Distribution name as used by pip/PyPI (``"scikit-image"``, not ``"skimage"``).
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def python_info() -> dict[str, str]:
    """Interpreter version, implementation and executable path."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def platform_info() -> dict[str, str]:
    """Operating system, kernel release and machine architecture."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def gromacs_info(gmx_command: str | Path | None = None) -> dict[str, str | None]:
    """GROMACS executable path and version string, or ``None`` values if absent.

    Uses :class:`conformatlas.gromacs.GromacsRunner` for discovery (custom path,
    then ``gmx``, then ``gmx_mpi`` on ``PATH``). Never raises: a missing or
    non-functional executable yields ``{"path": None, "version": None}``.
    """
    from conformatlas.gromacs import GromacsRunner

    try:
        runner = GromacsRunner(gmx_command)
    except Exception:
        return {"path": None, "version": None}
    return {"path": str(runner.gmx_path), "version": runner.version}


def collect_environment(
    gmx_command: str | Path | None = None,
    *,
    include_gromacs: bool = True,
    gromacs_version: str | None = None,
    extra_distributions: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Collect software versions and platform details for provenance records.

    Parameters
    ----------
    gmx_command : str or Path, optional
        Explicit GROMACS executable; ``None`` searches ``PATH``.
    include_gromacs : bool
        Set to ``False`` to skip the ``gmx --version`` subprocess (for example in
        unit tests or when GROMACS is known to be absent).
    gromacs_version : str, optional
        Version string already obtained by the caller (the pipeline detects GROMACS
        once and passes it here). When given, no subprocess is started and the
        executable path is resolved without running it.
    extra_distributions : sequence of str
        Additional distribution names to record under ``packages``.

    Returns
    -------
    dict
        ``{"collected_at", "python", "platform", "packages", "gromacs"}``. Every
        value is a plain ``str``, ``None`` or a nested ``dict`` of those, so the
        result serialises with the standard :mod:`json` module.
    """
    packages = {name: package_version(name) for name in TRACKED_DISTRIBUTIONS}
    for name in extra_distributions:
        packages[name] = package_version(name)

    if gromacs_version is not None:
        import shutil

        path = (
            shutil.which(str(gmx_command))
            if gmx_command
            else (shutil.which("gmx") or shutil.which("gmx_mpi"))
        )
        gromacs: dict[str, str | None] = {"path": path, "version": gromacs_version}
    elif include_gromacs:
        gromacs = gromacs_info(gmx_command)
    else:
        gromacs = {"path": None, "version": None}

    return {
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": python_info(),
        "platform": platform_info(),
        "packages": packages,
        "gromacs": gromacs,
    }


def format_environment(env: dict[str, Any]) -> str:
    """One-line-per-item text rendering of :func:`collect_environment` output."""
    lines = [
        f"Python {env['python']['version']} ({env['python']['implementation']})",
        f"Platform {env['platform']['platform']}",
    ]
    for name, version in env["packages"].items():
        lines.append(f"{name} {version if version is not None else 'not installed'}")
    gmx = env["gromacs"]
    lines.append(
        f"GROMACS {gmx['version']} ({gmx['path']})" if gmx["version"] else "GROMACS not found"
    )
    return "\n".join(lines)


__all__ = [
    "TRACKED_DISTRIBUTIONS",
    "collect_environment",
    "format_environment",
    "gromacs_info",
    "package_version",
    "platform_info",
    "python_info",
]
