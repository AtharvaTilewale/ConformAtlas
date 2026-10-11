"""Exceptions with user-facing messages and helpers to format common failures.

The CLI maps these to short messages without a traceback (exit code 2 for
usage/configuration problems, 1 for analysis failures).
"""

from __future__ import annotations

import re
from pathlib import Path

#: GROMACS-style group names understood without an index file
#: (see :func:`conformatlas.trajectory.map_group_to_mda_selection`).
BUILTIN_GROUPS: tuple[str, ...] = (
    "Backbone",
    "MainChain",
    "C-alpha",
    "Protein",
    "SideChain",
    "System",
)

_ZERO_ATOMS_RE = re.compile(r"[Ss]election '([^']+)'(?: \(query: '[^']*'\))? matched 0 atoms")


class ConformAtlasError(Exception):
    """Base class for errors that carry a user-facing message."""

    exit_code = 1


class UsageProblem(ConformAtlasError):
    """Configuration / command-line problems (exit code 2)."""

    exit_code = 2


class MissingDependencyError(UsageProblem):
    """A dependency needed for the requested feature is not importable.

    ``optional=False`` is for core dependencies (e.g. MDAnalysis) that are
    missing from a broken environment.
    """

    def __init__(self, package: str, pip_command: str, purpose: str, *, optional: bool = True):
        self.package = package
        self.pip_command = pip_command
        self.purpose = purpose
        kind = "optional package" if optional else "package"
        super().__init__(
            f"The {kind} '{package}' is required for {purpose} but is not installed.\n"
            f"  Install it with:  {pip_command}"
        )


class InputDataError(UsageProblem, ValueError):
    """The input data cannot support the requested analysis (exit code 2).

    Also a :class:`ValueError` so that library callers that catch ``ValueError``
    for bad inputs keep working.
    """


class MalformedConfigError(UsageProblem):
    """YAML syntax error with line/column information."""

    def __init__(self, path: str | Path, line: int | None, column: int | None, problem: str):
        self.path = Path(path)
        self.line = line
        self.column = column
        self.problem = problem
        where = ""
        if line is not None:
            where = f" at line {line}" + (f", column {column}" if column is not None else "")
        super().__init__(f"Malformed YAML in {self.path}{where}: {problem}")


class UnknownGroupError(UsageProblem):
    """A fit/analysis/ligand group name matched no atoms."""

    def __init__(self, name: str, available: list[str], source: str):
        self.name = name
        self.available = list(available)
        self.source = source
        listing = ", ".join(self.available) if self.available else "(none found)"
        super().__init__(
            f"Group '{name}' matched no atoms.\n  Available groups ({source}): {listing}"
        )


def malformed_config_from_yaml_error(path: str | Path, err: Exception) -> MalformedConfigError:
    """Convert a ``yaml.YAMLError`` into :class:`MalformedConfigError` (1-based line/column)."""
    mark = getattr(err, "problem_mark", None) or getattr(err, "context_mark", None)
    line = int(mark.line) + 1 if mark is not None else None
    col = int(mark.column) + 1 if mark is not None else None
    problem = getattr(err, "problem", None) or str(err).splitlines()[0]
    ctx = getattr(err, "context", None)
    if ctx:
        problem = f"{problem} ({ctx})"
    return MalformedConfigError(path, line, col, str(problem))


def selection_error_group(err: Exception) -> str | None:
    """Return the group name from a 'matched 0 atoms' ValueError, else ``None``."""
    m = _ZERO_ATOMS_RE.search(str(err))
    return m.group(1) if m else None


def available_groups(
    topology: str | Path | None, index_file: str | Path | None
) -> tuple[list[str], str]:
    """List group names a user may pass to ``-l/-c/-g``.

    Index-file groups come first (in file order), followed by the built-in
    GROMACS names and, when MDAnalysis can read the topology, the residue names
    of non-protein molecules (ligands) found there.
    """
    names: list[str] = []
    sources: list[str] = []
    if index_file and Path(index_file).exists():
        try:
            from conformatlas.ndx import parse_ndx

            names.extend(parse_ndx(index_file).keys())
            sources.append(f"index file {Path(index_file).name}")
        except Exception:
            pass
    names.extend(BUILTIN_GROUPS)
    sources.append("built-in")
    if topology and Path(topology).exists():
        try:
            import MDAnalysis as mda

            u = mda.Universe(str(topology))
            het = u.select_atoms("not protein and not resname SOL WAT HOH TIP3 NA CL K")
            lig = list(dict.fromkeys(het.residues.resnames.tolist()))[:20]
            if lig:
                names.extend(lig)
                sources.append("residue names in the topology")
        except Exception:
            pass
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen, "; ".join(sources)
