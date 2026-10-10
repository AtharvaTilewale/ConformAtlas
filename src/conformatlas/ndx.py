"""GROMACS index (.ndx) file support."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("conformatlas.ndx")


def parse_ndx(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a GROMACS index file into {group_name: 0-based atom indices}.

    GROMACS numbers atoms from 1; the returned indices are 0-based so they can be
    used directly with MDAnalysis ``universe.atoms[indices]``. When a group name
    appears more than once (GROMACS allows this) the first occurrence is kept and
    a warning is logged.
    """
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    current: str | None = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1].strip()
                if name in groups:
                    logger.warning(
                        f"Index group '{name}' is defined more than once; keeping the first."
                    )
                    current = None
                    continue
                groups[name] = []
                order.append(name)
                current = name
                continue
            if current is None:
                continue
            groups[current].extend(int(tok) - 1 for tok in line.split())
    return {name: np.asarray(groups[name], dtype=np.int64) for name in order}


def lookup_group(groups: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    """Case-insensitive group lookup; returns None when absent."""
    if name in groups:
        return groups[name]
    lower = {k.lower(): k for k in groups}
    key = lower.get(name.lower().strip())
    return groups[key] if key is not None else None
