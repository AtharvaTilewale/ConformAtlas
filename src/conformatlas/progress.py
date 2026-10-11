"""TTY-aware progress reporting.

A :class:`ProgressReporter` renders a progress bar with ``rich`` when that
package is importable and the stream is an interactive terminal, falls back to
a plain single-line counter on a TTY without ``rich``, and is completely silent
when the stream is not a TTY (pipes, log files, CI, notebooks executed with
nbconvert) or when ``quiet=True``. Library code can therefore call it
unconditionally.

Usage::

    with ProgressReporter(total=n_frames, description="Rep_1") as bar:
        for frame in frames:
            ...
            bar.update(1)

    for item in track(items, description="replicates"):
        ...
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any, TextIO

try:  # pragma: no cover - exercised only when rich is installed
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False

__all__ = ["HAS_RICH", "ProgressReporter", "is_interactive", "track"]


def is_interactive(stream: TextIO | None = None) -> bool:
    """True when ``stream`` (default ``sys.stderr``) is an interactive terminal.

    ``CONFORMATLAS_NO_PROGRESS=1`` in the environment forces False, which is
    convenient for test suites and batch schedulers.
    """
    if os.environ.get("CONFORMATLAS_NO_PROGRESS", "").strip() not in ("", "0"):
        return False
    stream = stream if stream is not None else sys.stderr
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty()) if callable(isatty) else False
    except Exception:  # pragma: no cover - exotic streams
        return False


class ProgressReporter:
    """Progress bar that degrades gracefully.

    Parameters
    ----------
    total : int, optional
        Number of work units; ``None`` renders an indeterminate counter.
    description : str
        Label shown before the bar.
    quiet : bool
        Suppress all output regardless of the stream.
    stream : text stream, optional
        Destination (default ``sys.stderr``). Output is produced only if the
        stream is a TTY.
    min_interval : float
        Minimum seconds between redraws of the plain fallback renderer.
    """

    def __init__(
        self,
        total: int | None = None,
        description: str = "",
        quiet: bool = False,
        stream: TextIO | None = None,
        min_interval: float = 0.1,
    ) -> None:
        self.total = int(total) if total is not None else None
        self.description = description
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = (not quiet) and is_interactive(self.stream)
        self.completed = 0
        self._min_interval = float(min_interval)
        self._last_draw = 0.0
        self._t0 = time.monotonic()
        self._rich: Any = None
        self._task: Any = None
        self.backend = "silent"
        if self.enabled:
            self.backend = "rich" if HAS_RICH else "plain"

    # ------------------------------------------------------------ lifecycle
    def __enter__(self) -> ProgressReporter:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if not self.enabled:
            return
        self._t0 = time.monotonic()
        if self.backend == "rich":
            columns = [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ]
            # rich writes to its own Console; bind it to the chosen stream.
            from rich.console import Console

            self._rich = Progress(*columns, console=Console(file=self.stream), transient=True)
            self._rich.start()
            self._task = self._rich.add_task(self.description, total=self.total)
        else:
            self._draw(force=True)

    def update(self, advance: int = 1) -> None:
        """Advance the bar by ``advance`` units."""
        self.completed += int(advance)
        if not self.enabled:
            return
        if self.backend == "rich":
            self._rich.update(self._task, advance=advance)
        else:
            self._draw()

    def set_description(self, description: str) -> None:
        self.description = description
        if not self.enabled:
            return
        if self.backend == "rich":
            self._rich.update(self._task, description=description)
        else:
            self._draw(force=True)

    def close(self) -> None:
        if not self.enabled:
            return
        if self.backend == "rich":
            if self._rich is not None:
                self._rich.stop()
                self._rich = None
        else:
            self._draw(force=True)
            self.stream.write("\n")
            self.stream.flush()
        self.enabled = False

    # ------------------------------------------------------------- plain UI
    def _draw(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_draw) < self._min_interval:
            return
        self._last_draw = now
        elapsed = now - self._t0
        if self.total:
            frac = min(1.0, self.completed / self.total)
            width = 30
            filled = int(round(frac * width))
            bar = "#" * filled + "-" * (width - filled)
            msg = f"\r{self.description} [{bar}] {self.completed}/{self.total} {elapsed:6.1f}s"
        else:
            msg = f"\r{self.description} {self.completed} {elapsed:6.1f}s"
        self.stream.write(msg)
        self.stream.flush()


def track(
    iterable: Iterable,
    description: str = "",
    total: int | None = None,
    quiet: bool = False,
    stream: TextIO | None = None,
) -> Iterator:
    """Iterate over ``iterable`` while reporting progress (see ProgressReporter)."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    with ProgressReporter(total=total, description=description, quiet=quiet, stream=stream) as bar:
        for item in iterable:
            yield item
            bar.update(1)
