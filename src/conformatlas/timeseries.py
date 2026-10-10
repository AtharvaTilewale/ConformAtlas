"""Time-series statistics: statistical inefficiency, effective sample size, equilibration.

The statistical inefficiency g = 1 + 2 tau (tau = integrated autocorrelation time in
frames) converts N correlated frames into N_eff = N / g independent samples. The
implementation follows the fast estimator of Chodera et al. (J. Chem. Theory
Comput. 2007, 3, 26) as used in pymbar: the normalised autocorrelation function is
accumulated with a geometrically growing lag until it first becomes non-positive.

Equilibration detection follows Chodera (J. Chem. Theory Comput. 2016, 12, 1799):
the production start t0 is the point that maximises the number of effectively
uncorrelated samples in the remainder of the series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def statistical_inefficiency(x: np.ndarray, mintime: int = 3, fast: bool = True) -> float:
    """Return g = 1 + 2 tau for a 1-D series (g >= 1).

    Parameters
    ----------
    x : array-like
        Observable time series (equally spaced frames).
    mintime : int
        Minimum lag before the autocorrelation may be declared zero.
    fast : bool
        Use geometrically increasing lags (sufficient for diagnostics).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 4:
        return 1.0
    dx = x - x.mean()
    sigma2 = np.dot(dx, dx) / n
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return 1.0

    g = 1.0
    t = 1
    increment = 1
    while t < n - 1:
        c = np.dot(dx[: n - t], dx[t:]) / ((n - t) * sigma2)
        if c <= 0.0 and t > mintime:
            break
        g += 2.0 * c * (1.0 - t / n) * increment
        t += increment
        if fast:
            increment += 1
    return float(max(g, 1.0))


def effective_sample_size(x: np.ndarray) -> float:
    """N / g for a 1-D series."""
    x = np.asarray(x).ravel()
    return len(x) / statistical_inefficiency(x)


@dataclass
class EquilibrationResult:
    t0_index: int  # first production frame (0-based index into the series)
    g: float  # statistical inefficiency of the production segment
    n_eff: float  # effective number of samples in the production segment
    n_total: int
    n_eff_full: float  # effective samples without discarding anything


def detect_equilibration(
    x: np.ndarray, nskip: int = 1, max_discard: float = 0.9
) -> EquilibrationResult:
    """Choose the production start that maximises the effective sample count.

    Candidate start points are scanned every ``nskip`` frames over the first
    ``max_discard`` fraction of the series; for each, g of the remainder is
    evaluated and N_eff = (N - t0) / g recorded. The maximum wins.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 10:
        g0 = statistical_inefficiency(x)
        return EquilibrationResult(0, g0, n / g0, n, n / g0)

    g_full = statistical_inefficiency(x)
    best = EquilibrationResult(0, g_full, n / g_full, n, n / g_full)
    last = int(n * max_discard)
    for t0 in range(nskip, last, nskip):
        seg = x[t0:]
        if np.allclose(seg, seg[0]):
            continue
        g = statistical_inefficiency(seg)
        n_eff = (n - t0) / g
        if n_eff > best.n_eff:
            best = EquilibrationResult(t0, g, n_eff, n, n / g_full)
    return best


def block_length_from_g(g: float, n: int, min_blocks: int = 5) -> int:
    """Block length (frames) for block bootstrap: a few correlation times, but leaving
    at least ``min_blocks`` blocks."""
    target = int(np.ceil(2.0 * g))
    return int(max(1, min(target, max(1, n // min_blocks))))
