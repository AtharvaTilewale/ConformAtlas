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


def statistical_inefficiency(x: np.ndarray, mintime: int = 3, fast: bool = False) -> float:
    """Return g = 1 + 2 tau for a 1-D series (g >= 1).

    g = 1 + 2 * sum_{t=1}^{T} (1 - t/N) C(t), with C(t) the normalised
    autocorrelation function, summed until C first becomes non-positive
    (Chodera et al. 2007). The full autocorrelation is evaluated with an FFT, so
    the estimate is exact at O(N log N) cost; ``fast`` is accepted for backwards
    compatibility and ignored (the geometric-lag approximation of earlier
    versions overestimated g by ~10-15 % for exponentially decaying correlations).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 4:
        return 1.0
    dx = x - x.mean()
    sigma2 = np.dot(dx, dx) / n
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return 1.0
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(dx, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[:n]  # sum_i dx_i dx_{i+t}
    counts = n - np.arange(n, dtype=np.float64)
    c = acov / (counts * sigma2)  # C(t), unbiased normalisation, C(0) = 1
    # truncate at the first non-positive C(t) with t > mintime
    nonpos = np.where((c[1:] <= 0.0) & (np.arange(1, n) > mintime))[0]
    t_max = int(nonpos[0]) + 1 if nonpos.size else n - 1
    t = np.arange(1, t_max, dtype=np.float64)
    g = 1.0 + 2.0 * np.sum(c[1:t_max] * (1.0 - t / n))
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


@dataclass
class MultiEquilibrationResult:
    """Equilibration decided jointly from several observables."""

    t0_index: int  # production start applied to all observables
    method: str  # 'max_neff' | 'joint'
    deciding_observable: str  # observable whose t0 was adopted ('joint' for the joint criterion)
    per_observable: dict[str, EquilibrationResult]
    n_total: int


def detect_equilibration_multi(
    series: dict[str, np.ndarray],
    method: str = "max_neff",
    nskip: int = 1,
    max_discard: float = 0.9,
) -> MultiEquilibrationResult:
    """Equilibration start from several observables (e.g. the two landscape CVs).

    ``method='max_neff'`` applies the Chodera (2016) criterion of
    :func:`detect_equilibration` to every observable independently and adopts the
    *latest* of the per-observable production starts, so no observable is still
    relaxing at ``t0`` (conservative). ``method='joint'`` scans the same candidate
    starts and chooses the one maximising ``min_k N_eff,k(t0)``, the effective
    sample count of the worst-equilibrated observable - a single criterion
    rather than a maximum over separate optima. All series must have the same
    length and be sampled on the same frames.
    """
    if method not in ("max_neff", "joint"):
        raise ValueError("method must be 'max_neff' or 'joint'")
    if not series:
        raise ValueError("series must contain at least one observable")
    arrays = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in series.items()}
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"all observables must have the same length (got {sorted(lengths)})")
    n = lengths.pop()
    per = {
        k: detect_equilibration(v, nskip=nskip, max_discard=max_discard) for k, v in arrays.items()
    }
    if method == "max_neff":
        deciding = max(per, key=lambda k: (per[k].t0_index, k))
        return MultiEquilibrationResult(
            t0_index=per[deciding].t0_index,
            method=method,
            deciding_observable=deciding,
            per_observable=per,
            n_total=n,
        )
    # joint: maximise the minimum effective sample size over observables
    if n < 10:
        return MultiEquilibrationResult(0, method, "joint", per, n)
    candidates = [0, *range(nskip, int(n * max_discard), nskip)]
    best_t0, best_val = 0, -np.inf
    for t0 in candidates:
        vals = []
        for v in arrays.values():
            seg = v[t0:]
            if np.allclose(seg, seg[0]):
                vals.append(0.0)
                continue
            vals.append((n - t0) / statistical_inefficiency(seg))
        val = min(vals)
        if val > best_val:
            best_t0, best_val = t0, val
    return MultiEquilibrationResult(best_t0, method, "joint", per, n)
