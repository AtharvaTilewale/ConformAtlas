"""Block-bootstrap uncertainty for the free energy landscape and derived quantities.

Frames are resampled with a circular block bootstrap *within each replicate* (block
length of a few correlation times, taken from the statistical inefficiency of the
leading projection), so temporal correlation and replicate structure are both
preserved. The basin definitions (watershed labels) are held fixed; what varies is
the histogram and therefore DeltaG(x, y), the state populations, the state free
energies relative to the most populated state, the basin minima and the saddle
heights between adjacent states.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from conformatlas.basins import frame_bin_indices, pairwise_saddles
from conformatlas.models import BasinState, FELGrid
from conformatlas.utils import R_GAS_CONSTANT


def circular_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of one circular block-bootstrap resample of length n."""
    block_len = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_len)[None, :]) % n
    return idx.ravel()[:n]


def _free_energy_from_counts(
    hist_t: np.ndarray, temperature: float, smooth_sigma: float
) -> np.ndarray:
    total = hist_t.sum()
    if total <= 0:
        return np.full_like(hist_t, np.nan, dtype=np.float64)
    prob = hist_t / total
    occ = hist_t > 0
    if smooth_sigma > 0:
        prob = gaussian_filter(prob, sigma=smooth_sigma, mode="nearest")
        prob = prob / prob.sum()
    p_max = prob[occ].max() if occ.any() else prob.max()
    fe = np.full_like(prob, np.nan, dtype=np.float64)
    valid = prob > 0
    if smooth_sigma > 0:
        valid &= prob >= 1e-6 * p_max
    fe[valid] = -R_GAS_CONSTANT * temperature * np.log(prob[valid] / p_max)
    return fe


@dataclass
class BootstrapResult:
    n_boot: int
    block_lengths: dict[str, int]
    fe_std: np.ndarray  # per-bin SD of DeltaG (NaN where rarely sampled)
    fe_ci_low: np.ndarray
    fe_ci_high: np.ndarray
    fe_support_fraction: np.ndarray  # fraction of resamples in which the bin was sampled
    state_table: pd.DataFrame  # per state: population, dG_state, min dG with CIs
    barrier_table: pd.DataFrame  # per adjacent state pair: saddle and barriers with CIs
    notes: list[str] = field(default_factory=list)


def bootstrap_landscape(
    cv1: np.ndarray,
    cv2: np.ndarray,
    replicate_ids: np.ndarray,
    fel_grid: FELGrid,
    basin_labels: np.ndarray,
    states: list[BasinState],
    block_lengths: dict[str, int],
    n_boot: int = 200,
    weights: np.ndarray | None = None,
    seed: int = 0,
) -> BootstrapResult:
    """Block-bootstrap DeltaG(x, y) and the state statistics derived from it."""
    rng = np.random.default_rng(seed)
    cv1 = np.asarray(cv1, dtype=np.float64)
    cv2 = np.asarray(cv2, dtype=np.float64)
    replicate_ids = np.asarray(replicate_ids)
    n = len(cv1)
    w_all = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    T = fel_grid.temperature
    sigma = fel_grid.smooth_sigma if fel_grid.smoothed else 0.0

    x_idx, y_idx, valid = frame_bin_indices(fel_grid, cv1, cv2)
    ny, nx = fel_grid.free_energy.shape
    frame_basin = np.full(n, 0, dtype=np.int64)
    frame_basin[valid] = basin_labels[y_idx[valid], x_idx[valid]]

    major = [s for s in states if s.basin_id > 0]
    basin_ids = np.array([s.basin_id for s in major], dtype=np.int64)
    masks = {s.basin_id: (basin_labels == s.basin_id) for s in major}
    pairs = [
        (a, b)
        for (a, b) in pairwise_saddles(basin_labels, fel_grid.free_energy)
        if a in masks and b in masks
    ]

    rep_index = {rep: np.where(replicate_ids == rep)[0] for rep in dict.fromkeys(replicate_ids)}

    fe_samples = np.full((n_boot, ny, nx), np.nan)
    pop_samples = np.full((n_boot, len(major)), np.nan)
    min_samples = np.full((n_boot, len(major)), np.nan)
    saddle_samples = np.full((n_boot, len(pairs)), np.nan)

    hist_range = [
        [fel_grid.x_edges[0], fel_grid.x_edges[-1]],
        [fel_grid.y_edges[0], fel_grid.y_edges[-1]],
    ]
    for b in range(n_boot):
        picks = []
        for rep, idx in rep_index.items():
            L = block_lengths.get(rep, 1)
            picks.append(idx[circular_block_indices(len(idx), L, rng)])
        sel = np.concatenate(picks)
        hist, _, _ = np.histogram2d(
            cv1[sel],
            cv2[sel],
            bins=[fel_grid.x_edges, fel_grid.y_edges],
            range=hist_range,
            weights=w_all[sel],
        )
        fe = _free_energy_from_counts(hist.T, T, sigma)
        fe_samples[b] = fe

        w_sel = w_all[sel]
        fb = frame_basin[sel]
        tot = w_sel.sum()
        for k, bid in enumerate(basin_ids):
            pop_samples[b, k] = w_sel[fb == bid].sum() / tot if tot > 0 else np.nan
            m = masks[bid]
            vals = fe[m]
            min_samples[b, k] = np.nanmin(vals) if np.any(np.isfinite(vals)) else np.nan
        if pairs:
            sad = pairwise_saddles(basin_labels, fe)
            for k, pr in enumerate(pairs):
                saddle_samples[b, k] = sad.get(pr, np.nan)

    def _ci(a, axis=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (
                np.nanpercentile(a, 2.5, axis=axis),
                np.nanpercentile(a, 97.5, axis=axis),
            )

    support = np.mean(np.isfinite(fe_samples), axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fe_std = np.nanstd(fe_samples, axis=0)
        lo, hi = _ci(fe_samples)
    rarely = support < 0.5
    fe_std[rarely] = np.nan
    lo[rarely] = np.nan
    hi[rarely] = np.nan

    # State table
    pop_point = np.array([s.population for s in major])
    ref_k = int(np.argmax(pop_point)) if len(major) else 0
    rows = []
    for k, s in enumerate(major):
        pops = pop_samples[:, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            dg_samples = -R_GAS_CONSTANT * T * np.log(pops / pop_samples[:, ref_k])
        dg_point = (
            -R_GAS_CONSTANT * T * np.log(pop_point[k] / pop_point[ref_k])
            if pop_point[ref_k] > 0
            else np.nan
        )
        plo, phi = _ci(pops)
        dlo, dhi = _ci(dg_samples)
        mlo, mhi = _ci(min_samples[:, k])
        rows.append(
            {
                "state": s.label,
                "population": pop_point[k],
                "population_sd": float(np.nanstd(pops)),
                "population_ci95_low": float(plo),
                "population_ci95_high": float(phi),
                "dG_state_kJ_mol": dg_point,
                "dG_state_ci95_low": float(dlo),
                "dG_state_ci95_high": float(dhi),
                "reference_state": major[ref_k].label,
                "min_dG_kJ_mol": s.min_free_energy,
                "min_dG_ci95_low": float(mlo),
                "min_dG_ci95_high": float(mhi),
            }
        )
    state_table = pd.DataFrame(rows)

    # Barrier table
    by_id = {s.basin_id: (k, s) for k, s in enumerate(major)}
    brow = []
    for k, (a, b) in enumerate(pairs):
        ka, sa = by_id[a]
        kb, sb = by_id[b]
        sad = saddle_samples[:, k]
        bar_a = sad - min_samples[:, ka]
        bar_b = sad - min_samples[:, kb]
        point_sad = pairwise_saddles(basin_labels, fel_grid.free_energy).get((a, b), np.nan)
        slo, shi = _ci(sad)
        alo, ahi = _ci(bar_a)
        blo, bhi = _ci(bar_b)
        brow.append(
            {
                "state_i": sa.label,
                "state_j": sb.label,
                "saddle_kJ_mol": point_sad,
                "saddle_ci95_low": float(slo),
                "saddle_ci95_high": float(shi),
                "barrier_from_i_kJ_mol": point_sad - sa.min_free_energy,
                "barrier_from_i_ci95_low": float(alo),
                "barrier_from_i_ci95_high": float(ahi),
                "barrier_from_j_kJ_mol": point_sad - sb.min_free_energy,
                "barrier_from_j_ci95_low": float(blo),
                "barrier_from_j_ci95_high": float(bhi),
            }
        )
    barrier_table = pd.DataFrame(
        brow,
        columns=[
            "state_i",
            "state_j",
            "saddle_kJ_mol",
            "saddle_ci95_low",
            "saddle_ci95_high",
            "barrier_from_i_kJ_mol",
            "barrier_from_i_ci95_low",
            "barrier_from_i_ci95_high",
            "barrier_from_j_kJ_mol",
            "barrier_from_j_ci95_low",
            "barrier_from_j_ci95_high",
        ],
    )

    notes = [
        f"{n_boot} circular block-bootstrap resamples within each replicate; block lengths (frames): "
        + ", ".join(f"{r}={L}" for r, L in block_lengths.items())
        + ". Basin definitions held fixed; intervals are 2.5-97.5 percentiles."
    ]
    return BootstrapResult(
        n_boot=n_boot,
        block_lengths=dict(block_lengths),
        fe_std=fe_std,
        fe_ci_low=lo,
        fe_ci_high=hi,
        fe_support_fraction=support,
        state_table=state_table,
        barrier_table=barrier_table,
        notes=notes,
    )
