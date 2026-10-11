"""Block-bootstrap uncertainty for the free energy landscape and derived quantities.

Frames are resampled with a circular block bootstrap *within each replicate* (block
length of a few correlation times, taken from the statistical inefficiency of the
leading projection), so temporal correlation and replicate structure are both
preserved. The basin definitions (watershed labels) are held fixed; what varies is
the histogram and therefore DeltaG(x, y), the state populations, the state free
energies relative to the most populated state, the basin minima and the saddle
heights between adjacent states.

A second mode (``mode='resegment'``) additionally re-runs the watershed
segmentation (:func:`conformatlas.basins.detect_energy_basins`) on every
resampled landscape, records how many major states are found, matches the
resampled states to the reference states by maximum frame overlap (Hungarian
assignment; a match is accepted when at least half of the resampled state's
frames come from the reference basin, or when the resampled minimum lies inside
the reference basin) and reports population intervals that include the
segmentation uncertainty together with the state-count distribution.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

from conformatlas.basins import detect_energy_basins, frame_bin_indices, pairwise_saddles
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
    mode: str = "fixed"  # 'fixed' (basin labels held) or 'resegment' (watershed re-run)
    n_states_per_resample: np.ndarray | None = None  # resegment: major states found per resample
    state_count_distribution: pd.DataFrame | None = None  # resegment: n_states, count, fraction
    resegment_state_table: pd.DataFrame | None = None  # resegment: per reference state


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
    mode: str = "fixed",
    resegment_kwargs: dict | None = None,
) -> BootstrapResult:
    """Block-bootstrap DeltaG(x, y) and the state statistics derived from it.

    ``mode='fixed'`` (default) keeps the watershed labels of the reference
    landscape; ``mode='resegment'`` re-runs :func:`detect_energy_basins` on every
    resampled landscape with ``resegment_kwargs`` (``min_state_population``,
    ``max_states``, ``min_barrier``, ``min_peak_distance``; defaults as in the
    reference analysis must be passed explicitly to reproduce it) and fills
    ``n_states_per_resample``, ``state_count_distribution`` and
    ``resegment_state_table``. The fixed-label tables are produced in both modes.
    """
    if mode not in ("fixed", "resegment"):
        raise ValueError("mode must be 'fixed' or 'resegment'")
    reseg_kw = dict(resegment_kwargs or {})
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
    n_states_b = np.full(n_boot, -1, dtype=np.int64)
    reseg_pop = np.full((n_boot, len(major)), np.nan)
    reseg_matched = np.zeros((n_boot, len(major)), dtype=bool)
    ref_min_bins = {}
    for s_ in major:
        xi, yi, ok = frame_bin_indices(fel_grid, np.array([s_.min_pc1]), np.array([s_.min_pc2]))
        ref_min_bins[s_.basin_id] = (int(yi[0]), int(xi[0])) if ok[0] else None

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
        if mode == "resegment":
            n_states_b[b], reseg_pop[b], reseg_matched[b] = _resegment_resample(
                sel,
                hist.T,
                fe,
                cv1,
                cv2,
                replicate_ids,
                w_all,
                fel_grid,
                basin_labels,
                basin_ids,
                frame_basin,
                reseg_kw,
            )

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
    count_df = reseg_table = None
    n_states_arr = None
    if mode == "resegment":
        n_states_arr = n_states_b
        vals, cnts = np.unique(n_states_b, return_counts=True)
        count_df = pd.DataFrame(
            {"n_states": vals, "count": cnts, "fraction": cnts / n_boot}
        ).sort_values("n_states", ignore_index=True)
        rrows = []
        for k, s in enumerate(major):
            pops = reseg_pop[:, k]
            plo, phi = _ci(pops)
            rrows.append(
                {
                    "state": s.label,
                    "population": pop_point[k],
                    "match_rate": float(reseg_matched[:, k].mean()),
                    "population_mean_resegmented": float(np.nanmean(pops))
                    if np.isfinite(pops).any()
                    else np.nan,
                    "population_sd_resegmented": float(np.nanstd(pops))
                    if np.isfinite(pops).any()
                    else np.nan,
                    "population_ci95_low_resegmented": float(plo),
                    "population_ci95_high_resegmented": float(phi),
                }
            )
        reseg_table = pd.DataFrame(rrows)
        mode_n = int(count_df.loc[count_df["count"].idxmax(), "n_states"]) if len(count_df) else 0
        notes.append(
            f"Re-segmentation mode: watershed re-run on every resample; {len(major)} reference "
            f"states, modal state count {mode_n} (found in "
            f"{100 * count_df['fraction'].max():.0f} % of resamples). Population intervals under "
            "re-segmentation use only resamples in which the state was matched."
        )
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
        mode=mode,
        n_states_per_resample=n_states_arr,
        state_count_distribution=count_df,
        resegment_state_table=reseg_table,
    )


def _resegment_resample(
    sel: np.ndarray,
    hist_t: np.ndarray,
    fe: np.ndarray,
    cv1: np.ndarray,
    cv2: np.ndarray,
    replicate_ids: np.ndarray,
    w_all: np.ndarray,
    fel_grid: FELGrid,
    basin_labels: np.ndarray,
    basin_ids: np.ndarray,
    frame_basin: np.ndarray,
    kwargs: dict,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Re-run watershed segmentation on one resample and match states to the reference.

    Returns (number of major states found, matched population per reference
    state (NaN if unmatched), matched flags). Matching maximises the total
    weighted frame overlap between resampled and reference states (Hungarian
    assignment); a pair is accepted when the overlap covers at least half of the
    resampled state's weight or when the resampled state's minimum bin lies in
    the reference basin of the original landscape.
    """
    n_ref = len(basin_ids)
    pops = np.full(n_ref, np.nan)
    matched = np.zeros(n_ref, dtype=bool)
    total = hist_t.sum()
    if total <= 0:
        return 0, pops, matched
    prob = hist_t / total
    grid_b = FELGrid(
        x_edges=fel_grid.x_edges,
        y_edges=fel_grid.y_edges,
        x_centers=fel_grid.x_centers,
        y_centers=fel_grid.y_centers,
        probability=gaussian_filter(prob, sigma=fel_grid.smooth_sigma, mode="nearest")
        if fel_grid.smoothed
        else prob,
        free_energy=fe,
        temperature=fel_grid.temperature,
        smoothed=fel_grid.smoothed,
        smooth_sigma=fel_grid.smooth_sigma,
        occupancy_mask=hist_t > 0,
    )
    df_b = pd.DataFrame(
        {
            "system": "boot",
            "replicate": replicate_ids[sel],
            "frame": sel,
            "time_ps": sel.astype(np.float64),
            "PC1": cv1[sel],
            "PC2": cv2[sel],
        }
    )
    w_sel = w_all[sel]
    # the resampled frame table always names its CV columns PC1/PC2
    kwargs = {k: v for k, v in kwargs.items() if k != "cv_columns"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger("conformatlas.basins").disabled = True
        try:
            states_b, _, assign_b, labels_b = detect_energy_basins(
                grid_b, df_b, weights=w_sel, **kwargs
            )
        finally:
            logging.getLogger("conformatlas.basins").disabled = False
    n_found = len(states_b)
    if n_found == 0:
        return 0, pops, matched
    lab_b = assign_b["state"].to_numpy()
    ref_b = frame_basin[sel]  # reference basin of every resampled frame
    overlap = np.zeros((n_ref, n_found))
    size_b = np.zeros(n_found)
    for j, sb in enumerate(states_b):
        m = lab_b == sb.label
        size_b[j] = w_sel[m].sum()
        for k, bid in enumerate(basin_ids):
            overlap[k, j] = w_sel[m & (ref_b == bid)].sum()
    rows, cols = linear_sum_assignment(-overlap)
    x_idx, y_idx, ok = frame_bin_indices(
        fel_grid,
        np.array([sb.min_pc1 for sb in states_b]),
        np.array([sb.min_pc2 for sb in states_b]),
    )
    for k, j in zip(rows, cols):
        frac = overlap[k, j] / size_b[j] if size_b[j] > 0 else 0.0
        in_ref_basin = bool(ok[j]) and basin_labels[y_idx[j], x_idx[j]] == basin_ids[k]
        if frac >= 0.5 or in_ref_basin:
            matched[k] = True
            pops[k] = states_b[j].population
    return n_found, pops, matched
