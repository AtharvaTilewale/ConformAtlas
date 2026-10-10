"""Statistics for multi-system comparisons on a shared landscape.

* per-system block bootstrap of state populations and ΔG grids
  (populations differences with confidence intervals, ΔΔG maps with
  standard deviations, state ΔΔG with confidence intervals);
* Welch's t-test on per-replicate populations (when every system has at
  least two replicates);
* landscape overlap metrics between systems (Bhattacharyya coefficient,
  Jensen–Shannon divergence) computed from the sampled probability
  densities on the shared grid;
* RMSIP between the essential subspaces of the individual systems.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from conformatlas.basins import frame_bin_indices
from conformatlas.bootstrap import _free_energy_from_counts, circular_block_indices
from conformatlas.convergence import calculate_subspace_rmsip
from conformatlas.models import BasinState, FELGrid
from conformatlas.timeseries import block_length_from_g, statistical_inefficiency

logger = logging.getLogger("conformatlas.comparison_stats")


@dataclass
class SystemBootstrap:
    """Bootstrap samples for one system on the shared grid."""

    name: str
    populations: np.ndarray  # (n_boot, n_states), fractions
    state_min_fe: np.ndarray  # (n_boot, n_states), kJ/mol
    fe_grids: np.ndarray  # (n_boot, ny, nx), kJ/mol
    block_lengths: dict[str, int] = field(default_factory=dict)
    n_frames: int = 0


def system_block_lengths(
    projections: pd.DataFrame, columns: tuple[str, str] = ("PC1", "PC2")
) -> dict[str, int]:
    """Block length per replicate from the statistical inefficiency of the CVs."""
    out: dict[str, int] = {}
    for rep, df in projections.groupby("replicate", sort=False):
        n = len(df)
        g = max(statistical_inefficiency(df[c].to_numpy()) for c in columns)
        out[str(rep)] = block_length_from_g(g, n)
    return out


def bootstrap_system(
    name: str,
    projections: pd.DataFrame,
    fel_grid: FELGrid,
    basin_labels: np.ndarray,
    states: list[BasinState],
    n_boot: int,
    rng: np.random.Generator,
    columns: tuple[str, str] = ("PC1", "PC2"),
) -> SystemBootstrap:
    """Circular block bootstrap of one system's frames on the shared grid."""
    cv1 = projections[columns[0]].to_numpy(dtype=np.float64)
    cv2 = projections[columns[1]].to_numpy(dtype=np.float64)
    reps = projections["replicate"].to_numpy()
    n = len(cv1)
    block_lengths = system_block_lengths(projections, columns)

    x_idx, y_idx, valid = frame_bin_indices(fel_grid, cv1, cv2)
    frame_basin = np.zeros(n, dtype=np.int64)
    frame_basin[valid] = basin_labels[y_idx[valid], x_idx[valid]]
    major = [s for s in states if s.basin_id > 0]
    basin_ids = np.array([s.basin_id for s in major], dtype=np.int64)
    masks = [basin_labels == bid for bid in basin_ids]

    ny, nx = fel_grid.free_energy.shape
    sigma = fel_grid.smooth_sigma if fel_grid.smoothed else 0.0
    rep_index = {rep: np.where(reps == rep)[0] for rep in dict.fromkeys(reps)}
    hist_range = [
        [fel_grid.x_edges[0], fel_grid.x_edges[-1]],
        [fel_grid.y_edges[0], fel_grid.y_edges[-1]],
    ]

    pops = np.full((n_boot, len(major)), np.nan)
    mins = np.full((n_boot, len(major)), np.nan)
    grids = np.full((n_boot, ny, nx), np.nan)
    for b in range(n_boot):
        picks = [
            idx[circular_block_indices(len(idx), block_lengths.get(str(rep), 1), rng)]
            for rep, idx in rep_index.items()
        ]
        sel = np.concatenate(picks)
        hist, _, _ = np.histogram2d(
            cv1[sel], cv2[sel], bins=[fel_grid.x_edges, fel_grid.y_edges], range=hist_range
        )
        fe = _free_energy_from_counts(hist.T, fel_grid.temperature, sigma)
        grids[b] = fe
        fb = frame_basin[sel]
        for k, bid in enumerate(basin_ids):
            pops[b, k] = np.mean(fb == bid)
            vals = fe[masks[k]]
            mins[b, k] = np.nanmin(vals) if np.any(np.isfinite(vals)) else np.nan
    return SystemBootstrap(name, pops, mins, grids, block_lengths, n)


def _ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        return float("nan"), float("nan")
    a = (1.0 - level) / 2.0
    return float(np.quantile(s, a)), float(np.quantile(s, 1.0 - a))


def _bootstrap_p_value(diff_samples: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (fraction of the
    bootstrap distribution on the other side of zero, doubled, floored at 1/B)."""
    s = diff_samples[np.isfinite(diff_samples)]
    if s.size == 0:
        return float("nan")
    frac = min(np.mean(s <= 0.0), np.mean(s >= 0.0))
    return float(min(1.0, max(2.0 * frac, 1.0 / s.size)))


def population_difference_table(
    states: list[BasinState],
    frame_assignments: pd.DataFrame,
    system_names: list[str],
    reference: str,
    boots: dict[str, SystemBootstrap],
) -> pd.DataFrame:
    """Per-state populations for every system, replicate-level spread, and the
    difference to the reference system with bootstrap CI / p-value and Welch's t."""
    try:
        from scipy import stats as sps
    except Exception:  # pragma: no cover
        sps = None

    major = [s for s in states if s.basin_id > 0]
    records = []
    for k, st in enumerate(major):
        row: dict[str, object] = {"State": st.label}
        rep_pops: dict[str, np.ndarray] = {}
        for name in system_names:
            s_frames = frame_assignments[frame_assignments["system"] == name]
            tot = len(s_frames)
            pop = 100.0 * np.sum(s_frames["state"] == st.label) / tot if tot else np.nan
            row[f"{name}_Population_Percent"] = pop
            per_rep = (
                s_frames.groupby("replicate", sort=False)["state"]
                .apply(lambda x, lab=st.label: 100.0 * np.mean(x == lab))
                .to_numpy()
            )
            rep_pops[name] = per_rep
            row[f"{name}_Replicate_SD_Percent"] = (
                float(np.std(per_rep, ddof=1)) if len(per_rep) > 1 else np.nan
            )
            if name in boots:
                lo, hi = _ci(100.0 * boots[name].populations[:, k])
                row[f"{name}_CI95_Low_Percent"] = lo
                row[f"{name}_CI95_High_Percent"] = hi
        for name in system_names:
            if name == reference:
                continue
            diff = row[f"{name}_Population_Percent"] - row[f"{reference}_Population_Percent"]
            row[f"{name}_minus_{reference}_Percent"] = diff
            if name in boots and reference in boots:
                d = 100.0 * (boots[name].populations[:, k] - boots[reference].populations[:, k])
                lo, hi = _ci(d)
                row[f"{name}_minus_{reference}_CI95_Low"] = lo
                row[f"{name}_minus_{reference}_CI95_High"] = hi
                row[f"{name}_minus_{reference}_Bootstrap_p"] = _bootstrap_p_value(d)
            a, b = rep_pops[reference], rep_pops[name]
            if sps is not None and len(a) >= 2 and len(b) >= 2:
                t = sps.ttest_ind(b, a, equal_var=False)
                row[f"{name}_minus_{reference}_Welch_t"] = float(t.statistic)
                row[f"{name}_minus_{reference}_Welch_p"] = float(t.pvalue)
                row[f"{name}_minus_{reference}_Welch_df"] = f"{len(b)} vs {len(a)} replicates"
            else:
                row[f"{name}_minus_{reference}_Welch_t"] = np.nan
                row[f"{name}_minus_{reference}_Welch_p"] = np.nan
                row[f"{name}_minus_{reference}_Welch_df"] = "n/a (<2 replicates)"
        if len(system_names) == 2:
            other = [n for n in system_names if n != reference][0]
            row["Difference_Percent"] = row[f"{other}_minus_{reference}_Percent"]
        records.append(row)
    return pd.DataFrame(records)


def state_free_energy_table(
    states: list[BasinState],
    basin_labels: np.ndarray,
    system_fels: dict[str, FELGrid],
    reference: str,
    boots: dict[str, SystemBootstrap],
) -> pd.DataFrame:
    """Minimum ΔG of every state in every system (shared grid, each system's own
    zero) and ΔΔG relative to the reference system with bootstrap CI."""
    major = [s for s in states if s.basin_id > 0]
    records = []
    for k, st in enumerate(major):
        mask = basin_labels == st.basin_id
        row: dict[str, object] = {"State": st.label}
        for name, grid in system_fels.items():
            vals = grid.free_energy[mask]
            row[f"{name}_dG_min_kJmol"] = (
                float(np.nanmin(vals)) if np.any(np.isfinite(vals)) else np.nan
            )
        for name in system_fels:
            if name == reference:
                continue
            row[f"{name}_minus_{reference}_ddG_kJmol"] = (
                row[f"{name}_dG_min_kJmol"] - row[f"{reference}_dG_min_kJmol"]
            )
            if name in boots and reference in boots:
                d = boots[name].state_min_fe[:, k] - boots[reference].state_min_fe[:, k]
                lo, hi = _ci(d)
                row[f"{name}_minus_{reference}_ddG_CI95_Low"] = lo
                row[f"{name}_minus_{reference}_ddG_CI95_High"] = hi
        records.append(row)
    return pd.DataFrame(records)


def ddg_map(
    fel_ref: FELGrid,
    fel_sys: FELGrid,
    boot_ref: SystemBootstrap | None,
    boot_sys: SystemBootstrap | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """ΔΔG = ΔG(system) − ΔG(reference) on the shared grid (NaN where either is
    unsampled) and its bootstrap standard deviation when bootstraps are given."""
    a, b = fel_ref.free_energy, fel_sys.free_energy
    ddg = np.where(np.isnan(a) | np.isnan(b), np.nan, b - a)
    sd = None
    if boot_ref is not None and boot_sys is not None:
        n = min(len(boot_ref.fe_grids), len(boot_sys.fe_grids))
        diffs = boot_sys.fe_grids[:n] - boot_ref.fe_grids[:n]
        with np.errstate(all="ignore"):
            sd = np.nanstd(diffs, axis=0)
        sd[np.isnan(ddg)] = np.nan
    return ddg, sd


def ddg_map_table(
    fel_ref: FELGrid, fel_sys: FELGrid, ddg: np.ndarray, sd: np.ndarray | None, ref: str, name: str
) -> pd.DataFrame:
    xc, yc = np.meshgrid(fel_ref.x_centers, fel_ref.y_centers)
    df = pd.DataFrame(
        {
            "CV1_center": xc.ravel(),
            "CV2_center": yc.ravel(),
            f"dG_{ref}_kJmol": fel_ref.free_energy.ravel(),
            f"dG_{name}_kJmol": fel_sys.free_energy.ravel(),
            "ddG_kJmol": ddg.ravel(),
        }
    )
    if sd is not None:
        df["ddG_bootstrap_SD_kJmol"] = sd.ravel()
    return df


def probability_grid(fel: FELGrid) -> np.ndarray:
    """Normalised probability density on the grid implied by the stored ΔG
    (unsampled bins are zero)."""
    fe = fel.free_energy
    kT = 0.0083144626 * fel.temperature
    with np.errstate(over="ignore", invalid="ignore"):
        p = np.where(np.isfinite(fe), np.exp(-np.nan_to_num(fe, nan=np.inf) / kT), 0.0)
    tot = p.sum()
    return p / tot if tot > 0 else p


def landscape_overlap(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Bhattacharyya coefficient (1 = identical), Hellinger distance and
    Jensen–Shannon divergence (bits, 0 = identical, 1 = disjoint) between two
    normalised grids, plus the fraction of bins sampled by both."""
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    bc = float(np.sum(np.sqrt(p * q)))
    hell = float(np.sqrt(max(0.0, 1.0 - bc)))
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    both = float(np.mean((p > 0) & (q > 0)))
    either = float(np.mean((p > 0) | (q > 0)))
    return {
        "Bhattacharyya_Coefficient": bc,
        "Hellinger_Distance": hell,
        "JS_Divergence_bits": float(js),
        "Sampled_Bin_Jaccard": both / either if either > 0 else float("nan"),
    }


def landscape_overlap_table(system_fels: dict[str, FELGrid]) -> pd.DataFrame:
    grids = {name: probability_grid(fel) for name, fel in system_fels.items()}
    rows = []
    for a, b in itertools.combinations(system_fels.keys(), 2):
        rows.append({"System_A": a, "System_B": b, **landscape_overlap(grids[a], grids[b])})
    return pd.DataFrame(rows)


def subspace_overlap_matrix(eigvecs: dict[str, np.ndarray], n_modes: int = 10) -> pd.DataFrame:
    """RMSIP between the leading ``n_modes`` eigenvectors of each system's own PCA."""
    names = list(eigvecs)
    k = min([n_modes] + [v.shape[1] for v in eigvecs.values()])
    mat = np.eye(len(names))
    for i, j in itertools.combinations(range(len(names)), 2):
        r = calculate_subspace_rmsip(eigvecs[names[i]][:, :k], eigvecs[names[j]][:, :k])
        mat[i, j] = mat[j, i] = r
    df = pd.DataFrame(mat, index=names, columns=names)
    df.index.name = f"RMSIP_top{k}"
    return df
