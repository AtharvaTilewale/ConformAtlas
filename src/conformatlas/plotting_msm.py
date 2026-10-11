"""Figures for the Markov state model and per-replicate landscapes.

All functions write a 300 dpi PNG (the matplotlib rcParams of the package) and
return the path. They consume the DataFrames produced by :mod:`conformatlas.msm`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from conformatlas.fel import calculate_fel  # noqa: E402
from conformatlas.models import FELGrid  # noqa: E402
from conformatlas.plotting import CV_LABELS, save_figure  # noqa: E402

logger = logging.getLogger("conformatlas.plotting")

_SET_COLORS = plt.cm.tab10.colors


def _finish(fig, out_path: str | Path) -> Path:
    fig.tight_layout()
    return save_figure(fig, out_path)  # every configured format; returns the primary path


# ----------------------------------------------------------------------------
# Implied timescales
# ----------------------------------------------------------------------------


def plot_implied_timescales(
    its: pd.DataFrame,
    out_path: str | Path,
    lag_report: dict[str, Any] | None = None,
    chosen_lag: int | None = None,
    n_show: int | None = None,
) -> Path:
    """Implied timescales versus lag time (log y). The grey region marks
    timescales shorter than the lag, which cannot be resolved; the dashed line
    is the lag reported by :func:`conformatlas.msm.suggest_markov_lag`, the
    dotted line the lag actually used for the model."""
    cols = [c for c in its.columns if c.startswith("ts_")]
    if n_show is not None:
        cols = cols[: int(n_show)]
    lag_ps = its["lag_ps"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, c in enumerate(cols):
        ax.plot(
            lag_ps,
            its[c].to_numpy(dtype=float),
            marker="o",
            ms=4,
            lw=1.4,
            color=_SET_COLORS[i % 10],
            label=f"t$_{{{i + 1}}}$",
        )
    xmax = lag_ps.max() * 1.05 if len(lag_ps) else 1.0
    if lag_report and lag_report.get("suggested_lag_frames") is not None and len(its):
        dt = float(its["lag_ps"].iloc[0] / its["lag_frames"].iloc[0])
        ax.axvline(
            lag_report["suggested_lag_frames"] * dt,
            color="k",
            ls="--",
            lw=1,
            label=f"plateau from {lag_report['suggested_lag_frames']} frames",
        )
    if chosen_lag is not None and len(its):
        dt = float(its["lag_ps"].iloc[0] / its["lag_frames"].iloc[0])
        ax.axvline(
            chosen_lag * dt, color="C3", ls=":", lw=1.2, label=f"model lag ({chosen_lag} frames)"
        )
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    finite = its[cols].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    y_lo = max(finite.min() * 0.5, 1e-3) if finite.size else 1e-3
    y_hi = finite.max() * 3 if finite.size else xmax
    ax.set_ylim(y_lo, y_hi)
    xs = np.linspace(y_lo, xmax, 200)
    ax.fill_between(xs, np.full_like(xs, y_lo), xs, color="0.82", zorder=1.5, label="t < lag")
    ax.set_xlabel("Lag time (ps)")
    ax.set_ylabel("Implied timescale (ps)")
    ax.set_title("Implied timescales vs lag (Markovian when flat)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    return _finish(fig, out_path)


# ----------------------------------------------------------------------------
# Chapman-Kolmogorov test
# ----------------------------------------------------------------------------


def plot_ck_test(
    ck: pd.DataFrame,
    out_path: str | Path,
    set_names: list[str] | None = None,
    diagonal_only: bool = True,
) -> Path:
    """One panel per metastable set: predicted ``T(tau)^k`` (line) versus the
    estimate at ``k tau`` (points with approximate 1-sigma error bars)."""
    if not len(ck):
        raise ValueError("empty CK table")
    sets = sorted(ck["from_set"].unique())
    names = set_names or [f"MS{i + 1}" for i in range(len(sets))]
    n = len(sets)
    ncols = min(n, 3) if diagonal_only else n
    nrows = int(np.ceil(n / ncols)) if diagonal_only else n
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False)
    k0 = ck["lag_ps"].to_numpy(dtype=float)
    panels = [(a, a) for a in sets] if diagonal_only else [(a, b) for a in sets for b in sets]
    for idx, (a, b) in enumerate(panels):
        ax = axes.flat[idx]
        sub = ck[(ck["from_set"] == a) & (ck["to_set"] == b)].sort_values("k")
        x = np.concatenate([[0.0], sub["lag_ps"].to_numpy(dtype=float)])
        pred = np.concatenate([[1.0 if a == b else 0.0], sub["predicted"].to_numpy(dtype=float)])
        est = sub["estimated"].to_numpy(dtype=float)
        se = sub["estimated_se"].to_numpy(dtype=float)
        ax.plot(x, pred, color="k", lw=1.5, label="model $T(\\tau)^k$")
        ax.errorbar(
            sub["lag_ps"],
            est,
            yerr=se,
            fmt="o",
            ms=4,
            color=_SET_COLORS[a % 10],
            capsize=2,
            label="estimate at $k\\tau$",
        )
        ax.set_title(f"{names[a]} → {names[b]}", fontsize=10)
        ax.set_xlabel("k·τ (ps)")
        ax.set_ylabel("probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, ls="--", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, frameon=False)
    for j in range(len(panels), axes.size):
        axes.flat[j].set_visible(False)
    fig.suptitle(
        f"Chapman–Kolmogorov test (lag {int(k0.min() / ck['k'].min())} ps; max |dev| "
        f"{ck['abs_deviation'].max():.3f})",
        fontsize=11,
    )
    return _finish(fig, out_path)


# ----------------------------------------------------------------------------
# PCCA+ memberships on the landscape
# ----------------------------------------------------------------------------


def plot_pcca_landscape(
    fel_grid: FELGrid,
    projections: pd.DataFrame,
    frame_table: pd.DataFrame,
    centers: np.ndarray,
    memberships: pd.DataFrame,
    out_path: str | Path,
    cv_columns: tuple[str, str] = ("PC1", "PC2"),
    max_points: int = 20000,
    seed: int = 0,
) -> Path:
    """Free-energy landscape with frames coloured by crisp PCCA+ set and the
    microstate centres drawn with membership-scaled pies (one panel per set
    shows the fuzzy membership as colour)."""
    names = [c for c in memberships.columns if c.startswith("MS")]
    m = len(names)
    fig, axes = plt.subplots(1, m + 1, figsize=(5.2 * (m + 1), 4.6), squeeze=False)
    X, Y = np.meshgrid(fel_grid.x_centers, fel_grid.y_centers)
    fe = np.ma.masked_invalid(fel_grid.free_energy)
    vmax = float(np.nanpercentile(fel_grid.free_energy, 99)) if np.isfinite(fe).any() else 1.0
    levels = np.linspace(0, max(vmax, 1e-3), 15)
    rng = np.random.default_rng(seed)
    n = len(projections)
    sel = np.arange(n) if n <= max_points else rng.choice(n, size=max_points, replace=False)
    x = projections[cv_columns[0]].to_numpy()[sel]
    y = projections[cv_columns[1]].to_numpy()[sel]
    labels = frame_table["metastable_set"].to_numpy()[sel]
    ax = axes[0, 0]
    cf0 = ax.contourf(X, Y, fe, levels=levels, cmap="Greys", extend="max", alpha=0.9)
    fig.colorbar(cf0, ax=ax, label="ΔG (kJ/mol)", pad=0.02)
    for i, nm in enumerate(names):
        mask = labels == nm
        ax.scatter(
            x[mask], y[mask], s=3, color=_SET_COLORS[i % 10], alpha=0.5, label=nm, rasterized=True
        )
    ax.scatter(
        centers[:, 0], centers[:, 1], marker="x", s=14, color="k", lw=0.8, label="microstates"
    )
    ax.set_title("Crisp PCCA+ assignment")
    ax.set_xlabel(CV_LABELS[0])
    ax.set_ylabel(CV_LABELS[1])
    ax.legend(fontsize=7, frameon=True, markerscale=3)
    micro_idx = memberships.index.to_numpy()
    for i, nm in enumerate(names):
        ax = axes[0, i + 1]
        ax.contourf(X, Y, fe, levels=levels, cmap="Greys", extend="max", alpha=0.9)
        sc = ax.scatter(
            centers[micro_idx, 0],
            centers[micro_idx, 1],
            c=memberships[nm].to_numpy(dtype=float),
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=40,
            edgecolor="k",
            lw=0.4,
        )
        ax.set_title(f"membership χ({nm})")
        ax.set_xlabel(CV_LABELS[0])
        if i == m - 1:
            fig.colorbar(sc, ax=ax, label="membership", pad=0.02)
    return _finish(fig, out_path)


# ----------------------------------------------------------------------------
# Metastable-state time series
# ----------------------------------------------------------------------------


def plot_metastable_timeseries(frame_table: pd.DataFrame, out_path: str | Path) -> Path:
    """Crisp metastable set along each replicate (step plot) with the membership
    of the occupied set as point colour intensity."""
    names = [
        c.replace("membership_", "") for c in frame_table.columns if c.startswith("membership_")
    ]
    code = {nm: i for i, nm in enumerate(names)}
    reps = list(dict.fromkeys(frame_table["replicate"]))
    n = len(reps)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.9 * n + 1.2), sharex=True, squeeze=False)
    for i, rep in enumerate(reps):
        ax = axes[i, 0]
        df = frame_table[frame_table["replicate"] == rep].sort_values("time_ps")
        t = df["time_ps"].to_numpy(dtype=float) / 1000.0
        lab = df["metastable_set"].to_numpy()
        y = np.array([code.get(s, np.nan) for s in lab], dtype=float)
        ax.step(t, y, where="post", color="#444444", lw=0.7)
        for nm, k in code.items():
            mask = lab == nm
            if mask.any():
                conf = df[f"membership_{nm}"].to_numpy(dtype=float)[mask]
                ax.scatter(
                    t[mask],
                    y[mask],
                    s=5,
                    c=conf,
                    cmap="Blues",
                    vmin=0,
                    vmax=1,
                    label=nm if i == 0 else None,
                )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_ylabel(rep, fontsize=9)
        ax.grid(axis="x", alpha=0.3, ls="--")
    axes[-1, 0].set_xlabel("Time (ns)")
    axes[0, 0].set_title(
        "PCCA+ metastable set along each replicate (colour: membership of the occupied set)"
    )
    return _finish(fig, out_path)


# ----------------------------------------------------------------------------
# Per-replicate landscapes on a shared grid
# ----------------------------------------------------------------------------


def plot_per_replicate_landscapes(
    projections_df: pd.DataFrame,
    fel_grid_kwargs: dict[str, Any],
    out_path: str | Path,
    cv_columns: tuple[str, str] = ("PC1", "PC2"),
    weights: np.ndarray | None = None,
    max_cols: int = 3,
) -> tuple[Path, dict[str, FELGrid]]:
    """Free-energy landscape of every replicate on the grid shared with the pooled landscape.

    ``fel_grid_kwargs`` are forwarded to :func:`conformatlas.fel.calculate_fel`
    (``temperature``, ``bins``, ``smooth_sigma``); the axis ranges are taken from
    the pooled data (or from ``x_range``/``y_range`` in the kwargs) so that all
    panels share bins and the colour scale. Each panel's ΔG is referenced to
    its own minimum (as the pooled landscape is), so panels compare the *shape*
    of the sampled landscape per replicate - a visual convergence diagnostic.
    Returns the figure path and the per-replicate grids.
    """
    kw = dict(fel_grid_kwargs)
    x_all = projections_df[cv_columns[0]].to_numpy(dtype=float)
    y_all = projections_df[cv_columns[1]].to_numpy(dtype=float)
    kw.setdefault("x_range", (float(x_all.min()), float(x_all.max())))
    kw.setdefault("y_range", (float(y_all.min()), float(y_all.max())))
    reps = list(dict.fromkeys(projections_df["replicate"]))
    grids: dict[str, FELGrid] = {}
    for rep in reps:
        mask = (projections_df["replicate"] == rep).to_numpy()
        w = None if weights is None else np.asarray(weights)[mask]
        grids[str(rep)] = calculate_fel(x_all[mask], y_all[mask], weights=w, **kw)
    vmax = max(
        float(np.nanpercentile(g.free_energy, 99))
        for g in grids.values()
        if np.isfinite(g.free_energy).any()
    )
    levels = np.linspace(0, max(vmax, 1e-3), 16)
    n = len(reps)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), squeeze=False)
    cf = None
    for i, rep in enumerate(reps):
        ax = axes.flat[i]
        g = grids[str(rep)]
        Xg, Yg = np.meshgrid(g.x_centers, g.y_centers)
        cf = ax.contourf(
            Xg, Yg, np.ma.masked_invalid(g.free_energy), levels=levels, cmap="turbo", extend="max"
        )
        n_rep = int(np.sum(projections_df["replicate"] == rep))
        ax.set_title(f"{rep} ({n_rep} frames)", fontsize=10)
        ax.set_xlabel(CV_LABELS[0])
        if i % ncols == 0:
            ax.set_ylabel(CV_LABELS[1])
        ax.set_xlim(kw["x_range"])
        ax.set_ylim(kw["y_range"])
    for j in range(n, axes.size):
        axes.flat[j].set_visible(False)
    if cf is not None:
        fig.colorbar(cf, ax=axes.ravel().tolist(), label="ΔG (kJ/mol)", pad=0.02, fraction=0.025)
    fig.suptitle("Per-replicate free-energy landscapes on the shared grid", fontsize=12)
    return save_figure(fig, out_path), grids
