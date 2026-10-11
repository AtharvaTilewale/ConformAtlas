"""Additional figures: 1-D profiles with bootstrap bands, landscape uncertainty,
descriptors, state time series and kinetics, mode contributions, comparison maps."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from conformatlas.bootstrap import circular_block_indices  # noqa: E402
from conformatlas.models import BasinState, FELGrid  # noqa: E402
from conformatlas.plotting import CV_LABELS, _fs, landscape_cmap, save_figure  # noqa: E402
from conformatlas.utils import R_GAS_CONSTANT  # noqa: E402

logger = logging.getLogger("conformatlas.plotting")

_STATE_COLORS = plt.cm.tab10.colors


def _finish(fig, out_path: str | Path) -> Path:
    fig.tight_layout()
    return save_figure(fig, out_path)


# ----------------------------------------------------------------------------
# 1-D free energy profiles with block-bootstrap bands
# ----------------------------------------------------------------------------


def _profile(values, weights, edges, temperature):
    h, _ = np.histogram(values, bins=edges, weights=weights)
    p = h / h.sum() if h.sum() > 0 else h
    fe = np.full(len(p), np.nan)
    m = p > 0
    fe[m] = -R_GAS_CONSTANT * temperature * np.log(p[m] / p.max())
    return fe


def plot_fel_1d(
    cv1: np.ndarray,
    cv2: np.ndarray,
    replicate_ids: np.ndarray,
    temperature: float,
    out_path: str | Path,
    bins: int = 50,
    block_lengths: dict[str, int] | None = None,
    weights: np.ndarray | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> Path:
    """Free energy along each CV separately, with 95 % block-bootstrap bands."""
    rng = np.random.default_rng(seed)
    cv1 = np.asarray(cv1, float)
    cv2 = np.asarray(cv2, float)
    reps = np.asarray(replicate_ids)
    w = np.ones(len(cv1)) if weights is None else np.asarray(weights, float)
    rep_index = {r: np.where(reps == r)[0] for r in dict.fromkeys(reps)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, cv, label in zip(axes, (cv1, cv2), CV_LABELS):
        edges = np.linspace(cv.min(), cv.max(), bins + 1)
        centers = 0.5 * (edges[1:] + edges[:-1])
        point = _profile(cv, w, edges, temperature)
        samples = np.full((n_boot, bins), np.nan)
        for b in range(n_boot):
            sel = np.concatenate(
                [
                    idx[circular_block_indices(len(idx), (block_lengths or {}).get(r, 1), rng)]
                    for r, idx in rep_index.items()
                ]
            )
            samples[b] = _profile(cv[sel], w[sel], edges, temperature)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanpercentile(samples, 2.5, axis=0)
            hi = np.nanpercentile(samples, 97.5, axis=0)
        ax.fill_between(centers, lo, hi, color="#2b5c8f", alpha=0.25, label="95 % bootstrap band")
        ax.plot(centers, point, color="#2b5c8f", lw=2.2, label="ΔG")
        ax.set_xlabel(label)
        ax.set_ylabel("ΔG (kJ/mol)")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(frameon=False)
    fig.suptitle(f"One-dimensional free energy profiles (T = {temperature:.1f} K)")
    return _finish(fig, out_path)


# ----------------------------------------------------------------------------
# Landscape uncertainty map
# ----------------------------------------------------------------------------


def plot_fel_uncertainty(
    fel_grid: FELGrid, fe_std: np.ndarray, out_path: str | Path, states=None
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x, y = fel_grid.x_centers, fel_grid.y_centers
    im0 = axes[0].pcolormesh(x, y, fel_grid.free_energy, cmap=landscape_cmap(), shading="auto")
    fig.colorbar(im0, ax=axes[0], label="ΔG (kJ/mol)")
    axes[0].set_title("Free energy landscape")
    im1 = axes[1].pcolormesh(x, y, fe_std, cmap="magma", shading="auto")
    fig.colorbar(im1, ax=axes[1], label="bootstrap SD of ΔG (kJ/mol)")
    axes[1].set_title("Statistical uncertainty of ΔG")
    for ax in axes:
        ax.set_xlabel(CV_LABELS[0])
        ax.set_ylabel(CV_LABELS[1])
        if states:
            for st in states:
                ax.plot(st.min_pc1, st.min_pc2, marker="*", ms=12, color="white", mec="black")
    return _finish(fig, out_path)


# ----------------------------------------------------------------------------
# Descriptors: RMSD(t), Rg(t), RMSF, equilibration markers
# ----------------------------------------------------------------------------


def plot_descriptors(
    descriptors: dict[str, pd.DataFrame],
    out_path: str | Path,
    equilibration: pd.DataFrame | None = None,
) -> Path:
    ts = descriptors["timeseries"]
    res = descriptors["rmsf_residues"]
    has_lig = "Ligand_RMSD" in ts.columns
    n_rows = 3 if has_lig else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.6 * n_rows))
    axes = np.atleast_2d(axes)
    t0 = {}
    if equilibration is not None and len(equilibration):
        t0 = dict(zip(equilibration["Replicate"], equilibration["t0_ps"]))
    for i, (rep, df) in enumerate(ts.groupby("replicate", sort=False)):
        c = _STATE_COLORS[i % 10]
        t_ns = df["time_ps"] / 1000.0
        axes[0, 0].plot(t_ns, df["RMSD"], lw=1.0, color=c, label=rep)
        axes[0, 1].plot(t_ns, df["Rg"], lw=1.0, color=c, label=rep)
        if has_lig:
            axes[2, 0].plot(t_ns, df["Ligand_RMSD"], lw=1.0, color=c, label=rep)
        if rep in t0 and t0[rep] > 0:
            for ax in (axes[0, 0], axes[0, 1]):
                ax.axvline(t0[rep] / 1000.0, color=c, ls="--", lw=1.2)
    axes[0, 0].set_xlabel("Time (ns)")
    axes[0, 0].set_ylabel("RMSD to reference (Å)")
    axes[0, 0].set_title("Backbone RMSD" if not has_lig else "Protein RMSD")
    axes[0, 1].set_xlabel("Time (ns)")
    axes[0, 1].set_ylabel("Rg (Å)")
    axes[0, 1].set_title("Radius of gyration")
    if t0:
        axes[0, 0].plot([], [], color="grey", ls="--", label="detected equilibration")
    axes[0, 0].legend(frameon=False, fontsize=_fs(8))

    x = np.arange(len(res))
    axes[1, 0].plot(x, res["RMSF"], color="#2b5c8f", lw=1.5)
    axes[1, 0].fill_between(x, 0, res["RMSF"], color="#2b5c8f", alpha=0.25)
    step = max(1, len(res) // 10)
    axes[1, 0].set_xticks(x[::step])
    axes[1, 0].set_xticklabels(res["residue_number"].iloc[::step], rotation=0)
    axes[1, 0].set_xlabel("Residue")
    axes[1, 0].set_ylabel("RMSF (Å)")
    axes[1, 0].set_title("Per-residue RMSF (all frames)")

    # RMSD and Rg distributions
    for rep, df in ts.groupby("replicate", sort=False):
        axes[1, 1].hist(df["RMSD"], bins=40, alpha=0.5, label=rep, density=True)
    axes[1, 1].set_xlabel("RMSD (Å)")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title("RMSD distribution")
    axes[1, 1].legend(frameon=False, fontsize=_fs(8))
    if has_lig:
        axes[2, 0].set_xlabel("Time (ns)")
        axes[2, 0].set_ylabel("Ligand RMSD (Å)")
        axes[2, 0].set_title("Ligand RMSD after protein fit")
        for rep, df in ts.groupby("replicate", sort=False):
            axes[2, 1].hist(df["Ligand_RMSD"], bins=40, alpha=0.5, label=rep, density=True)
        axes[2, 1].set_xlabel("Ligand RMSD (Å)")
        axes[2, 1].set_title("Ligand RMSD distribution")
    for ax in axes.ravel():
        ax.grid(alpha=0.3, ls="--")
    return _finish(fig, out_path)


# ----------------------------------------------------------------------------
# State time series and dwell times
# ----------------------------------------------------------------------------


def plot_state_timeseries(
    frame_assignments: pd.DataFrame,
    states: list[BasinState],
    out_path: str | Path,
    dwell_times: pd.DataFrame | None = None,
) -> Path:
    labels = [s.label for s in states]
    code = {s: i for i, s in enumerate(labels)}
    reps = list(dict.fromkeys(frame_assignments["replicate"]))
    n = len(reps)
    fig = plt.figure(figsize=(13, 2.2 * n + 3.5))
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[1] * n + [1.4])
    for i, rep in enumerate(reps):
        ax = fig.add_subplot(gs[i, :])
        df = frame_assignments[frame_assignments["replicate"] == rep]
        t = df["time_ps"].to_numpy() / 1000.0
        y = df["state"].map(code).to_numpy(dtype=float)
        ax.step(t, y, where="post", color="#444444", lw=0.8)
        for s, k in code.items():
            m = df["state"].to_numpy() == s
            ax.scatter(t[m], y[m], s=6, color=_STATE_COLORS[k % 10], label=s if i == 0 else None)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=_fs(8))
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.set_ylabel(rep, fontsize=_fs(9))
        ax.grid(axis="x", alpha=0.3, ls="--")
        if i == n - 1:
            ax.set_xlabel("Time (ns)")
        if i == 0:
            ax.legend(ncol=min(len(labels), 5), fontsize=_fs(8), frameon=False, loc="upper right")
            ax.set_title("State occupancy along each replicate (unassigned frames omitted)")
    ax_d = fig.add_subplot(gs[n, 0])
    ax_c = fig.add_subplot(gs[n, 1])
    if dwell_times is not None and len(dwell_times):
        for s, k in code.items():
            d = dwell_times[(dwell_times["state"] == s) & (~dwell_times["censored"].astype(bool))]
            if len(d):
                ax_d.hist(
                    d["dwell_ps"] / 1000.0,
                    bins=20,
                    alpha=0.55,
                    color=_STATE_COLORS[k % 10],
                    label=s,
                )
        ax_d.set_xlabel("Dwell time (ns)")
        ax_d.set_ylabel("visits")
        ax_d.set_title("Dwell-time distribution (uncensored visits)")
        if ax_d.get_legend_handles_labels()[0]:
            ax_d.legend(fontsize=_fs(8), frameon=False)
    # transition count heat map
    counts = np.zeros((len(labels), len(labels)))
    for rep in reps:
        seq = (
            frame_assignments[
                (frame_assignments["replicate"] == rep) & frame_assignments["state"].isin(labels)
            ]["state"]
            .map(code)
            .to_numpy()
        )
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1
    im = ax_c.imshow(counts, cmap="Blues")
    ax_c.set_xticks(range(len(labels)))
    ax_c.set_xticklabels(labels, rotation=45, ha="right", fontsize=_fs(8))
    ax_c.set_yticks(range(len(labels)))
    ax_c.set_yticklabels(labels, fontsize=_fs(8))
    ax_c.set_xlabel("to")
    ax_c.set_ylabel("from")
    ax_c.set_title("Transition counts (lag = 1 frame)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax_c.text(
                j,
                i,
                f"{int(counts[i, j])}",
                ha="center",
                va="center",
                fontsize=_fs(8),
                color="white" if counts[i, j] > counts.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax_c, fraction=0.046)
    return _finish(fig, out_path)


# ----------------------------------------------------------------------------
# Mode contributions per residue
# ----------------------------------------------------------------------------


def plot_mode_contributions(res_df: pd.DataFrame, out_path: str | Path, n_modes: int = 3) -> Path:
    cols = [c for c in res_df.columns if c.startswith("PC")][:n_modes]
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 2.6 * len(cols)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(res_df))
    for ax, col, color in zip(axes, cols, ("#2b5c8f", "#d95f02", "#1b9e77")):
        ax.fill_between(x, 0, res_df[col], color=color, alpha=0.35)
        ax.plot(x, res_df[col], color=color, lw=1.4)
        ax.set_ylabel(f"{col} weight")
        ax.grid(alpha=0.3, ls="--")
        top = res_df.nlargest(5, col)
        for _, r in top.iterrows():
            ax.annotate(
                f"{r['residue_name']}{int(r['residue_number'])}",
                (res_df.index.get_loc(r.name), r[col]),
                fontsize=_fs(7),
                ha="center",
                va="bottom",
            )
    step = max(1, len(res_df) // 12)
    axes[-1].set_xticks(x[::step])
    axes[-1].set_xticklabels(res_df["residue_number"].iloc[::step])
    axes[-1].set_xlabel("Residue")
    axes[0].set_title(
        "Per-residue contribution to the leading modes (sqrt of summed squared eigenvector components)"
    )
    return _finish(fig, out_path)


# ----------------------------------------------------------------------------
# Comparison: ΔΔG map
# ----------------------------------------------------------------------------


def plot_ddg_map(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    ddg: np.ndarray,
    name_a: str,
    name_b: str,
    out_path: str | Path,
    sd: np.ndarray | None = None,
) -> Path:
    """ΔΔG map (system B minus reference A); with ``sd`` a second panel shows the
    bootstrap standard deviation and hatches bins where |ΔΔG| < 2 SD."""
    ncols = 2 if sd is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.2 * ncols, 6), squeeze=False)
    ax = axes[0, 0]
    lim = np.nanmax(np.abs(ddg)) if np.any(np.isfinite(ddg)) else 1.0
    im = ax.pcolormesh(
        x_centers, y_centers, ddg, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto"
    )
    fig.colorbar(im, ax=ax, label=f"ΔΔG = ΔG({name_b}) − ΔG({name_a}) (kJ/mol)")
    ax.set_xlabel(CV_LABELS[0])
    ax.set_ylabel(CV_LABELS[1])
    ax.set_title(
        f"Free energy difference map: {name_b} vs {name_a}\n"
        f"(blue: more favourable in {name_b}; white: sampled by one system only)"
    )
    if sd is not None:
        insignificant = np.isfinite(ddg) & (np.abs(ddg) < 2.0 * np.nan_to_num(sd, nan=np.inf))
        if insignificant.any():
            ax.contourf(
                x_centers,
                y_centers,
                insignificant.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["////"],
            )
            ax.text(
                0.02,
                0.02,
                "hatched: |ΔΔG| < 2 bootstrap SD",
                transform=ax.transAxes,
                fontsize=_fs(8),
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        ax2 = axes[0, 1]
        im2 = ax2.pcolormesh(x_centers, y_centers, sd, cmap="magma", shading="auto")
        fig.colorbar(im2, ax=ax2, label="bootstrap SD of ΔΔG (kJ/mol)")
        ax2.set_xlabel(CV_LABELS[0])
        ax2.set_ylabel(CV_LABELS[1])
        ax2.set_title("Statistical uncertainty of ΔΔG")
    return _finish(fig, out_path)
