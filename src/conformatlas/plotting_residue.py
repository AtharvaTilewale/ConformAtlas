"""Residue-level figures: DCCM heatmap, per-residue state-difference profiles and
ligand-contact persistence per state."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from conformatlas.plotting import save_figure, set_cv_labels  # noqa: E402,F401  (applies the shared rcParams)

logger = logging.getLogger("conformatlas.plotting")

_STATE_COLORS = plt.cm.tab10.colors


def _finish(fig, out_path: str | Path) -> Path:
    fig.tight_layout()
    return save_figure(fig, out_path)  # every configured format; returns the primary path


def _residue_ticks(residue_numbers: np.ndarray, max_ticks: int = 12):
    n = len(residue_numbers)
    step = max(1, int(np.ceil(n / max_ticks)))
    pos = np.arange(0, n, step)
    return pos, [str(residue_numbers[i]) for i in pos]


def plot_dccm(
    matrix: np.ndarray,
    residue_numbers: np.ndarray | list[int],
    out_path: str | Path,
    title: str = "Dynamic cross-correlation map",
    chains: np.ndarray | list[str] | None = None,
) -> Path:
    """Heatmap of a (residue- or atom-level) cross-correlation matrix in [-1, 1].

    Chain boundaries (when ``chains`` is given) are drawn as thin lines.
    """
    m = np.asarray(matrix, dtype=float)
    residue_numbers = np.asarray(residue_numbers)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", interpolation="nearest")
    pos, labels = _residue_ticks(residue_numbers)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Residue")
    ax.set_title(title)
    if chains is not None:
        ch = np.asarray(chains)
        bounds = np.flatnonzero(ch[1:] != ch[:-1]) + 0.5
        for b in bounds:
            ax.axhline(b, color="k", lw=0.6)
            ax.axvline(b, color="k", lw=0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("C$_{ij}$")
    return _finish(fig, out_path)


def plot_residue_state_differences(
    diff: pd.DataFrame,
    out_path: str | Path,
    n_highlight: int = 10,
) -> Path:
    """Three-panel per-residue profile for each (state_a, state_b) pair in ``diff``:
    RMSF of both states, delta RMSF, and the RMSD between the state-mean positions.
    The ``n_highlight`` residues with the largest ``mean_rmsd`` are labelled."""
    pairs = list(dict.fromkeys(zip(diff["state_a"], diff["state_b"])))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    for k, (a, b) in enumerate(pairs):
        d = diff[(diff["state_a"] == a) & (diff["state_b"] == b)].reset_index(drop=True)
        x = np.arange(len(d))
        color = _STATE_COLORS[(k + 1) % len(_STATE_COLORS)]
        if k == 0:
            axes[0].plot(x, d[f"rmsf_{a}"], color=_STATE_COLORS[0], label=f"{a} (reference)")
        axes[0].plot(x, d[f"rmsf_{b}"], color=color, label=str(b))
        axes[1].plot(x, d["delta_rmsf"], color=color, label=f"{b} − {a}")
        axes[2].plot(x, d["mean_rmsd"], color=color, label=f"{b} vs {a}")
        hi = d.nlargest(n_highlight, "mean_rmsd")
        axes[2].scatter(hi.index, hi["mean_rmsd"], color=color, s=18, zorder=3)
        for i, r in hi.iterrows():
            axes[2].annotate(
                f"{r['residue_name']}{int(r['residue_number'])}",
                (i, r["mean_rmsd"]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7,
            )
    axes[1].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("RMSF (Å)")
    axes[1].set_ylabel("ΔRMSF (Å)")
    axes[2].set_ylabel("RMSD of state means (Å)")
    axes[2].set_xlabel("Residue")
    first = diff[(diff["state_a"] == pairs[0][0]) & (diff["state_b"] == pairs[0][1])]
    pos, labels = _residue_ticks(first["residue_number"].to_numpy(), max_ticks=20)
    axes[2].set_xticks(pos)
    axes[2].set_xticklabels(labels, rotation=90)
    for ax in axes:
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    axes[0].set_title("Per-residue differences between states")
    return _finish(fig, out_path)


def plot_contact_persistence(
    contacts: pd.DataFrame,
    out_path: str | Path,
    min_persistence: float = 0.1,
    max_residues: int = 40,
    title: str = "Ligand–residue contact persistence",
) -> Path:
    """Grouped bars of contact persistence per state for the residues whose
    persistence reaches ``min_persistence`` in any state (top ``max_residues`` by
    maximum persistence)."""
    state_cols = [
        c for c in contacts.columns if c.startswith("persistence_") and c != "persistence_all"
    ]
    cols = state_cols if state_cols else ["persistence_all"]
    d = contacts[contacts["max_persistence"] >= min_persistence].copy()
    d = d.sort_values("max_persistence", ascending=False).head(max_residues)
    d = d.sort_values(["chain", "residue_number"]).reset_index(drop=True)
    labels = [f"{r}{n}" for r, n in zip(d["residue_name"], d["residue_number"])]
    n_states = len(cols)
    width = 0.8 / n_states
    fig, ax = plt.subplots(figsize=(max(7, 0.3 * len(d) + 2), 4.5))
    x = np.arange(len(d))
    for k, c in enumerate(cols):
        ax.bar(
            x + (k - (n_states - 1) / 2) * width,
            d[c].to_numpy(),
            width=width,
            color=_STATE_COLORS[k % len(_STATE_COLORS)],
            label=c.replace("persistence_", ""),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction of frames in contact")
    cutoff = contacts.attrs.get("cutoff")
    ax.set_title(title + (f" (cutoff {cutoff:g} Å)" if cutoff else ""))
    ax.legend(title="State", fontsize=8)
    return _finish(fig, out_path)
