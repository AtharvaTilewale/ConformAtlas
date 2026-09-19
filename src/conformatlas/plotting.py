"""Publication-quality visualization of PCA, Free Energy Landscapes, and Convergence."""

import logging
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from conformatlas.models import BasinState, ConvergenceResults, FELGrid, PCAResults

logger = logging.getLogger("conformatlas.plotting")

# Configure publication-grade styling
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Bitstream Vera Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.0,
    }
)


def _compute_smart_label_offsets(
    points: list[tuple[float, float]],
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """Compute label offset vectors (in points) that point away from neighboring minima
    and away from axis boundaries/titles to prevent any overlap."""
    offsets = []
    scale = 30.0

    for i, (xi, yi) in enumerate(points):
        rx, ry = 0.0, 0.0

        # 1. Repulsion from other state minima
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy + 1e-4
            rx += dx / dist_sq
            ry += dy / dist_sq

        # 2. Boundary repulsion (prevents pushing into title, y-axis, or borders)
        if xlim is not None:
            xmin, xmax = min(xlim), max(xlim)
            w = max(xmax - xmin, 1e-4)
            f_left = (xi - xmin) / w
            if f_left < 0.28:
                rx += 3.0 / ((f_left + 0.06) ** 2)
            f_right = (xmax - xi) / w
            if f_right < 0.28:
                rx -= 3.0 / ((f_right + 0.06) ** 2)

        if ylim is not None:
            ymin, ymax = min(ylim), max(ylim)
            h = max(ymax - ymin, 1e-4)
            f_bottom = (yi - ymin) / h
            if f_bottom < 0.25:
                ry += 3.0 / ((f_bottom + 0.06) ** 2)
            f_top = (ymax - yi) / h
            if f_top < 0.35:
                # Strong force pushing downwards away from title
                ry -= 5.0 / ((f_top + 0.06) ** 2)

        norm = np.hypot(rx, ry)
        if norm < 1e-3:
            if ylim is not None and (max(ylim) - yi) / max(max(ylim) - min(ylim), 1e-4) < 0.35:
                offsets.append((20.0, -22.0))
            else:
                alts = [(22.0, 18.0), (-24.0, 18.0), (22.0, -22.0), (-24.0, -22.0)]
                offsets.append(alts[i % len(alts)])
        else:
            dx_pt = (rx / norm) * scale
            dy_pt = (ry / norm) * scale
            # Strictly enforce downward push near top boundary
            if ylim is not None:
                ymin, ymax = min(ylim), max(ylim)
                if (ymax - yi) / max(ymax - ymin, 1e-4) < 0.22 and dy_pt > -12.0:
                    dy_pt = -22.0
            # Strictly enforce inward push near left boundary
            if xlim is not None:
                xmin, xmax = min(xlim), max(xlim)
                if (xi - xmin) / max(xmax - xmin, 1e-4) < 0.22 and dx_pt < 12.0:
                    dx_pt = 22.0
            offsets.append((float(np.round(dx_pt)), float(np.round(dy_pt))))

    return offsets


def _save_unannotated_copy(out_path: Path, unann_path: Path) -> None:
    """Ensure non-annotated figure is accessible as both _unannotated and _non_annotated."""
    try:
        non_ann_path = out_path.parent / f"{out_path.stem}_non_annotated{out_path.suffix}"
        if unann_path.exists() and unann_path != non_ann_path:
            shutil.copyfile(unann_path, non_ann_path)
    except Exception as e:
        logger.debug(f"Could not copy non-annotated figure: {e}")


# ==============================================================================
# 1. PCA Projections & Variance (Separate Figures)
# ==============================================================================


def plot_eigenvalues_and_variance(
    pca_res: PCAResults,
    out_path: str | Path,
    annotate: bool = True,
    save_unannotated: bool = True,
) -> Path:
    """Plot eigenvalue spectrum scree plot and cumulative explained variance."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7.5, 5.2))
    modes = np.arange(1, len(pca_res.eigenvalues) + 1)

    color_bar = "#2b5c8f"
    color_line = "#d95f02"

    # Bar chart for individual explained variance
    bars = ax1.bar(
        modes,
        pca_res.explained_variance,
        color=color_bar,
        alpha=0.8,
        width=0.6,
        label="Individual %",
    )
    ax1.set_xlabel("Principal Component Mode", labelpad=8)
    ax1.set_ylabel("Explained Variance (%)", color=color_bar, labelpad=8)
    ax1.tick_params(axis="y", labelcolor=color_bar)
    ax1.set_xticks(modes)
    ax1.set_xlim(0.4, len(modes) + 0.6)
    max_bar = max(pca_res.explained_variance) if len(pca_res.explained_variance) > 0 else 50
    ax1.set_ylim(0, max_bar * 1.35)
    ax1.spines["top"].set_visible(False)

    # Annotate top modes directly above bars if enabled
    if annotate:
        for i in range(min(4, len(bars))):
            val = pca_res.explained_variance[i]
            ax1.text(
                modes[i],
                val + (max_bar * 0.03),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
                weight="bold",
                color=color_bar,
            )

    # Twin axis for cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(
        modes,
        pca_res.cumulative_variance,
        color=color_line,
        marker="o",
        markersize=6,
        linewidth=2.2,
        label="Cumulative %",
    )
    ax2.set_ylabel("Cumulative Variance (%)", color=color_line, labelpad=8)
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(0, 108)
    ax2.spines["top"].set_visible(False)

    # Combined legend positioned cleanly
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9, edgecolor="#cccccc"
    )

    ax1.set_title("PCA Eigenvalue Spectrum & Cumulative Variance", pad=12)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if annotate and save_unannotated:
        unann_path = out_path.parent / f"{out_path.stem}_unannotated{out_path.suffix}"
        plot_eigenvalues_and_variance(pca_res, unann_path, annotate=False, save_unannotated=False)
        _save_unannotated_copy(out_path, unann_path)

    return out_path


def plot_pca_scatter(
    projections_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Plot standalone 2D PCA projection scatter plot colored by replicate or system."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    color_col = "replicate" if projections_df["replicate"].nunique() > 1 else "system"
    categories = projections_df[color_col].unique()

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    for idx, cat in enumerate(categories):
        sub = projections_df[projections_df[color_col] == cat]
        c = palette[idx % len(palette)]
        ax.scatter(
            sub["PC1"], sub["PC2"], label=str(cat), alpha=0.55, s=16, color=c, edgecolors="none"
        )

    ax.set_xlabel("PC1 Projection", labelpad=8)
    ax.set_ylabel("PC2 Projection", labelpad=8)
    ax.set_title("Principal Component Projections (Scatter)", pad=12)
    ax.legend(
        loc="best", frameon=True, framealpha=0.9, edgecolor="#cccccc", title=color_col.capitalize()
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pca_density(
    projections_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Plot standalone 2D PCA conformation ensemble density map (hexbin)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    hb = ax.hexbin(
        projections_df["PC1"],
        projections_df["PC2"],
        gridsize=45,
        cmap="viridis",
        mincnt=1,
        edgecolors="none",
    )
    ax.set_xlabel("PC1 Projection", labelpad=8)
    ax.set_ylabel("PC2 Projection", labelpad=8)
    ax.set_title("Conformational Ensemble Density", pad=12)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cb = fig.colorbar(hb, ax=ax, label="Frame Count per Bin", pad=0.03, aspect=20)
    cb.ax.tick_params(labelsize=9.5)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pca_projections(
    projections_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Legacy dual-view plot of PCA projections (scatter and density)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

    color_col = "replicate" if projections_df["replicate"].nunique() > 1 else "system"
    categories = projections_df[color_col].unique()

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, cat in enumerate(categories):
        sub = projections_df[projections_df[color_col] == cat]
        ax1.scatter(
            sub["PC1"],
            sub["PC2"],
            label=str(cat),
            alpha=0.5,
            s=12,
            color=palette[idx % len(palette)],
        )

    ax1.set_xlabel("PC1 Projection")
    ax1.set_ylabel("PC2 Projection")
    ax1.set_title("PCA Projections (Scatter)")
    ax1.legend(loc="best", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.35)

    hb = ax2.hexbin(
        projections_df["PC1"],
        projections_df["PC2"],
        gridsize=45,
        cmap="viridis",
        mincnt=1,
    )
    ax2.set_xlabel("PC1 Projection")
    ax2.set_ylabel("PC2 Projection")
    ax2.set_title("Conformational Density")
    fig.colorbar(hb, ax=ax2, label="Frame Count", pad=0.03)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ==============================================================================
# 2. Free Energy Landscapes (Separate Figures)
# ==============================================================================


def plot_fel_2d(
    fel_grid: FELGrid,
    states: list[BasinState] | None = None,
    out_path: str | Path = "figures/fel_2d.png",
    annotate: bool = True,
    save_unannotated: bool = True,
) -> Path:
    """Plot standalone, publication-quality 2D Free Energy Landscape with collision-free labels."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    ax.set_facecolor("#f2f2f2")  # Unoccupied bins show neutral clean grey

    X, Y = np.meshgrid(fel_grid.x_centers, fel_grid.y_centers)
    Z = fel_grid.free_energy.copy()

    valid_z = Z[~np.isnan(Z)]
    vmax = min(np.percentile(valid_z, 98), 35.0) if len(valid_z) > 0 else 25.0
    levels = np.linspace(0.0, vmax, 40)

    cf = ax.contourf(X, Y, Z, levels=levels, cmap="turbo", extend="max")
    cs = ax.contour(X, Y, Z, levels=levels[::4], colors="black", linewidths=0.5, alpha=0.55)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # Annotate states with collision avoidance if enabled
    if states and annotate:
        points = [(st.min_pc1, st.min_pc2) for st in states]
        offsets = _compute_smart_label_offsets(points, xlim=ax.get_xlim(), ylim=ax.get_ylim())

        for idx, st in enumerate(states):
            ax.plot(
                st.min_pc1,
                st.min_pc2,
                marker="*",
                markersize=14,
                color="#ffffff",
                markeredgecolor="#000000",
                markeredgewidth=1.2,
                zorder=5,
            )
            dx, dy = offsets[idx]
            ax.annotate(
                f"{st.label}\n({st.population * 100:.1f}%)",
                xy=(st.min_pc1, st.min_pc2),
                xytext=(dx, dy),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.35,rounding_size=0.3",
                    fc="#ffffff",
                    ec="#333333",
                    lw=0.9,
                    alpha=0.92,
                ),
                arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9, shrinkA=2, shrinkB=3),
                fontsize=9.5,
                weight="bold",
                ha="center" if abs(dx) < 8 else ("left" if dx > 0 else "right"),
                va="center" if abs(dy) < 8 else ("bottom" if dy > 0 else "top"),
                zorder=6,
            )

    ax.set_xlabel("PC1 Projection", labelpad=8)
    ax.set_ylabel("PC2 Projection", labelpad=8)
    ax.set_title(f"2D Free Energy Landscape (T = {fel_grid.temperature:.1f} K)", pad=12)

    cb = fig.colorbar(cf, ax=ax, label="Free Energy ΔG (kJ/mol)", pad=0.03, aspect=22)
    cb.ax.tick_params(labelsize=9.5)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if states and annotate and save_unannotated:
        unann_path = out_path.parent / f"{out_path.stem}_unannotated{out_path.suffix}"
        plot_fel_2d(
            fel_grid, states=states, out_path=unann_path, annotate=False, save_unannotated=False
        )
        _save_unannotated_copy(out_path, unann_path)

    return out_path


def plot_fel_3d(
    fel_grid: FELGrid,
    out_path: str | Path = "figures/fel_3d.png",
) -> Path:
    """Plot standalone, publication-quality 3D Free Energy Landscape perspective surface."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.2, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    X, Y = np.meshgrid(fel_grid.x_centers, fel_grid.y_centers)
    Z = fel_grid.free_energy.copy()

    valid_z = Z[~np.isnan(Z)]
    vmax = min(np.percentile(valid_z, 98), 35.0) if len(valid_z) > 0 else 25.0

    # Replace NaNs for smooth surface edge without tearing
    Z_3d = np.where(np.isnan(Z), vmax * 1.05, Z)
    surf = ax.plot_surface(
        X, Y, Z_3d, cmap="turbo", edgecolor="none", alpha=0.92, vmin=0, vmax=vmax
    )

    ax.set_zlim(0, vmax)
    ax.set_xlabel("PC1 Projection", labelpad=10, weight="bold")
    ax.set_ylabel("PC2 Projection", labelpad=10, weight="bold")
    ax.set_zlabel("ΔG (kJ/mol)", labelpad=10, weight="bold")
    ax.set_title(f"3D Free Energy Landscape (T = {fel_grid.temperature:.1f} K)", pad=16)
    ax.view_init(elev=32, azim=-60)

    cb = fig.colorbar(surf, ax=ax, shrink=0.62, aspect=15, pad=0.08, label="ΔG (kJ/mol)")
    cb.ax.tick_params(labelsize=9.5)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fel(
    fel_grid: FELGrid,
    states: list[BasinState] | None = None,
    out_path: str | Path = "figures/fel_2d_3d.png",
    annotate: bool = True,
    save_unannotated: bool = True,
) -> Path:
    """Combined 2D and 3D Free Energy Landscape visualization (legacy dual view)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.1])

    X, Y = np.meshgrid(fel_grid.x_centers, fel_grid.y_centers)
    Z = fel_grid.free_energy.copy()

    valid_z = Z[~np.isnan(Z)]
    vmax = min(np.percentile(valid_z, 98), 35.0) if len(valid_z) > 0 else 25.0
    levels = np.linspace(0.0, vmax, 40)

    # Left: 2D FEL
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#f0f0f0")

    cf = ax1.contourf(X, Y, Z, levels=levels, cmap="turbo", extend="max")
    cs = ax1.contour(X, Y, Z, levels=levels[::4], colors="black", linewidths=0.5, alpha=0.6)
    ax1.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    if states and annotate:
        points = [(st.min_pc1, st.min_pc2) for st in states]
        offsets = _compute_smart_label_offsets(points, xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
        for idx, st in enumerate(states):
            ax1.plot(
                st.min_pc1,
                st.min_pc2,
                marker="*",
                markersize=14,
                color="white",
                markeredgecolor="black",
            )
            dx, dy = offsets[idx]
            ax1.annotate(
                f"{st.label}\n({st.population * 100:.1f}%)",
                xy=(st.min_pc1, st.min_pc2),
                xytext=(dx, dy),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88, ec="gray"),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
                fontsize=9,
                weight="bold",
                ha="center" if abs(dx) < 6 else ("left" if dx > 0 else "right"),
            )

    ax1.set_xlabel("PC1 Projection")
    ax1.set_ylabel("PC2 Projection")
    ax1.set_title(f"2D Free Energy Landscape (T = {fel_grid.temperature:.1f} K)")
    fig.colorbar(cf, ax=ax1, label="Free Energy ΔG (kJ/mol)", pad=0.03)

    # Right: 3D Surface
    ax2 = fig.add_subplot(gs[1], projection="3d")
    Z_3d = np.where(np.isnan(Z), vmax * 1.05, Z)
    surf = ax2.plot_surface(
        X, Y, Z_3d, cmap="turbo", edgecolor="none", alpha=0.9, vmin=0, vmax=vmax
    )
    ax2.set_zlim(0, vmax)
    ax2.set_xlabel("PC1", labelpad=8)
    ax2.set_ylabel("PC2", labelpad=8)
    ax2.set_zlabel("ΔG (kJ/mol)", labelpad=8)
    ax2.set_title(f"3D Free Energy Landscape (T = {fel_grid.temperature:.1f} K)")
    ax2.view_init(elev=35, azim=-60)
    fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=12, label="ΔG (kJ/mol)", pad=0.08)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if states and annotate and save_unannotated:
        unann_path = out_path.parent / f"{out_path.stem}_unannotated{out_path.suffix}"
        plot_fel(
            fel_grid, states=states, out_path=unann_path, annotate=False, save_unannotated=False
        )
        _save_unannotated_copy(out_path, unann_path)

    return out_path


# ==============================================================================
# 3. Basin Segmentation (Separate Figure)
# ==============================================================================


def plot_basin_map(
    fel_grid: FELGrid,
    basin_map: np.ndarray,
    states: list[BasinState],
    out_path: str | Path = "figures/basin_map.png",
    annotate: bool = True,
    save_unannotated: bool = True,
) -> Path:
    """Plot standalone Topological Watershed Basin Map with clear state labels."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    ax.set_facecolor("#e6e6e6")

    masked_basins = np.where(basin_map == 0, np.nan, basin_map)
    im = ax.pcolormesh(
        fel_grid.x_edges,
        fel_grid.y_edges,
        masked_basins,
        cmap="tab20",
        shading="flat",
    )

    if states and annotate:
        points = [(st.min_pc1, st.min_pc2) for st in states]
        offsets = _compute_smart_label_offsets(points, xlim=ax.get_xlim(), ylim=ax.get_ylim())
        for idx, st in enumerate(states):
            ax.plot(
                st.min_pc1,
                st.min_pc2,
                marker="P",
                markersize=11,
                color="black",
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=5,
            )
            dx, dy = offsets[idx]
            ax.annotate(
                st.label,
                xy=(st.min_pc1, st.min_pc2),
                xytext=(dx, dy),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.3,rounding_size=0.3",
                    fc="white",
                    ec="black",
                    lw=0.9,
                    alpha=0.92,
                ),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9, shrinkA=2, shrinkB=3),
                fontsize=9.5,
                weight="bold",
                ha="center" if abs(dx) < 6 else ("left" if dx > 0 else "right"),
                va="center" if abs(dy) < 6 else ("bottom" if dy > 0 else "top"),
                zorder=6,
            )

    ax.set_xlabel("PC1 Projection", labelpad=8)
    ax.set_ylabel("PC2 Projection", labelpad=8)
    ax.set_title("Topological Watershed Basin Map", pad=12)

    cb = fig.colorbar(im, ax=ax, label="Basin Identifier", pad=0.03, aspect=22)
    cb.ax.tick_params(labelsize=9.5)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if states and annotate and save_unannotated:
        unann_path = out_path.parent / f"{out_path.stem}_unannotated{out_path.suffix}"
        plot_basin_map(
            fel_grid, basin_map, states, out_path=unann_path, annotate=False, save_unannotated=False
        )
        _save_unannotated_copy(out_path, unann_path)

    return out_path


def plot_basin_segmentation(
    fel_grid: FELGrid,
    basin_map: np.ndarray,
    states: list[BasinState],
    out_path: str | Path,
) -> Path:
    """Legacy dual view of FEL alongside discrete basin map."""
    plot_basin_map(fel_grid, basin_map, states, out_path)
    return Path(out_path)


# ==============================================================================
# 4. State Populations & Uncertainty (Separate Figure)
# ==============================================================================


def plot_state_populations(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame | None,
    out_path: str | Path,
    annotate: bool = True,
    save_unannotated: bool = True,
) -> Path:
    """Plot state populations with uncertainty error bars and clean non-overlapping labels."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if summary_df.empty or "State" not in summary_df.columns or len(summary_df) == 0:
        ax.text(
            0.5,
            0.5,
            "No major conformational states to display",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("Conformational State Populations")
        fig.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    states = summary_df["State"].tolist()
    means = summary_df["Mean_Population_Percent"].to_numpy()

    if "SD" in summary_df.columns:
        errors = summary_df["SD"].to_numpy()
        err_label = "Replicate SD"
    elif "Block_SD" in summary_df.columns:
        errors = summary_df["Block_SD"].to_numpy()
        err_label = "Block SD"
    else:
        errors = np.zeros_like(means)
        err_label = ""

    x_pos = np.arange(len(states))
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bar_colors = [color_palette[i % len(color_palette)] for i in range(len(states))]

    bars = ax.bar(
        x_pos,
        means,
        yerr=errors,
        capsize=6,
        error_kw={"elinewidth": 1.4, "capthick": 1.4, "ecolor": "#333333"},
        color=bar_colors,
        alpha=0.82,
        width=0.55,
        edgecolor="#222222",
        linewidth=1.0,
        label=f"Mean ± {err_label}" if err_label else "Mean Population",
    )

    # Overlay replicate dots if available
    if detailed_df is not None and "Replicate" in detailed_df.columns and len(detailed_df) > 1:
        for st_idx, st in enumerate(states):
            if st in detailed_df.columns:
                rep_vals = detailed_df[st].to_numpy()
                jitter = (
                    np.linspace(-0.08, 0.08, len(rep_vals))
                    if len(rep_vals) > 1
                    else np.array([0.0])
                )
                ax.scatter(
                    np.full_like(rep_vals, st_idx) + jitter,
                    rep_vals,
                    color="#222222",
                    s=32,
                    zorder=5,
                    edgecolors="white",
                    linewidth=0.8,
                    label="Replicates" if st_idx == 0 else "",
                )

    # Dynamic headroom to prevent text clipping and avoid empty space
    y_max = float(np.max(means + errors)) if len(means) > 0 else 50.0
    if detailed_df is not None and "Replicate" in detailed_df.columns and len(detailed_df) > 1:
        for st in states:
            if st in detailed_df.columns:
                y_max = max(y_max, float(detailed_df[st].max()))

    ylim_top = min(105.0, max(y_max * 1.35, 12.0))
    ax.set_ylim(0, ylim_top)

    # Position text cleanly above bars, error bars, and replicate points if enabled
    if annotate:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            err = errors[i] if i < len(errors) else 0.0
            top_feature = height + err
            if (
                detailed_df is not None
                and "Replicate" in detailed_df.columns
                and states[i] in detailed_df.columns
            ):
                top_feature = max(top_feature, float(detailed_df[states[i]].max()))
            y_text = top_feature + (ylim_top * 0.035)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                color="#111111",
                weight="bold",
                fontsize=10.5,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(states, weight="bold", fontsize=11)
    ax.set_ylabel("Conformational Population (%)", labelpad=8)
    ax.set_title("Conformational State Populations & Uncertainty", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if annotate and save_unannotated:
        unann_path = out_path.parent / f"{out_path.stem}_unannotated{out_path.suffix}"
        plot_state_populations(
            summary_df, detailed_df, out_path=unann_path, annotate=False, save_unannotated=False
        )
        _save_unannotated_copy(out_path, unann_path)

    return out_path


# ==============================================================================
# 5. Convergence Diagnostics (Separate Figures)
# ==============================================================================


def plot_convergence_progressive(
    conv: ConvergenceResults,
    out_path: str | Path,
    annotate: bool = True,
    save_unannotated: bool = True,
) -> Path:
    """Plot standalone Progressive Subspace Overlap (RMSIP vs trajectory fraction)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fracs = sorted(conv.progressive_overlap.keys())
    scores = [conv.progressive_overlap[f] for f in fracs]

    ax.plot(
        fracs,
        scores,
        marker="s",
        markersize=7,
        color="#2ca02c",
        linewidth=2.4,
        label="Progressive RMSIP",
    )
    ax.axhline(
        0.70, color="#d95f02", linestyle="--", linewidth=1.8, label="Heuristic threshold (0.70)"
    )

    if annotate:
        for f, s in zip(fracs, scores):
            ax.text(
                f,
                s + 0.03,
                f"{s:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                weight="bold",
                color="#1b6e1b",
            )

    ax.set_xlabel("Cumulative Trajectory Fraction (%)", labelpad=8)
    ax.set_ylabel("Subspace Overlap (RMSIP)", labelpad=8)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(min(fracs) - 5 if fracs else 0, max(fracs) + 5 if fracs else 105)
    ax.set_title("Progressive Subspace Convergence", pad=12)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if annotate and save_unannotated:
        unann_path = out_path.parent / f"{out_path.stem}_unannotated{out_path.suffix}"
        plot_convergence_progressive(
            conv, out_path=unann_path, annotate=False, save_unannotated=False
        )
        _save_unannotated_copy(out_path, unann_path)

    return out_path


def plot_convergence_cosine(
    conv: ConvergenceResults,
    out_path: str | Path,
    annotate: bool = True,
    save_unannotated: bool = True,
) -> Path:
    """Plot standalone PC Cosine Content (Diffusion Test) with threshold."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    pcs = [f"PC{k}" for k in sorted(conv.cosine_content.keys())]
    c_vals = [conv.cosine_content[k] for k in sorted(conv.cosine_content.keys())]
    bar_colors = ["#d62728" if c >= 0.5 else "#2b5c8f" for c in c_vals]

    bars = ax.bar(
        pcs, c_vals, color=bar_colors, alpha=0.82, width=0.55, edgecolor="#222222", linewidth=1.0
    )
    ax.axhline(
        0.50, color="#d62728", linestyle="--", linewidth=1.8, label="Diffusive threshold (c = 0.50)"
    )

    if annotate:
        for bar, val in zip(bars, c_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                weight="bold",
                color="#222222",
            )

    ax.set_xlabel("Principal Component", labelpad=8)
    ax.set_ylabel("Cosine Content", labelpad=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("Principal Component Cosine Content (Diffusion Test)", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if annotate and save_unannotated:
        unann_path = out_path.parent / f"{out_path.stem}_unannotated{out_path.suffix}"
        plot_convergence_cosine(conv, out_path=unann_path, annotate=False, save_unannotated=False)
        _save_unannotated_copy(out_path, unann_path)

    return out_path


def plot_convergence_stability(
    conv: ConvergenceResults,
    out_path: str | Path,
) -> Path:
    """Plot standalone State Population Stability curves over cumulative sampling."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    if conv.population_stability is not None:
        p_df = conv.population_stability
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        col_idx = 0
        max_pop = 0.0
        for col in p_df.columns:
            if col not in ["Fraction_Percent", "Frames"]:
                c = palette[col_idx % len(palette)]
                ax.plot(
                    p_df["Fraction_Percent"],
                    p_df[col],
                    marker="o",
                    markersize=6,
                    label=col,
                    linewidth=2.2,
                    color=c,
                )
                col_idx += 1
                max_pop = max(max_pop, float(p_df[col].max()))
        ax.set_xlabel("Cumulative Trajectory Fraction (%)", labelpad=8)
        ax.set_ylabel("State Population (%)", labelpad=8)
        ax.set_ylim(0, min(105.0, max(max_pop * 1.30, 15.0)))
        ax.set_title("Conformational State Population Stability", pad=12)
        ax.legend(loc="best", framealpha=0.9, edgecolor="#cccccc", title="Basin States")
        ax.grid(True, linestyle="--", alpha=0.35)
    else:
        ax.text(
            0.5,
            0.5,
            "Population stability data not available",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_title("State Population Stability Curves")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_convergence_diagnostics(
    conv: ConvergenceResults,
    out_path: str | Path,
) -> Path:
    """Legacy multi-panel convergence assessment dashboard (also outputs individual plots)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Progressive overlap
    fracs = sorted(conv.progressive_overlap.keys())
    scores = [conv.progressive_overlap[f] for f in fracs]
    ax1.plot(fracs, scores, marker="s", color="#2ca02c", linewidth=2)
    ax1.axhline(0.70, color="grey", linestyle="--", alpha=0.7, label="Heuristic threshold (0.70)")
    ax1.set_xlabel("Trajectory Fraction (%)")
    ax1.set_ylabel("Subspace RMSIP Overlap")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Progressive Subspace Convergence")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Cosine Content
    pcs = [f"PC{k}" for k in sorted(conv.cosine_content.keys())]
    c_vals = [conv.cosine_content[k] for k in sorted(conv.cosine_content.keys())]
    bar_colors = ["#d62728" if c > 0.5 else "#1f77b4" for c in c_vals]
    ax2.bar(pcs, c_vals, color=bar_colors, alpha=0.8)
    ax2.axhline(0.50, color="red", linestyle="--", label="Diffusive threshold (0.50)")
    ax2.set_ylabel("Cosine Content")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("PC Cosine Content (Diffusion Test)")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: Population Stability
    if conv.population_stability is not None:
        p_df = conv.population_stability
        for col in p_df.columns:
            if col not in ["Fraction_Percent", "Frames"]:
                ax3.plot(p_df["Fraction_Percent"], p_df[col], marker="o", label=col, linewidth=2)
        ax3.set_xlabel("Cumulative Trajectory (%)")
        ax3.set_ylabel("State Population (%)")
        ax3.set_title("State Population Stability Curves")
        ax3.legend(loc="best")
        ax3.grid(True, linestyle="--", alpha=0.4)
    else:
        ax3.text(0.5, 0.5, "Population stability data not available", ha="center", va="center")

    # Panel 4: Overall Assessment Card
    ax4.axis("off")
    eval_text = (
        f"Sampling Assessment:\n"
        f"● {conv.sampling_assessment}\n\n"
        f"Key Diagnostics:\n"
        f"• First vs Second Half RMSIP: {conv.rmsip_halves:.2f}\n"
        f"• PC1 Cosine Content: {conv.cosine_content.get(1, 0.0):.2f}\n\n"
        f"Interpretative Notes:\n"
    )
    for note in conv.assessment_notes:
        eval_text += f"• {note}\n"

    eval_text += (
        "\nNote: Convergence metrics are empirical diagnostics.\n"
        "They do not guarantee that all conformational states were explored."
    )

    ax4.text(
        0.05,
        0.95,
        eval_text,
        transform=ax4.transAxes,
        fontsize=10.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da"),
    )

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ==============================================================================
# 6. Condition Comparison (Separate Figures)
# ==============================================================================


def plot_comparison_shared_pca(
    comp_dict: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Plot standalone Shared PCA Coordinate Space scatter for WT vs Mutant."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    master_df = comp_dict["pca_results"].projections
    systems = master_df["system"].unique()

    palette = ["#1f77b4", "#e41a1c", "#4daf4a", "#984ea3"]
    for idx, s_name in enumerate(systems):
        sub = master_df[master_df["system"] == s_name]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            label=s_name,
            alpha=0.55,
            s=16,
            color=palette[idx % len(palette)],
        )

    ax.set_xlabel("Shared PC1 Projection", labelpad=8)
    ax.set_ylabel("Shared PC2 Projection", labelpad=8)
    ax.set_title("Shared PCA Coordinate Space", pad=12)
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_comparison_populations(
    comp_dict: dict[str, Any],
    out_path: str | Path,
    annotate: bool = True,
    save_unannotated: bool = True,
) -> Path:
    """Plot standalone Grouped Bar Chart of state population shifts across conditions."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    master_df = comp_dict["pca_results"].projections
    systems = list(master_df["system"].unique())
    pop_df = comp_dict["population_comparison"]

    states = pop_df["State"].tolist()
    x_pos = np.arange(len(states))
    n_sys = len(systems)
    width = 0.8 / n_sys

    palette = ["#1f77b4", "#e41a1c", "#4daf4a", "#984ea3"]

    y_max = 0.0
    for s_name in systems:
        col = f"{s_name}_Population_Percent"
        if col in pop_df.columns:
            vals = pop_df[col].to_numpy()
            if len(vals) > 0:
                y_max = max(y_max, float(np.max(vals)))
    ylim_top = min(105.0, max(y_max * 1.35, 12.0))
    ax.set_ylim(0, ylim_top)

    for idx, s_name in enumerate(systems):
        col = f"{s_name}_Population_Percent"
        if col in pop_df.columns:
            vals = pop_df[col].to_numpy()
            offset = (idx - (n_sys - 1) / 2.0) * width
            bars = ax.bar(
                x_pos + offset,
                vals,
                width * 0.92,
                label=s_name,
                color=palette[idx % len(palette)],
                alpha=0.85,
                edgecolor="#222222",
                linewidth=0.9,
            )
            if annotate:
                for bar, val in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        val + (ylim_top * 0.03),
                        f"{val:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        weight="bold",
                        color="#111111",
                    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(states, weight="bold", fontsize=11)
    ax.set_ylabel("Population (%)", labelpad=8)
    ax.set_title("Conformational Population Shifts", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if annotate and save_unannotated:
        unann_path = out_path.parent / f"{out_path.stem}_unannotated{out_path.suffix}"
        plot_comparison_populations(
            comp_dict, out_path=unann_path, annotate=False, save_unannotated=False
        )
        _save_unannotated_copy(out_path, unann_path)

    return out_path


def plot_comparison_system_fel(
    comp_dict: dict[str, Any],
    system_name: str,
    out_path: str | Path,
) -> Path:
    """Plot standalone 2D FEL for a single condition within shared PCA space."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    ax.set_facecolor("#f2f2f2")

    global_fel = comp_dict["global_fel"]
    X, Y = np.meshgrid(global_fel.x_centers, global_fel.y_centers)
    valid_fe = global_fel.free_energy[~np.isnan(global_fel.free_energy)]
    vmax = min(np.percentile(valid_fe, 98), 35.0) if len(valid_fe) > 0 else 25.0
    levels = np.linspace(0.0, vmax, 40)

    sys_fels = comp_dict["system_fels"]
    if system_name in sys_fels:
        fe = sys_fels[system_name].free_energy
        cf = ax.contourf(X, Y, fe, levels=levels, cmap="turbo", extend="max")
        cs = ax.contour(X, Y, fe, levels=levels[::4], colors="black", linewidths=0.5, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    else:
        cf = ax.contourf(X, Y, global_fel.free_energy, levels=levels, cmap="turbo", extend="max")

    ax.set_xlabel("Shared PC1 Projection", labelpad=8)
    ax.set_ylabel("Shared PC2 Projection", labelpad=8)
    ax.set_title(f"Free Energy Landscape: {system_name}", pad=12)

    cb = fig.colorbar(cf, ax=ax, label="Free Energy ΔG (kJ/mol)", pad=0.03, aspect=22)
    cb.ax.tick_params(labelsize=9.5)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_condition_comparison(
    comp_dict: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Legacy multi-panel comparative figure for WT vs Mutant in shared PCA space."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    master_df = comp_dict["pca_results"].projections
    systems = master_df["system"].unique()

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)

    # Panel 1: Shared PCA Scatter
    ax1 = fig.add_subplot(gs[0, 0])
    for s_name in systems:
        sub = master_df[master_df["system"] == s_name]
        ax1.scatter(sub["PC1"], sub["PC2"], label=s_name, alpha=0.4, s=8)
    ax1.set_xlabel("Shared PC1")
    ax1.set_ylabel("Shared PC2")
    ax1.set_title("Shared PCA Coordinate Space")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: State Population Comparison Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    pop_df = comp_dict["population_comparison"]
    states = pop_df["State"].tolist()
    x_pos = np.arange(len(states))
    width = 0.35

    s1_name = systems[0]
    s2_name = systems[1] if len(systems) > 1 else systems[0]

    ax2.bar(
        x_pos - width / 2, pop_df[f"{s1_name}_Population_Percent"], width, label=s1_name, alpha=0.8
    )
    if len(systems) > 1:
        ax2.bar(
            x_pos + width / 2,
            pop_df[f"{s2_name}_Population_Percent"],
            width,
            label=s2_name,
            alpha=0.8,
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(states, weight="bold")
    ax2.set_ylabel("Population (%)")
    ax2.set_title("Conformational Population Shifts")
    ax2.legend(loc="best")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3 & 4: Per-system 2D FELs on identical color scales
    global_fel = comp_dict["global_fel"]
    X, Y = np.meshgrid(global_fel.x_centers, global_fel.y_centers)
    valid_fe = global_fel.free_energy[~np.isnan(global_fel.free_energy)]
    vmax = min(np.percentile(valid_fe, 98), 35.0) if len(valid_fe) > 0 else 25.0
    levels = np.linspace(0.0, vmax, 35)

    sys_fels = comp_dict["system_fels"]

    ax3 = fig.add_subplot(gs[1, 0])
    fe1 = sys_fels[s1_name].free_energy
    cf1 = ax3.contourf(X, Y, fe1, levels=levels, cmap="turbo", extend="max")
    ax3.set_xlabel("Shared PC1")
    ax3.set_ylabel("Shared PC2")
    ax3.set_title(f"FEL: {s1_name}")
    fig.colorbar(cf1, ax=ax3, label="ΔG (kJ/mol)")

    ax4 = fig.add_subplot(gs[1, 1])
    if len(systems) > 1:
        fe2 = sys_fels[s2_name].free_energy
        cf2 = ax4.contourf(X, Y, fe2, levels=levels, cmap="turbo", extend="max")
        ax4.set_xlabel("Shared PC1")
        ax4.set_ylabel("Shared PC2")
        ax4.set_title(f"FEL: {s2_name}")
        fig.colorbar(cf2, ax=ax4, label="ΔG (kJ/mol)")

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
