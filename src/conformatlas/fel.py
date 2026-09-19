"""Free Energy Landscape (FEL) calculation and Boltzmann inversion."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from conformatlas.models import FELGrid
from conformatlas.utils import R_GAS_CONSTANT


def calculate_fel(
    pc1: np.ndarray,
    pc2: np.ndarray,
    temperature: float = 300.0,
    bins: int = 64,
    smooth_sigma: float = 0.0,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> FELGrid:
    """Calculate 2D Free Energy Landscape from PC1 and PC2 coordinates.

    Formula:
        ΔG(x, y) = -R * T * ln( P(x, y) / P_max )
        where R = 0.008314462618 kJ / (mol * K)
        Unobserved bins (P == 0) are assigned NaN.

    Parameters
    ----------
    pc1 : np.ndarray
        Array of PC1 projections.
    pc2 : np.ndarray
        Array of PC2 projections.
    temperature : float
        Temperature in Kelvin (must be > 0).
    bins : int
        Number of histogram bins along each axis.
    smooth_sigma : float
        Standard deviation for Gaussian kernel smoothing. If 0.0, no smoothing.
    x_range : Optional[Tuple[float, float]]
        Fixed range (min, max) for PC1 axis. If None, inferred from data.
    y_range : Optional[Tuple[float, float]]
        Fixed range (min, max) for PC2 axis. If None, inferred from data.

    Returns
    -------
    FELGrid
        Structured grid containing probability and free energy arrays.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive Kelvin (got {temperature})")

    pc1 = np.asarray(pc1, dtype=np.float64)
    pc2 = np.asarray(pc2, dtype=np.float64)

    if len(pc1) == 0 or len(pc2) == 0:
        raise ValueError("Cannot calculate FEL from empty coordinate arrays.")
    if len(pc1) != len(pc2):
        raise ValueError(f"pc1 and pc2 lengths differ: {len(pc1)} vs {len(pc2)}")

    hist_range = None
    if x_range is not None and y_range is not None:
        hist_range = [list(x_range), list(y_range)]

    # np.histogram2d: row corresponds to x, column to y
    hist, x_edges, y_edges = np.histogram2d(
        pc1, pc2, bins=bins, range=hist_range, density=False
    )

    # Transpose so rows correspond to y (PC2) and cols to x (PC1)
    # matching 2D matrix indexing [y_idx, x_idx]
    hist_2d = hist.T

    total_counts = np.sum(hist_2d)
    if total_counts == 0:
        raise ValueError("Histogram is completely empty; zero occupied bins.")

    raw_prob = hist_2d / total_counts
    occupancy_mask = hist_2d > 0

    if smooth_sigma > 0.0:
        smoothed_prob = gaussian_filter(raw_prob, sigma=smooth_sigma, mode="nearest")
        # Mask out regions that had no samples if outside occupancy buffer
        # Re-normalize
        prob = smoothed_prob / np.sum(smoothed_prob)
    else:
        prob = raw_prob

    # Maximum probability for relative reference ΔG = 0 at minimum
    p_max = np.max(prob[occupancy_mask]) if np.any(occupancy_mask) else np.max(prob)
    if p_max <= 0:
        raise ValueError("Invalid probability distribution: P_max <= 0")

    # Boltzmann inversion: ΔG = -R * T * ln(P / P_max)
    free_energy = np.full_like(prob, np.nan, dtype=np.float64)

    # Calculate where prob > 0
    valid_mask = (prob > 0)
    if smooth_sigma > 0.0:
        # Avoid non-zero artifacts infinitely far from observed data
        # Keep valid mask within smoothed density >= 1e-6 * p_max
        valid_mask = valid_mask & (prob >= 1e-6 * p_max)

    free_energy[valid_mask] = -R_GAS_CONSTANT * temperature * np.log(prob[valid_mask] / p_max)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    return FELGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        probability=prob,
        free_energy=free_energy,
        temperature=temperature,
        smoothed=(smooth_sigma > 0.0),
        smooth_sigma=smooth_sigma,
        occupancy_mask=occupancy_mask,
    )


def save_fel_grid(fel_grid: FELGrid, output_dir: str | Path) -> Path:
    """Save FEL numerical arrays and CSV table to output directory.

    Parameters
    ----------
    fel_grid : FELGrid
        The computed FEL grid.
    output_dir : Union[str, Path]
        Directory under which fel/ will be created.

    Returns
    -------
    Path
        Directory containing the FEL outputs.
    """
    fel_dir = Path(output_dir) / "fel"
    fel_dir.mkdir(parents=True, exist_ok=True)

    np.save(fel_dir / "probability.npy", fel_grid.probability)
    np.save(fel_dir / "free_energy.npy", fel_grid.free_energy)
    np.save(fel_dir / "x_edges.npy", fel_grid.x_edges)
    np.save(fel_dir / "y_edges.npy", fel_grid.y_edges)

    # Create 4-column CSV: PC1, PC2, FreeEnergy_kJ_mol, Probability
    records = []
    for y_idx, y_val in enumerate(fel_grid.y_centers):
        for x_idx, x_val in enumerate(fel_grid.x_centers):
            fe = fel_grid.free_energy[y_idx, x_idx]
            pr = fel_grid.probability[y_idx, x_idx]
            records.append((x_val, y_val, fe, pr))

    df = pd.DataFrame(records, columns=["PC1", "PC2", "FreeEnergy_kJ_mol", "Probability"])
    df.to_csv(fel_dir / "free_energy.csv", index=False)

    return fel_dir
