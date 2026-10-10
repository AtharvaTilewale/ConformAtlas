"""Least-squares superposition (Kabsch) of trajectory frames onto a reference.

Cartesian PCA is only meaningful once overall translation and rotation have been
removed: otherwise the leading principal components describe how the molecule
drifts and tumbles in the box, not how it changes shape. GROMACS does this for
you inside ``gmx covar`` (``-fit``); this module provides the same step for the
MDAnalysis streaming backend.

The fit is a rigid-body transform determined on a *fit group* (``lsq_group``,
e.g. Backbone) and applied to the *analysis group* (``cov_group``). Both groups
may differ, exactly as in ``gmx covar``.
"""

from __future__ import annotations

import numpy as np


def kabsch_rotation(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return the proper rotation matrix R minimising ``|| (mobile @ R) - reference ||``.

    Both inputs must already be centred (their weighted centroids at the origin) and
    have shape (n_atoms, 3). ``weights`` optionally mass-weights the fit.

    The returned matrix acts on row vectors: ``aligned = centred_mobile @ R``.
    Reflections are excluded (det(R) = +1), so chirality is preserved.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(
            f"kabsch_rotation expects two (n_atoms, 3) arrays of equal shape, "
            f"got {mobile.shape} and {reference.shape}"
        )
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        if w.shape[0] != mobile.shape[0]:
            raise ValueError("weights must have one entry per atom")
        covariance = (mobile * w).T @ reference
    else:
        covariance = mobile.T @ reference

    u, _, vt = np.linalg.svd(covariance)
    # Correct for an improper rotation (reflection).
    d = np.sign(np.linalg.det(u @ vt))
    if d < 0:
        u[:, -1] *= -1.0
    return u @ vt


def weighted_centroid(coords: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Centroid of an (n_atoms, 3) array, optionally weighted."""
    coords = np.asarray(coords, dtype=np.float64)
    if weights is None:
        return coords.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return (coords * w[:, None]).sum(axis=0) / total


def superpose_frame(
    fit_coords: np.ndarray,
    apply_coords: np.ndarray,
    reference_fit_coords: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fit one frame onto the reference and return the transformed analysis coordinates.

    Parameters
    ----------
    fit_coords : (n_fit, 3)
        Coordinates of the fit-group atoms in this frame.
    apply_coords : (n_apply, 3)
        Coordinates of the analysis-group atoms in this frame; receive the same
        rigid-body transform that superposes ``fit_coords`` on the reference.
    reference_fit_coords : (n_fit, 3)
        Fit-group coordinates of the reference structure (not required to be centred).
    weights : (n_fit,), optional
        Per-atom fit weights (e.g. masses).

    Returns
    -------
    np.ndarray
        ``apply_coords`` after superposition, expressed in the reference frame.
    """
    fit_coords = np.asarray(fit_coords, dtype=np.float64)
    apply_coords = np.asarray(apply_coords, dtype=np.float64)
    reference_fit_coords = np.asarray(reference_fit_coords, dtype=np.float64)

    ref_centroid = weighted_centroid(reference_fit_coords, weights)
    mob_centroid = weighted_centroid(fit_coords, weights)

    rotation = kabsch_rotation(
        fit_coords - mob_centroid, reference_fit_coords - ref_centroid, weights
    )
    return (apply_coords - mob_centroid) @ rotation + ref_centroid


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Plain (unweighted) RMSD between two (n_atoms, 3) arrays, no fitting."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
