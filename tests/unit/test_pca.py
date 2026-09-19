"""Unit tests for Cartesian PCA and shared basis projections."""

import numpy as np

from conformatlas.pca import compute_cartesian_pca, project_coordinates


def test_pca_recovers_known_dominant_motion():
    """Synthetic coordinates with dominant motion along x-axis must recover PC1 along x."""
    np.random.seed(42)
    n_frames = 200
    n_atoms = 5

    # Base coords: stationary
    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)

    # Atom 0 oscillates strongly along x (amplitude = 5.0)
    # Atom 1 oscillates mildly along y (amplitude = 1.0)
    t = np.linspace(0, 4 * np.pi, n_frames)
    coords[:, 0, 0] = 5.0 * np.sin(t)
    coords[:, 1, 1] = 1.0 * np.cos(t)

    evals, evecs, exp_var, cum_var, mean_struct = compute_cartesian_pca(
        {"rep1": coords}, weighting="frames", n_components=3
    )

    # Dominant mode must have highest eigenvalue
    assert evals[0] > evals[1]
    assert exp_var[0] > 80.0  # Dominant motion accounts for > 80%

    # Eigenvector 1 should have its largest component at atom 0, x-coord (index 0)
    pc1_vec = evecs[:, 0]
    max_atom_dim = np.argmax(np.abs(pc1_vec))
    assert max_atom_dim == 0  # Atom 0, X


def test_shared_pca_projection_consistency():
    """Two trajectories projected on the same shared basis maintain identical coordinate systems."""
    np.random.seed(123)
    n_frames = 100
    n_atoms = 4

    coords_wt = np.random.normal(0, 1, (n_frames, n_atoms, 3))
    coords_mut = np.random.normal(2, 1, (n_frames, n_atoms, 3))

    # Compute shared basis
    evals, evecs, exp_var, cum_var, mean_struct = compute_cartesian_pca(
        {"WT": coords_wt, "MUT": coords_mut},
        weighting="equal-replicate",
        n_components=5,
    )

    proj_wt = project_coordinates(coords_wt, mean_struct, evecs)
    proj_mut = project_coordinates(coords_mut, mean_struct, evecs)

    assert proj_wt.shape == (n_frames, 5)
    assert proj_mut.shape == (n_frames, 5)

    # Frame 0 of WT projected independently must match frame 0 from batch
    single_proj = project_coordinates(coords_wt[0:1], mean_struct, evecs)
    np.testing.assert_allclose(single_proj[0], proj_wt[0], atol=1e-10)


def test_replicate_weighting():
    """Equal-replicate weighting prevents an unusually long replicate from dominating mean."""
    n_atoms = 2
    # Rep1: 10 frames centered at 0.0
    rep1 = np.zeros((10, n_atoms, 3))
    # Rep2: 1000 frames centered at 10.0
    rep2 = np.full((1000, n_atoms, 3), 10.0)

    # Under equal-replicate weighting, mean should be (0 + 10) / 2 = 5.0
    _, _, _, _, mean_equal = compute_cartesian_pca(
        {"rep1": rep1, "rep2": rep2}, weighting="equal-replicate", n_components=2
    )
    np.testing.assert_allclose(mean_equal, 5.0, atol=1e-5)

    # Under frames weighting, mean should be dominated by rep2: (10*0 + 1000*10) / 1010 ≈ 9.901
    _, _, _, _, mean_frames = compute_cartesian_pca(
        {"rep1": rep1, "rep2": rep2}, weighting="frames", n_components=2
    )
    np.testing.assert_allclose(mean_frames, 10000.0 / 1010.0, atol=1e-3)
