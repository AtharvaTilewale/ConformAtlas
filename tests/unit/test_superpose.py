"""Unit tests for least-squares superposition and its use in trajectory streaming."""

import importlib.util

import numpy as np
import pytest

from conformatlas.pca import compute_cartesian_pca
from conformatlas.superpose import kabsch_rotation, rmsd, superpose_frame

HAS_MDA = importlib.util.find_spec("MDAnalysis") is not None


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_kabsch_recovers_known_rotation():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(30, 3))
    ref -= ref.mean(axis=0)
    rot_true = _random_rotation(rng)
    mobile = ref @ rot_true.T  # column-vector convention: x' = R x
    rot = kabsch_rotation(mobile, ref)
    assert np.allclose(mobile @ rot, ref, atol=1e-10)
    assert pytest.approx(np.linalg.det(rot), abs=1e-10) == 1.0


def test_kabsch_rejects_reflection():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(20, 3))
    ref -= ref.mean(axis=0)
    mirrored = ref * np.array([1.0, 1.0, -1.0])
    rot = kabsch_rotation(mirrored, ref)
    assert pytest.approx(np.linalg.det(rot), abs=1e-10) == 1.0
    # A mirror image cannot be superposed exactly by a proper rotation.
    assert rmsd(mirrored @ rot, ref) > 0.1


def test_superpose_frame_removes_rigid_body_motion_from_subset():
    rng = np.random.default_rng(2)
    all_atoms = rng.normal(size=(40, 3)) * 5.0
    fit_idx = np.arange(0, 40, 2)
    apply_idx = np.arange(1, 40, 2)
    rot = _random_rotation(rng)
    shift = np.array([12.0, -7.0, 3.0])
    moved = all_atoms @ rot.T + shift

    aligned = superpose_frame(moved[fit_idx], moved[apply_idx], all_atoms[fit_idx])
    assert rmsd(aligned, all_atoms[apply_idx]) < 1e-9


def test_pca_on_fitted_frames_ignores_tumbling():
    """Pure rigid-body motion must carry (almost) no variance once frames are fitted."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(25, 3)) * 4.0
    frames_raw, frames_fit = [], []
    for _ in range(60):
        pos = base @ _random_rotation(rng).T + rng.uniform(-10, 10, size=3)
        frames_raw.append(pos)
        frames_fit.append(superpose_frame(pos, pos, base))
    ev_raw, *_ = compute_cartesian_pca({"r": np.array(frames_raw)}, n_components=3)
    ev_fit, *_ = compute_cartesian_pca({"r": np.array(frames_fit)}, n_components=3)
    assert ev_raw[0] > 1.0
    assert ev_fit[0] < 1e-8


@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis required")
def test_stream_trajectory_coordinates_fit_flag(tmp_path):
    from conformatlas.examples_gen import generate_mini_example_dataset
    from conformatlas.trajectory import stream_trajectory_coordinates

    generate_mini_example_dataset(tmp_path)
    topo = tmp_path / "WT" / "topology.pdb"
    traj = tmp_path / "WT" / "rep1.xtc"

    raw, _ = stream_trajectory_coordinates(topo, traj, group="Backbone", fit=False)
    fitted, times = stream_trajectory_coordinates(
        topo, traj, group="Backbone", fit=True, fit_group="Backbone"
    )
    assert raw.shape == fitted.shape == (100, 15, 3)  # GROMACS Backbone = N, CA, C (5 residues)
    assert len(times) == 100

    # The demo trajectory tumbles freely: centroids wander by Angstroms before
    # fitting and are pinned to the reference afterwards.
    raw_centroid_spread = np.std(raw.mean(axis=1), axis=0).max()
    fit_centroid_spread = np.std(fitted.mean(axis=1), axis=0).max()
    assert raw_centroid_spread > 2.0
    assert fit_centroid_spread < 0.2

    # Fitting must shrink the dominant Cartesian variance by a large factor.
    ev_raw, *_ = compute_cartesian_pca({"r": raw}, n_components=1)
    ev_fit, *_ = compute_cartesian_pca({"r": fitted}, n_components=1)
    assert ev_fit[0] < 0.2 * ev_raw[0]
