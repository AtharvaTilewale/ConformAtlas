"""Unit tests for Root Mean Square Inner Product (RMSIP) calculation."""

import numpy as np
import pytest

from conformatlas.convergence import calculate_subspace_rmsip


def test_rmsip_identical_subspaces():
    """Identical orthonormal subspaces must produce RMSIP = 1.0."""
    np.random.seed(42)
    # Generate random orthonormal basis of dimension (50, 10)
    random_mat = np.random.randn(50, 10)
    q, _ = np.linalg.qr(random_mat)

    score = calculate_subspace_rmsip(q, q)
    assert pytest.approx(score, abs=1e-6) == 1.0


def test_rmsip_orthogonal_subspaces():
    """Mutually orthogonal synthetic subspaces must produce RMSIP = 0.0."""
    d = 20
    s = 5
    # Standard basis: first s vectors vs next s vectors
    eye = np.eye(d)
    u = eye[:, :s]
    v = eye[:, s : 2 * s]

    score = calculate_subspace_rmsip(u, v)
    assert pytest.approx(score, abs=1e-6) == 0.0


def test_rmsip_intermediate():
    """Partially overlapping subspaces should have intermediate RMSIP values."""
    eye = np.eye(10)
    # u shares 3 out of 5 vectors with v
    u = eye[:, [0, 1, 2, 3, 4]]
    v = eye[:, [0, 1, 2, 8, 9]]

    # Inner products matrix has 3 ones on diagonal, rest zero
    # sum of squares = 3
    # RMSIP = sqrt(3 / 5) = sqrt(0.6) ≈ 0.774596669
    expected = np.sqrt(3.0 / 5.0)
    score = calculate_subspace_rmsip(u, v)
    assert pytest.approx(score, rel=1e-5) == expected
