"""Parsing of gmx anaeig -proj output (one data set per eigenvector)."""

import numpy as np

from conformatlas.pca import _read_anaeig_projection


def test_multiset_xvg(tmp_path):
    p = tmp_path / "proj.xvg"
    p.write_text(
        '# anaeig\n@ s0 legend "vec 1"\n0 0.1\n10 0.2\n20 0.3\n&\n'
        '@ s1 legend "vec 2"\n0 1.1\n10 1.2\n20 1.3\n&\n'
    )
    t, m = _read_anaeig_projection(p)
    np.testing.assert_allclose(t, [0, 10, 20])
    assert m.shape == (3, 2)
    np.testing.assert_allclose(m[:, 1], [1.1, 1.2, 1.3])


def test_single_set_multicolumn(tmp_path):
    p = tmp_path / "proj.xvg"
    p.write_text("0 0.1 1.1 2.1\n10 0.2 1.2 2.2\n")
    t, m = _read_anaeig_projection(p)
    assert m.shape == (2, 3)
    np.testing.assert_allclose(m[1], [0.2, 1.2, 2.2])


def test_backbone_group_follows_gromacs_definition():
    """GROMACS 'Backbone' is N, CA, C; 'MainChain' adds O."""
    from conformatlas.trajectory import map_group_to_mda_selection

    assert "name N CA C" in map_group_to_mda_selection("Backbone")
    assert map_group_to_mda_selection("MainChain").startswith("backbone")
