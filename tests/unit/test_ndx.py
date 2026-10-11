"""GROMACS index-file parsing."""

import numpy as np
import pytest

from conformatlas.ndx import lookup_group, parse_ndx


def test_parse_ndx_groups_are_zero_based_and_ordered(tmp_path):
    p = tmp_path / "index.ndx"
    p.write_text(
        "[ System ]\n1 2 3 4 5 6\n[ C-alpha ]\n 2   5\n[ Empty ]\n[ Ligand ]\n6\n", encoding="utf-8"
    )
    groups = parse_ndx(p)
    assert list(groups) == ["System", "C-alpha", "Empty", "Ligand"]
    np.testing.assert_array_equal(groups["C-alpha"], [1, 4])
    np.testing.assert_array_equal(groups["Ligand"], [5])
    assert groups["Empty"].size == 0


def test_lookup_is_case_and_space_tolerant(tmp_path):
    p = tmp_path / "index.ndx"
    p.write_text("[ Protein_ZMR ]\n1 2\n[ C-alpha ]\n1\n", encoding="utf-8")
    groups = parse_ndx(p)
    assert lookup_group(groups, "c-alpha") is not None
    assert lookup_group(groups, "protein_zmr") is not None
    assert lookup_group(groups, "Backbone") is None


def test_duplicate_group_names_keep_first(tmp_path):
    p = tmp_path / "index.ndx"
    p.write_text("[ ZMR ]\n1 2 3\n[ ZMR ]\n9\n", encoding="utf-8")
    groups = parse_ndx(p)
    np.testing.assert_array_equal(groups["ZMR"], [0, 1, 2])


def test_missing_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        parse_ndx(tmp_path / "nope.ndx")
