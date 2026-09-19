"""Unit tests for GROMACS XPM parsing and conversion."""

import numpy as np
import pytest

from conformatlas.xpm import parse_xpm, xpm_to_dat, xpm_to_dataframe

SAMPLE_GROMACS_XPM = """/* XPM */
/* This file can be converted to EPS by the GROMACS program xpm2ps */
/* title: "Gibbs Energy Landscape" */
/* legend: "G (kJ/mol)" */
/* x-label: "PC1" */
/* y-label: "PC2" */
/* type: "Continuous" */
static char *gromacs_xpm[] = {
"3 2   3 1",
/* x-axis:  0.0 1.0 2.0 */
/* y-axis:  10.0 20.0 */
/* colors */
"a c #FFFFFF "/* "0.0" */,
"b c #7F7F7F "/* "5.5" */,
"c c #000000 "/* "11.0" */,
/* pixels */
"abc",
"cca"
};
"""

MULTI_CHAR_XPM = """/* XPM */
static char *gromacs_xpm[] = {
"2 2   2 2",
/* x-axis:  1.0 2.0 */
/* y-axis:  3.0 4.0 */
"aa c #FFFFFF "/* "0.0" */,
"bb c #000000 "/* "10.0" */,
"aabb",
"bbaa"
};
"""

CORRUPTED_UNKNOWN_CHAR_XPM = """/* XPM */
static char *gromacs_xpm[] = {
"2 2   2 1",
/* x-axis:  1.0 2.0 */
/* y-axis:  3.0 4.0 */
"a c #FFFFFF "/* "0.0" */,
"b c #000000 "/* "10.0" */,
"ab",
"aX"
};
"""


def test_parse_gromacs_xpm(tmp_path):
    """Verify parsing of GROMACS XPM header, axes, colors, and row values."""
    xpm_file = tmp_path / "test.xpm"
    xpm_file.write_text(SAMPLE_GROMACS_XPM)

    x_axis, y_axis, matrix = parse_xpm(xpm_file)

    assert len(x_axis) == 3
    assert len(y_axis) == 2
    np.testing.assert_allclose(x_axis, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(y_axis, [10.0, 20.0])

    # In XPM, row 0 in text is highest y (20.0), which corresponds to matrix[1, :]
    # row 1 in text is lowest y (10.0), which corresponds to matrix[0, :]
    # Text row 1 is "cca" -> [11.0, 11.0, 0.0]
    np.testing.assert_allclose(matrix[0, :], [11.0, 11.0, 0.0])
    # Text row 0 is "abc" -> [0.0, 5.5, 11.0]
    np.testing.assert_allclose(matrix[1, :], [0.0, 5.5, 11.0])


def test_parse_multi_char_cpp(tmp_path):
    """Verify parsing with characters-per-pixel > 1."""
    xpm_file = tmp_path / "multi.xpm"
    xpm_file.write_text(MULTI_CHAR_XPM)

    x_axis, y_axis, matrix = parse_xpm(xpm_file)
    assert matrix.shape == (2, 2)
    # Row 0 (bottom): "bbaa" -> [10.0, 0.0]
    np.testing.assert_allclose(matrix[0, :], [10.0, 0.0])
    # Row 1 (top): "aabb" -> [0.0, 10.0]
    np.testing.assert_allclose(matrix[1, :], [0.0, 10.0])


def test_unknown_character_raises_error(tmp_path):
    """Unknown characters in XPM matrix must raise ValueError instead of silently becoming 0.0."""
    xpm_file = tmp_path / "corrupt.xpm"
    xpm_file.write_text(CORRUPTED_UNKNOWN_CHAR_XPM)

    with pytest.raises(ValueError, match="Unknown symbol 'X'"):
        parse_xpm(xpm_file)


def test_xpm_to_dataframe_and_dat(tmp_path):
    """Verify conversion from XPM to DataFrame and 3-column DAT file."""
    xpm_file = tmp_path / "test.xpm"
    dat_file = tmp_path / "output.dat"
    xpm_file.write_text(SAMPLE_GROMACS_XPM)

    df = xpm_to_dataframe(xpm_file)
    assert list(df.columns) == ["x", "y", "z"]
    assert len(df) == 6  # 3 * 2

    xpm_to_dat(xpm_file, dat_file)
    assert dat_file.exists()
    loaded = np.loadtxt(str(dat_file))
    assert loaded.shape == (6, 3)
