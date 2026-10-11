"""CV / weight file readers."""

import numpy as np
import pytest

from conformatlas.features import read_cv_file, read_weights_file


def test_read_xvg_skips_headers_and_selects_columns(tmp_path):
    p = tmp_path / "gyrate.xvg"
    p.write_text(
        '# comment\n@ title "Rg"\n@ s0 legend "Rg"\n'
        "0.0 1.5 0.1 0.2 0.3\n10.0 1.6 0.1 0.2 0.3\n20.0 1.7 0.1 0.2 0.3\n",
        encoding="utf-8",
    )
    cv, t = read_cv_file(p, columns=(1, 4))
    assert cv.shape == (3, 2)
    np.testing.assert_allclose(cv[:, 0], [1.5, 1.6, 1.7])
    np.testing.assert_allclose(cv[:, 1], 0.3)
    np.testing.assert_allclose(t, [0.0, 10.0, 20.0])


def test_read_csv_with_header_line(tmp_path):
    p = tmp_path / "cv.csv"
    p.write_text("time,d1,d2\n0,1,2\n1,3,4\n", encoding="utf-8")
    cv, t = read_cv_file(p, columns=(1, 2))
    np.testing.assert_allclose(cv, [[1, 2], [3, 4]])
    np.testing.assert_allclose(t, [0, 1])


def test_bad_column_request_raises(tmp_path):
    p = tmp_path / "cv.dat"
    p.write_text("0 1\n1 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_cv_file(p, columns=(1, 5))


def test_weights_file_uses_last_column_by_default(tmp_path):
    p = tmp_path / "w.dat"
    p.write_text("# t w\n0 2.0\n1 0.5\n2 1.0\n", encoding="utf-8")
    w = read_weights_file(p)
    np.testing.assert_allclose(w, [2.0, 0.5, 1.0])
    np.testing.assert_allclose(read_weights_file(p, column=0), [0, 1, 2])
