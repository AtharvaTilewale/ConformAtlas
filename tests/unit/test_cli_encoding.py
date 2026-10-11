"""The CLI must not depend on the platform's default code page.

On Windows (cp1252) with redirected output, printing ΔG or ✔ raised
``UnicodeEncodeError`` in the middle of an analysis, in ``doctor`` and in
``--help``. The tests below reproduce that condition on every platform:
streams encoded as cp1252 and no UTF-8 mode / ASCII locale for file I/O.
"""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys

import pytest

from conformatlas.cli import _harden_text_streams

HAS_MDA = importlib.util.find_spec("MDAnalysis") is not None

SYMBOLS = "ΔG (kJ/mol) Å ✔ ✘ ○ • φ"


def _stream(encoding: str, tty: bool = False):
    class _Stream(io.TextIOWrapper):
        def isatty(self):  # noqa: D401
            return tty

    return _Stream(io.BytesIO(), encoding=encoding, write_through=True)


def test_control_cp1252_stream_cannot_print_the_symbols():
    with pytest.raises(UnicodeEncodeError):
        _stream("cp1252").write(SYMBOLS)


def test_redirected_stream_is_switched_to_utf8():
    s = _stream("cp1252")
    _harden_text_streams([s])
    s.write(SYMBOLS)
    assert s.buffer.getvalue().decode("utf-8") == SYMBOLS


def test_terminal_keeps_its_encoding_but_never_raises():
    s = _stream("cp1252", tty=True)
    _harden_text_streams([s])
    s.write(SYMBOLS)  # unencodable characters are replaced, not fatal
    assert s.encoding == "cp1252"
    assert s.buffer.getvalue().decode("cp1252").startswith("?G (kJ/mol)")


def test_utf8_stream_is_left_untouched():
    s = _stream("utf-8")
    _harden_text_streams([s])
    s.write(SYMBOLS)
    assert s.buffer.getvalue().decode("utf-8") == SYMBOLS


def test_non_text_streams_are_ignored():
    _harden_text_streams([object(), None])


def _run_cli(args, cwd):
    env = {
        **os.environ,
        "PYTHONIOENCODING": "cp1252",  # what a redirected Windows console uses
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        "LC_ALL": "C",  # ASCII default encoding for file I/O on POSIX
    }
    code = "import sys; from conformatlas.cli import main; sys.exit(main())"
    return subprocess.run(
        [sys.executable, "-c", code, *args], cwd=cwd, env=env, capture_output=True, timeout=600
    )


@pytest.mark.parametrize("args", [["--help"], ["analyze", "--help"], ["doctor"]])
def test_commands_survive_a_cp1252_redirected_stdout(args, tmp_path):
    res = _run_cli(args, tmp_path)
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")[-1500:]
    assert b"Traceback" not in res.stderr
    res.stdout.decode("utf-8")  # valid UTF-8, whatever PYTHONIOENCODING says


@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis is required")
def test_example_and_full_analysis_survive_a_cp1252_redirected_stdout(tmp_path):
    ex = _run_cli(["example", "-o", "ex"], tmp_path)
    assert ex.returncode == 0, ex.stderr.decode("utf-8", "replace")[-1500:]
    out = _run_cli(
        [
            "analyze",
            "-s",
            "ex/WT/topology.pdb",
            "-f",
            "ex/WT/rep1.xtc",
            "-f",
            "ex/WT/rep2.xtc",
            "-T",
            "300",
            "--bins",
            "16",
            "--bootstrap",
            "0",
            "-o",
            "ex/res",
        ],
        tmp_path,
    )
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-1500:]
    text = out.stdout.decode("utf-8")  # progress output is printed (not --quiet)
    assert "kJ/mol" in text
    assert (tmp_path / "ex" / "res" / "summary.json").exists()
