"""`conformatlas example --replicates/--frames`: size options.

That the designed populations (WT 65 %, MUT 30 % State A) are recovered is checked
end to end through `conformatlas compare` by the documentation validation
(docs/tutorial_adk.md, section 7)."""

from __future__ import annotations

import importlib.util

import pytest
import yaml
from click.testing import CliRunner

from conformatlas.cli import main
from conformatlas.examples_gen import generate_mini_example_dataset

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("MDAnalysis") is None, reason="MDAnalysis is required"
)


def test_cli_writes_requested_replicates_and_frames(tmp_path):
    out = tmp_path / "ex"
    r = CliRunner().invoke(main, ["example", "-o", str(out), "--replicates", "3", "--frames", "40"])
    assert r.exit_code == 0, r.output
    import MDAnalysis as mda

    for system in ("WT", "MUT"):
        reps = sorted((out / system).glob("rep*.xtc"))
        assert [p.name for p in reps] == ["rep1.xtc", "rep2.xtc", "rep3.xtc"]
        u = mda.Universe(str(out / system / "topology.pdb"), str(reps[0]))
        assert len(u.trajectory) == 40
    cfg = yaml.safe_load((out / "comparison.yaml").read_text())
    assert [len(s["trajectories"]) for s in cfg["systems"]] == [3, 3]
    assert "3 replicates (40 frames each)" in (out / "README.md").read_text()


def test_invalid_sizes_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate_mini_example_dataset(tmp_path / "a", n_replicates=0)
    r = CliRunner().invoke(main, ["example", "-o", str(tmp_path / "b"), "--frames", "1"])
    assert r.exit_code != 0
