"""YAML + CLI merging for the analyze command."""

import pytest

from conformatlas.config import build_analysis_config, generate_example_config


def test_yaml_values_with_cli_override(tmp_path):
    topo = tmp_path / "top.pdb"
    traj = tmp_path / "run.xtc"
    topo.write_text("")
    traj.write_bytes(b"")
    cfg = tmp_path / "analyze.yaml"
    cfg.write_text(
        "topology: top.pdb\ntrajectories: [run.xtc]\ntemperature: 310\nbegin_ps: 4000\n"
        "end_ps: 16000\nfel_pcs: [1, 3]\nbins: auto\nequilibration: auto\n"
    )
    config = build_analysis_config({"structure": None, "trajectory": (), "bins": "32"}, cfg)
    assert config.temperature == 310
    assert config.begin_ps == 4000 and config.end_ps == 16000
    assert tuple(config.fel_pcs) == (1, 3)
    assert config.bins == 32  # CLI override beats YAML 'auto'
    assert config.equilibration == "auto"
    assert config.topology.name == "top.pdb"


def test_unknown_key_rejected(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("topology: a.pdb\ntrajectories: [b.xtc]\nnot_a_key: 1\n")
    with pytest.raises(ValueError, match="Unknown key"):
        build_analysis_config({}, cfg)


def test_missing_inputs_rejected(tmp_path):
    with pytest.raises(ValueError, match="structure"):
        build_analysis_config({"structure": None, "trajectory": ()}, None)


def test_templates_parse_for_both_types():
    import yaml

    for kind in ("compare", "analyze"):
        data = yaml.safe_load(generate_example_config(kind))
        assert isinstance(data, dict) and data
