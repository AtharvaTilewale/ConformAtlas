"""Integration tests running end-to-end analysis and comparison pipelines."""

import importlib.util

import pytest
from click.testing import CliRunner

from conformatlas.cli import main
from conformatlas.examples_gen import generate_mini_example_dataset

HAS_MDA = importlib.util.find_spec("MDAnalysis") is not None


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis is required for trajectory integration tests")
def test_end_to_end_analyze(tmp_path):
    """Run full 'conformatlas analyze' on multi-replicate mini dataset and verify outputs."""
    demo_dir = tmp_path / "mini_demo"
    generate_mini_example_dataset(demo_dir)

    topo = demo_dir / "WT" / "topology.pdb"
    rep1 = demo_dir / "WT" / "rep1.xtc"
    rep2 = demo_dir / "WT" / "rep2.xtc"
    out_res = demo_dir / "results_wt"

    runner = CliRunner()
    cmd = [
        "analyze",
        "-s",
        str(topo),
        "-f",
        str(rep1),
        "-f",
        str(rep2),
        "-T",
        "310",
        "--bins",
        "24",
        "-o",
        str(out_res),
    ]

    res = runner.invoke(main, cmd)
    assert res.exit_code == 0, f"Command failed: {res.output}"

    # Verify directory structure and outputs
    assert (out_res / "pca" / "eigenvalues.csv").exists()
    assert (out_res / "pca" / "projections.csv").exists()
    assert (out_res / "fel" / "free_energy.csv").exists()
    assert (out_res / "fel" / "free_energy.npy").exists()
    assert (out_res / "fel" / "probability.npy").exists()
    assert (out_res / "states" / "basins.csv").exists()
    assert (out_res / "states" / "frame_assignments.csv").exists()
    assert (out_res / "convergence" / "rmsip.csv").exists()
    assert (out_res / "statistics" / "state_populations.csv").exists()
    assert (out_res / "figures" / "fel_2d_3d.png").exists()
    assert (out_res / "figures" / "pca.png").exists()
    assert (out_res / "figures" / "convergence.png").exists()
    assert (out_res / "report.html").exists()
    assert (out_res / "run_metadata.json").exists()
    assert (out_res / "summary.json").exists()


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis is required for trajectory integration tests")
def test_end_to_end_compare(tmp_path):
    """Run full 'conformatlas compare' on mini dataset and verify comparison outputs."""
    demo_dir = tmp_path / "mini_demo"
    generate_mini_example_dataset(demo_dir)

    config_yaml = demo_dir / "comparison.yaml"
    out_comp = demo_dir / "results_comparison"

    runner = CliRunner()
    cmd = [
        "compare",
        "--config",
        str(config_yaml),
        "-o",
        str(out_comp),
    ]

    res = runner.invoke(main, cmd)
    assert res.exit_code == 0, f"Command failed: {res.output}"

    assert (out_comp / "pca" / "common_atom_mapping.csv").exists()
    assert (out_comp / "pca" / "projections.csv").exists()
    assert (out_comp / "comparison" / "state_population_comparison.csv").exists()
    assert (out_comp / "figures" / "condition_comparison.png").exists()
