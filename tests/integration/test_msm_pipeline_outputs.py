"""MSM and re-segmentation bootstrap on the outputs of a real `conformatlas analyze` run.

The real neuraminidase fixture (81 frames) is far too short for a meaningful
MSM; this test only verifies that the MSM layer consumes the pipeline's
projections / frame assignments, handles 'Unassigned' frames, writes the msm/
tree and that the re-segmentation bootstrap runs on the saved landscape.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from conformatlas import msm as M
from conformatlas import plotting_msm as PM
from conformatlas.cli import main

HAS_MDA = importlib.util.find_spec("MDAnalysis") is not None
DATA = Path(__file__).resolve().parents[1] / "data" / "h1n1_ca"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis required")
def test_msm_on_real_pipeline_projections(tmp_path):
    out = tmp_path / "res"
    r = CliRunner().invoke(
        main,
        [
            "analyze",
            "-s",
            str(DATA / "neuraminidase_ca.pdb"),
            "-f",
            str(DATA / "neuraminidase_ca.xtc"),
            "-n",
            str(DATA / "index.ndx"),
            "-l",
            "C-alpha",
            "-c",
            "C-alpha",
            "-T",
            "300",
            "--bins",
            "12",
            "--bootstrap",
            "0",
            "-o",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    proj = pd.read_csv(out / "pca" / "projections.csv")
    assign = pd.read_csv(out / "states" / "frame_assignments.csv")
    assert len(proj) == len(assign) == 81
    res = M.build_msm(
        proj,
        ["PC1", "PC2", "PC3"],
        lag=1,
        lags=[1, 2, 3],
        n_clusters=6,
        n_metastable=2,
        basin_states=assign["state"].to_numpy(),
    )
    assert res.dt_ps == pytest.approx(250.0)
    assert res.msm.n_active >= 2
    assert res.basin_overlap is not None
    msm_dir = M.save_msm(res, out)
    assert (msm_dir / "transition_matrix.csv").exists()
    fig_dir = out / "figures"
    PM.plot_implied_timescales(res.implied_timescales, fig_dir / "msm_its.png", res.lag_report, 1)
    PM.plot_metastable_timeseries(res.frame_table, fig_dir / "msm_ts.png")
    p, grids = PM.plot_per_replicate_landscapes(
        proj, {"temperature": 300.0, "bins": 12}, fig_dir / "per_replicate.png"
    )
    assert p.exists() and len(grids) == 1
    # re-segmentation bootstrap on the saved landscape
    from conformatlas.basins import detect_energy_basins
    from conformatlas.bootstrap import bootstrap_landscape
    from conformatlas.fel import calculate_fel

    fel = calculate_fel(proj["PC1"].to_numpy(), proj["PC2"].to_numpy(), 300.0, bins=12)
    states, _, _, labels = detect_energy_basins(fel, proj, min_state_population=0.05)
    boot = bootstrap_landscape(
        proj["PC1"].to_numpy(),
        proj["PC2"].to_numpy(),
        proj["replicate"].to_numpy(),
        fel,
        labels,
        states,
        {proj["replicate"].iloc[0]: 4},
        n_boot=20,
        mode="resegment",
        resegment_kwargs={"min_state_population": 0.05},
    )
    assert boot.state_count_distribution is not None
    assert np.isclose(boot.state_count_distribution["fraction"].sum(), 1.0)
    assert len(boot.resegment_state_table) == len(states)
