"""End-to-end runs on a real protein: influenza A/H1N1 neuraminidase with zanamivir
(Cα atoms + ligand heavy atoms, 81 frames at 250 ps spacing, 20 ns)."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from conformatlas.cli import main

HAS_MDA = importlib.util.find_spec("MDAnalysis") is not None
DATA = Path(__file__).resolve().parents[1] / "data" / "h1n1_ca"
TOPO = DATA / "neuraminidase_ca.pdb"
TRAJ = DATA / "neuraminidase_ca.xtc"
NDX = DATA / "index.ndx"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis is required"),
]


def _run(args):
    res = CliRunner().invoke(main, args)
    assert res.exit_code == 0, res.output
    return res


def test_full_feature_run_with_index_ligand_window_and_bootstrap(tmp_path):
    out = tmp_path / "res"
    _run(
        [
            "analyze",
            "-s",
            str(TOPO),
            "-f",
            str(TRAJ),
            "-n",
            str(NDX),
            "--cov-group",
            "C-alpha",
            "--lsq-group",
            "C-alpha",
            "--lig",
            "ZMR",
            "-T",
            "300",
            "-b",
            "2000",
            "-e",
            "18000",
            "--bins",
            "12",
            "--bootstrap",
            "25",
            "--n-representatives",
            "2",
            "--representative",
            "medoid",
            "-o",
            str(out),
        ]
    )
    meta = json.loads((out / "run_metadata.json").read_text())
    proj = pd.read_csv(out / "pca" / "projections.csv")
    # 81 frames at 250 ps -> frames with 2000 <= t <= 18000 ps: 65
    assert len(proj) == 65
    assert proj["time_ps"].min() >= 2000 and proj["time_ps"].max() <= 18000
    assert meta["time_window_ps"] == [2000.0, 18000.0] or tuple(meta["time_window_ps"]) == (
        2000.0,
        18000.0,
    )
    assert meta["lig_group"] == "ZMR"

    # descriptors incl. ligand RMSD, RMSF per residue (387 residues)
    ts = pd.read_csv(out / "descriptors" / "rmsd_rg_timeseries.csv")
    assert {"RMSD", "Rg"} <= set(ts.columns)
    assert "Ligand_RMSD" in ts.columns
    assert 2.0 < ts["Ligand_RMSD"].iloc[1:].mean() < 8.0  # zanamivir after protein fit, Å
    assert 18.5 < ts["Rg"].mean() < 20.0  # Cα Rg of the NA head, Å
    rmsf = pd.read_csv(out / "descriptors" / "rmsf_residues.csv")
    assert (rmsf["residue_name"] != "ZMR").sum() == 387  # protein residues (+ ligand row)

    # uncertainty merged into the population table
    pops = pd.read_csv(out / "statistics" / "state_populations.csv")
    assert {"Bootstrap_CI_95_Lower", "Bootstrap_CI_95_Upper"} <= set(pops.columns)
    assert (out / "fel" / "free_energy_bootstrap_sd.npy").exists()

    # effective samples / equilibration reports
    eff = pd.read_csv(out / "convergence" / "effective_samples.csv")
    assert eff["g_PC1"].iloc[0] >= 1.0 and eff["N_eff"].iloc[0] <= 65
    eq = pd.read_csv(out / "convergence" / "equilibration.csv")
    assert not eq["applied"].astype(bool).any()

    # mode interpretation
    assert (out / "pca" / "modes.nmd").exists()
    assert (out / "pca" / "average.pdb").exists()
    contrib = pd.read_csv(out / "pca" / "mode_contributions_residues.csv")
    # --lig adds the ligand to the analysis group: 387 protein residues + ZMR
    assert len(contrib) == 388 and (contrib["residue_name"] == "ZMR").sum() == 1

    # medoid representatives: two structures per state
    states = pd.read_csv(out / "states" / "basins.csv")
    first = states.iloc[0]["State"].replace(" ", "_")
    reps = sorted((out / "states" / first).glob("representative*.pdb"))
    assert len(reps) >= 1
    assert (out / "states" / first / "average.pdb").exists()
    assert (out / "states" / "state_ensemble_summary.csv").exists()
    assert (out / "states" / "state_free_energies.csv").exists()
    assert (out / "kinetics" / "transition_matrix.csv").exists()
    assert (out / "report.html").exists()


def test_rmsd_rg_landscape_and_auto_equilibration(tmp_path):
    out = tmp_path / "rr"
    res = _run(
        [
            "analyze",
            "-s",
            str(TOPO),
            "-f",
            str(TRAJ),
            "--cov-group",
            "C-alpha",
            "--lsq-group",
            "C-alpha",
            "-T",
            "300",
            "--cv",
            "rmsd-rg",
            "--bins",
            "12",
            "--bootstrap",
            "0",
            "--equilibration",
            "auto",
            "-o",
            str(out),
        ]
    )
    assert "discarded" in res.output.lower()
    eq = pd.read_csv(out / "convergence" / "equilibration.csv")
    assert eq["applied"].astype(bool).all() and eq["frames_discarded"].iloc[0] > 0
    fel = np.load(out / "fel" / "free_energy.npy")
    assert fel.shape == (12, 12)
    assert np.nanmin(fel) == pytest.approx(0.0)


def test_tica_and_auto_bins(tmp_path):
    out = tmp_path / "tica"
    _run(
        [
            "analyze",
            "-s",
            str(TOPO),
            "-f",
            str(TRAJ),
            "--cov-group",
            "C-alpha",
            "--lsq-group",
            "C-alpha",
            "-T",
            "300",
            "--pca-type",
            "tica",
            "--tica-lag",
            "4",
            "--bins",
            "auto",
            "--bootstrap",
            "0",
            "-o",
            str(out),
        ]
    )
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["pca_type"] == "tica" and meta["tica_lag_frames"] == 4
    assert meta["bins_mode"].startswith("auto") and int(meta["bins"]) >= 16
    assert (out / "pca" / "eigenvalues.csv").exists()


def test_external_cv_file_with_weights(tmp_path):
    # build a CV file from the frame times: two smooth functions + unit weights
    n = 81
    t = 250.0 * np.arange(n)
    cv = tmp_path / "cv.dat"
    np.savetxt(cv, np.column_stack([t, np.sin(t / 3000.0), np.cos(t / 5000.0)]))
    w = tmp_path / "w.dat"
    np.savetxt(w, np.column_stack([t, np.ones(n)]))
    out = tmp_path / "cvout"
    _run(
        [
            "analyze",
            "-s",
            str(TOPO),
            "-f",
            str(TRAJ),
            "--cov-group",
            "C-alpha",
            "--lsq-group",
            "C-alpha",
            "-T",
            "300",
            "--cv",
            "file",
            "--cv-file",
            str(cv),
            "--cv-cols",
            "1,2",
            "--cv-labels",
            "sinT,cosT",
            "--weights-file",
            str(w),
            "--bins",
            "10",
            "--bootstrap",
            "0",
            "-o",
            str(out),
        ]
    )
    assert (out / "fel" / "free_energy.npy").exists()
    assert not (out / "kinetics").exists() or not any((out / "kinetics").iterdir())


@pytest.mark.skipif(shutil.which("gmx") is None, reason="GROMACS not installed")
def test_gromacs_pca_backend_matches_python_route(tmp_path):
    from conformatlas.examples_gen import generate_mini_example_dataset

    demo = tmp_path / "demo"
    generate_mini_example_dataset(demo)
    topo, r1, r2 = demo / "WT" / "topology.pdb", demo / "WT" / "rep1.xtc", demo / "WT" / "rep2.xtc"
    outs = {}
    for backend in ("python", "gromacs"):
        out = tmp_path / backend
        _run(
            [
                "analyze",
                "-s",
                str(topo),
                "-f",
                str(r1),
                "-f",
                str(r2),
                "-T",
                "300",
                "--pca-backend",
                backend,
                "--bins",
                "16",
                "--smooth-sigma",
                "1.0",
                "--bootstrap",
                "0",
                "-o",
                str(out),
            ]
        )
        outs[backend] = pd.read_csv(out / "statistics" / "state_populations.csv")
    a = outs["python"].sort_values(outs["python"].columns[0])
    b = outs["gromacs"].sort_values(outs["gromacs"].columns[0])
    assert len(a) == len(b) == 2
    col = [c for c in a.columns if "Mean_Population" in c][0]
    np.testing.assert_allclose(np.sort(a[col].to_numpy()), np.sort(b[col].to_numpy()), atol=3.0)
