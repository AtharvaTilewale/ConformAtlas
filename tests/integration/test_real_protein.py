"""End-to-end runs on a real protein: E. coli adenylate kinase (tests/data/adk_ca).

214 Cα atoms from a published 1 µs equilibrium MD run (every 40th frame: 105 frames,
9.6 ns apart) plus 57 heavy atoms of Ap5A whose coordinates are synthetic (placed from
the crystal structure 1AKE by superposition on the CORE domain, 0.3 Å noise); see
examples/adk/make_adk_data.py."""

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
DATA = Path(__file__).resolve().parents[1] / "data" / "adk_ca"
TOPO = DATA / "adk_ap5_ca.pdb"
TRAJ = DATA / "adk_ap5_ca.xtc"
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
            "AP5",
            "-T",
            "300",
            "-b",
            "96000",
            "-e",
            "864000",
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
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    proj = pd.read_csv(out / "pca" / "projections.csv")
    # 105 frames at 9600 ps -> frames with 96000 <= t <= 864000 ps: indices 10..90
    assert len(proj) == 81
    assert proj["time_ps"].min() >= 96000 and proj["time_ps"].max() <= 864000
    assert meta["time_window_ps"] == [96000.0, 864000.0] or tuple(meta["time_window_ps"]) == (
        96000.0,
        864000.0,
    )
    assert meta["lig_group"] == "AP5"

    # descriptors incl. ligand RMSD, RMSF per residue (214 residues)
    ts = pd.read_csv(out / "descriptors" / "rmsd_rg_timeseries.csv")
    assert {"RMSD", "Rg"} <= set(ts.columns)
    assert "Ligand_RMSD" in ts.columns
    # Ap5A is rigid on the CORE plus 0.3 Å noise per coordinate: noise alone gives
    # sqrt(6) * 0.3 = 0.73 Å between two frames; the CORE moving relative to the all-Cα
    # fit as the lids open adds a few Å (2.6 Å measured on this window)
    assert 1.0 < ts["Ligand_RMSD"].iloc[1:].mean() < 4.0
    assert 18.5 < ts["Rg"].mean() < 21.0  # Cα Rg of ADK, Å (19.7 measured; open > closed)
    rmsf = pd.read_csv(out / "descriptors" / "rmsf_residues.csv")
    assert (rmsf["residue_name"] != "AP5").sum() == 214  # protein residues (+ ligand row)

    # uncertainty merged into the population table
    pops = pd.read_csv(out / "statistics" / "state_populations.csv")
    assert {"Bootstrap_CI_95_Lower", "Bootstrap_CI_95_Upper"} <= set(pops.columns)
    assert (out / "fel" / "free_energy_bootstrap_sd.npy").exists()

    # effective samples / equilibration reports
    eff = pd.read_csv(out / "convergence" / "effective_samples.csv")
    assert eff["g_PC1"].iloc[0] >= 1.0 and eff["N_eff"].iloc[0] <= 81
    eq = pd.read_csv(out / "convergence" / "equilibration.csv")
    assert not eq["applied"].astype(bool).any()

    # mode interpretation
    assert (out / "pca" / "modes.nmd").exists()
    assert (out / "pca" / "average.pdb").exists()
    contrib = pd.read_csv(out / "pca" / "mode_contributions_residues.csv")
    # --lig adds the ligand to the analysis group: 214 protein residues + AP5
    assert len(contrib) == 215 and (contrib["residue_name"] == "AP5").sum() == 1

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
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["pca_type"] == "tica" and meta["tica_lag_frames"] == 4
    assert meta["bins_mode"].startswith("auto") and int(meta["bins"]) >= 16
    assert (out / "pca" / "eigenvalues.csv").exists()


def test_external_cv_file_with_weights(tmp_path):
    # build a CV file from the frame times: two smooth functions + unit weights
    n = 105
    t = 9600.0 * np.arange(n)
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


def test_window_with_a_single_frame_is_a_clear_error(tmp_path):
    res = CliRunner().invoke(
        main,
        [
            "analyze",
            "-s",
            str(TOPO),
            "-f",
            str(TRAJ),
            "-n",
            str(NDX),
            "-c",
            "C-alpha",
            "-l",
            "C-alpha",
            "-T",
            "300",
            "-b",
            "2000",
            "-e",
            "18000",
            "-o",
            str(tmp_path / "one"),
        ],
    )
    assert res.exit_code != 0
    assert "Only 1 frame" in res.output and "at least 2 frames" in res.output
