"""End-to-end coverage of the analysis paths added in 1.0.0.

One full `analyze` run on the bundled Cα neuraminidase + zanamivir fixture
exercises MSM construction, resegment bootstrap, CV-based equilibration
detection, residue analysis, the HDF5 container and provenance; a second run
re-analyses it with ``--from``; a third repeats the PCA with the streaming
engine and must reproduce the in-memory projections.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from conformatlas.cli import main

ROOT = Path(__file__).resolve().parents[2]
H1N1 = ROOT / "tests" / "data" / "h1n1_ca"
HAS_H5PY = importlib.util.find_spec("h5py") is not None

BASE = [
    "analyze",
    "-s",
    str(H1N1 / "neuraminidase_ca.pdb"),
    "-f",
    str(H1N1 / "neuraminidase_ca.xtc"),
    "-n",
    str(H1N1 / "index.ndx"),
    "-T",
    "300",
    "--bins",
    "16",
    "--smooth-sigma",
    "1.0",
    "--quiet",
]


def _run(args, out: Path):
    res = CliRunner().invoke(main, [*args, "-o", str(out)])
    assert res.exit_code == 0, res.output[-2000:]
    return res


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    _run(
        [
            *BASE,
            "--lig",
            "ZMR",
            "--bootstrap",
            "10",
            "--bootstrap-mode",
            "resegment",
            "--equilibration-on",
            "cv",
            "--msm",
            "--msm-clusters",
            "12",
            "--msm-lag",
            "2",
            "--msm-lags",
            "1,2,4",
            "--msm-states",
            "2",
            "--residue-analysis",
            *(["--save-hdf5"] if HAS_H5PY else []),
        ],
        out,
    )
    return out


@pytest.mark.integration
def test_new_output_families_exist(full_run):
    out = full_run
    assert (out / "msm").is_dir() and any((out / "msm").iterdir())
    assert (out / "residues").is_dir()
    for p in (out / "residues").rglob("*"):
        assert " " not in p.name, f"output file name contains a space: {p}"
    assert (out / "residues" / "dccm.npy").exists()
    assert (out / "residues" / "ligand_contacts_per_state.csv").exists()
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["conformatlas_version"] == "1.0.0"
    env = meta["environment"]  # provenance.collect_environment record
    assert env.get("python") or env.get("python_version")
    assert any("numpy" in str(k).lower() for k in json.dumps(env).split('"'))
    if HAS_H5PY:
        assert (out / "results.h5").exists()


@pytest.mark.integration
def test_resegment_bootstrap_table(full_run):
    stats = pd.read_csv(full_run / "statistics" / "state_populations.csv")
    assert len(stats) >= 1
    boot_cols = [c for c in stats.columns if "ci" in c.lower() or "boot" in c.lower()]
    assert boot_cols, stats.columns.tolist()
    assert np.isfinite(stats[boot_cols].to_numpy(dtype=float)).all()


@pytest.mark.integration
def test_from_results_matches_original(full_run, tmp_path):
    out = tmp_path / "from"
    _run(["analyze", "--from", str(full_run), "--quiet"], out)
    a = pd.read_csv(full_run / "states" / "basins.csv")
    b = pd.read_csv(out / "states" / "basins.csv")
    pd.testing.assert_frame_equal(
        a.sort_values(a.columns[0]).reset_index(drop=True),
        b.sort_values(b.columns[0]).reset_index(drop=True),
        check_dtype=False,
        atol=1e-9,
    )
    fa = pd.read_csv(full_run / "states" / "frame_assignments.csv")
    fb = pd.read_csv(out / "states" / "frame_assignments.csv")
    assert (fa["state"].to_numpy() == fb["state"].to_numpy()).all()


@pytest.mark.integration
def test_streaming_engine_reproduces_memory_projections(tmp_path):
    mem = tmp_path / "mem"
    stream = tmp_path / "stream"
    _run([*BASE, "--bootstrap", "0", "--engine", "memory"], mem)
    _run([*BASE, "--bootstrap", "0", "--engine", "streaming", "--chunk-size", "20"], stream)
    pm = pd.read_csv(mem / "pca" / "projections.csv")
    ps = pd.read_csv(stream / "pca" / "projections.csv")
    for pc in ("PC1", "PC2", "PC3"):
        x, y = pm[pc].to_numpy(), ps[pc].to_numpy()
        # eigenvector sign is arbitrary
        assert np.allclose(x, y, atol=1e-6) or np.allclose(x, -y, atol=1e-6), pc
    ev_m = pd.read_csv(mem / "pca" / "eigenvalues.csv")
    ev_s = pd.read_csv(stream / "pca" / "eigenvalues.csv")
    np.testing.assert_allclose(ev_m.iloc[:, 1].to_numpy(), ev_s.iloc[:, 1].to_numpy(), rtol=1e-8)


@pytest.mark.integration
def test_dihedral_set_backbone_chi(tmp_path):
    out = tmp_path / "dpca"
    res = CliRunner().invoke(
        main,
        [
            *BASE,
            "--pca-type",
            "dihedral",
            "--dihedral-set",
            "backbone+chi",
            "--bootstrap",
            "0",
            "-o",
            str(out),
        ],
    )
    # The Cα-only fixture carries none of the torsion-defining atoms: this must be a
    # usage error (exit 2) with guidance, never a traceback.
    assert res.exit_code == 2, res.output[-1500:]
    assert "Traceback" not in res.output
    assert "--pca-type cartesian" in res.output
