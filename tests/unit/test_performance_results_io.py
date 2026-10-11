"""HDF5 container round trip, results-directory loading, rebin and the high-level API."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HAS_MDA = importlib.util.find_spec("MDAnalysis") is not None
HAS_H5PY = importlib.util.find_spec("h5py") is not None
ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "demo_dataset" / "WT"
H1N1 = ROOT / "tests" / "data" / "h1n1_ca"

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis is required")


def deep_equal(a, b, path="", problems=None):
    """Recursive equality for AnalysisResults (DataFrames, arrays, dataclasses, dicts)."""
    problems = [] if problems is None else problems
    if isinstance(a, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b, check_dtype=False)
        except AssertionError as e:
            problems.append((path, str(e)[:120]))
    elif isinstance(a, np.ndarray):
        ok = (
            isinstance(b, np.ndarray)
            and a.shape == b.shape
            and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
        )
        if not ok:
            problems.append((path, "array"))
    elif dataclasses.is_dataclass(a):
        if type(a) is not type(b):
            problems.append((path, type(b)))
        for f in dataclasses.fields(a):
            deep_equal(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}", problems)
    elif isinstance(a, dict):
        if set(a) != set(b):
            problems.append((path, set(a) ^ set(b)))
        for k in a:
            if k in b:
                deep_equal(a[k], b[k], f"{path}[{k!r}]", problems)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            problems.append((path, "len"))
        for i, (x, y) in enumerate(zip(a, b)):
            deep_equal(x, y, f"{path}[{i}]", problems)
    elif isinstance(a, float) and np.isnan(a):
        if not (isinstance(b, float) and np.isnan(b)):
            problems.append((path, b))
    elif isinstance(a, Path):
        if Path(b) != a:
            problems.append((path, b))
    elif a != b:
        problems.append((path, (a, b)))
    return problems


@pytest.fixture(scope="module")
def demo_results(tmp_path_factory):
    os.environ["CONFORMATLAS_NO_PROGRESS"] = "1"
    from conformatlas.models import AnalysisConfig
    from conformatlas.pipeline import run_analysis

    out = tmp_path_factory.mktemp("demo") / "res"
    cfg = AnalysisConfig(
        topology=DEMO / "topology.pdb",
        trajectories=[DEMO / "rep1.xtc", DEMO / "rep2.xtc"],
        output_dir=out,
        bins=32,
        smooth_sigma=1.0,
        n_bootstrap=15,
    )
    return cfg, run_analysis(cfg)


@pytest.mark.skipif(not HAS_H5PY, reason="h5py is required")
def test_hdf5_round_trip_restores_everything(demo_results, tmp_path):
    from conformatlas.results_io import load_results, save_results

    _, res = demo_results
    path = save_results(res, tmp_path / "r.h5")
    back = load_results(path)
    assert type(back) is type(res)
    assert deep_equal(res, back) == []
    assert type(back.extras["bootstrap"]).__name__ == "BootstrapResult"
    assert isinstance(back.convergence.progressive_overlap, dict)
    assert all(isinstance(k, int) for k in back.convergence.progressive_overlap)


@pytest.mark.skipif(not HAS_H5PY, reason="h5py is required")
def test_hdf5_refuses_newer_format(tmp_path, demo_results):
    import h5py

    from conformatlas.results_io import FORMAT_VERSION, load_results, save_results

    _, res = demo_results
    p = save_results(res, tmp_path / "v.h5")
    with h5py.File(p, "a") as h5:
        h5.attrs["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError):
        load_results(p)


def test_load_results_dir_reconstructs_run(demo_results):
    from conformatlas.results_io import load_results_dir

    cfg, res = demo_results
    back = load_results_dir(cfg.output_dir)
    np.testing.assert_allclose(
        back.pca.projections.filter(like="PC").to_numpy(),
        res.pca.projections.filter(like="PC").to_numpy(),
    )
    np.testing.assert_array_equal(back.fel.free_energy, res.fel.free_energy)
    np.testing.assert_array_equal(back.fel.occupancy_mask, res.fel.occupancy_mask)
    assert [(s.label, s.basin_id) for s in back.states] == [
        (s.label, s.basin_id) for s in res.states
    ]
    np.testing.assert_allclose(
        [s.population for s in back.states], [s.population for s in res.states]
    )
    assert (back.frame_assignments["state"] == res.frame_assignments["state"]).all()
    assert back.config.bins == 32 and back.config.smooth_sigma == 1.0
    assert back.convergence.rmsip_halves == pytest.approx(res.convergence.rmsip_halves)
    assert "kinetics" in back.extras and "barriers" in back.extras
    with pytest.raises(FileNotFoundError):
        load_results_dir(cfg.output_dir / "does_not_exist")


@pytest.mark.parametrize("bins,sigma", [(24, 0.0), (40, 1.5)])
def test_rebin_equals_fresh_run(demo_results, tmp_path, bins, sigma):
    from conformatlas.pipeline import run_analysis
    from conformatlas.results_io import load_results_dir, rebin

    cfg, res = demo_results
    fresh = run_analysis(
        dataclasses.replace(cfg, output_dir=tmp_path / "fresh", bins=bins, smooth_sigma=sigma)
    )
    for source in (res, load_results_dir(cfg.output_dir)):
        rb = rebin(source, bins, sigma)
        np.testing.assert_array_equal(rb.fel.free_energy, fresh.fel.free_energy)
        np.testing.assert_array_equal(rb.fel.probability, fresh.fel.probability)
        assert [(s.label, s.population) for s in rb.states] == [
            (s.label, s.population) for s in fresh.states
        ]
        assert (rb.frame_assignments["state"] == fresh.frame_assignments["state"]).all()
        np.testing.assert_array_equal(
            rb.extras["basin_map"], np.load(tmp_path / "fresh" / "states" / "basin_map.npy")
        )
        assert rb.config.bins == bins and rb.config.smooth_sigma == sigma
        # populations table (numeric part) equals the fresh run's
        num = rb.replicate_statistics.select_dtypes("number")
        np.testing.assert_allclose(
            num.to_numpy(), fresh.replicate_statistics[num.columns].to_numpy(), equal_nan=True
        )
    # input untouched
    assert res.config.bins == 32


# --------------------------------------------------------------------------- api
def test_api_building_blocks_match_pipeline(tmp_path):
    os.environ["CONFORMATLAS_NO_PROGRESS"] = "1"
    import matplotlib

    matplotlib.use("Agg")
    from conformatlas import api

    top, traj, ndx = (
        H1N1 / "neuraminidase_ca.pdb",
        H1N1 / "neuraminidase_ca.xtc",
        H1N1 / "index.ndx",
    )
    ens = api.load_trajectories(top, [traj], group="C-alpha", lig_group="ZMR", index_file=ndx)
    assert ens.n_frames == 81 and ens.ligand_mask.sum() == 24 and ens.dt_ps == 250.0
    pcs = api.pca(ens, n_components=10)
    land = api.landscape(pcs, temperature=300, bins=12)
    st = api.states(land)
    kin = api.kinetics(st)
    conv = api.convergence(ens, pcs, st)
    res = api.analyze(
        top,
        [traj],
        output_dir=tmp_path / "run",
        cov_group="C-alpha",
        lsq_group="C-alpha",
        lig_group="ZMR",
        index_file=ndx,
        bins=12,
        n_bootstrap=5,
    )
    np.testing.assert_allclose(res.pca.eigenvalues, pcs.eigenvalues, rtol=1e-12)
    np.testing.assert_array_equal(res.fel.free_energy, land.fel.free_energy)
    assert [(s.label, s.population) for s in res.states] == [
        (s.label, s.population) for s in st.states
    ]
    pd.testing.assert_frame_equal(res.extras["kinetics"]["dwell_summary"], kin["dwell_summary"])
    assert res.convergence.rmsip_halves == pytest.approx(conv.rmsip_halves)
    assert list(st.populations().columns)[:2] == ["state", "population"]
    with pytest.raises(TypeError):
        api.analyze(top, [traj], not_an_option=1)
    fig, ax, cf = api.plot_landscape(land, states=st.states)
    assert ax.get_xlabel().startswith("PC1")
    fig2, ax2, _ = api.plot_landscape(res, ax=ax)
    assert fig2 is fig
    pcs2 = api.pca({"a": ens.coords["Rep_1"]}, n_components=3, method="tica", lag=4)
    assert pcs2.eigenvalues.shape == (3,)
    with pytest.raises(ValueError):
        api.pca(ens, method="svd")
