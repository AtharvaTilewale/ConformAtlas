"""DCCM, per-residue state differences, ligand contacts, binding-site RMSF and the
residue-level figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conformatlas.plotting_residue import (
    plot_contact_persistence,
    plot_dccm,
    plot_residue_state_differences,
)
from conformatlas.residue_analysis import (
    all_state_residue_differences,
    binding_site_residues,
    binding_site_rmsf,
    collapse_to_residues,
    contact_persistence,
    dccm,
    dccm_per_state,
    ligand_residue_contacts,
    per_residue_rmsf,
    state_residue_differences,
    write_residue_outputs,
)

ROOT = Path(__file__).resolve().parents[2]
TOP = ROOT / "tests/data/h1n1_ca/neuraminidase_ca.pdb"
TRJ = ROOT / "tests/data/h1n1_ca/neuraminidase_ca.xtc"


def _atom_table(n_res, chain="A"):
    return pd.DataFrame(
        {
            "atom_index": np.arange(n_res),
            "chain": chain,
            "residue_number": np.arange(1, n_res + 1),
            "residue_name": "GLY",
            "atom_name": "CA",
        }
    )


# ----------------------------------------------------------------------------
# DCCM
# ----------------------------------------------------------------------------


def test_dccm_matches_explicit_definition_and_limits():
    rng = np.random.default_rng(0)
    nf, na = 200, 6
    x = rng.normal(size=(nf, na, 3))
    x[:, 1] = x[:, 0] * 2 + 5  # perfectly correlated with atom 0
    x[:, 2] = -x[:, 0] + 1  # perfectly anti-correlated
    c = dccm(x)
    d = x - x.mean(0)
    ref = np.array(
        [
            [
                np.mean(np.sum(d[:, i] * d[:, j], 1))
                / np.sqrt(np.mean(np.sum(d[:, i] ** 2, 1)) * np.mean(np.sum(d[:, j] ** 2, 1)))
                for j in range(na)
            ]
            for i in range(na)
        ]
    )
    np.testing.assert_allclose(c, ref, atol=1e-12)
    assert c[0, 1] == pytest.approx(1.0) and c[0, 2] == pytest.approx(-1.0)
    np.testing.assert_allclose(np.diag(c), 1.0)
    # flattened input accepted
    np.testing.assert_allclose(dccm(x.reshape(nf, -1)), c)


def test_dccm_weights_and_static_atoms():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 4, 3))
    w = rng.integers(1, 4, 50).astype(float)
    np.testing.assert_allclose(dccm(x, w), dccm(np.repeat(x, w.astype(int), axis=0)), atol=1e-12)
    x[:, 3] = 0.0
    c = dccm(x)
    assert np.isnan(c[3]).all() and np.isnan(c[:, 3]).all()
    with pytest.raises(ValueError):
        dccm(x, np.ones(49))


def test_dccm_per_state_and_residue_collapse():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 4, 3))
    states = np.array(["A"] * 30 + ["B"] * 30)
    per = dccm_per_state(x, states)
    np.testing.assert_allclose(per["A"], dccm(x[:30]))
    table = pd.DataFrame(
        {
            "chain": "A",
            "residue_number": [1, 1, 2, 2],
            "residue_name": "ALA",
            "atom_name": ["N", "CA"] * 2,
        }
    )
    m, res = collapse_to_residues(dccm(x), table)
    assert m.shape == (2, 2) and list(res["n_atoms"]) == [2, 2]
    assert m[0, 1] == pytest.approx(np.mean(dccm(x)[:2, 2:]))


# ----------------------------------------------------------------------------
# per-residue differences
# ----------------------------------------------------------------------------


def test_state_residue_differences_detects_known_shift():
    rng = np.random.default_rng(3)
    n_res, nf = 12, 400
    base = rng.normal(size=(n_res, 3)) * 5
    a = base + rng.normal(0, 0.3, (nf, n_res, 3))
    b = base.copy()
    b[[3, 4]] += np.array([2.0, 0, 0])
    b = b + rng.normal(0, 0.3, (nf, n_res, 3))
    b[:, 8] += rng.normal(0, 1.0, (nf, 3))  # extra fluctuation at residue 9
    coords = np.concatenate([a, b])
    states = np.array(["open"] * nf + ["closed"] * nf)
    df = state_residue_differences(coords, _atom_table(n_res), states, "open", "closed")
    assert list(df.columns[:4]) == ["chain", "residue_number", "residue_name", "n_atoms"]
    d = df.set_index("residue_number")
    assert d.loc[4, "mean_rmsd"] == pytest.approx(2.0, abs=0.1)
    assert d.loc[5, "centroid_displacement"] == pytest.approx(2.0, abs=0.1)
    assert d.drop([4, 5])["mean_rmsd"].max() < 0.15
    assert d.loc[9, "delta_rmsf"] > 0.8 and d.drop([9])["delta_rmsf"].abs().max() < 0.15
    assert (df["n_frames_open"] == nf).all()
    with pytest.raises(ValueError):
        state_residue_differences(coords, _atom_table(n_res), states, "open", "missing")
    allp = all_state_residue_differences(coords, _atom_table(n_res), states, reference_state="open")
    assert set(allp["state_b"]) == {"closed"}
    r = per_residue_rmsf(a, _atom_table(n_res))
    assert r["rmsf"].mean() == pytest.approx(0.3 * np.sqrt(3), rel=0.05)


# ----------------------------------------------------------------------------
# contacts on the neuraminidase fixture
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contacts():
    return ligand_residue_contacts(TOP, TRJ, "resname ZMR", "protein", cutoff=4.5, use_pbc=False)


def test_contacts_vs_brute_force(contacts):
    mda = pytest.importorskip("MDAnalysis")
    u = mda.Universe(str(TOP), str(TRJ))
    lig = u.select_atoms("resname ZMR and not name H*")
    prot = u.select_atoms("protein and not name H*")
    resids = prot.resids
    uniq = list(dict.fromkeys(resids.tolist()))
    assert contacts.contacts.shape == (len(u.trajectory), len(uniq))
    for k in (0, 40, 80):
        u.trajectory[k]
        d = np.sqrt(((lig.positions[:, None] - prot.positions[None]) ** 2).sum(-1)).min(0)
        md = np.array([d[resids == r].min() for r in uniq])
        np.testing.assert_allclose(contacts.min_distance[k], md, atol=1e-4)
        np.testing.assert_array_equal(contacts.contacts[k], md < 4.5)
    assert contacts.contacts.any()


def test_contact_persistence_and_binding_site(contacts):
    n = contacts.contacts.shape[0]
    states = np.array(["S1" if i < n // 2 else "S2" for i in range(n)])
    pers = contact_persistence(contacts, states)
    np.testing.assert_allclose(pers["persistence_all"], contacts.contacts.mean(0))
    np.testing.assert_allclose(pers["persistence_S2"], contacts.contacts[n // 2 :].mean(0))
    assert pers.attrs["states"] == ["S1", "S2"]
    site = binding_site_residues(pers, threshold=0.5)
    assert len(site) >= 1 and (site["max_persistence"] >= 0.5).all()
    # weighted persistence with weights concentrated on S1 frames equals the S1 fraction
    w = (states == "S1").astype(float)
    pw = contact_persistence(contacts, states, weights=w)
    np.testing.assert_allclose(pw["persistence_all"], pers["persistence_S1"])
    with pytest.raises(ValueError):
        ligand_residue_contacts(TOP, TRJ, "resname ZMR", "resname ZMR")


def test_binding_site_rmsf_from_fitted_coordinates(contacts):
    from conformatlas.trajectory import atom_info_for_indices, stream_trajectory_coordinates

    coords, times = stream_trajectory_coordinates(TOP, TRJ, group="C-alpha")
    import MDAnalysis as mda

    u = mda.Universe(str(TOP))
    table = atom_info_for_indices(TOP, u.select_atoms("name CA").indices)
    states = np.array(["S1" if i % 2 else "S2" for i in range(coords.shape[0])])
    pers = contact_persistence(contacts, states)
    site = binding_site_residues(pers, 0.5)
    r = binding_site_rmsf(coords, table, site, states)
    assert len(r) == len(site)
    assert {"rmsf_all", "rmsf_S1", "rmsf_S2"} <= set(r.columns)
    assert np.isfinite(r["rmsf_all"]).all() and (r["rmsf_all"] > 0).all()


# ----------------------------------------------------------------------------
# outputs and figures
# ----------------------------------------------------------------------------


def test_write_outputs_and_plots(tmp_path, contacts):
    rng = np.random.default_rng(4)
    n_res, nf = 10, 80
    coords = rng.normal(size=(2 * nf, n_res, 3)) + np.arange(n_res)[None, :, None] * 3.8
    states = np.array(["A"] * nf + ["B"] * nf)
    table = _atom_table(n_res)
    m = dccm(coords)
    diff = state_residue_differences(coords, table, states, "A", "B")
    pers = contact_persistence(contacts, np.array(["A", "B"] * 40 + ["A"]))
    written = write_residue_outputs(
        tmp_path / "residues", table, m, dccm_per_state(coords, states), diff, pers
    )
    for key in (
        "dccm",
        "dccm_npy",
        "dccm_residues",
        "dccm_A",
        "state_residue_differences",
        "ligand_contacts_per_state",
    ):
        assert written[key].exists(), key
    back = pd.read_csv(written["dccm"], index_col=0).to_numpy()
    np.testing.assert_allclose(back, m, atol=1e-12)
    assert np.load(written["dccm_npy"]).shape == (n_res, n_res)
    f1 = plot_dccm(m, table["residue_number"], tmp_path / "f" / "dccm.png", chains=table["chain"])
    f2 = plot_residue_state_differences(diff, tmp_path / "f" / "diff.png")
    f3 = plot_contact_persistence(pers, tmp_path / "f" / "contacts.png", min_persistence=0.05)
    for f in (f1, f2, f3):
        assert f.exists() and f.stat().st_size > 1000


def test_analyze_residues_end_to_end(tmp_path):
    import MDAnalysis as mda

    from conformatlas.residue_analysis import analyze_residues

    u = mda.Universe(str(TOP))
    ca = u.select_atoms("name CA").indices
    n = len(mda.Universe(str(TOP), str(TRJ)).trajectory)
    fa = pd.DataFrame(
        {
            "replicate": ["rep1"] * n,
            "frame": np.arange(n),
            "state": ["S1" if i < n // 2 else "S2" for i in range(n)],
        }
    )
    written = analyze_residues(
        TOP,
        {"rep1": TRJ},
        fa,
        tmp_path / "residues",
        ca,
        ligand_selection="resname ZMR",
        reference_state="S1",
        figure_dir=tmp_path / "figures",
    )
    for key in (
        "dccm",
        "dccm_S1",
        "state_residue_differences",
        "ligand_contacts_per_state",
        "binding_site_rmsf",
        "fig_dccm",
        "fig_state_residue_differences",
        "fig_contact_persistence",
    ):
        assert key in written and written[key].exists(), key
    diff = pd.read_csv(written["state_residue_differences"])
    assert set(diff["state_a"]) == {"S1"} and len(diff) == len(ca)
    with pytest.raises(ValueError):
        analyze_residues(TOP, {"other": TRJ}, fa, tmp_path / "x", ca)


def test_per_state_dccm_file_names_have_no_spaces(tmp_path):
    from conformatlas.residue_analysis import state_file_slug

    assert state_file_slug("State A") == "state_A"
    assert state_file_slug("Unassigned") == "Unassigned"
    assert " " not in state_file_slug("my state / 1")
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(40, 6, 3))
    states = np.array(["State A"] * 20 + ["State B"] * 20)
    table = pd.DataFrame(
        {
            "atom_index": np.arange(6),
            "residue_number": np.repeat([1, 2, 3], 2),
            "residue_name": "ALA",
            "chain": "A",
            "atom_name": ["N", "CA"] * 3,
        }
    )
    written = write_residue_outputs(
        tmp_path / "residues", table, None, dccm_per_state(coords, states), None, None
    )
    names = sorted(p.name for p in (tmp_path / "residues").iterdir())
    assert names == ["dccm_state_A.npy", "dccm_state_B.npy"]
    assert written["dccm_State A"].name == "dccm_state_A.npy"
