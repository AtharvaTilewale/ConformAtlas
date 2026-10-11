"""Extended dihedral features (chi1/chi2, nucleic acids) and the chain-aware
common-atom mapping."""

from __future__ import annotations

import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis")
from MDAnalysis.lib.distances import calc_dihedrals  # noqa: E402

from conformatlas.features import (  # noqa: E402
    compute_dihedral,
    dihedral_definitions,
    stream_backbone_dihedrals,
    stream_dihedrals,
)
from conformatlas.trajectory import build_common_atom_mapping  # noqa: E402

PEPTIDE = [
    ("GLY", ["N", "CA", "C", "O"]),
    ("LEU", ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"]),
    ("ASP", ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"]),
    ("MET", ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"]),
    ("ILE", ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"]),
    ("SER", ["N", "CA", "C", "O", "CB", "OG"]),
    ("ALA", ["N", "CA", "C", "O", "CB"]),
]
DNA = [
    ("DA", ["O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C4"]),
    ("DC", ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2"]),
]


def _build(residues, path, chains=("A",), seed=0):
    rng = np.random.default_rng(seed)
    names, resnames, resids, resindex, segindex = [], [], [], [], []
    ri = 0
    for c_i, _ in enumerate(chains):
        for r_i, (rn, atoms) in enumerate(residues):
            resnames.append(rn)
            resids.append(r_i + 1)
            segindex.append(c_i)
            for a in atoms:
                names.append(a)
                resindex.append(ri)
            ri += 1
    n = len(names)
    u = mda.Universe.empty(
        n,
        n_residues=ri,
        n_segments=len(chains),
        atom_resindex=np.array(resindex),
        residue_segindex=np.array(segindex),
        trajectory=True,
    )
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", resids)
    u.add_TopologyAttr("segid", list(chains))
    u.add_TopologyAttr("chainIDs", [chains[segindex[resindex[i]]] for i in range(n)])
    u.atoms.positions = rng.normal(size=(n, 3)) * 2.0 + np.arange(n)[:, None] * 0.7
    u.atoms.write(str(path))
    return path


def test_compute_dihedral_hand_geometry_and_mdanalysis():
    th = np.deg2rad(70.0)
    p0, p1, p2 = np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 0, 1.0])
    p3 = np.array([np.cos(th), np.sin(th), 1.0])
    assert np.rad2deg(compute_dihedral(p0, p1, p2, p3)) == pytest.approx(70.0, abs=1e-9)
    assert np.rad2deg(
        compute_dihedral(p0, p1, p2, np.array([np.cos(th), -np.sin(th), 1.0]))
    ) == pytest.approx(-70.0)
    assert np.rad2deg(compute_dihedral(p0, p1, p2, np.array([-1.0, 0, 1.0]))) == pytest.approx(
        180.0
    )
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(4, 100, 3))
    assert (
        np.abs(compute_dihedral(*pts) - calc_dihedrals(*pts)).max() < 1e-5
    )  # MDAnalysis is float32


def test_chi_definitions_match_janin_and_backbone_matches_ramachandran(tmp_path):
    from MDAnalysis.analysis.dihedrals import Dihedral, Janin, Ramachandran

    pdb = _build(PEPTIDE, tmp_path / "pep.pdb")
    u = mda.Universe(str(pdb))
    prot = u.select_atoms("protein")
    labels, groups = dihedral_definitions(prot.residues, "backbone+chi")
    kinds = [lab["angle"] for lab in labels]
    assert kinds.count("phi") == 5 and kinds.count("psi") == 5  # residues 2-6
    assert kinds.count("chi1") == 5  # LEU ASP MET ILE SER
    assert kinds.count("chi2") == 4  # LEU ASP MET ILE
    ang = Dihedral(groups).run().results.angles[0]
    ours = {(lab["residue_number"], lab["angle"]): a for lab, a in zip(labels, ang)}
    jan = Janin(prot).run()
    ja = jan.results.angles[0]
    for k, res in enumerate(jan.ag1.residues):
        for col, name in ((0, "chi1"), (1, "chi2")):
            d = (ours[(res.resid, name)] - ja[k, col] + 180) % 360 - 180
            assert abs(d) < 1e-9
    ram = Ramachandran(prot).run()
    ra = ram.results.angles[0]
    for k, res in enumerate(ram.ag2.residues):
        assert ours[(res.resid, "phi")] == pytest.approx(ra[k, 0], abs=1e-9)
        assert ours[(res.resid, "psi")] == pytest.approx(ra[k, 1], abs=1e-9)
    # SER chi1 uses OG; ILE chi2 uses CG1-CD1
    ser = [
        g
        for lab, g in zip(labels, groups)
        if lab["residue_name"] == "SER" and lab["angle"] == "chi1"
    ][0]
    assert list(ser.names) == ["N", "CA", "CB", "OG"]
    ile2 = [
        g
        for lab, g in zip(labels, groups)
        if lab["residue_name"] == "ILE" and lab["angle"] == "chi2"
    ][0]
    assert list(ile2.names) == ["CA", "CB", "CG1", "CD1"]


def test_stream_dihedrals_layout_and_backbone_labels(tmp_path):
    pdb = _build(PEPTIDE, tmp_path / "pep.pdb")
    feats, times, labels, angles = stream_dihedrals(pdb, pdb, "protein", "backbone+chi")
    n_dih = 5 + 5 + 5 + 4
    assert feats.shape == (1, 2 * n_dih) and angles.shape == (1, n_dih)
    assert list(labels["angle"].iloc[:10]) == ["phi"] * 10  # cos block then sin block
    assert list(labels["component"].iloc[:10]) == ["cos"] * 5 + ["sin"] * 5
    np.testing.assert_allclose(feats[0, :5], np.cos(np.deg2rad(angles[0, :5])), atol=1e-6)
    np.testing.assert_allclose(feats[0, 5:10], np.sin(np.deg2rad(angles[0, :5])), atol=1e-6)
    # legacy backbone function: same values and labels now point to residues 2..6
    f_old, _, lab_old = stream_backbone_dihedrals(pdb, pdb)
    f_new, _, lab_new, _ = stream_dihedrals(pdb, pdb, "protein", "backbone")
    np.testing.assert_allclose(f_old, f_new, atol=1e-6)
    assert sorted(set(lab_old["residue_number"])) == [2, 3, 4, 5, 6]
    assert lab_old[["residue_number", "angle", "component"]].equals(
        lab_new[["residue_number", "angle", "component"]]
    )
    with pytest.raises(ValueError):
        stream_dihedrals(pdb, pdb, "protein", "nucleic")
    with pytest.raises(ValueError):
        dihedral_definitions(mda.Universe(str(pdb)).residues, "sidechain")


def test_nucleic_dihedrals_against_hand_computation(tmp_path):
    pdb = _build(DNA, tmp_path / "dna.pdb")
    u = mda.Universe(str(pdb))
    feats, _, labels, angles = stream_dihedrals(pdb, pdb, "nucleic", "nucleic")
    per_res = (
        labels[labels["component"] == "cos"]
        .groupby("residue_number")["angle"]
        .apply(list)
        .to_dict()
    )
    assert per_res[1] == ["gamma", "delta", "epsilon", "zeta", "chi"]  # no P -> no alpha/beta
    assert per_res[2] == ["alpha", "beta", "gamma", "delta", "chi"]  # no next residue
    r1, r2 = u.residues[0], u.residues[1]

    def pos(res, name):
        return res.atoms.select_atoms(f"name {name}").positions[0]

    expected = {
        (1, "gamma"): (pos(r1, "O5'"), pos(r1, "C5'"), pos(r1, "C4'"), pos(r1, "C3'")),
        (1, "delta"): (pos(r1, "C5'"), pos(r1, "C4'"), pos(r1, "C3'"), pos(r1, "O3'")),
        (1, "epsilon"): (pos(r1, "C4'"), pos(r1, "C3'"), pos(r1, "O3'"), pos(r2, "P")),
        (1, "zeta"): (pos(r1, "C3'"), pos(r1, "O3'"), pos(r2, "P"), pos(r2, "O5'")),
        (1, "chi"): (pos(r1, "O4'"), pos(r1, "C1'"), pos(r1, "N9"), pos(r1, "C4")),
        (2, "alpha"): (pos(r1, "O3'"), pos(r2, "P"), pos(r2, "O5'"), pos(r2, "C5'")),
        (2, "beta"): (pos(r2, "P"), pos(r2, "O5'"), pos(r2, "C5'"), pos(r2, "C4'")),
        (2, "chi"): (pos(r2, "O4'"), pos(r2, "C1'"), pos(r2, "N1"), pos(r2, "C2")),
    }
    lab_cos = labels[labels["component"] == "cos"].reset_index(drop=True)
    for i, row in lab_cos.iterrows():
        key = (int(row["residue_number"]), row["angle"])
        if key in expected:
            hand = np.rad2deg(compute_dihedral(*expected[key]))
            assert angles[0, i] == pytest.approx(hand, abs=1e-4), key


def test_chain_aware_mapping_two_chains(tmp_path):
    a = _build(PEPTIDE, tmp_path / "a.pdb", chains=("A", "B"))
    b = _build(PEPTIDE, tmp_path / "b.pdb", chains=("B", "A"), seed=1)  # chains swapped
    idx, df = build_common_atom_mapping({"a": a, "b": b}, group="C-alpha")
    assert len(idx["a"]) == 14 and len(idx["b"]) == 14
    ua, ub = mda.Universe(str(a)), mda.Universe(str(b))
    for i, j in zip(idx["a"], idx["b"]):
        assert (ua.atoms[i].chainID, ua.atoms[i].resid) == (ub.atoms[j].chainID, ub.atoms[j].resid)
    assert set(df["chain"]) == {"A", "B"} and "key" in df.columns
    # legacy key collapses chains
    idx_l, _ = build_common_atom_mapping({"a": a, "b": b}, group="C-alpha", chain_aware=False)
    assert len(idx_l["a"]) == 7
    # different chain label sets -> automatic fallback with a warning; forcing raises
    c = _build(PEPTIDE, tmp_path / "c.pdb", chains=("X",))
    idx_f, _ = build_common_atom_mapping({"a": a, "c": c}, group="C-alpha")
    assert len(idx_f["a"]) == 7
    with pytest.raises(ValueError):
        build_common_atom_mapping({"a": a, "c": c}, group="C-alpha", chain_aware=True)
