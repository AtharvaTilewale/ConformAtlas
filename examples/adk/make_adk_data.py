"""Build the adenylate kinase (ADK) example data shipped with ConformAtlas.

Source (only the reduced files written here are redistributed):

* Equilibrium MD of apo E. coli adenylate kinase, 1.004 µs on Anton, CHARMM27, explicit
  water and ions, NPT at 300 K and 1 bar, protein only, 4,187 frames every 240 ps,
  superposed on the CORE domain (the deposit's description). The starting structure is
  not stated; the file name (1ake_007) and the near-closed first frame indicate 1AKE.
  S. L. Seyler and O. Beckstein, "Molecular dynamics trajectory for benchmarking
  MDAnalysis", figshare (2017), doi:10.6084/m9.figshare.5108170, licence CC BY 4.0.
  Files ``adk4AKE.psf`` and ``1ake_007-nowater-core-dt240ps.dcd``.
* Crystal structures 1AKE (closed, with the inhibitor Ap5A) and 4AKE (open) from the
  RCSB PDB (public domain, CC0).

Written:

``examples/adk/`` (tutorial data)
    ``adk_ca.pdb``          Cα atoms of the 214 residues (frame 0), topology
    ``adk_ca.xtc``          all 4,187 frames, Cα only, time in ps
    ``index.ndx``           System, C-alpha, CORE, NMP, LID
    ``domain_angles.dat``   time (ps), θ_NMP, θ_LID (degrees) from the backbone of the
                            full trajectory (Beckstein et al. 2009 definitions)
    ``crystal/1ake_A_ca.pdb``, ``crystal/4ake_A_ca.pdb``  Cα of chain A, residues 1-214
    ``crystal/crystal_angles.json``  θ_NMP, θ_LID of 1AKE and 4AKE (same definition)

``tests/data/adk_ca/`` (regression fixture)
    ``adk_ap5_ca.pdb`` / ``.xtc``  Cα (214) + Ap5A heavy atoms (57, residue AP5),
                            every 40th frame (105 frames, 9.6 ns apart). The Ap5A
                            coordinates are SYNTHETIC: the 1AKE ligand is placed in
                            every frame by superposing the 1AKE CORE Cα atoms on the
                            frame's CORE Cα atoms, plus 0.3 Å Gaussian noise per atom
                            (seed 0). The protein coordinates are the real MD frames.
    ``index.ndx``           System, Protein, C-alpha, AP5, CORE, NMP, LID
    ``4ake_A.pdb``          protein heavy atoms of 4AKE chain A (all-atom structure for
                            dihedral checks)

Usage::

    python examples/adk/make_adk_data.py --psf adk4AKE.psf \\
        --dcd 1ake_007-nowater-core-dt240ps.dcd --pdb-dir DIR_WITH_1AKE_4AKE

``--download DIR`` fetches the four source files (figshare, RCSB) into DIR first; the
figshare md5 sums are always checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FIGSHARE_ARTICLE = 5108170
MD5 = {
    "adk4AKE.psf": "c6f6b0711709cdfebe4620191b6d8e23",
    "1ake_007-nowater-core-dt240ps.dcd": "8790cbd6810dc7c591e135d86442b372",
}
DT_PS = 240.0
# Domains of E. coli ADK (NMP-binding and LID domains; everything else is CORE)
NMP = (30, 59)
LID = (122, 159)
# Angle definitions (Beckstein et al. 2009, J. Mol. Biol. 394, 160): residue ranges whose
# backbone centres define the vertex and arms of theta_NMP and theta_LID
ANGLE_RES = {
    "core_a": (90, 100),
    "hinge": (115, 125),
    "nmp": (35, 55),
    "core_b": (179, 185),
    "lid": (125, 153),
}
FIXTURE_STRIDE = 40
LIGAND_NOISE_A = 0.3


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    meta = json.load(
        urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}")
    )
    for f in meta["files"]:
        if f["name"] in MD5 and not (target / f["name"]).exists():
            print("downloading", f["name"])
            urllib.request.urlretrieve(f["download_url"], target / f["name"])
    for pdb in ("1AKE", "4AKE"):
        if not (target / f"{pdb}.pdb").exists():
            urllib.request.urlretrieve(
                f"https://files.rcsb.org/download/{pdb}.pdb", target / f"{pdb}.pdb"
            )


def angle(vertex: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v1, v2 = a - vertex, b - vertex
    c = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def domain_angles(xyz: np.ndarray, resid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """theta_NMP, theta_LID from backbone coordinates ``xyz`` (..., n_atoms, 3)."""
    c = {
        k: xyz[..., (resid >= lo) & (resid <= hi), :].mean(axis=-2)
        for k, (lo, hi) in ANGLE_RES.items()
    }
    return (
        angle(c["core_a"], c["hinge"], c["nmp"]),
        angle(c["hinge"], c["core_b"], c["lid"]),
    )


def pdb_chain_a(path: Path, names: tuple[str, ...] | None, hetero: str | None = None):
    """Heavy-atom ATOM records (or HETATM records of residue ``hetero``) of chain A,
    first alternate location only, protein residues 1-214. ``names=None`` keeps every
    heavy atom."""
    out = []
    for ln in path.read_text().splitlines():
        if len(ln) < 54 or ln[21] != "A":
            continue
        rec = ln[:6].strip()
        if hetero is None and rec != "ATOM":
            continue
        if hetero is not None and (rec != "HETATM" or ln[17:20].strip() != hetero):
            continue
        if ln[16] not in (" ", "A"):
            continue
        name = ln[12:16].strip()
        element = ln[76:78].strip() or name[0]
        if element == "H":
            continue
        if names is not None and name not in names:
            continue
        resid = int(ln[22:26])
        if hetero is None and not 1 <= resid <= 214:
            continue
        out.append(
            dict(
                name=name,
                resname=ln[17:20].strip(),
                resid=resid,
                element=element,
                xyz=(float(ln[30:38]), float(ln[38:46]), float(ln[46:54])),
            )
        )
    return out


def write_pdb(path: Path, atoms: list[dict], remark: str) -> None:
    lines = [f"REMARK   1 {remark}"]
    for i, a in enumerate(atoms, 1):
        rec = "HETATM" if a.get("het") else "ATOM  "
        nm = a["name"]
        nm = f" {nm:<3}" if len(nm) < 4 else nm
        x, y, z = a["xyz"]
        lines.append(
            f"{rec}{i:5d} {nm:<4} {a['resname']:>3} A{a['resid']:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {a['element']:>2}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def write_ndx(path: Path, groups: dict[str, np.ndarray]) -> None:
    out = []
    for name, idx in groups.items():
        out.append(f"[ {name} ]")
        idx = np.asarray(idx) + 1
        for k in range(0, len(idx), 15):
            out.append(" ".join(f"{i:4d}" for i in idx[k : k + 15]))
    path.write_text("\n".join(out) + "\n")


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation R and centroids such that (x - cm) @ R + ct maps mobile onto target."""
    cm, ct = mobile.mean(0), target.mean(0)
    h = (mobile - cm).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return r, cm, ct


def _with_frame0(atoms: list[dict], pdb: Path, xtc: Path) -> list[dict]:
    """Copy of ``atoms`` with the coordinates of frame 0 as stored in ``xtc``."""
    import MDAnalysis as mda

    u = mda.Universe(str(pdb), str(xtc))
    pos = u.trajectory[0].positions.astype(np.float64)
    return [dict(a, xyz=tuple(pos[i])) for i, a in enumerate(atoms)]


def domain_groups(resid: np.ndarray) -> dict[str, np.ndarray]:
    nmp = (resid >= NMP[0]) & (resid <= NMP[1])
    lid = (resid >= LID[0]) & (resid <= LID[1])
    return {"CORE": np.where(~nmp & ~lid)[0], "NMP": np.where(nmp)[0], "LID": np.where(lid)[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--psf", type=Path)
    ap.add_argument("--dcd", type=Path)
    ap.add_argument("--pdb-dir", type=Path)
    ap.add_argument("--download", type=Path, help="fetch the source files into this directory")
    ap.add_argument("--examples-out", type=Path, default=HERE)
    ap.add_argument("--fixture-out", type=Path, default=ROOT / "tests" / "data" / "adk_ca")
    args = ap.parse_args()
    if args.download:
        download(args.download)
        args.psf = args.psf or args.download / "adk4AKE.psf"
        args.dcd = args.dcd or args.download / "1ake_007-nowater-core-dt240ps.dcd"
        args.pdb_dir = args.pdb_dir or args.download
    for p in (args.psf, args.dcd):
        got = md5(p)
        if got != MD5[p.name]:
            raise SystemExit(f"md5 mismatch for {p}: {got}")

    import MDAnalysis as mda

    u = mda.Universe(str(args.psf), str(args.dcd))
    ca = u.select_atoms("name CA")
    bb = u.select_atoms("name N CA C")
    assert len(ca) == 214 and u.trajectory.n_frames == 4187, (len(ca), u.trajectory.n_frames)
    xyz_ca = u.trajectory.timeseries(ca, order="fac").astype(np.float64)
    xyz_bb = u.trajectory.timeseries(bb, order="fac").astype(np.float64)
    boxes = np.array([ts.dimensions for ts in u.trajectory], dtype=np.float64)  # NPT box per frame
    n = len(xyz_ca)
    times = np.arange(n) * DT_PS
    resid = ca.resids
    resn = ca.resnames

    ex = args.examples_out
    (ex / "crystal").mkdir(parents=True, exist_ok=True)
    ca_atoms = [
        dict(name="CA", resname=resn[i], resid=int(resid[i]), element="C", xyz=tuple(xyz_ca[0, i]))
        for i in range(214)
    ]
    write_pdb(
        ex / "adk_ca.pdb",
        ca_atoms,
        "ADK apo equilibrium MD, CA only (Seyler & Beckstein 2017, "
        "doi:10.6084/m9.figshare.5108170, CC BY 4.0)",
    )
    top = mda.Universe(str(ex / "adk_ca.pdb"))
    with mda.Writer(str(ex / "adk_ca.xtc"), n_atoms=214) as w:
        for f in range(n):
            top.atoms.positions = xyz_ca[f].astype(np.float32)
            top.trajectory.ts.dimensions = boxes[f]
            top.trajectory.ts.time = float(times[f])
            w.write(top.atoms)
    # the topology carries exactly the (XTC-rounded) coordinates of frame 0
    write_pdb(
        ex / "adk_ca.pdb",
        _with_frame0(ca_atoms, ex / "adk_ca.pdb", ex / "adk_ca.xtc"),
        "ADK apo equilibrium MD, CA only, frame 0 (Seyler & Beckstein 2017, "
        "doi:10.6084/m9.figshare.5108170, CC BY 4.0)",
    )
    groups = {"System": np.arange(214), "C-alpha": np.arange(214), **domain_groups(resid)}
    write_ndx(ex / "index.ndx", groups)

    th_nmp, th_lid = domain_angles(xyz_bb, bb.resids)
    with open(ex / "domain_angles.dat", "w") as fh:
        fh.write(
            "# ADK domain angles (degrees) from the backbone (N, CA, C) of the full trajectory\n"
            "# theta_NMP: angle(centre res 115-125, centre res 90-100, centre res 35-55)\n"
            "# theta_LID: angle(centre res 179-185, centre res 115-125, centre res 125-153)\n"
            "# time_ps theta_NMP theta_LID\n"
        )
        for t, a, b in zip(times, th_nmp, th_lid):
            fh.write(f"{t:10.1f} {a:9.4f} {b:9.4f}\n")

    cryst = {}
    for pdb in ("1AKE", "4AKE"):
        src = args.pdb_dir / f"{pdb}.pdb"
        atoms = pdb_chain_a(src, ("CA",))
        assert len(atoms) == 214, (pdb, len(atoms))
        write_pdb(
            ex / "crystal" / f"{pdb.lower()}_A_ca.pdb",
            atoms,
            f"PDB {pdb} chain A, CA atoms of residues 1-214 (RCSB PDB, CC0)",
        )
        bba = pdb_chain_a(src, ("N", "CA", "C"))
        bxyz = np.array([a["xyz"] for a in bba])
        bres = np.array([a["resid"] for a in bba])
        a, b = domain_angles(bxyz, bres)
        cryst[pdb] = {"theta_NMP": round(float(a), 2), "theta_LID": round(float(b), 2)}
    (ex / "crystal" / "crystal_angles.json").write_text(json.dumps(cryst, indent=2) + "\n")

    # ---------------- regression fixture: CA + synthetic Ap5A
    fx = args.fixture_out
    fx.mkdir(parents=True, exist_ok=True)
    src1 = args.pdb_dir / "1AKE.pdb"
    lig = pdb_chain_a(src1, None, hetero="AP5")
    assert len(lig) == 57, len(lig)
    lig_xyz = np.array([a["xyz"] for a in lig])
    ca1 = np.array([a["xyz"] for a in pdb_chain_a(src1, ("CA",))])
    core = domain_groups(resid)["CORE"]
    rng = np.random.default_rng(0)
    frames = np.arange(0, n, FIXTURE_STRIDE)
    fix_xyz = np.empty((len(frames), 214 + 57, 3))
    for k, f in enumerate(frames):
        r, cm, ct = kabsch(ca1[core], xyz_ca[f, core])
        fix_xyz[k, :214] = xyz_ca[f]
        fix_xyz[k, 214:] = (lig_xyz - cm) @ r + ct + rng.normal(0, LIGAND_NOISE_A, lig_xyz.shape)
    atoms = [dict(a, xyz=tuple(fix_xyz[0, i])) for i, a in enumerate(ca_atoms)] + [
        dict(a, resid=215, het=True, xyz=tuple(fix_xyz[0, 214 + i])) for i, a in enumerate(lig)
    ]
    write_pdb(
        fx / "adk_ap5_ca.pdb",
        atoms,
        "ADK CA from MD (Seyler & Beckstein 2017, CC BY 4.0) + SYNTHETIC Ap5A placed from 1AKE",
    )
    ftop = mda.Universe(str(fx / "adk_ap5_ca.pdb"))
    with mda.Writer(str(fx / "adk_ap5_ca.xtc"), n_atoms=len(atoms)) as w:
        for k, f in enumerate(frames):
            ftop.atoms.positions = fix_xyz[k].astype(np.float32)
            ftop.trajectory.ts.dimensions = boxes[f]
            ftop.trajectory.ts.time = float(times[f])
            w.write(ftop.atoms)
    write_pdb(
        fx / "adk_ap5_ca.pdb",
        _with_frame0(atoms, fx / "adk_ap5_ca.pdb", fx / "adk_ap5_ca.xtc"),
        "ADK CA from MD (Seyler & Beckstein 2017, CC BY 4.0) + SYNTHETIC Ap5A placed from 1AKE",
    )
    write_ndx(
        fx / "index.ndx",
        {
            "System": np.arange(len(atoms)),
            "Protein": np.arange(214),
            "C-alpha": np.arange(214),
            "AP5": np.arange(214, len(atoms)),
            **domain_groups(resid),
        },
    )
    heavy = pdb_chain_a(args.pdb_dir / "4AKE.pdb", None)
    write_pdb(fx / "4ake_A.pdb", heavy, "PDB 4AKE chain A, protein heavy atoms (RCSB PDB, CC0)")
    print(
        json.dumps(
            {
                "frames": n,
                "fixture_frames": len(frames),
                "crystal_angles": cryst,
                "theta_NMP_range": [float(th_nmp.min()), float(th_nmp.max())],
                "theta_LID_range": [float(th_lid.min()), float(th_lid.max())],
                "heavy_4ake_atoms": len(heavy),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
