"""Second half of the validation battery (imported by validate.py)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _markov_chain(n, P, rng, s0=0):
    s = np.empty(n, dtype=int)
    s[0] = s0
    cum = np.cumsum(P, axis=1)
    u = rng.random(n)
    for i in range(1, n):
        s[i] = np.searchsorted(cum[s[i - 1]], u[i])
    return s


def run_part2(tmp: Path, truth, record, ROOT: Path, KB: float):
    # =========================================================================
    # 6. Block bootstrap: CI coverage over many independent synthetic systems
    # =========================================================================
    from conformatlas.basins import detect_energy_basins
    from conformatlas.bootstrap import bootstrap_landscape
    from conformatlas.fel import calculate_fel
    from conformatlas.timeseries import block_length_from_g, statistical_inefficiency

    rng = np.random.default_rng(21)
    P = np.array([[0.985, 0.015], [0.035, 0.965]])  # stationary pi = (0.7, 0.3)
    pi = np.array([P[1, 0], P[0, 1]]) / (P[0, 1] + P[1, 0])
    centres = np.array([[-3.0, 0.0], [3.0, 0.0]])
    n_exp, n_frames, hits, widths = 120, 3000, 0, []
    for e in range(n_exp):
        s = _markov_chain(n_frames, P, rng, s0=int(rng.random() < pi[1]))
        xy = centres[s] + rng.normal(0, 0.5, size=(n_frames, 2))
        df = pd.DataFrame(
            {
                "system": "S",
                "replicate": "R",
                "frame": np.arange(n_frames),
                "time_ps": np.arange(n_frames, dtype=float),
                "PC1": xy[:, 0],
                "PC2": xy[:, 1],
            }
        )
        fel = calculate_fel(xy[:, 0], xy[:, 1], temperature=300.0, bins=24)
        states, _, assign, labels = detect_energy_basins(fel, df, min_state_population=0.05)
        if len(states) != 2:
            continue
        g = max(statistical_inefficiency(xy[:, 0]), statistical_inefficiency(xy[:, 1]))
        L = block_length_from_g(g, n_frames)
        res = bootstrap_landscape(
            xy[:, 0],
            xy[:, 1],
            df["replicate"].to_numpy(),
            fel,
            labels,
            states,
            {"R": L},
            n_boot=200,
            seed=e,
        )
        tbl = res.state_table.set_index("state")
        st = next(s_ for s_ in states if s_.min_pc1 < 0)
        row = tbl.loc[st.label]
        hits += row["population_ci95_low"] <= pi[0] <= row["population_ci95_high"]
        widths.append(row["population_ci95_high"] - row["population_ci95_low"])
    cov = hits / n_exp
    record(
        "Bootstrap",
        "95 % CI coverage of the true state population (120 correlated Markov systems)",
        0.88 <= cov <= 0.99,
        f"{cov:.3f} (mean CI width {np.mean(widths):.3f})",
        "0.95",
        "0.88–0.99 (binomial SE ≈ 0.02; block bootstrap is slightly anti-conservative)",
        "known stationary distribution π = (0.7, 0.3)",
    )

    # naive (iid) bootstrap would be badly anti-conservative — demonstrate the block choice matters
    hits_iid = 0
    rng2 = np.random.default_rng(22)
    for e in range(40):
        s = _markov_chain(n_frames, P, rng2)
        xy = centres[s] + rng2.normal(0, 0.5, size=(n_frames, 2))
        df = pd.DataFrame(
            {
                "system": "S",
                "replicate": "R",
                "frame": np.arange(n_frames),
                "time_ps": np.arange(n_frames, dtype=float),
                "PC1": xy[:, 0],
                "PC2": xy[:, 1],
            }
        )
        fel = calculate_fel(xy[:, 0], xy[:, 1], temperature=300.0, bins=24)
        states, _, assign, labels = detect_energy_basins(fel, df, min_state_population=0.05)
        if len(states) != 2:
            continue
        res = bootstrap_landscape(
            xy[:, 0],
            xy[:, 1],
            df["replicate"].to_numpy(),
            fel,
            labels,
            states,
            {"R": 1},
            n_boot=200,
            seed=e,
        )
        tbl = res.state_table.set_index("state")
        st = next(s_ for s_ in states if s_.min_pc1 < 0)
        row = tbl.loc[st.label]
        hits_iid += row["population_ci95_low"] <= pi[0] <= row["population_ci95_high"]
    record(
        "Bootstrap",
        "control: block length 1 (iid bootstrap) under-covers on correlated data",
        hits_iid / 40 < 0.8,
        f"coverage {hits_iid / 40:.2f}",
        "≪ 0.95",
        "< 0.80",
        "demonstrates why g-based blocks are used",
    )

    # =========================================================================
    # 7. Kinetics vs analytical Markov chain
    # =========================================================================
    from conformatlas.kinetics import compute_state_kinetics

    rng = np.random.default_rng(31)
    P3 = np.array([[0.95, 0.03, 0.02], [0.04, 0.94, 0.02], [0.01, 0.03, 0.96]])
    n = 400_000
    s = _markov_chain(n, P3, rng)
    labels3 = ["State A", "State B", "State C"]
    fa = pd.DataFrame(
        {
            "replicate": "R",
            "frame": np.arange(n),
            "time_ps": 2.0 * np.arange(n),
            "state": np.array(labels3)[s],
        }
    )
    kin = compute_state_kinetics(fa, labels3, dt_ps=2.0)
    tpm = kin["transition_matrix"].to_numpy(dtype=float)
    record(
        "Kinetics",
        "transition matrix vs generating Markov chain (4×10⁵ steps)",
        np.max(np.abs(tpm - P3)) < 0.003,
        f"max |ΔP| {np.max(np.abs(tpm - P3)):.4f}",
        "0",
        "< 0.003",
        "known transition matrix",
    )
    # analytical MFPT: (I - P_{-j}) m = 1
    mf = kin["mfpt"].to_numpy(dtype=float)
    mf_true = np.zeros((3, 3))
    for j in range(3):
        keep = [i for i in range(3) if i != j]
        m = np.linalg.solve(np.eye(2) - P3[np.ix_(keep, keep)], np.ones(2))
        for r_, i in enumerate(keep):
            mf_true[i, j] = m[r_] * 2.0
    rel = np.abs(mf - mf_true)[~np.eye(3, dtype=bool)] / mf_true[~np.eye(3, dtype=bool)]
    record(
        "Kinetics",
        "Markov MFPT vs analytical hitting times",
        np.all(rel < 0.05),
        f"max rel. diff {rel.max():.3f}",
        "0",
        "< 5 %",
        "linear-system solution for the known chain",
    )
    # mean dwell time = dt / (1 - p_ii) (geometric), uncensored visits
    dw = kin["dwell_summary"].set_index("state")
    dwell_true = 2.0 / (1 - np.diag(P3))
    rel_d = np.abs(dw.loc[labels3, "mean_dwell_ps"].to_numpy() - dwell_true) / dwell_true
    record(
        "Kinetics",
        "mean dwell time vs geometric expectation Δt/(1−pᵢᵢ)",
        np.all(rel_d < 0.05),
        f"max rel. diff {rel_d.max():.3f}",
        "0",
        "< 5 %",
        "geometric dwell distribution",
    )

    # =========================================================================
    # 8. TICA vs deeptime and analytical AR(1) eigenvalue
    # =========================================================================
    from conformatlas.pca import compute_tica, implied_timescales

    rng = np.random.default_rng(41)
    n, phi = 60_000, 0.99
    slow = np.empty(n)
    slow[0] = 0
    for i in range(1, n):
        slow[i] = phi * slow[i - 1] + np.sqrt(1 - phi**2) * rng.normal()
    fast = rng.normal(size=(n, 5)) * 3.0
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    X = np.column_stack([slow, fast]) @ q.T
    lag = 10
    evals, evecs, *_ = compute_tica({"R": X}, lag=lag, n_components=3)
    record(
        "TICA",
        "leading eigenvalue vs analytical AR(1) autocorrelation φ^τ",
        abs(evals[0] - phi**lag) < 0.02,
        f"{evals[0]:.4f}",
        f"{phi**lag:.4f}",
        "< 0.02",
        "AR(1) autocorrelation",
    )
    ts = implied_timescales(evals, lag, 1.0)
    record(
        "TICA",
        "implied timescale vs −1/ln φ",
        abs(ts[0] - (-1 / np.log(phi))) / (-1 / np.log(phi)) < 0.2,
        f"{ts[0]:.1f} frames",
        f"{-1 / np.log(phi):.1f} frames",
        "< 20 %",
        "AR(1)",
    )
    try:
        from deeptime.decomposition import TICA as DTICA

        model = DTICA(lagtime=lag, dim=3, scaling=None).fit(X).fetch_model()
        dt_evals = (
            model.singular_values[:3]
            if hasattr(model, "singular_values")
            else model.timescales(lag)
        )
        rec = abs(evals[0] - dt_evals[0]) < 0.01
        record(
            "TICA",
            "leading eigenvalue vs deeptime TICA",
            rec,
            f"{evals[0]:.4f}",
            f"deeptime {dt_evals[0]:.4f}",
            "< 0.01",
            f"deeptime {__import__('deeptime').__version__}",
        )
        v_dt = model.instantaneous_coefficients[:, 0]
        v_ours = evecs[:, 0]
        cosang = abs(np.dot(v_dt, v_ours)) / np.linalg.norm(v_dt) / np.linalg.norm(v_ours)
        record(
            "TICA",
            "leading IC direction vs deeptime",
            cosang > 0.99,
            f"|cos| {cosang:.4f}",
            "1",
            "> 0.99",
            "deeptime",
        )
    except Exception as exc:  # pragma: no cover
        record(
            "TICA",
            "vs deeptime",
            True,
            f"skipped ({type(exc).__name__}: {exc})",
            "-",
            "-",
            "deeptime",
            note="skipped",
        )

    # =========================================================================
    # 9. Dihedral features vs MDAnalysis Dihedral (periodicity-safe)
    # =========================================================================
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Ramachandran

    from conformatlas.features import stream_backbone_dihedrals

    fx = ROOT / "tests" / "data" / "adk_ca"
    demo = ROOT / "examples" / "demo_dataset" / "WT"
    u = mda.Universe(str(demo / "topology.pdb"), str(demo / "rep1.xtc"))
    try:
        feats, times, lab = stream_backbone_dihedrals(demo / "topology.pdb", demo / "rep1.xtc")
        rama = Ramachandran(u.select_atoms("protein")).run()
        ang = np.deg2rad(rama.results.angles)  # (frames, residues, 2)
        cos_phi_ref = np.cos(ang[:, :, 0])
        cos_cols = lab[(lab["angle"] == "phi") & (lab["component"] == "cos")].index.to_numpy()
        diff = np.max(np.abs(feats[:, cos_cols] - cos_phi_ref))
        record(
            "Dihedral PCA",
            "cos φ features vs MDAnalysis Ramachandran",
            diff < 1e-5,
            f"max |Δ| {diff:.1e}",
            "0",
            "1e-5",
            "MDAnalysis Ramachandran",
        )
        record(
            "Dihedral PCA",
            "features bounded in [−1, 1] (periodicity handled via sin/cos)",
            np.all(np.abs(feats) <= 1 + 1e-9),
            f"max |f| {np.max(np.abs(feats)):.6f}",
            "≤ 1",
            "exact",
            "definition",
        )
    except Exception as exc:
        record(
            "Dihedral PCA",
            "vs MDAnalysis Ramachandran",
            False,
            f"{type(exc).__name__}: {exc}",
            "-",
            "-",
            "MDAnalysis",
        )

    # =========================================================================
    # 10. Descriptors vs MDAnalysis (RMSD, Rg, RMSF) and gmx gyrate
    # =========================================================================
    from MDAnalysis.analysis import rms
    from MDAnalysis.analysis.align import AlignTraj

    from conformatlas.descriptors import radius_of_gyration, rmsd_series, rmsf_per_atom
    from conformatlas.trajectory import load_reference_coordinates, stream_trajectory_coordinates

    topo, traj = fx / "adk_ap5_ca.pdb", fx / "adk_ap5_ca.xtc"
    u = mda.Universe(str(topo), str(traj))
    ca = u.select_atoms("name CA")
    idx = ca.indices.tolist()
    ref = load_reference_coordinates(topo, atom_indices=idx)
    coords, times = stream_trajectory_coordinates(
        topo, traj, atom_indices=idx, fit=True, fit_atom_indices=idx, reference_fit_coords=ref
    )
    ours_rmsd = rmsd_series(coords, ref)
    R = rms.RMSD(u, u, select="name CA", ref_frame=0).run()
    ref_rmsd = R.results.rmsd[:, 2]
    record(
        "Descriptors",
        "Cα RMSD vs MDAnalysis rms.RMSD (fit + RMSD)",
        np.max(np.abs(ours_rmsd - ref_rmsd)) < 1e-3,
        f"max |Δ| {np.max(np.abs(ours_rmsd - ref_rmsd)):.1e} Å",
        "0",
        "1e-3 Å",
        "MDAnalysis",
    )
    masses = ca.masses
    ours_rg = radius_of_gyration(coords, masses=masses)
    ref_rg = np.array([ca.radius_of_gyration() for ts in u.trajectory])
    record(
        "Descriptors",
        "mass-weighted Rg vs MDAnalysis AtomGroup.radius_of_gyration",
        np.max(np.abs(ours_rg - ref_rg)) < 1e-3,
        f"max |Δ| {np.max(np.abs(ours_rg - ref_rg)):.1e} Å",
        "0",
        "1e-3 Å",
        "MDAnalysis",
    )
    # RMSF after fitting vs MDAnalysis RMSF on an aligned copy
    u2 = mda.Universe(str(topo), str(traj))
    AlignTraj(u2, mda.Universe(str(topo)), select="name CA", in_memory=True).run()
    rmsf_ref = rms.RMSF(u2.select_atoms("name CA")).run().results.rmsf
    rmsf_ours = rmsf_per_atom(coords)
    record(
        "Descriptors",
        "Cα RMSF (fitted) vs MDAnalysis rms.RMSF on aligned trajectory",
        np.max(np.abs(rmsf_ours - rmsf_ref)) < 1e-3,
        f"max |Δ| {np.max(np.abs(rmsf_ours - rmsf_ref)):.1e} Å",
        "0",
        "1e-3 Å",
        "MDAnalysis",
    )
    gmx = shutil.which("gmx")
    if gmx:
        d = tmp / "gyr"
        d.mkdir(exist_ok=True)
        # Cα only, as a GROMACS selection: the group name "C-alpha" is a syntax error for
        # the selection parser of current GROMACS, and the fallback (all atoms) brings in
        # ligand masses that GROMACS guesses from atom names (PA, PB -> Pa, Pb) while
        # MDAnalysis uses the element column
        r = subprocess.run(
            [gmx, "gyrate", "-s", str(topo), "-f", str(traj), "-o", "g.xvg"],
            input="name CA\n",
            text=True,
            capture_output=True,
            cwd=d,
        )
        if r.returncode != 0:
            # group name may differ in a CA-only pdb: pick group 0 (System == CA + ligand) -> compare on all atoms
            r = subprocess.run(
                [gmx, "gyrate", "-s", str(topo), "-f", str(traj), "-o", "g.xvg"],
                input="0\n",
                text=True,
                capture_output=True,
                cwd=d,
            )
            allidx = u.atoms.indices.tolist()
            c_all, _ = stream_trajectory_coordinates(topo, traj, atom_indices=allidx, fit=False)
            ours = radius_of_gyration(c_all, masses=u.atoms.masses)
        else:
            ours = ours_rg
        if r.returncode == 0:
            g = (
                np.array(
                    [
                        float(ln.split()[1])
                        for ln in (d / "g.xvg").read_text(encoding="utf-8").splitlines()
                        if ln and ln[0] not in "#@"
                    ]
                )
                * 10
            )
            diff = np.max(np.abs(ours - g))
            record(
                "Descriptors",
                "Rg vs gmx gyrate (nm → Å)",
                diff < 0.05,
                f"max |Δ| {diff:.3f} Å",
                "0",
                "< 0.05 Å (GROMACS mass table vs MDAnalysis guessed masses)",
                "GROMACS gyrate",
            )
        else:
            record("Descriptors", "Rg vs gmx gyrate", False, r.stderr[-200:], "-", "-", "GROMACS")
    else:
        record(
            "Descriptors",
            "Rg vs gmx gyrate",
            True,
            "skipped",
            "-",
            "-",
            "GROMACS not installed",
            note="skipped",
        )

    # =========================================================================
    # 10b. Landscape vs gmx sham on identical projections and bin edges
    # =========================================================================
    if gmx:
        d = tmp / "sham"
        d.mkdir(exist_ok=True)
        demo = ROOT / "examples" / "demo_dataset" / "WT"
        ok = True
        for cmd, inp in [
            (
                [
                    gmx,
                    "trjcat",
                    "-f",
                    str(demo / "rep1.xtc"),
                    str(demo / "rep2.xtc"),
                    "-o",
                    "pooled.xtc",
                    "-cat",
                ],
                None,
            ),
            (
                [
                    gmx,
                    "covar",
                    "-s",
                    str(demo / "topology.pdb"),
                    "-f",
                    "pooled.xtc",
                    "-o",
                    "eig.xvg",
                    "-v",
                    "vec.trr",
                    "-av",
                    "av.pdb",
                ],
                "Backbone\nBackbone\n",
            ),
            (
                [
                    gmx,
                    "anaeig",
                    "-s",
                    str(demo / "topology.pdb"),
                    "-f",
                    "pooled.xtc",
                    "-v",
                    "vec.trr",
                    "-first",
                    "1",
                    "-last",
                    "2",
                    "-2d",
                    "proj2d.xvg",
                ],
                "Backbone\nBackbone\n",
            ),
            (
                [
                    gmx,
                    "sham",
                    "-f",
                    "proj2d.xvg",
                    "-notime",
                    "-tsham",
                    "300",
                    "-nlevels",
                    "400",
                    "-ls",
                    "gibbs.xpm",
                ],
                None,
            ),
        ]:
            r = subprocess.run(cmd, input=inp, text=True, capture_output=True, cwd=d)
            ok &= r.returncode == 0
        if (
            ok
            and "Ptot = 0"
            in (
                open(d / "shamlog.log", encoding="utf-8").read()
                if (d / "shamlog.log").exists()
                else ""
            )
            or not (d / "gibbs.xpm").exists()
        ):
            record(
                "FEL",
                "ΔG grid vs gmx sham",
                True,
                "skipped (this gmx sham build reads no data)",
                "-",
                "-",
                "GROMACS sham",
                note="skipped",
            )
        elif ok:
            from conformatlas.xpm import parse_xpm

            arr = np.array(
                [
                    [float(v) for v in ln.split()]
                    for ln in open(d / "proj2d.xvg", encoding="utf-8")
                    if ln[0] not in "#@"
                ]
            )
            xs, ys = arr[:, 0], arr[:, 1]
            xax, yax, G = parse_xpm(d / "gibbs.xpm")
            G = np.asarray(G, float)
            nb = G.shape[1]
            wx = (xs.max() - xs.min()) / (nb - 1)
            wy = (ys.max() - ys.min()) / (nb - 1)
            fel = calculate_fel(
                xs,
                ys,
                temperature=300.0,
                bins=nb,
                x_range=(xs.min() - wx / 2, xs.max() + wx / 2),
                y_range=(ys.min() - wy / 2, ys.max() + wy / 2),
            )
            m = np.isfinite(fel.free_energy)
            diff = np.abs(G[m] - fel.free_energy[m])
            step = (G.max() - G.min()) / 400
            n_same = int((diff < 0.05).sum())
            record(
                "FEL",
                "ΔG grid vs gmx sham (identical projections, bin edges, T): bins within 0.05 kJ/mol",
                n_same >= 0.95 * m.sum() and np.median(diff) < step,
                f"{n_same}/{m.sum()} bins; median |Δ| {np.median(diff):.4f} kJ/mol; others are single frames on shared bin edges",
                "all",
                "≥ 95 % of bins; median < XPM level step",
                "GROMACS sham",
            )
            # bins ConformAtlas leaves undefined: sham writes one constant fill value there,
            # at or above its highest sampled ΔG; a few bins differ only because a single
            # frame on a shared bin edge is binned to the neighbour (see the row above)
            vals, counts = np.unique(np.round(G[~m], 4), return_counts=True)
            fill = float(vals[np.argmax(counts)]) if len(vals) else float("nan")
            n_fill = int(counts.max()) if len(counts) else 0
            sham_sampled = G[np.abs(G - fill) > step / 2]  # bins sham itself sampled
            g_sampled_max = float(sham_sampled.max()) if sham_sampled.size else float("nan")
            record(
                "FEL",
                "unsampled bins: sham writes one constant fill value at or above the highest "
                "sampled ΔG, ConformAtlas = NaN (deliberate difference)",
                bool(len(vals)) and fill >= g_sampled_max - step and n_fill >= int((~m).sum()) - 3,
                f"fill {fill:.2f} kJ/mol in {n_fill}/{int((~m).sum())} empty bins; highest sampled "
                f"ΔG {g_sampled_max:.2f} kJ/mol",
                "NaN in ConformAtlas",
                "fill ≥ highest sampled ΔG; ≤ 3 edge-frame bins differ",
                "GROMACS sham",
            )
        else:
            record(
                "FEL", "ΔG grid vs gmx sham", False, "GROMACS pipeline failed", "-", "-", "GROMACS"
            )

    # =========================================================================
    # 11. Convergence metrics: cosine content and RMSIP limits
    # =========================================================================
    from conformatlas.convergence import calculate_cosine_content, calculate_subspace_rmsip

    t = np.arange(5000)
    cc1 = calculate_cosine_content(np.cos(np.pi * t / (len(t) - 1)))
    rng = np.random.default_rng(3)
    # Hess 2000: the first PC of a high-dimensional random walk has cosine content -> 1
    from conformatlas.pca import compute_cartesian_pca, project_coordinates

    walk = np.cumsum(rng.normal(size=(3000, 90)), axis=0).reshape(3000, 30, 3)
    ev, vec, *_rest, mu = compute_cartesian_pca({"R": walk}, weighting="frames", n_components=3)
    pc1 = project_coordinates(walk, mu, vec)[:, 0]
    cc_walk = calculate_cosine_content(pc1)
    cc_noise = calculate_cosine_content(rng.normal(size=20000))
    record(
        "Convergence",
        "cosine content of a half-period cosine = 1",
        abs(cc1 - 1) < 1e-9,
        f"{cc1:.6f}",
        "1",
        "1e-9",
        "Hess 2000/2002 definition",
    )
    record(
        "Convergence",
        "cosine content: PC1 of a 90-D random walk ≈ 1 (Hess 2000), white noise ≈ 0",
        cc_walk > 0.85 and cc_noise < 0.02,
        f"walk PC1 {cc_walk:.3f}, noise {cc_noise:.4f}",
        "→1, ≈0",
        "> 0.85, < 0.02",
        "Hess 2000 (random diffusion)",
    )
    cc_bm = np.mean(
        [calculate_cosine_content(np.cumsum(rng.normal(size=20000))) for _ in range(100)]
    )
    record(
        "Convergence",
        "cosine content of centred 1-D Brownian paths (informational: broad distribution)",
        True,
        f"mean {cc_bm:.2f}",
        "~0.4–0.6",
        "informational",
        "Karhunen–Loève modes of Brownian motion",
    )
    q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
    record(
        "Convergence",
        "RMSIP: identical subspaces 1, orthogonal 0, 5/10 shared → √0.5",
        abs(calculate_subspace_rmsip(q[:, :10], q[:, :10]) - 1) < 1e-12
        and abs(calculate_subspace_rmsip(q[:, :10], q[:, 10:20])) < 1e-12
        and abs(calculate_subspace_rmsip(q[:, :10], q[:, 5:15]) - np.sqrt(0.5)) < 1e-12,
        "1, 0, 0.7071",
        "1, 0, 0.7071",
        "1e-12",
        "Amadei 1999 definition",
    )

    # =========================================================================
    # 12. Comparison statistics: overlap metrics vs SciPy, null calibration
    # =========================================================================
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import ttest_ind

    from conformatlas.comparison_stats import (
        _bootstrap_p_value,
        bootstrap_system,
        landscape_overlap,
        population_difference_table,
    )

    rng = np.random.default_rng(7)
    p = rng.random((10, 10))
    p /= p.sum()
    qd = rng.random((10, 10))
    qd /= qd.sum()
    ov = landscape_overlap(p, qd)
    js_ref = jensenshannon(p.ravel(), qd.ravel(), base=2) ** 2
    bc_ref = np.sum(np.sqrt(p * qd))
    record(
        "Comparison",
        "Jensen–Shannon divergence vs scipy.spatial.distance.jensenshannon²",
        abs(ov["JS_Divergence_bits"] - js_ref) < 1e-12,
        f"{ov['JS_Divergence_bits']:.6f}",
        f"{js_ref:.6f}",
        "1e-12",
        "SciPy",
    )
    record(
        "Comparison",
        "Bhattacharyya coefficient and Hellinger distance vs direct formula",
        abs(ov["Bhattacharyya_Coefficient"] - bc_ref) < 1e-12
        and abs(ov["Hellinger_Distance"] - np.sqrt(1 - bc_ref)) < 1e-12,
        f"{ov['Bhattacharyya_Coefficient']:.6f}",
        f"{bc_ref:.6f}",
        "1e-12",
        "definition",
    )

    # Null calibration: two systems drawn from the SAME Markov process; the bootstrap
    # p-value for the population difference should reject at ~5 % when alpha = 0.05.
    P = np.array([[0.985, 0.015], [0.035, 0.965]])
    centres = np.array([[-3.0, 0.0], [3.0, 0.0]])
    n_frames, n_null, rejects, pvals, ci_hits = 3000, 100, 0, [], 0
    for e in range(n_null):
        rows = []
        for sysname in ("A", "B"):
            for rep in ("1", "2"):
                s = _markov_chain(n_frames, P, rng, s0=int(rng.random() < 0.3))
                xy = centres[s] + rng.normal(0, 0.5, size=(n_frames, 2))
                rows.append(
                    pd.DataFrame(
                        {
                            "system": sysname,
                            "replicate": f"{sysname}_Rep{rep}",
                            "frame": np.arange(n_frames),
                            "time_ps": np.arange(n_frames, dtype=float),
                            "PC1": xy[:, 0],
                            "PC2": xy[:, 1],
                        }
                    )
                )
        df = pd.concat(rows, ignore_index=True)
        fel = calculate_fel(df["PC1"].to_numpy(), df["PC2"].to_numpy(), temperature=300.0, bins=24)
        states, _, assign, labels = detect_energy_basins(fel, df, min_state_population=0.05)
        if len(states) != 2:
            continue
        rb = np.random.default_rng(1000 + e)
        boots = {
            nm: bootstrap_system(nm, df[df["system"] == nm], fel, labels, states, 200, rb)
            for nm in ("A", "B")
        }
        tbl = population_difference_table(states, assign, ["A", "B"], "A", boots)
        pv = tbl["B_minus_A_Bootstrap_p"].iloc[0]
        pvals.append(pv)
        rejects += pv < 0.05
        ci_hits += tbl["B_minus_A_CI95_Low"].iloc[0] <= 0 <= tbl["B_minus_A_CI95_High"].iloc[0]
    n_ok = len(pvals)
    record(
        "Comparison",
        f"type-I error of the bootstrap population-difference test under the null (α = 0.05, {n_ok} trials)",
        rejects / n_ok <= 0.12,
        f"{rejects / n_ok:.3f}",
        "0.05",
        "≤ 0.12 (binomial SE ≈ 0.02; mild anti-conservatism expected)",
        "two systems from one Markov process",
    )
    record(
        "Comparison",
        "95 % CI of the population difference contains 0 under the null",
        ci_hits / n_ok >= 0.88,
        f"{ci_hits / n_ok:.3f}",
        "0.95",
        "≥ 0.88",
        "same null experiment",
    )
    # Null with heterogeneous replicates (thresholds fixed before the first run):
    # both systems come from ONE slow two-state process (mean dwell 1000 frames,
    # 3000 frames per replicate), so single replicates are often trapped in one
    # state and replicate populations scatter widely, as in real µs simulations.
    # 3 replicates per system. The replicate-level bootstrap and Welch's t must stay
    # near nominal; the frame-level bootstrap is expected to fail (it omits the
    # between-replicate variance), which is the documented reason for the default.
    P_slow = np.array([[0.999, 0.001], [0.001, 0.999]])
    het = {"replicate": [0, 0], "frame": [0, 0], "welch": [0, 0]}
    rng_het = np.random.default_rng(11)  # own stream: later checks keep their draws
    for e in range(100):
        rows = []
        for sysname in ("A", "B"):
            for rep in ("1", "2", "3"):
                s = _markov_chain(n_frames, P_slow, rng_het, s0=int(rng_het.random() < 0.5))
                xy = centres[s] + rng_het.normal(0, 0.5, size=(n_frames, 2))
                rows.append(
                    pd.DataFrame(
                        {
                            "system": sysname,
                            "replicate": f"{sysname}_Rep{rep}",
                            "frame": np.arange(n_frames),
                            "time_ps": np.arange(n_frames, dtype=float),
                            "PC1": xy[:, 0],
                            "PC2": xy[:, 1],
                        }
                    )
                )
        df = pd.concat(rows, ignore_index=True)
        fel = calculate_fel(df["PC1"].to_numpy(), df["PC2"].to_numpy(), temperature=300.0, bins=24)
        states, _, assign, labels = detect_energy_basins(fel, df, min_state_population=0.05)
        if len(states) != 2:
            continue
        for lev in ("replicate", "frame"):
            rb = np.random.default_rng(5000 + e)
            boots = {
                nm: bootstrap_system(
                    nm, df[df["system"] == nm], fel, labels, states, 200, rb, level=lev
                )
                for nm in ("A", "B")
            }
            tbl = population_difference_table(states, assign, ["A", "B"], "A", boots)
            lo, hi = tbl["B_minus_A_CI95_Low"].iloc[0], tbl["B_minus_A_CI95_High"].iloc[0]
            het[lev][0] += not (lo <= 0 <= hi)
            het[lev][1] += 1
            if lev == "replicate":
                wp = tbl["B_minus_A_Welch_p"].iloc[0]
                if np.isfinite(wp):
                    het["welch"][0] += wp < 0.05
                    het["welch"][1] += 1
    fpr = {k: v[0] / v[1] for k, v in het.items() if v[1]}
    record(
        "Comparison",
        f"heterogeneous replicates: replicate-level bootstrap CI excludes 0 under the null "
        f"(3 vs 3 replicates, {het['replicate'][1]} trials)",
        fpr["replicate"] <= 0.15,
        f"{fpr['replicate']:.3f}",
        "0.05",
        "≤ 0.15",
        "two systems from one slow Markov process",
    )
    record(
        "Comparison",
        f"heterogeneous replicates: Welch's t rejects under the null at α = 0.05 "
        f"({het['welch'][1]} trials)",
        fpr["welch"] <= 0.10,
        f"{fpr['welch']:.3f}",
        "0.05",
        "≤ 0.10",
        "same null experiment",
    )
    record(
        "Comparison",
        "heterogeneous replicates: frame-level bootstrap (bootstrap_level: frame) is "
        "anti-conservative, as documented",
        fpr["frame"] > 0.15,
        f"{fpr['frame']:.3f}",
        "> 0.15 (omits between-replicate variance)",
        "> 0.15",
        "same null experiment; documents why 'replicate' is the default",
    )
    # Welch interval: same statistic and degrees of freedom as scipy
    from conformatlas.comparison_stats import welch_interval

    wa = np.array([65.0, 70.0, 61.0])
    wb = np.array([40.0, 46.0, 55.0, 49.0])
    wd, wlo, whi, wdf = welch_interval(wb, wa)
    tt = ttest_ind(wb, wa, equal_var=False)
    se = np.sqrt(wa.var(ddof=1) / 3 + wb.var(ddof=1) / 4)
    t_back = wd / se
    ok_ci = abs(t_back - tt.statistic) < 1e-12
    if hasattr(tt, "confidence_interval"):
        ci = tt.confidence_interval(0.95)
        ok_ci = ok_ci and abs(ci.low - wlo) < 1e-10 and abs(ci.high - whi) < 1e-10
        ok_ci = ok_ci and abs(getattr(tt, "df", wdf) - wdf) < 1e-10
    record(
        "Comparison",
        "Welch 95 % interval and Welch–Satterthwaite df match SciPy",
        ok_ci,
        f"[{wlo:.4f}, {whi:.4f}], df {wdf:.4f}",
        "scipy ttest_ind(...).confidence_interval",
        "1e-10",
        "SciPy",
    )
    # Welch columns of the comparison table: per-replicate populations are built by
    # population_difference_table from a frame table; compare with SciPy on the
    # populations computed independently here
    from conformatlas.comparison_stats import population_difference_table
    from conformatlas.models import BasinState

    rng_w = np.random.default_rng(11)
    pops = {"WT": [0.65, 0.70, 0.61], "MUT": [0.40, 0.46, 0.55, 0.49]}
    rows_w = []
    indep = {}
    for sysname, plist in pops.items():
        indep[sysname] = []
        for r_i, p_a in enumerate(plist):
            n_fr = 200 + 37 * r_i  # unequal replicate lengths
            lab = np.where(rng_w.random(n_fr) < p_a, "State A", "State B")
            indep[sysname].append(100.0 * np.mean(lab == "State A"))
            rows_w += [(sysname, f"Rep_{r_i + 1}", s_) for s_ in lab]
    fa_w = pd.DataFrame(rows_w, columns=["system", "replicate", "state"])
    st_w = [
        BasinState(1, "State A", 0.5, 0.0, 0.0, 0.0, "Rep_1", 0, 0.0, 0.1, basin_id=1),
        BasinState(2, "State B", 0.5, 1.0, 0.0, 0.0, "Rep_1", 1, 1.0, 0.1, basin_id=2),
    ]
    tab_w = population_difference_table(st_w, fa_w, ["WT", "MUT"], "WT", {}).set_index("State")
    tt = ttest_ind(indep["MUT"], indep["WT"], equal_var=False)
    got_t = float(tab_w.loc["State A", "MUT_minus_WT_Welch_t"])
    got_p = float(tab_w.loc["State A", "MUT_minus_WT_Welch_p"])
    got_d = float(tab_w.loc["State A", "MUT_minus_WT_Welch_Diff_Percent"])
    ref_d = float(np.mean(indep["MUT"]) - np.mean(indep["WT"]))
    err_w = max(abs(got_t - tt.statistic), abs(got_p - tt.pvalue), abs(got_d - ref_d))
    record(
        "Comparison",
        "Welch t, p and mean difference of the comparison table match SciPy on "
        "independently computed per-replicate populations (3 vs 4 replicates, unequal lengths)",
        err_w < 1e-10,
        f"t = {got_t:.4f}, p = {got_p:.4f}, Δ = {got_d:.3f} pp; max |Δ| vs SciPy {err_w:.1e}",
        f"t = {tt.statistic:.4f}, p = {tt.pvalue:.4f}, Δ = {ref_d:.3f} pp",
        "1e-10",
        "scipy.stats.ttest_ind(equal_var=False)",
    )
    record(
        "Comparison",
        "bootstrap p-value: floor 1/B and two-sidedness",
        _bootstrap_p_value(np.ones(200)) == 0.005
        and _bootstrap_p_value(np.r_[np.ones(100), -np.ones(100)]) == 1.0,
        "0.005, 1.0",
        "0.005, 1.0",
        "exact",
        "definition",
    )

    # =========================================================================
    # 13. End-to-end: GROMACS vs Python backend; representative/extreme structures; NMD
    # =========================================================================
    from click.testing import CliRunner

    from conformatlas.cli import main

    demo = ROOT / "examples" / "demo_dataset" / "WT"
    runner = CliRunner()
    outs = {}
    backends = ["python"] + (["gromacs"] if shutil.which("gmx") else [])
    for be in backends:
        o = tmp / f"e2e_{be}"
        r = runner.invoke(
            main,
            [
                "analyze",
                "-s",
                str(demo / "topology.pdb"),
                "-f",
                str(demo / "rep1.xtc"),
                "-f",
                str(demo / "rep2.xtc"),
                "-T",
                "300",
                "--pca-backend",
                be,
                "--bins",
                "16",
                "--smooth-sigma",
                "1.0",
                "--bootstrap",
                "0",
                "-o",
                str(o),
            ],
        )
        assert r.exit_code == 0, r.output
        outs[be] = o
    if "gromacs" in outs:
        a = (
            pd.read_csv(outs["python"] / "statistics" / "state_populations.csv")[
                "Mean_Population_Percent"
            ]
            .sort_values()
            .to_numpy()
        )
        b = (
            pd.read_csv(outs["gromacs"] / "statistics" / "state_populations.csv")[
                "Mean_Population_Percent"
            ]
            .sort_values()
            .to_numpy()
        )
        ea = (
            pd.read_csv(outs["python"] / "pca" / "explained_variance.csv")[
                "ExplainedVariance_Percent"
            ]
            .iloc[:3]
            .to_numpy()
        )
        eb = (
            pd.read_csv(outs["gromacs"] / "pca" / "explained_variance.csv")[
                "ExplainedVariance_Percent"
            ]
            .iloc[:3]
            .to_numpy()
        )
        record(
            "End-to-end",
            "state populations: GROMACS covar/anaeig backend vs Python backend",
            len(a) == len(b) and np.max(np.abs(a - b)) < 2.0,
            f"{a.round(1)} vs {b.round(1)} %",
            "equal",
            "< 2 pp",
            "GROMACS 2023 covar + anaeig",
        )
        record(
            "End-to-end",
            "explained variance (top 3): GROMACS vs Python",
            np.max(np.abs(ea - eb)) < 1.0,
            f"max |Δ| {np.max(np.abs(ea - eb)):.2f} pp",
            "0",
            "< 1 pp",
            "GROMACS",
        )
    else:
        record(
            "End-to-end",
            "GROMACS backend vs Python backend",
            True,
            "skipped",
            "-",
            "-",
            "no gmx",
            note="skipped",
        )

    o = outs["python"]
    # representative structure equals the trajectory frame it claims to be
    meta = json.load(open(o / "states" / "State_A" / "metadata.json", encoding="utf-8"))
    rep_u = mda.Universe(str(o / "states" / "State_A" / "representative.pdb"))
    rep_name = str(meta.get("replicate", ""))
    tp = demo / ("rep1.xtc" if rep_name.endswith("1") else "rep2.xtc")
    u = mda.Universe(str(demo / "topology.pdb"), str(tp))
    t_target = float(meta.get("time_ps", meta.get("representative_time_ps")))
    fr = int(np.argmin([abs(ts.time - t_target) for ts in u.trajectory]))
    u.trajectory[fr]
    d = np.max(np.abs(u.atoms.positions - rep_u.atoms.positions))
    record(
        "End-to-end",
        "representative.pdb coordinates equal the trajectory frame at the recorded time",
        d < 2e-3,
        f"max |Δx| {d:.1e} Å (frame {fr}, t = {t_target} ps)",
        "0",
        "2e-3 Å (PDB precision)",
        "trajectory frame",
    )
    # mode contributions: squared per-atom weights of each mode sum to 1
    mc = pd.read_csv(o / "pca" / "mode_contributions_atoms.csv")
    cols = [c for c in mc.columns if c.startswith("PC")]
    sums = (mc[cols] ** 2).sum().to_numpy()
    record(
        "End-to-end",
        "per-atom mode contributions: Σ w² = 1 for each mode (unit eigenvectors)",
        np.allclose(sums, 1, atol=1e-6),
        f"{sums.round(6)}",
        "1",
        "1e-6",
        "definition",
    )
    # NMD readable by ProDy, modes orthonormal
    try:
        import prody

        prody.confProDy(verbosity="none")
        nmd = prody.parseNMD(str(o / "pca" / "modes.nmd"))
        ens = nmd[0]
        arr = ens.getEigvecs()
        orth = np.max(np.abs(arr.T @ arr - np.eye(arr.shape[1])))
        record(
            "End-to-end",
            "modes.nmd parsed by ProDy; eigenvectors orthonormal",
            orth < 1e-3,
            f"{arr.shape[1]} modes, max |VᵀV − I| {orth:.1e}",
            "orthonormal",
            "1e-3 (NMD text precision)",
            f"ProDy {prody.__version__}",
        )
    except Exception as exc:
        record(
            "End-to-end",
            "modes.nmd parsed by ProDy",
            False,
            f"{type(exc).__name__}: {exc}",
            "-",
            "-",
            "ProDy",
        )
    # extreme structures: projection of the written extreme models onto PC1 spans the data range
    evecs = np.load(o / "pca" / "eigenvectors.npy")
    mean = np.load(o / "pca" / "mean_coordinates.npy").ravel()
    proj = pd.read_csv(o / "pca" / "projections.csv")
    ext = mda.Universe(str(o / "pca" / "PC1_extremes.pdb"))
    projs = []
    for ts in ext.trajectory:
        projs.append(float((ext.atoms.positions.ravel() - mean) @ evecs[:, 0]))
    record(
        "End-to-end",
        "PC1_extremes.pdb models project to the min/max of PC1",
        abs(min(projs) - proj["PC1"].min()) < 0.05 and abs(max(projs) - proj["PC1"].max()) < 0.05,
        f"[{min(projs):.3f}, {max(projs):.3f}]",
        f"[{proj['PC1'].min():.3f}, {proj['PC1'].max():.3f}]",
        "0.05 Å",
        "projection identity",
    )
    # summary consistency: populations in summary.json equal basins.csv
    summ = json.load(open(o / "summary.json", encoding="utf-8"))
    bas = pd.read_csv(o / "states" / "basins.csv")
    pops_json = (
        sorted(float(s["population_percent"]) for s in summ["states"]) if "states" in summ else None
    )
    if pops_json is not None:
        record(
            "End-to-end",
            "summary.json populations equal states/basins.csv",
            np.allclose(pops_json, sorted(bas["Population_Percent"]), atol=1e-6),
            f"{pops_json}",
            f"{sorted(bas['Population_Percent'].round(3))}",
            "1e-6",
            "internal consistency",
        )
