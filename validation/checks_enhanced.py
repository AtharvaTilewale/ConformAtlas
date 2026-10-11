"""Validation checks for the enhanced-sampling reweighting and residue-level
analysis track (``conformatlas.reweighting``, ``conformatlas.residue_analysis``,
extended ``conformatlas.features`` and the chain-aware common-atom mapping).

Exposes ``run(record, tmp, ROOT, KB)`` for ``validation/validate.py``; can also be
executed standalone::

    .venv/bin/python validation/checks_enhanced.py

Every check compares against an analytical result or an independent
implementation (pymbar, PLUMED ``sum_hills`` when a ``plumed`` executable is
available, MDAnalysis, ProDy, explicit numpy formulas) with the tolerance stated.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

R_GAS = 0.008314462618


# =============================================================================
# Synthetic systems
# =============================================================================


def double_well(x: np.ndarray) -> np.ndarray:
    """Asymmetric 1-D double well, kJ/mol (barrier ~10 kJ/mol, basin offset 3 kJ/mol)."""
    return 10.0 * (x**2 - 1.0) ** 2 + 1.5 * x


def bin_integrated_free_energy(edges: np.ndarray, kT: float, potential=double_well) -> np.ndarray:
    """Exact ``-kT ln int_bin exp(-U/kT) dx`` per bin, minimum set to 0."""
    from scipy.integrate import quad

    f = np.array(
        [
            -kT * np.log(quad(lambda s: np.exp(-potential(s) / kT), a, b)[0])
            for a, b in zip(edges[:-1], edges[1:])
        ]
    )
    return f - f.min()


def sample_umbrella_windows(
    centers: np.ndarray,
    kappa: float,
    temperature: float,
    n_steps: int,
    step: float = 0.2,
    burn: int = 2000,
    seed: int = 1,
    potential=double_well,
) -> np.ndarray:
    """Metropolis sampling of ``U + 1/2 kappa (x - c)^2`` in every window at once.

    Returns ``(n_windows, n_steps)`` samples (correlated; no subsampling).
    """
    rng = np.random.default_rng(seed)
    kT = R_GAS * temperature
    x = centers.astype(float).copy()
    e = potential(x) + 0.5 * kappa * (x - centers) ** 2
    out = np.empty((len(centers), n_steps))
    for i in range(n_steps + burn):
        xn = x + rng.normal(0, step, len(centers))
        en = potential(xn) + 0.5 * kappa * (xn - centers) ** 2
        acc = rng.random(len(centers)) < np.exp(-(en - e) / kT)
        x = np.where(acc, xn, x)
        e = np.where(acc, en, e)
        if i >= burn:
            out[:, i - burn] = x
    return out


def simulate_metadynamics(
    hills_path: Path,
    colvar_path: Path,
    temperature: float = 300.0,
    n_steps: int = 400_000,
    pace: int = 200,
    print_stride: int = 10,
    bias_factor: float | None = 8.0,
    height: float = 1.0,
    sigma: float = 0.1,
    step: float = 0.3,
    dt: float = 0.002,
    seed: int = 3,
    potential=double_well,
    x0: float = -1.0,
) -> None:
    """Metropolis walker on ``potential`` under (well-tempered) metadynamics, writing
    PLUMED-format HILLS and COLVAR files.

    Conventions follow PLUMED METAD: hills are stretched Gaussians truncated at
    ``d^2/2sigma^2 = 6.25``; the HILLS ``height`` column is multiplied by
    ``gamma/(gamma-1)`` in well-tempered runs; ``metad.bias`` printed at a
    deposition step excludes the hill deposited at that step.
    """
    from conformatlas.reweighting import _STRETCH_A, _STRETCH_B, DP2CUTOFF

    rng = np.random.default_rng(seed)
    kT = R_GAS * temperature
    n_max = n_steps // pace + 1
    hc = np.empty(n_max)
    hh = np.empty(n_max)
    n = 0

    def bias(x):
        if n == 0:
            return 0.0
        dp2 = 0.5 * ((x - hc[:n]) / sigma) ** 2
        k = _STRETCH_A * np.exp(-dp2) + _STRETCH_B
        k[dp2 >= DP2CUTOFF] = 0.0
        return float(k @ hh[:n])

    x = x0
    ux, vx = potential(x), 0.0
    normals = rng.normal(0, step, n_steps)
    unif = rng.random(n_steps)
    col, hills = [], []
    for i in range(n_steps):
        t = i * dt
        if i % print_stride == 0:
            col.append((t, x, vx))
        if i % pace == 0:
            if bias_factor is None:
                h = height
                biasf_out = -1.0
                h_out = h
            else:
                h = height * np.exp(-vx / (R_GAS * (bias_factor - 1.0) * temperature))
                biasf_out = bias_factor
                h_out = h * bias_factor / (bias_factor - 1.0)
            hc[n], hh[n] = x, h
            n += 1
            hills.append((t, x, sigma, h_out, biasf_out))
            vx = bias(x)
        xn = x + normals[i]
        un, vn = potential(xn), bias(xn)
        if unif[i] < np.exp(-(un + vn - ux - vx) / kT):
            x, ux, vx = xn, un, vn
    with open(hills_path, "w") as fh:
        fh.write("#! FIELDS time x sigma_x height biasf\n#! SET multivariate false\n")
        for r in hills:
            fh.write("%14.6f %16.10f %14.9f %16.10f %14.6f\n" % r)
    with open(colvar_path, "w") as fh:
        fh.write("#! FIELDS time x metad.bias\n")
        for r in col:
            fh.write("%14.6f %16.10f %16.10f\n" % r)


def sample_static_bias(
    hills, temperature: float, n_steps: int, step: float = 0.3, seed: int = 5, potential=double_well
) -> np.ndarray:
    """Equilibrium Metropolis samples of ``U + V_final`` (static final bias)."""
    from conformatlas.reweighting import hills_bias

    rng = np.random.default_rng(seed)
    kT = R_GAS * temperature
    x = -1.0
    e = potential(x) + hills_bias(hills, np.array([x]))[0]
    out = np.empty(n_steps)
    for i in range(n_steps + 2000):
        xn = x + rng.normal(0, step)
        en = potential(xn) + hills_bias(hills, np.array([xn]))[0]
        if rng.random() < np.exp(-(en - e) / kT):
            x, e = xn, en
        if i >= 2000:
            out[i - 2000] = x
    return out


def _find_plumed() -> str | None:
    exe = shutil.which("plumed")
    if exe:
        return exe
    for cand in (
        Path.home() / ".claude-science/conda/envs/plumed-ref/bin/plumed",
        Path("/usr/local/bin/plumed"),
    ):
        if cand.exists():
            return str(cand)
    return None


def _pmf_deviation(pmf_df, f_true, mask):
    f = pmf_df["pmf_kJ_mol"].to_numpy()
    ok = np.isfinite(f) & mask
    d = f[ok] - f_true[ok]
    d -= d.mean()
    return float(np.abs(d).max()), float(np.sqrt(np.mean(d**2))), int(ok.sum())


# =============================================================================
# Checks
# =============================================================================


def check_plumed_readers(record, tmp: Path):
    from conformatlas.reweighting import read_colvar, read_hills, read_plumed_file

    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 4))
    p = tmp / "COLVAR_test"
    with open(p, "w") as fh:
        fh.write("#! FIELDS time d1 d2 metad.bias\n#! SET min_d1 -pi\n#! SET max_d1 pi\n")
        for row in data:
            fh.write(" ".join(f"{v:.12g}" for v in row) + "\n")
    ref = np.loadtxt(p, comments="#")
    tab = read_plumed_file(p)
    vals, t, _ = read_colvar(p, ["d2", "d1"])
    ok = (
        np.array_equal(tab.data, ref)
        and np.array_equal(vals, ref[:, [2, 1]])
        and np.array_equal(t, ref[:, 0])
        and tab.constants == {"min_d1": "-pi", "max_d1": "pi"}
    )
    record(
        "Reweighting",
        "COLVAR reader vs numpy.loadtxt",
        ok,
        f"max|diff| = {np.abs(tab.data - ref).max():.1e}, columns by name",
        "identical",
        "exact",
        "numpy.loadtxt(comments='#')",
    )
    h = tmp / "HILLS_test"
    with open(h, "w") as fh:
        fh.write("#! FIELDS time phi sigma_phi height biasf\n#! SET multivariate false\n")
        fh.write("#! SET min_phi -pi\n#! SET max_phi pi\n")
        fh.write("1.0 0.5 0.3 1.2 10.0\n2.0 -3.0 0.3 1.1 10.0\n")
    hills = read_hills(h)
    exp_h = np.array([1.2, 1.1]) * 0.9
    ok = (
        np.allclose(hills.deposited_height, exp_h)
        and hills.periodic[0] == (-np.pi, np.pi)
        and hills.bias_factor == 10.0
    )
    record(
        "Reweighting",
        "HILLS reader: well-tempered height rescaling and periodicity",
        ok,
        f"deposited heights {hills.deposited_height.round(4).tolist()}, period {hills.periodic[0]}",
        f"{exp_h.round(4).tolist()} (= height*(gamma-1)/gamma), (-pi, pi)",
        "1e-12",
        "PLUMED METAD source: writeGaussian multiplies by biasf/(biasf-1)",
    )


def check_metadynamics(record, tmp: Path, KB: float):
    from conformatlas.reweighting import (
        hills_bias,
        hills_bias_at_times,
        metad_weights,
        pmf_1d,
        read_colvar,
        read_hills,
        read_plumed_file,
    )

    T = 300.0
    kT = KB * T
    hills_path, colvar_path = tmp / "HILLS", tmp / "COLVAR"
    simulate_metadynamics(hills_path, colvar_path, temperature=T)
    hills = read_hills(hills_path)
    cv, t, tab = read_colvar(colvar_path, ["x"])
    gamma = hills.bias_factor

    # (a) time-dependent bias reconstruction vs the metad.bias column written by the run
    b_rec = hills_bias_at_times(hills, cv, t)
    err = float(np.abs(b_rec - tab.column("metad.bias")).max())
    record(
        "Reweighting",
        "Metadynamics V(s_t, t) from HILLS vs metad.bias column",
        err < 1e-6,
        f"max |dV| = {err:.1e} kJ/mol over {len(t)} frames ({hills.n_hills} hills)",
        "0",
        "1e-6 kJ/mol (file precision)",
        "synthetic WT-metaD run writing PLUMED-format HILLS/COLVAR",
    )

    # (b) final bias vs plumed sum_hills
    plumed = _find_plumed()
    if plumed is None:
        record(
            "Reweighting",
            "Final bias vs plumed sum_hills",
            True,
            "plumed executable not found",
            "-",
            "-",
            "PLUMED sum_hills",
            note="skipped",
        )
    else:
        env = dict(os.environ, OMP_NUM_THREADS="1", KMP_AFFINITY="disabled")
        fes = tmp / "fes.dat"
        r = subprocess.run(
            [
                plumed,
                "sum_hills",
                "--hills",
                str(hills_path),
                "--outfile",
                str(fes),
                "--min",
                "-2.5",
                "--max",
                "2.5",
                "--bin",
                "500",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp,
        )
        if r.returncode != 0 or not fes.exists():
            record(
                "Reweighting",
                "Final bias vs plumed sum_hills",
                False,
                r.stderr[-200:],
                "-",
                "-",
                "PLUMED sum_hills",
            )
        else:
            ft = read_plumed_file(fes)
            x = ft.column("x")
            f_plumed = ft.column(ft.fields[1])
            v = hills_bias(hills, x, hill_shape="plumed")
            dev = float(np.abs(-gamma / (gamma - 1.0) * v - f_plumed).max())
            v_g = hills_bias(hills, x, hill_shape="gaussian")
            dev_g = float(np.abs(-gamma / (gamma - 1.0) * v_g - f_plumed).max())
            record(
                "Reweighting",
                "Final bias vs plumed sum_hills",
                dev < 1e-6,
                f"max |(-gamma/(gamma-1)) V - F_sum_hills| = {dev:.1e} kJ/mol (max |V| = {np.abs(v).max():.1f}); "
                f"plain Gaussian hills would deviate by {dev_g:.3f}",
                "0",
                "1e-6 kJ/mol",
                "PLUMED sum_hills (stretched Gaussian hills, DP2CUTOFF 6.25)",
            )

    # (c) reweighted histogram vs Boltzmann distribution of the known potential
    edges = np.linspace(-1.7, 1.7, 51)
    f_true = bin_integrated_free_energy(edges, kT)
    mask = f_true < 8.0  # sampled range: bins within ~3 kT of the global minimum
    res_tp = metad_weights(cv, t, hills, T, method="tiwary-parrinello", bins=400)
    pmf_tp = pmf_1d(
        cv[res_tp.kept, 0], res_tp.weights[res_tp.kept], T, bins=50, value_range=(-1.7, 1.7)
    )
    mx, rms, nb = _pmf_deviation(pmf_tp, f_true, mask)
    record(
        "Reweighting",
        "Tiwary-Parrinello reweighting recovers the Boltzmann distribution",
        mx < 0.3,
        f"max |dF| = {mx:.3f} kJ/mol, RMS {rms:.3f} over {nb} bins with F < 8 kJ/mol",
        "0",
        "0.3 kJ/mol",
        "analytical bin-integrated free energy of the double well",
    )
    pmf_raw = pmf_1d(cv[:, 0], np.ones(len(cv)), T, bins=50, value_range=(-1.7, 1.7))
    mx_raw, _, _ = _pmf_deviation(pmf_raw, f_true, mask)
    res_fb = metad_weights(cv, t, hills, T, method="final-bias", skip_time=0.3 * t[-1])
    pmf_fb = pmf_1d(
        cv[res_fb.kept, 0], res_fb.weights[res_fb.kept], T, bins=50, value_range=(-1.7, 1.7)
    )
    mx_fb, rms_fb, _ = _pmf_deviation(pmf_fb, f_true, mask)
    record(
        "Reweighting",
        "Final-bias reweighting of the (unconverged) WT run",
        mx_fb < 1.5 and mx_fb < 0.2 * mx_raw,
        f"max |dF| = {mx_fb:.3f} kJ/mol (RMS {rms_fb:.3f}); unweighted histogram deviates by {mx_raw:.1f}",
        "0 (expected residual ~kT/gamma-level error until the bias has converged)",
        "1.5 kJ/mol and >5x better than unweighted",
        "analytical double well; Branduardi et al. JCTC 2012 discuss the static-bias assumption",
    )
    # (d) final-bias reweighting is exact for an equilibrium sample of U + V_final
    xs = sample_static_bias(hills, T, n_steps=150_000)
    res_eq = metad_weights(xs, np.arange(xs.size, dtype=float), hills, T, method="final-bias")
    pmf_eq = pmf_1d(xs, res_eq.weights, T, bins=50, value_range=(-1.7, 1.7))
    mx_eq, rms_eq, nb_eq = _pmf_deviation(pmf_eq, f_true, mask)
    record(
        "Reweighting",
        "Final-bias reweighting of an equilibrium sample under the static final bias",
        mx_eq < 0.3,
        f"max |dF| = {mx_eq:.3f} kJ/mol, RMS {rms_eq:.3f} over {nb_eq} bins",
        "0",
        "0.3 kJ/mol",
        "analytical double well (exp(beta V) reweighting is exact for a static bias)",
    )
    # (e) c(t) in the untempered limit vs direct quadrature for one hill
    from conformatlas.reweighting import Hills, metad_rct

    one = Hills(
        time=np.array([0.0]),
        centers=np.array([[0.3]]),
        sigma=np.array([[0.2]]),
        height=np.array([2.0]),
        biasf=np.array([-1.0]),
        cv_names=["x"],
        periodic=[None],
        heights_rescaled=False,
    )
    g = np.linspace(-2, 2, 4001)
    _, c = metad_rct(one, T, grid_axes=[g])
    v = hills_bias(one, g)
    c_ref = kT * np.log(np.mean(np.exp(v / kT)))
    record(
        "Reweighting",
        "c(t) untempered limit vs direct grid average",
        abs(c[0] - c_ref) < 1e-10,
        f"{c[0]:.10f} vs {c_ref:.10f} kJ/mol",
        "equal",
        "1e-10",
        "Tiwary & Parrinello 2015 eq. 12 with gamma -> inf",
    )


def check_umbrella(record, tmp: Path, KB: float):
    from conformatlas.reweighting import UmbrellaWindow, mbar, mbar_solve, mbar_weights, wham

    T = 300.0
    kT = KB * T
    centers = np.arange(-1.6, 1.61, 0.2)
    kappa = 250.0
    n_steps = 60_000
    samples = sample_umbrella_windows(centers, kappa, T, n_steps)
    windows = [
        UmbrellaWindow(samples[k], centers[k], kappa, name=f"w{k}") for k in range(len(centers))
    ]
    rng_ = (-1.7, 1.7)
    edges = np.linspace(*rng_, 61)
    f_true = bin_integrated_free_energy(edges, kT)
    res_w = wham(windows, T, bins=60, value_range=rng_, tol=1e-9)
    res_m = mbar(windows, T, bins=60, value_range=rng_, tol=1e-12)
    for res, name, ref in (
        (res_w, "WHAM", "Kumar et al. 1992"),
        (res_m, "MBAR", "Shirts & Chodera 2008"),
    ):
        counts = res.pmf["count"].to_numpy()
        mx, rms, nb = _pmf_deviation(res.pmf, f_true, counts > 500)
        record(
            "Reweighting",
            f"{name} PMF vs analytical double well",
            mx < 0.3 and res.converged,
            f"max |dF| = {mx:.3f} kJ/mol, RMS {rms:.3f} over {nb} bins (converged in {res.n_iter} it.)",
            "0",
            "0.3 kJ/mol over the sampled range (bins with > 500 samples)",
            f"analytical bin-integrated free energy; {ref}",
        )
    # MBAR vs pymbar (free energies and unbiased weights)
    try:
        import pymbar
    except ImportError:
        record(
            "Reweighting",
            "MBAR vs pymbar",
            True,
            "pymbar missing",
            "-",
            "-",
            "pymbar",
            note="skipped",
        )
        return
    x_all = np.concatenate(samples)
    n_k = np.array([n_steps] * len(centers) + [0])
    u_kn = np.array([0.5 * kappa * (x_all - c) ** 2 / kT for c in centers] + [np.zeros(x_all.size)])
    ref = pymbar.MBAR(u_kn, n_k, relative_tolerance=1e-12)
    f_ref = ref.f_k - ref.f_k[0]
    f_me, _, conv = mbar_solve(u_kn, n_k, tol=1e-12)
    w_ref = ref.weights()[:, -1]
    w_me = mbar_weights(u_kn, n_k, f_me)
    df = float(np.abs(f_me - f_ref).max())
    dw = float(np.abs(w_me - w_ref).max() / w_ref.max())
    record(
        "Reweighting",
        "MBAR free energies vs pymbar.MBAR",
        df < 1e-3 and conv,
        f"max |df_k| = {df:.1e} kT (incl. unsampled unbiased state)",
        "0",
        "1e-3 kT",
        "pymbar 4 MBAR (adaptive solver, rtol 1e-12)",
    )
    record(
        "Reweighting",
        "MBAR unbiased weights vs pymbar.MBAR.weights()",
        dw < 1e-6,
        f"max relative |dw_n| = {dw:.1e}",
        "0",
        "1e-6 (relative to the largest weight)",
        "pymbar 4",
    )
    dwm = float(np.abs(res_w.reduced_free_energies - res_m.reduced_free_energies).max())
    record(
        "Reweighting",
        "WHAM vs MBAR window free energies",
        dwm < 0.05,
        f"max |df_k| = {dwm:.4f} kT",
        "0 (histogram discretisation only)",
        "0.05 kT",
        "internal consistency",
    )


def check_dccm(record, tmp: Path):
    from conformatlas.residue_analysis import dccm

    rng = np.random.default_rng(0)
    nf, na = 400, 15
    base = rng.normal(size=(nf, na, 3))
    mode = rng.normal(size=(nf, 1, 1)) * np.linspace(-1, 1, na)[None, :, None]
    x = base + 2 * mode + 10 * rng.normal(size=(1, na, 3))
    c = dccm(x)
    d = x - x.mean(0)
    ref = np.empty((na, na))
    for i in range(na):
        for j in range(na):
            ref[i, j] = np.mean(np.sum(d[:, i] * d[:, j], 1)) / np.sqrt(
                np.mean(np.sum(d[:, i] ** 2, 1)) * np.mean(np.sum(d[:, j] ** 2, 1))
            )
    err = float(np.abs(c - ref).max())
    record(
        "Residues",
        "DCCM vs explicit <dr_i.dr_j>/sqrt(<dr_i^2><dr_j^2>)",
        err < 1e-12,
        f"max |dC| = {err:.1e}",
        "0",
        "1e-12",
        "explicit numpy loop (Ichiye & Karplus 1991)",
    )
    try:
        import prody

        prody.confProDy(verbosity="none")
        ens = prody.Ensemble()
        ens.setCoords(x.mean(0))
        ens.addCoordset(x)
        eda = prody.EDA()
        eda.buildCovariance(ens)
        eda.calcModes(n_modes=None)
        cp = prody.calcCrossCorr(eda, norm=True)
        errp = float(np.abs(c - cp).max())
        record(
            "Residues",
            "DCCM vs ProDy calcCrossCorr (EDA, all modes)",
            errp < 1e-10,
            f"max |dC| = {errp:.1e}",
            "0",
            "1e-10",
            "ProDy EDA cross-correlation from the full covariance",
        )
    except ImportError:
        record(
            "Residues", "DCCM vs ProDy", True, "prody missing", "-", "-", "ProDy", note="skipped"
        )
    w = rng.integers(1, 4, nf).astype(float)
    xd = np.repeat(x, w.astype(int), axis=0)
    errw = float(np.abs(dccm(x, w) - dccm(xd)).max())
    record(
        "Residues",
        "Weighted DCCM equals DCCM of frame-replicated ensemble",
        errw < 1e-12,
        f"max |dC| = {errw:.1e}",
        "0",
        "1e-12",
        "integer weights = frame multiplicity",
    )


def check_state_differences(record):
    import pandas as pd

    from conformatlas.residue_analysis import state_residue_differences

    rng = np.random.default_rng(2)
    n_res, nf = 30, 2000
    base = np.cumsum(rng.normal(size=(n_res, 3)), axis=0) * 3.8
    atom_table = pd.DataFrame(
        {
            "chain": "A",
            "residue_number": np.arange(1, n_res + 1),
            "residue_name": "GLY",
            "atom_name": "CA",
        }
    )
    moved = [5, 6, 7, 8]
    floppy = [15, 16, 17]
    shift = np.array([3.0, 0.0, 0.0])
    sig_a, sig_b_floppy = 0.5, 1.5
    coords = np.empty((2 * nf, n_res, 3))
    coords[:nf] = base + rng.normal(0, sig_a, (nf, n_res, 3))
    cb = base.copy()
    cb[[m - 1 for m in moved]] += shift
    noise_b = rng.normal(0, sig_a, (nf, n_res, 3))
    noise_b[:, [f - 1 for f in floppy]] *= sig_b_floppy / sig_a
    coords[nf:] = cb + noise_b
    states = np.array(["A"] * nf + ["B"] * nf)
    df = state_residue_differences(coords, atom_table, states, "A", "B")
    disp = df.set_index("residue_number")["mean_rmsd"]
    d_rmsf = df.set_index("residue_number")["delta_rmsf"]
    # expectation: moved residues ~3 A, others ~ sqrt(2*3*sig^2/nf) ~ 0.05 A
    exp_disp = float(np.linalg.norm(shift))
    err_moved = float(np.abs(disp.loc[moved] - exp_disp).max())
    max_other = float(disp.drop(moved).max())
    exp_drmsf = np.sqrt(3) * (sig_b_floppy - sig_a)
    err_floppy = float(np.abs(d_rmsf.loc[floppy] - exp_drmsf).max())
    max_other_rmsf = float(np.abs(d_rmsf.drop(floppy)).max())
    record(
        "Residues",
        "Per-residue RMSD of state means detects displaced residues",
        err_moved < 0.1 and max_other < 0.15,
        f"moved residues: |d - 3.0 A| <= {err_moved:.3f}; max elsewhere {max_other:.3f} A",
        "3.0 A for residues 5-8, ~0.03 A elsewhere (noise)",
        "0.1 A / 0.15 A",
        "synthetic two-state ensemble with a known 3 A shift",
    )
    record(
        "Residues",
        "Delta RMSF detects residues with increased fluctuation",
        err_floppy < 0.1 and max_other_rmsf < 0.1,
        f"floppy residues: |dRMSF - {exp_drmsf:.3f}| <= {err_floppy:.3f}; max elsewhere {max_other_rmsf:.3f} A",
        f"sqrt(3)*(1.5-0.5) = {exp_drmsf:.3f} A for residues 15-17, 0 elsewhere",
        "0.1 A (~4 standard errors for 2000 frames per state)",
        "synthetic ensemble with known per-residue isotropic noise",
    )


def check_contacts(record, ROOT: Path):
    import MDAnalysis as mda
    import pandas as pd

    from conformatlas.residue_analysis import contact_persistence, ligand_residue_contacts

    top = ROOT / "tests/data/h1n1_ca/neuraminidase_ca.pdb"
    trj = ROOT / "tests/data/h1n1_ca/neuraminidase_ca.xtc"
    res = ligand_residue_contacts(top, trj, "resname ZMR", "protein", cutoff=4.5, use_pbc=False)
    # brute-force numpy minimum distances, no PBC
    u = mda.Universe(str(top), str(trj))
    lig = u.select_atoms("resname ZMR and not name H*")
    prot = u.select_atoms("protein and not name H*")
    resids = prot.resids
    uniq = list(dict.fromkeys(resids.tolist()))
    err = 0.0
    mism = 0
    for k, ts in enumerate(u.trajectory):
        d = np.sqrt(((lig.positions[:, None, :] - prot.positions[None, :, :]) ** 2).sum(-1)).min(0)
        md = np.array([d[resids == r].min() for r in uniq])
        err = max(err, float(np.abs(md - res.min_distance[k]).max()))
        mism += int(np.sum((md < 4.5) != res.contacts[k]))
    record(
        "Residues",
        "Ligand-residue minimum distances vs brute-force numpy (neuraminidase fixture)",
        err < 1e-4 and mism == 0,
        f"max |dd| = {err:.1e} A, {mism} contact mismatches over {res.contacts.shape[0]} frames x {len(uniq)} residues",
        "0",
        "1e-4 A (float32 coordinates)",
        "explicit pairwise distances",
    )
    # MDAnalysis distance_array with the triclinic box: contacts of a compact
    # complex must not depend on the minimum-image convention
    res_pbc = ligand_residue_contacts(top, trj, "resname ZMR", "protein", cutoff=4.5, use_pbc=True)
    same = int(np.sum(res_pbc.contacts != res.contacts))
    record(
        "Residues",
        "Contacts with and without minimum image agree on the fixture",
        same == 0 and res_pbc.used_pbc,
        f"{same} differing residue-frame entries",
        "0",
        "exact",
        "MDAnalysis.lib.distances.distance_array(box=...)",
    )
    # persistence arithmetic vs direct fraction
    states = np.array(["S1" if i % 2 else "S2" for i in range(res.contacts.shape[0])])
    pers = contact_persistence(res, states)
    direct = res.contacts[states == "S1"].mean(0)
    errp = float(np.abs(pers["persistence_S1"].to_numpy() - direct).max())
    record(
        "Residues",
        "Contact persistence per state equals the fraction of state frames in contact",
        errp < 1e-12,
        f"max |dp| = {errp:.1e}; {int((pers['max_persistence'] >= 0.5).sum())} residues >= 50 %",
        "0",
        "1e-12",
        "direct boolean mean",
    )
    del pd


def check_dihedrals(record, ROOT: Path, tmp: Path):
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_dihedrals

    from conformatlas.features import compute_dihedral, dihedral_definitions

    th = np.deg2rad(70.0)
    p0, p1, p2 = np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 0, 1.0])
    p3 = np.array([np.cos(th), np.sin(th), 1.0])
    hand = np.rad2deg(compute_dihedral(p0, p1, p2, p3))
    record(
        "Dihedrals",
        "Dihedral of a hand-built geometry (IUPAC sign)",
        abs(hand - 70.0) < 1e-9,
        f"{hand:.9f} deg",
        "+70 deg (p3 rotated clockwise by 70 deg looking along p1->p2)",
        "1e-9 deg",
        "IUPAC 1974 torsion-angle convention",
    )
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(4, 200, 3))
    dev = float(np.abs(compute_dihedral(*pts) - calc_dihedrals(*pts)).max())
    record(
        "Dihedrals",
        "compute_dihedral vs MDAnalysis calc_dihedrals (random geometries)",
        dev < 1e-5,
        f"max |d| = {dev:.1e} rad",
        "0",
        "1e-5 rad (MDAnalysis evaluates in float32)",
        "MDAnalysis.lib.distances.calc_dihedrals",
    )
    gro = Path.home() / "Downloads/H1N1_ZAN/md_0_1.gro"
    if not gro.exists():
        record(
            "Dihedrals",
            "chi1/chi2 vs MDAnalysis Janin",
            True,
            "H1N1 all-atom structure not available",
            "-",
            "-",
            "MDAnalysis Janin",
            note="skipped",
        )
        return
    from MDAnalysis.analysis.dihedrals import Dihedral, Janin, Ramachandran

    u = mda.Universe(str(gro))
    prot = u.select_atoms("protein")
    labels, groups = dihedral_definitions(prot.residues, "backbone+chi")
    ang = Dihedral(groups).run().results.angles[0]
    ours = {(lab["residue_number"], lab["angle"]): a for lab, a in zip(labels, ang)}
    jan = Janin(prot).run()
    ja = jan.results.angles[0]
    jres = jan.ag1.residues.resids

    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0

    chi1 = np.array([ours[(r, "chi1")] for r in jres])
    chi2 = np.array([ours[(r, "chi2")] for r in jres])
    d1 = float(np.abs(wrap(chi1 - ja[:, 0])).max())
    d2 = float(np.abs(wrap(chi2 - ja[:, 1])).max())
    record(
        "Dihedrals",
        "chi1/chi2 vs MDAnalysis Janin (neuraminidase, all-atom)",
        d1 < 1e-9 and d2 < 1e-9,
        f"max |dchi1| = {d1:.1e}, max |dchi2| = {d2:.1e} deg over {len(jres)} residues",
        "0",
        "1e-9 deg",
        "MDAnalysis.analysis.dihedrals.Janin",
    )
    ram = Ramachandran(prot).run()
    ra = ram.results.angles[0]
    rres = ram.ag2.residues.resids
    phi = np.array([ours[(r, "phi")] for r in rres])
    psi = np.array([ours[(r, "psi")] for r in rres])
    dp = float(max(np.abs(phi - ra[:, 0]).max(), np.abs(psi - ra[:, 1]).max()))
    record(
        "Dihedrals",
        "phi/psi of explicit definitions vs MDAnalysis Ramachandran",
        dp < 1e-9 and len(rres) == sum(1 for lab in labels if lab["angle"] == "phi"),
        f"max |d| = {dp:.1e} deg over {len(rres)} residues",
        "0",
        "1e-9 deg",
        "MDAnalysis.analysis.dihedrals.Ramachandran",
    )


def check_chain_mapping(record, tmp: Path):
    import MDAnalysis as mda

    from conformatlas.trajectory import build_common_atom_mapping

    def write_pdb(path, chains, offset=0.0):
        n_res = 4
        atoms_per_res = ["N", "CA", "C"]
        n_atoms = len(chains) * n_res * len(atoms_per_res)
        u = mda.Universe.empty(
            n_atoms,
            n_residues=len(chains) * n_res,
            n_segments=len(chains),
            atom_resindex=np.repeat(np.arange(len(chains) * n_res), len(atoms_per_res)),
            residue_segindex=np.repeat(np.arange(len(chains)), n_res),
            trajectory=True,
        )
        u.add_TopologyAttr("name", atoms_per_res * (len(chains) * n_res))
        u.add_TopologyAttr("resname", ["ALA"] * (len(chains) * n_res))
        u.add_TopologyAttr("resid", list(range(1, n_res + 1)) * len(chains))
        u.add_TopologyAttr("segid", list(chains))
        u.add_TopologyAttr("chainIDs", np.repeat(list(chains), n_res * len(atoms_per_res)))
        u.atoms.positions = np.arange(n_atoms * 3).reshape(n_atoms, 3) + offset
        u.atoms.write(str(path))

    a, b = tmp / "two_chain_a.pdb", tmp / "two_chain_b.pdb"
    write_pdb(a, ["A", "B"])
    write_pdb(b, ["B", "A"])  # chains in opposite order
    idx, df = build_common_atom_mapping({"a": a, "b": b}, group="Backbone")
    n_common = len(idx["a"])
    # the mapping must pair residue 1 of chain A in a with residue 1 of chain A in b
    ua, ub = mda.Universe(str(a)), mda.Universe(str(b))
    pairs_ok = all(
        (ua.atoms[i].chainID, ua.atoms[i].resid, ua.atoms[i].name)
        == (ub.atoms[j].chainID, ub.atoms[j].resid, ub.atoms[j].name)
        for i, j in zip(idx["a"], idx["b"])
    )
    record(
        "Mapping",
        "Chain-aware common-atom mapping on two chains with repeated residue numbers",
        n_common == 24 and pairs_ok,
        f"{n_common} common atoms, chain/resid/name pairs consistent: {pairs_ok}",
        "24 atoms (2 chains x 4 residues x N,CA,C), all pairs consistent",
        "exact",
        "constructed topology",
    )
    idx_legacy, _ = build_common_atom_mapping({"a": a, "b": b}, group="Backbone", chain_aware=False)
    record(
        "Mapping",
        "Legacy (residue number, atom name) key collapses repeated residue numbers",
        len(idx_legacy["a"]) == 12,
        f"{len(idx_legacy['a'])} atoms",
        "12 (one copy per residue number)",
        "exact",
        "documents the previous behaviour the chain-aware key fixes",
    )


def run(record, tmp: Path, ROOT: Path, KB: float):
    """Run every check of this track; ``record`` follows validation/validate.py."""
    tmp = Path(tmp)
    check_plumed_readers(record, tmp)
    check_umbrella(record, tmp, KB)
    check_metadynamics(record, tmp, KB)
    check_dccm(record, tmp)
    check_state_differences(record)
    check_contacts(record, ROOT)
    check_dihedrals(record, ROOT, tmp)
    check_chain_mapping(record, tmp)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    results: list[dict] = []

    def record(section, name, passed, measured, expected, tolerance, reference, note=""):
        results.append(
            {
                "section": section,
                "check": name,
                "pass": bool(passed),
                "measured": measured,
                "expected": expected,
                "tolerance": tolerance,
                "reference": reference,
                "note": note,
            }
        )
        flag = "SKIP" if note == "skipped" else ("PASS" if passed else "FAIL")
        print(f"[{flag}] {section} / {name}: {measured} vs {expected} ({tolerance})", flush=True)

    with tempfile.TemporaryDirectory() as td:
        run(record, Path(td), ROOT, R_GAS)
    n_fail = sum(1 for r in results if not r["pass"])
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    import json

    out = ROOT / "validation" / "results_enhanced.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    sys.exit(1 if n_fail else 0)
