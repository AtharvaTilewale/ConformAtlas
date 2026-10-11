"""PLUMED readers, metadynamics reweighting, WHAM and MBAR."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from conformatlas.features import read_weights_file
from conformatlas.reweighting import (
    Hills,
    UmbrellaWindow,
    default_grid,
    hills_bias,
    hills_bias_at_times,
    match_times,
    mbar,
    mbar_solve,
    mbar_weights,
    metad_rct,
    metad_weights,
    pmf_1d,
    read_colvar,
    read_hills,
    read_plumed_file,
    umbrella_windows_from_colvars,
    wham,
    write_umbrella_outputs,
    write_weights_file,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
from checks_enhanced import (  # noqa: E402
    bin_integrated_free_energy,
    sample_umbrella_windows,
    simulate_metadynamics,
)

KB = 0.008314462618
T = 300.0
kT = KB * T


# ----------------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------------


def test_read_plumed_file_fields_and_constants(tmp_path):
    p = tmp_path / "COLVAR"
    p.write_text(
        "#! FIELDS time d1 d2\n#! SET min_d1 -pi\n#! SET max_d1 pi\n"
        "0.0 1.0 2.0\n# a comment\n1.0 1.5 2.5\n2.0 2.0 3.0\n"
    )
    tab = read_plumed_file(p)
    assert tab.fields == ["time", "d1", "d2"]
    assert tab.constants == {"min_d1": "-pi", "max_d1": "pi"}
    np.testing.assert_array_equal(tab.data, np.loadtxt(p, comments="#"))
    np.testing.assert_array_equal(tab.column("d2"), [2.0, 2.5, 3.0])
    vals, t, _ = read_colvar(p, ["d2", "d1"])
    np.testing.assert_array_equal(vals[:, 0], [2.0, 2.5, 3.0])
    np.testing.assert_array_equal(t, [0.0, 1.0, 2.0])
    with pytest.raises(KeyError):
        tab.column("nope")


def test_read_plumed_file_requires_fields_header(tmp_path):
    p = tmp_path / "bad"
    p.write_text("0 1 2\n")
    with pytest.raises(ValueError):
        read_plumed_file(p)


def test_read_hills_well_tempered_rescaling_and_periodic(tmp_path):
    p = tmp_path / "HILLS"
    p.write_text(
        "#! FIELDS time phi sigma_phi height biasf\n#! SET multivariate false\n"
        "#! SET min_phi -pi\n#! SET max_phi pi\n"
        "1.0 0.5 0.3 1.2 10.0\n2.0 -3.0 0.3 1.1 10.0\n"
    )
    h = read_hills(p)
    assert h.cv_names == ["phi"]
    assert h.bias_factor == 10.0
    np.testing.assert_allclose(h.deposited_height, np.array([1.2, 1.1]) * 0.9)
    assert h.periodic[0] == (-np.pi, np.pi)
    # forcing no rescaling keeps the file values
    assert np.allclose(read_hills(p, heights_rescaled=False).deposited_height, [1.2, 1.1])


def test_read_hills_standard_metad_and_multivariate_rejected(tmp_path):
    p = tmp_path / "HILLS"
    p.write_text(
        "#! FIELDS time d sigma_d height biasf\n#! SET multivariate false\n1 0.5 0.1 1.0 -1\n"
    )
    h = read_hills(p)
    assert h.bias_factor is None
    np.testing.assert_allclose(h.deposited_height, [1.0])
    p.write_text(
        "#! FIELDS time d sigma_d height biasf\n#! SET multivariate true\n1 0.5 0.1 1.0 -1\n"
    )
    with pytest.raises(ValueError, match="multivariate"):
        read_hills(p)


# ----------------------------------------------------------------------------
# bias reconstruction
# ----------------------------------------------------------------------------


def _one_hill(center=0.0, sigma=0.2, height=2.0, periodic=None, t=0.0):
    return Hills(
        time=np.array([t]),
        centers=np.array([[center]]),
        sigma=np.array([[sigma]]),
        height=np.array([height]),
        biasf=np.array([-1.0]),
        cv_names=["x"],
        periodic=[periodic],
        heights_rescaled=False,
    )


def test_hill_shape_plumed_and_gaussian():
    h = _one_hill()
    assert hills_bias(h, np.array([0.0]))[0] == pytest.approx(2.0)
    # plain Gaussian at one sigma
    assert hills_bias(h, np.array([0.2]), hill_shape="gaussian")[0] == pytest.approx(
        2.0 * np.exp(-0.5)
    )
    # stretched Gaussian is zero at and beyond the cutoff (d^2/2s^2 = 6.25 -> d = sqrt(12.5) sigma)
    d_cut = np.sqrt(12.5) * 0.2
    assert hills_bias(h, np.array([d_cut + 1e-9]))[0] == 0.0
    assert hills_bias(h, np.array([d_cut - 1e-6]))[0] == pytest.approx(0.0, abs=1e-5)
    assert hills_bias(h, np.array([d_cut + 1e-9]), hill_shape="gaussian")[0] > 0


def test_periodic_wrapping():
    h = _one_hill(center=np.pi - 0.1, periodic=(-np.pi, np.pi))
    v_near = hills_bias(h, np.array([-np.pi + 0.1]))[0]  # 0.2 away through the boundary
    v_ref = hills_bias(_one_hill(center=0.0), np.array([0.2]))[0]
    assert v_near == pytest.approx(v_ref)


def test_time_dependent_bias_excludes_hill_at_frame_time():
    hills = Hills(
        time=np.array([0.0, 1.0, 2.0]),
        centers=np.zeros((3, 1)),
        sigma=np.full((3, 1), 0.2),
        height=np.ones(3),
        biasf=np.full(3, -1.0),
        cv_names=["x"],
        periodic=[None],
        heights_rescaled=False,
    )
    pts = np.zeros(4)
    v = hills_bias_at_times(hills, pts, np.array([0.0, 1.0, 1.5, 2.5]))
    np.testing.assert_allclose(v, [0.0, 1.0, 2.0, 3.0])
    assert hills_bias(hills, np.array([0.0]), t_max=1.0)[0] == pytest.approx(2.0)


def test_rct_untempered_and_well_tempered_limits():
    h = _one_hill()
    g = np.linspace(-2, 2, 2001)
    _, c = metad_rct(h, T, grid_axes=[g])
    v = hills_bias(h, g)
    assert c[0] == pytest.approx(kT * np.log(np.mean(np.exp(v / kT))), abs=1e-10)
    # well-tempered formula with gamma given explicitly
    gamma = 5.0
    _, cw = metad_rct(h, T, grid_axes=[g], bias_factor=gamma)
    num = np.log(np.sum(np.exp(gamma / (gamma - 1) * v / kT)))
    den = np.log(np.sum(np.exp(1 / (gamma - 1) * v / kT)))
    assert cw[0] == pytest.approx(kT * (num - den), abs=1e-10)
    # a very large gamma approaches the untempered limit
    _, cl = metad_rct(h, T, grid_axes=[g], bias_factor=1e6)
    assert cl[0] == pytest.approx(c[0], abs=1e-4)


def test_default_grid_covers_hills_and_period():
    h = _one_hill(center=1.0, sigma=0.1)
    (ax,) = default_grid(h, bins=11)
    assert ax[0] < 1.0 - 0.3 and ax[-1] > 1.0 + 0.3
    hp = _one_hill(center=1.0, periodic=(-np.pi, np.pi))
    (axp,) = default_grid(hp, bins=10)
    assert axp[0] == pytest.approx(-np.pi) and len(axp) == 10


# ----------------------------------------------------------------------------
# metadynamics end to end on a synthetic run
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def metad_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("metad")
    simulate_metadynamics(d / "HILLS", d / "COLVAR", n_steps=150_000, pace=150)
    return d


def test_metad_bias_matches_colvar_column(metad_run):
    hills = read_hills(metad_run / "HILLS")
    cv, t, tab = read_colvar(metad_run / "COLVAR", ["x"])
    v = hills_bias_at_times(hills, cv, t)
    assert np.abs(v - tab.column("metad.bias")).max() < 1e-6


def test_tiwary_parrinello_recovers_boltzmann(metad_run):
    hills = read_hills(metad_run / "HILLS")
    cv, t, _ = read_colvar(metad_run / "COLVAR", ["x"])
    res = metad_weights(cv, t, hills, T, method="tiwary-parrinello", bins=300)
    assert res.weights[res.kept].mean() == pytest.approx(1.0)
    assert res.bias_factor == 8.0
    pmf = pmf_1d(cv[:, 0], res.weights, T, bins=40, value_range=(-1.7, 1.7))
    edges = np.linspace(-1.7, 1.7, 41)
    f_true = bin_integrated_free_energy(edges, kT)
    f = pmf["pmf_kJ_mol"].to_numpy()
    m = np.isfinite(f) & (f_true < 8)
    d = f[m] - f_true[m]
    d -= d.mean()
    assert np.abs(d).max() < 0.6  # shorter run than the validation check (0.3 kJ/mol there)
    raw = pmf_1d(cv[:, 0], np.ones(len(cv)), T, bins=40, value_range=(-1.7, 1.7))[
        "pmf_kJ_mol"
    ].to_numpy()
    draw = raw[m] - f_true[m]
    assert np.abs(draw - draw.mean()).max() > 5 * np.abs(d).max()


def test_final_bias_and_skip_time(metad_run):
    hills = read_hills(metad_run / "HILLS")
    cv, t, _ = read_colvar(metad_run / "COLVAR", ["x"])
    res = metad_weights(cv, t, hills, T, method="final-bias", skip_time=50.0)
    assert np.all(res.weights[t < 50.0] == 0)
    assert res.weights[t >= 50.0].mean() == pytest.approx(1.0)
    np.testing.assert_allclose(res.bias, hills_bias(hills, cv))
    with pytest.raises(ValueError):
        metad_weights(cv, t, hills, T, method="nonsense")


def test_weights_file_roundtrip_and_time_matching(tmp_path):
    t = np.arange(0, 100, 2.0)
    w = np.linspace(0.5, 1.5, t.size)
    p = write_weights_file(tmp_path / "w.dat", t, w, header="test")
    np.testing.assert_allclose(read_weights_file(p), w, rtol=1e-9)
    frame_times = np.array([0.0, 10.0, 20.5, 98.0])
    idx = match_times(frame_times, t, tolerance=1.0)
    np.testing.assert_array_equal(t[idx], [0.0, 10.0, 20.0, 98.0])
    with pytest.raises(ValueError):
        match_times(np.array([150.0]), t, tolerance=1.0)


# ----------------------------------------------------------------------------
# umbrella sampling
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def umbrella_samples():
    centers = np.arange(-1.6, 1.61, 0.2)
    return centers, 250.0, sample_umbrella_windows(centers, 250.0, T, 20_000)


def test_wham_and_mbar_pmf_vs_analytical(umbrella_samples):
    centers, kappa, samples = umbrella_samples
    windows = [UmbrellaWindow(samples[k], centers[k], kappa) for k in range(len(centers))]
    edges = np.linspace(-1.7, 1.7, 41)
    f_true = bin_integrated_free_energy(edges, kT)
    for fn in (wham, mbar):
        res = fn(windows, T, bins=40, value_range=(-1.7, 1.7))
        assert res.converged
        f = res.pmf["pmf_kJ_mol"].to_numpy()
        m = np.isfinite(f) & (res.pmf["count"].to_numpy() > 300)
        d = f[m] - f_true[m]
        d -= d.mean()
        assert np.abs(d).max() < 0.4, fn.__name__
        assert np.isclose(sum(w.sum() for w in res.weights), 1.0)
        assert res.window_free_energies[0] == 0.0


def test_mbar_matches_pymbar(umbrella_samples):
    pymbar = pytest.importorskip("pymbar")
    centers, kappa, samples = umbrella_samples
    x = np.concatenate(samples)
    n_k = np.array([samples.shape[1]] * len(centers) + [0])
    u_kn = np.array([0.5 * kappa * (x - c) ** 2 / kT for c in centers] + [np.zeros(x.size)])
    ref = pymbar.MBAR(u_kn, n_k, relative_tolerance=1e-12)
    f_me, _, conv = mbar_solve(u_kn, n_k, tol=1e-12)
    assert conv
    assert np.abs(f_me - (ref.f_k - ref.f_k[0])).max() < 1e-6
    w_ref = ref.weights()[:, -1]
    assert np.abs(mbar_weights(u_kn, n_k, f_me) - w_ref).max() / w_ref.max() < 1e-8


def test_mbar_solve_identical_states_and_input_checks():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(1, 100))
    u_kn = np.vstack([u, u])
    f, _, conv = mbar_solve(u_kn, np.array([50, 50]))
    assert conv and f[1] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        mbar_solve(u_kn, np.array([50, 40]))


def test_umbrella_from_colvars_and_outputs(tmp_path, umbrella_samples):
    centers, kappa, samples = umbrella_samples
    paths = []
    for k in range(len(centers)):
        p = tmp_path / f"COLVAR_{k}"
        with open(p, "w") as fh:
            fh.write("#! FIELDS time d\n")
            for i, v in enumerate(samples[k][:2000]):
                fh.write(f"{i * 0.5:.3f} {v:.8f}\n")
        paths.append(p)
    windows, times = umbrella_windows_from_colvars(
        paths, "d", list(centers), kappa, skip_time=100.0
    )
    assert all(len(w.samples) == 1800 for w in windows)
    assert times[0][0] == pytest.approx(100.0)
    res = mbar(windows, T, bins=30)
    written = write_umbrella_outputs(res, tmp_path / "out", times, [p.name for p in paths])
    assert (tmp_path / "out" / "pmf.csv").exists()
    w0 = read_weights_file(written["weights_COLVAR_0"])
    assert len(w0) == 1800
    n_total = sum(len(w.samples) for w in windows)
    all_w = np.concatenate([read_weights_file(written[f"weights_{p.name}"]) for p in paths])
    assert all_w.mean() == pytest.approx(1.0, rel=1e-6)
    assert len(all_w) == n_total
