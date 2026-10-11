"""Validation checks for the MSM / statistics track (conformatlas.msm,
conformatlas.plotting_msm, bootstrap re-segmentation mode,
timeseries.detect_equilibration_multi).

Exposes ``run(record, tmp, ROOT, KB)`` with the ``record(section, check,
passed, measured, expected, tolerance, reference, note="")`` convention of
validation/validate.py. Every check compares against an analytical result or an
independent implementation (deeptime 0.4.x, pymbar 4, SciPy) on synthetic
systems with known truth. Run standalone with ``python validation/checks_msm.py``.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def _markov_chain(T, n, rng, s0=0):
    cum = np.cumsum(T, axis=1)
    s = np.empty(n, dtype=np.int64)
    s[0] = s0
    u = rng.random(n)
    for i in range(1, n):
        s[i] = np.searchsorted(cum[s[i - 1]], u[i])
    return s


def _reversible(T):
    from conformatlas.msm import stationary_distribution

    pi = stationary_distribution(T)
    X = pi[:, None] * T
    X = 0.5 * (X + X.T)
    return X / X.sum(axis=1)[:, None]


def _projection_df(xy, rep="R", dt=1.0):
    n = len(xy)
    return pd.DataFrame(
        {
            "system": "S",
            "replicate": rep,
            "frame": np.arange(n),
            "time_ps": dt * np.arange(n, dtype=float),
            "PC1": xy[:, 0],
            "PC2": xy[:, 1],
        }
    )


# =============================================================================
# MSM estimators
# =============================================================================
def check_msm(record, KB):
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov import pcca as dpcca
    from deeptime.markov.msm import MaximumLikelihoodMSM
    from deeptime.markov.tools.analysis import mfpt as dmfpt
    from scipy.optimize import linear_sum_assignment

    from conformatlas import msm as M

    dt_version = __import__("deeptime").__version__
    rng = np.random.default_rng(100)
    T4 = _reversible(
        np.array(
            [
                [0.90, 0.07, 0.02, 0.01],
                [0.05, 0.88, 0.05, 0.02],
                [0.01, 0.04, 0.90, 0.05],
                [0.02, 0.02, 0.06, 0.90],
            ]
        )
    )
    pi4 = M.stationary_distribution(T4)
    n = 1_000_000
    s = _markov_chain(T4, n, rng)
    dtrajs = [s[: n // 2], s[n // 2 :]]

    # --- counts vs deeptime (exact)
    exact = True
    for lag in (1, 3, 7):
        C = M.count_matrix(dtrajs, lag)
        ref = TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit_fetch(dtrajs)
        exact &= bool(np.array_equal(C, ref.count_matrix))
    record(
        "MSM",
        "sliding count matrix within replicates vs deeptime (lags 1, 3, 7)",
        exact,
        "identical" if exact else "differs",
        "identical",
        "exact",
        f"deeptime {dt_version} TransitionCountEstimator(count_mode='sliding')",
    )

    # --- reversible MLE vs deeptime and vs truth
    C1 = M.count_matrix(dtrajs, 1)
    Trev, info = M.transition_matrix_reversible(C1, return_info=True)
    dm = MaximumLikelihoodMSM(reversible=True).fit_fetch(
        TransitionCountEstimator(lagtime=1, count_mode="sliding").fit_fetch(dtrajs)
    )
    d = float(np.max(np.abs(Trev - dm.transition_matrix)))
    record(
        "MSM",
        "reversible maximum-likelihood T vs deeptime MaximumLikelihoodMSM(reversible=True)",
        d < 1e-6,
        f"max |ΔT| = {d:.1e} ({info['iterations']} fixed-point iterations)",
        "0",
        "< 1e-6",
        f"deeptime {dt_version}",
    )
    abs_err = float(np.max(np.abs(Trev - T4)))
    se = np.sqrt(T4 * (1 - T4) / C1.sum(axis=1)[:, None])
    z = float(np.max(np.abs(Trev - T4) / se))
    record(
        "MSM",
        "reversible T vs true T of a 4-state chain (10^6 steps)",
        abs_err < 0.005 and z < 4.5,
        f"max |ΔT_ij| = {abs_err:.4f}; max |z| = {z:.2f} (binomial SE)",
        "true T; |ΔT_ij| < 0.005 (0.5 % absolute)",
        "< 0.005 and |z| < 4.5",
        "analytical (known generating matrix)",
    )
    Tnr = M.transition_matrix_nonreversible(C1)
    abs_nr = float(np.max(np.abs(Tnr - T4)))
    record(
        "MSM",
        "row-normalised (non-reversible) T vs true T (10^6 steps)",
        abs_nr < 0.005,
        f"max |ΔT_ij| = {abs_nr:.4f}",
        "true T",
        "< 0.005",
        "analytical",
    )
    dpi = float(np.max(np.abs(info["stationary"] - dm.stationary_distribution)))
    dpi_true = float(np.max(np.abs(info["stationary"] - pi4)))
    record(
        "MSM",
        "stationary distribution vs deeptime and vs true π",
        dpi < 1e-6 and dpi_true < 0.005,
        f"max |Δπ| vs deeptime {dpi:.1e}; vs truth {dpi_true:.4f}",
        "π of the generating chain",
        "< 1e-6 (deeptime); < 0.005 (truth)",
        f"deeptime {dt_version}; analytical",
    )

    # --- implied timescales vs analytical -tau/ln(lambda^tau) = -1/ln(lambda)
    lags = [1, 2, 3, 5, 8, 10]
    its = M.implied_timescales_scan(dtrajs, lags, n_timescales=3, dt_ps=1.0)
    truth = -1.0 / np.log(np.sort(np.linalg.eigvals(T4).real)[::-1][1:])
    rel = max(
        float(np.max(np.abs(its[f"ts_{i + 1}"].to_numpy() - truth[i]) / truth[i])) for i in range(3)
    )
    dts = float(
        np.max(np.abs(M.implied_timescales(M.eigen_decomposition(Trev)[0], 1) - dm.timescales()))
    )
    record(
        "MSM",
        "implied timescales at lags 1-10 vs analytical −τ/ln λ(τ) (constant for a Markov chain)",
        rel < 0.05 and dts < 1e-5,
        f"max relative deviation {rel:.3f} over 3 timescales × 6 lags; vs deeptime {dts:.1e} ps",
        f"t = {np.array2string(truth, precision=2)} frames at every lag",
        "< 5 % (statistical); < 1e-5 vs deeptime",
        "analytical eigenvalues; deeptime MaximumLikelihoodMSM.timescales()",
    )
    rep = M.suggest_markov_lag(its, rel_tol=0.1)
    record(
        "MSM",
        "lag plateau report identifies lag 1 as Markovian for a true Markov chain",
        rep["plateau_found"] and rep["suggested_lag_frames"] == 1,
        f"suggested {rep['suggested_lag_frames']} frames",
        "1 frame",
        "exact",
        "construction (process is Markov at lag 1)",
        note="heuristic report; definition-level check only",
    )

    # --- PCCA+ vs deeptime
    def match(chi, ref):
        r, c = linear_sum_assignment(-(chi.T @ ref))
        return float(np.max(np.abs(chi[:, r] - ref[:, c]))), r, c

    B = rng.random((6, 6)) * 0.02
    np.fill_diagonal(B, 0)
    for a, b in ((0, 1), (2, 3), (4, 5)):
        B[a, b] = B[b, a] = 0.2
    T6 = B.copy()
    np.fill_diagonal(T6, 1 - B.sum(axis=1))
    T6 = _reversible(T6)
    d2, _, _ = match(M.pcca_plus(T4, 2), dpcca(T4, 2).memberships)
    chi3 = M.pcca_plus(T6, 3)
    ref3 = dpcca(T6, 3)
    d3, r, c = match(chi3, ref3.memberships)
    record(
        "MSM",
        "PCCA+ memberships vs deeptime.markov.pcca (m = 2 on 4 states; m = 3 on 6 states)",
        d2 < 1e-3 and d3 < 1e-3,
        f"max |Δχ| = {d2:.1e} (m=2), {d3:.1e} (m=3)",
        "deeptime memberships (columns matched)",
        "< 1e-3",
        f"deeptime {dt_version} (Roeblitz & Weber 2013 algorithm)",
    )
    Tc, _ = M.coarse_grain(T6, M.stationary_distribution(T6), chi3)
    dc = float(
        np.max(np.abs(Tc[np.ix_(r, r)] - ref3.coarse_grained_transition_matrix[np.ix_(c, c)]))
    )
    record(
        "MSM",
        "PCCA+ coarse-grained transition matrix vs deeptime",
        dc < 1e-3,
        f"max |ΔT_c| = {dc:.1e}",
        "deeptime coarse_grained_transition_matrix",
        "< 1e-3",
        f"deeptime {dt_version} (Kube & Weber 2007 projection)",
    )

    # --- MFPT: analytical 2-state and deeptime for sets
    T2 = np.array([[0.9, 0.1], [0.3, 0.7]])
    m2 = M.mfpt_between_sets(T2, M.stationary_distribution(T2), [np.array([0]), np.array([1])], 1)
    e2 = max(abs(m2[0, 1] - 10.0), abs(m2[1, 0] - 1 / 0.3))
    sets = [np.array([0, 1]), np.array([2, 3])]
    m4 = M.mfpt_between_sets(T4, pi4, sets, 1)
    e4 = max(
        abs(m4[0, 1] - dmfpt(T4, target=[2, 3], origin=[0, 1])),
        abs(m4[1, 0] - dmfpt(T4, target=[0, 1], origin=[2, 3])),
    )
    record(
        "MSM",
        "MFPT between sets: analytical 2-state (1/p_ij) and deeptime mfpt for sets",
        e2 < 1e-9 and e4 < 1e-8,
        f"|Δ| = {e2:.1e} (analytical), {e4:.1e} (deeptime)",
        "10 and 3.333 frames; deeptime",
        "< 1e-8",
        "analytical; deeptime.markov.tools.analysis.mfpt",
    )

    # --- Chapman-Kolmogorov: Markov passes, hidden-state process fails
    _, ck_ok = M.chapman_kolmogorov_test(dtrajs, 1, sets, k_max=5)
    H = np.array([[0.98, 0.02, 0.0], [0.02, 0.78, 0.20], [0.0, 0.20, 0.80]])
    h = _markov_chain(H, 400_000, rng)
    obs = np.where(h == 2, 1, 0)
    _, ck_bad = M.chapman_kolmogorov_test([obs], 1, [np.array([0]), np.array([1])], k_max=8)
    record(
        "MSM",
        "Chapman–Kolmogorov test passes for a Markov chain and fails for a hidden-state process",
        ck_ok["passed"] and ck_ok["max_abs_deviation"] < 0.01 and not ck_bad["passed"],
        f"Markov: max |dev| {ck_ok['max_abs_deviation']:.4f} (pass); hidden-state: "
        f"{ck_bad['max_abs_deviation']:.3f} ({ck_bad['n_violations']} violations, fail)",
        "pass / fail",
        "max |dev| < 0.01 vs > 0.05 (criterion max(0.05, 3 SE))",
        "construction: 3 hidden states (fast/slow exit) projected onto 2 observed states",
    )

    # --- clustering
    from scipy.cluster.vq import kmeans2
    from scipy.spatial.distance import cdist, pdist

    X = rng.normal(size=(5000, 3))
    X[:2500] += 3.0
    init = X[:4].copy()
    ours = M.kmeans_cluster(X, 4, init=init)
    cent, lab = kmeans2(X, init.copy(), minit="matrix", iter=300)
    dcent = float(np.max(np.abs(np.sort(cent, axis=0) - np.sort(ours.centers, axis=0))))
    in_scipy = float(cdist(X, cent, "sqeuclidean").min(axis=1).sum())
    record(
        "MSM",
        "k-means (Lloyd) vs scipy.cluster.vq.kmeans2 from the same initial centres",
        dcent < 1e-8 and abs(in_scipy - ours.inertia) < 1e-6 * in_scipy,
        f"max |Δcentre| = {dcent:.1e}; inertia {ours.inertia:.3f} vs {in_scipy:.3f}",
        "identical centres and objective",
        "< 1e-8; relative 1e-6",
        "SciPy kmeans2",
    )
    rs = M.regular_space_cluster(X, dmin=1.5)
    dmin_pair = float(pdist(rs.centers).min())
    dmax_assign = float(cdist(X, rs.centers).min(axis=1).max())
    record(
        "MSM",
        "regular-space clustering: centres > dmin apart and every frame within dmin of a centre",
        dmin_pair > 1.5 and dmax_assign <= 1.5,
        f"{rs.n_clusters} centres; min pair {dmin_pair:.3f}; max frame-centre {dmax_assign:.3f}",
        "> 1.5 and ≤ 1.5",
        "exact",
        "Prinz et al. 2011 definition",
    )

    # --- end-to-end on a 3-well landscape with known kinetics
    T3 = _reversible(np.array([[0.98, 0.015, 0.005], [0.02, 0.97, 0.01], [0.01, 0.02, 0.97]]))
    pi3 = M.stationary_distribution(T3)
    centres = np.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    dfs = []
    for r in range(3):
        st = _markov_chain(T3, 20000, rng, s0=r)
        xy = centres[st] + rng.normal(0, 0.5, size=(20000, 2))
        dfs.append(_projection_df(xy, rep=f"Rep_{r + 1}", dt=10.0))
    df = pd.concat(dfs, ignore_index=True)
    true_state = np.concatenate(
        [np.where(d["PC2"] > 2, 2, np.where(d["PC1"] < 0, 0, 1)) for d in dfs]
    )
    res = M.build_msm(
        df,
        ["PC1", "PC2"],
        lag=2,
        lags=[1, 2, 4, 8],
        n_clusters=60,
        n_metastable=3,
        basin_states=np.array(["State A", "State B", "State C"])[true_state],
    )
    t_true = -10.0 * 1.0 / np.log(np.sort(np.linalg.eigvals(T3).real)[::-1][1:3])
    t_est = res.msm.timescales_ps[:2]
    rel_t = float(np.max(np.abs(t_est - t_true) / t_true))
    pic = res.coarse_stationary["stationary_probability"].to_numpy()
    dpi3 = float(np.max(np.abs(np.sort(pic) - np.sort(pi3))))
    ovl = float(res.basin_overlap["row_fraction_max"].min())
    record(
        "MSM",
        "end-to-end 3-well landscape: slow timescales, PCCA+ populations and set/basin overlap",
        rel_t < 0.1 and dpi3 < 0.02 and ovl > 0.98 and res.ck_summary["passed"],
        f"t1,t2 = {t_est[0]:.0f}, {t_est[1]:.0f} ps (truth {t_true[0]:.0f}, {t_true[1]:.0f}; "
        f"max rel {rel_t:.3f}); max |Δπ_set| = {dpi3:.3f}; min set/basin overlap {ovl:.3f}; "
        f"CK {'pass' if res.ck_summary['passed'] else 'fail'}",
        "generating 3-state chain",
        "timescales < 10 %; π < 0.02; overlap > 0.98",
        "analytical (known T, 60 k-means microstates, 3 × 20 000 frames)",
    )


# =============================================================================
# Bootstrap re-segmentation: coverage and state-count distribution
# =============================================================================
def _reseg_experiment(P, centres, sd, n_sys, n_frames, n_boot, seed, min_pop=0.05):
    from conformatlas.basins import detect_energy_basins
    from conformatlas.bootstrap import bootstrap_landscape
    from conformatlas.fel import calculate_fel
    from conformatlas.msm import stationary_distribution
    from conformatlas.timeseries import block_length_from_g, statistical_inefficiency

    rng = np.random.default_rng(seed)
    pi = stationary_distribution(P)
    k = len(pi)
    hits = np.zeros(k)
    tries = np.zeros(k)
    widths = []
    modal_frac = []
    n_ref_counts = []
    for e in range(n_sys):
        s = _markov_chain(P, n_frames, rng, s0=int(rng.choice(k, p=pi)))
        xy = centres[s] + rng.normal(0, sd, size=(n_frames, 2))
        df = _projection_df(xy)
        fel = calculate_fel(xy[:, 0], xy[:, 1], temperature=300.0, bins=24)
        states, _, _, labels = detect_energy_basins(fel, df, min_state_population=min_pop)
        n_ref_counts.append(len(states))
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
            n_boot=n_boot,
            seed=e,
            mode="resegment",
            resegment_kwargs={"min_state_population": min_pop},
        )
        dist = res.state_count_distribution
        modal_frac.append(float(dist.loc[dist["n_states"] == k, "fraction"].sum()))
        if len(states) != k:
            continue
        tbl = res.resegment_state_table.set_index("state")
        # map each reference state to the generating well by minimum position
        for st in states:
            w = int(np.argmin(np.sum((centres - np.array([st.min_pc1, st.min_pc2])) ** 2, axis=1)))
            row = tbl.loc[st.label]
            lo, hi = row["population_ci95_low_resegmented"], row["population_ci95_high_resegmented"]
            tries[w] += 1
            hits[w] += lo <= pi[w] <= hi
            widths.append(hi - lo)
    return {
        "coverage": hits.sum() / max(tries.sum(), 1),
        "coverage_per_state": hits / np.maximum(tries, 1),
        "n_used": int(tries.sum() / k) if k else 0,
        "mean_width": float(np.mean(widths)) if widths else np.nan,
        "modal_fraction_mean": float(np.mean(modal_frac)),
        "modal_fraction_min": float(np.min(modal_frac)),
        "ref_counts": np.bincount(n_ref_counts),
        "pi": pi,
    }


def check_bootstrap_resegment(record, KB):
    # two-well, well separated
    P2 = np.array([[0.985, 0.015], [0.035, 0.965]])  # pi = (0.7, 0.3)
    c2 = np.array([[-3.0, 0.0], [3.0, 0.0]])
    r2 = _reseg_experiment(P2, c2, 0.5, n_sys=120, n_frames=3000, n_boot=100, seed=31)
    record(
        "Bootstrap (re-segmentation)",
        "95 % CI coverage of true populations under re-segmentation, two-well (120 systems)",
        0.88 <= r2["coverage"] <= 0.99,
        f"{r2['coverage']:.3f} (per state {np.array2string(r2['coverage_per_state'], precision=2)}; "
        f"mean width {r2['mean_width']:.3f}; {r2['n_used']} systems with 2 reference states)",
        "0.95",
        "0.88–0.99 (binomial SE ≈ 0.02; block bootstrap slightly anti-conservative)",
        "known stationary distribution π = (0.7, 0.3)",
    )
    record(
        "Bootstrap (re-segmentation)",
        "state-count distribution concentrated on the true count for well-separated two-well",
        r2["modal_fraction_mean"] > 0.95,
        f"mean fraction of resamples with 2 states {r2['modal_fraction_mean']:.3f} "
        f"(min over systems {r2['modal_fraction_min']:.2f})",
        "≈ 1",
        "> 0.95",
        "construction (wells 6σ apart)",
    )
    # three-well
    P3 = _reversible(np.array([[0.975, 0.015, 0.01], [0.02, 0.97, 0.01], [0.02, 0.015, 0.965]]))
    c3 = np.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    r3 = _reseg_experiment(P3, c3, 0.5, n_sys=100, n_frames=4000, n_boot=100, seed=32)
    record(
        "Bootstrap (re-segmentation)",
        "95 % CI coverage of true populations under re-segmentation, three-well (100 systems)",
        0.88 <= r3["coverage"] <= 0.99,
        f"{r3['coverage']:.3f} (per state {np.array2string(r3['coverage_per_state'], precision=2)}; "
        f"mean width {r3['mean_width']:.3f}; {r3['n_used']} systems with 3 reference states)",
        "0.95",
        "0.88–0.99",
        f"known π = {np.array2string(r3['pi'], precision=3)}",
    )
    record(
        "Bootstrap (re-segmentation)",
        "state-count distribution concentrated on 3 for well-separated three-well",
        r3["modal_fraction_mean"] > 0.95,
        f"mean fraction with 3 states {r3['modal_fraction_mean']:.3f} "
        f"(min {r3['modal_fraction_min']:.2f})",
        "≈ 1",
        "> 0.95",
        "construction",
    )
    # shallow two-well: the count must spread
    c_sh = np.array([[-1.2, 0.0], [1.2, 0.0]])
    rs = _reseg_experiment(P2, c_sh, 0.6, n_sys=40, n_frames=3000, n_boot=100, seed=33)
    record(
        "Bootstrap (re-segmentation)",
        "state-count distribution spreads for shallow wells (centres 2σ apart)",
        rs["modal_fraction_mean"] < 0.9 and rs["modal_fraction_min"] < 0.8,
        f"mean fraction with 2 states {rs['modal_fraction_mean']:.3f} (min {rs['modal_fraction_min']:.2f}); "
        f"reference analyses found {rs['ref_counts'].tolist()} states (index = count)",
        "< 0.9",
        "mean < 0.9 and min < 0.8",
        "construction (barrier of order kT)",
        note="demonstrates sensitivity; the exact spread is system-specific",
    )


# =============================================================================
# Multi-observable equilibration vs pymbar
# =============================================================================
def check_equilibration_multi(record, KB):
    import pymbar.timeseries as pts

    from conformatlas.timeseries import detect_equilibration_multi

    rng = np.random.default_rng(41)
    n, phi = 40_000, 0.95

    def ar1():
        x = np.empty(n)
        x[0] = 0.0
        for i in range(1, n):
            x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * rng.normal()
        return x

    a, b = ar1(), ar1()
    burn_a, burn_b = 1500, 5000
    a[:burn_a] += 10.0 * np.linspace(1, 0, burn_a) ** 2
    b[:burn_b] += 10.0 * np.linspace(1, 0, burn_b) ** 2
    res = detect_equilibration_multi({"CV1": a, "CV2": b}, nskip=25)
    t_pm = {
        k: pts.detect_equilibration(v, nskip=25, fast=False)[0] for k, v in (("CV1", a), ("CV2", b))
    }
    dev = max(abs(res.per_observable[k].t0_index - t_pm[k]) for k in t_pm)
    record(
        "Timeseries",
        "detect_equilibration_multi: per-observable t₀ vs pymbar.detect_equilibration",
        dev <= 25,
        f"ours {[res.per_observable[k].t0_index for k in ('CV1', 'CV2')]}, pymbar {[t_pm[k] for k in ('CV1', 'CV2')]}",
        f"true burn-in {burn_a}, {burn_b}",
        "≤ 25 frames (one nskip)",
        "pymbar 4 (Chodera 2016)",
    )
    record(
        "Timeseries",
        "detect_equilibration_multi adopts the latest t₀ (conservative)",
        res.t0_index == max(t_pm.values()) and res.deciding_observable == "CV2",
        f"t₀ = {res.t0_index} from {res.deciding_observable}",
        f"max over observables = {max(t_pm.values())}",
        "exact",
        "definition",
    )
    joint = detect_equilibration_multi({"CV1": a, "CV2": b}, method="joint", nskip=25)
    # brute-force the joint criterion independently
    cands = [0, *range(25, int(n * 0.9), 25)]
    best = max(
        cands,
        key=lambda t0: min(
            (n - t0) / pts.statistical_inefficiency(a[t0:], fast=False),
            (n - t0) / pts.statistical_inefficiency(b[t0:], fast=False),
        ),
    )
    record(
        "Timeseries",
        "detect_equilibration_multi(method='joint') vs brute-force max-min N_eff with pymbar g",
        abs(joint.t0_index - best) <= 50,
        f"ours {joint.t0_index}, brute force {best}",
        "same t₀",
        "≤ 50 frames (two nskip; g estimators differ by < 5 %)",
        "pymbar statistical_inefficiency",
    )


def run(record, tmp, ROOT, KB):
    check_msm(record, KB)
    check_bootstrap_resegment(record, KB)
    check_equilibration_multi(record, KB)


if __name__ == "__main__":
    import tempfile

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    RESULTS = []

    def record(section, name, passed, measured, expected, tolerance, reference, note=""):
        RESULTS.append(
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
        print(
            f"[{'PASS' if passed else 'FAIL'}] {section} / {name}: {measured} vs {expected} ({tolerance})",
            flush=True,
        )

    with tempfile.TemporaryDirectory() as td:
        run(record, Path(td), ROOT, 0.008314462618)
    n_pass = sum(r["pass"] for r in RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed")
    import json

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(json.dumps(RESULTS, indent=2, default=str))
    sys.exit(0 if n_pass == len(RESULTS) else 1)
