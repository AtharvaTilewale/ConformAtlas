"""Markov state model: clustering, counting, estimators, PCCA+, CK test, MFPT."""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist

from conformatlas import msm as M

deeptime = pytest.importorskip("deeptime")


def _markov_chain(T, n, rng, s0=0):
    cum = np.cumsum(T, axis=1)
    s = np.empty(n, dtype=np.int64)
    s[0] = s0
    u = rng.random(n)
    for i in range(1, n):
        s[i] = np.searchsorted(cum[s[i - 1]], u[i])
    return s


def _reversible(T):
    pi = M.stationary_distribution(T)
    X = pi[:, None] * T
    X = 0.5 * (X + X.T)
    return X / X.sum(axis=1)[:, None]


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


@pytest.fixture(scope="module")
def chain4():
    rng = np.random.default_rng(0)
    s = _markov_chain(T4, 200_000, rng)
    return [s[:100_000], s[100_000:]]


# --------------------------------------------------------------------- clustering


def test_kmeans_matches_scipy_from_same_init():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 3))
    X[:1500] += 3.0
    init = X[:4].copy()
    ours = M.kmeans_cluster(X, 4, init=init)
    cent, lab = kmeans2(X, init.copy(), minit="matrix", iter=300)
    assert np.allclose(np.sort(cent, axis=0), np.sort(ours.centers, axis=0), atol=1e-8)
    assert np.mean(lab == ours.labels) > 0.999
    inertia_scipy = cdist(X, cent, "sqeuclidean").min(axis=1).sum()
    assert abs(inertia_scipy - ours.inertia) < 1e-6 * inertia_scipy


def test_kmeans_deterministic_and_pluspluss_not_worse():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 2))
    X[:1000] += 4.0
    a = M.kmeans_cluster(X, 5, seed=3)
    b = M.kmeans_cluster(X, 5, seed=3)
    assert np.array_equal(a.labels, b.labels)
    assert a.n_clusters == 5
    assert len(np.unique(a.labels)) == 5


def test_regular_space_properties():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4000, 2)) * 2
    res = M.regular_space_cluster(X, dmin=0.8)
    assert pdist(res.centers).min() > 0.8
    assert cdist(X, res.centers).min(axis=1).max() <= 0.8 + 1e-12
    with pytest.raises(ValueError):
        M.regular_space_cluster(X, dmin=0.05, max_centers=10)


# --------------------------------------------------------------------- counting


def test_count_matrix_within_replicates_matches_deeptime(chain4):
    from deeptime.markov import TransitionCountEstimator

    for lag in (1, 3, 7):
        C = M.count_matrix(chain4, lag)
        ref = TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit_fetch(chain4)
        assert np.array_equal(C, ref.count_matrix)
    # no junction counts: two trajectories ending/starting in different states
    C = M.count_matrix([np.array([0, 0, 0]), np.array([1, 1, 1])], 1)
    assert C[0, 1] == 0 and C[1, 0] == 0 and C[0, 0] == 2 and C[1, 1] == 2
    # dict input and negative labels ignored
    C = M.count_matrix({"a": np.array([0, -1, 1])}, 1, n_states=2)
    assert C.sum() == 0


def test_split_discrete_trajectories_orders_by_time():
    labels = np.array([0, 1, 2, 3, 4, 5])
    reps = np.array(["b", "a", "b", "a", "b", "a"])
    times = np.array([2.0, 1.0, 0.0, 0.0, 1.0, 2.0])
    d = M.split_discrete_trajectories(labels, reps, order=times)
    assert list(d) == ["b", "a"]
    assert d["b"].tolist() == [2, 4, 0]
    assert d["a"].tolist() == [3, 1, 5]


# --------------------------------------------------------------------- estimators


def test_reversible_estimator_matches_deeptime(chain4):
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM

    C = M.count_matrix(chain4, 2)
    T, info = M.transition_matrix_reversible(C, return_info=True)
    assert info["converged"]
    ref = MaximumLikelihoodMSM(reversible=True).fit_fetch(
        TransitionCountEstimator(lagtime=2, count_mode="sliding").fit_fetch(chain4)
    )
    assert np.max(np.abs(T - ref.transition_matrix)) < 1e-6
    assert np.max(np.abs(info["stationary"] - ref.stationary_distribution)) < 1e-6
    assert M.is_reversible(T)
    # analytic truth at lag 2
    T_true = np.linalg.matrix_power(T4, 2)
    assert np.max(np.abs(T - T_true)) < 0.01


def test_nonreversible_estimator_and_stationary(chain4):
    C = M.count_matrix(chain4, 1)
    T = M.transition_matrix_nonreversible(C)
    assert np.allclose(T.sum(axis=1), 1.0)
    pi = M.stationary_distribution(T)
    assert np.allclose(pi @ T, pi)
    assert np.max(np.abs(pi - M.stationary_distribution(T4))) < 0.01


def test_eigen_decomposition_and_timescales():
    vals, R, L = M.eigen_decomposition(T4)
    assert np.allclose(vals, np.sort(np.linalg.eigvals(T4).real)[::-1])
    assert np.allclose(R[:, 0], 1.0)
    assert np.allclose(L.T @ R, np.eye(4), atol=1e-10)
    ts = M.implied_timescales(vals, lag=3, dt_ps=2.0)
    assert np.allclose(ts, -3 * 2.0 / np.log(vals[1:]))
    # non-reversible path
    Tn = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    v, Rn, Ln = M.eigen_decomposition(Tn, reversible=False)
    assert np.allclose(np.sort(v), np.sort(np.linalg.eigvals(Tn).real))
    assert np.allclose(Ln.T @ Rn, np.eye(3), atol=1e-10)


def test_implied_timescales_flat_for_markov_chain(chain4):
    its = M.implied_timescales_scan(chain4, [1, 2, 4, 8], n_timescales=3, dt_ps=1.0)
    truth = -1.0 / np.log(np.sort(np.linalg.eigvals(T4).real)[::-1][1:])
    for i in range(3):
        col = its[f"ts_{i + 1}"].to_numpy()
        assert np.all(np.abs(col - truth[i]) / truth[i] < 0.05)
    rep = M.suggest_markov_lag(its, rel_tol=0.1)
    assert rep["plateau_found"] and rep["suggested_lag_frames"] == 1
    assert isinstance(rep["relative_changes"], pd.DataFrame)


def test_connected_set_restriction():
    C = np.array([[10, 2, 0], [3, 10, 0], [0, 0, 5]])
    sets = M.connected_sets(C)
    assert sets[0].tolist() == [0, 1]
    with pytest.raises(ValueError):
        M.transition_matrix_reversible(C)
    est = M.estimate_msm([np.array([0, 1, 0, 1, 1, 0])], 1)
    assert est.n_active == 2


# --------------------------------------------------------------------- PCCA+


def _match(chi, ref):
    r, c = linear_sum_assignment(-(chi.T @ ref))
    return np.max(np.abs(chi[:, r] - ref[:, c]))


def test_pcca_matches_deeptime_two_and_three_sets():
    from deeptime.markov import pcca as dpcca

    rng = np.random.default_rng(4)
    B = rng.random((6, 6)) * 0.02
    np.fill_diagonal(B, 0)
    for a, b in ((0, 1), (2, 3), (4, 5)):
        B[a, b] = B[b, a] = 0.2
    T6 = B.copy()
    np.fill_diagonal(T6, 1 - B.sum(axis=1))
    T6 = _reversible(T6)
    chi2 = M.pcca_plus(T4, 2)
    assert chi2.shape == (4, 2) and np.all(chi2 >= 0) and np.allclose(chi2.sum(axis=1), 1)
    assert _match(chi2, dpcca(T4, 2).memberships) < 1e-3
    chi3 = M.pcca_plus(T6, 3)
    assert _match(chi3, dpcca(T6, 3).memberships) < 1e-3
    crisp = M.crisp_assignment(chi3)
    assert crisp[0] == crisp[1] and crisp[2] == crisp[3] and crisp[4] == crisp[5]
    assert len(set(crisp.tolist())) == 3
    Tc, pic = M.coarse_grain(T6, M.stationary_distribution(T6), chi3)
    ref = dpcca(T6, 3)
    r, c = linear_sum_assignment(-(chi3.T @ ref.memberships))
    assert (
        np.max(np.abs(Tc[np.ix_(r, r)] - ref.coarse_grained_transition_matrix[np.ix_(c, c)])) < 1e-3
    )
    assert np.allclose(pic.sum(), 1.0)


def test_pcca_identity_when_m_equals_n():
    assert np.array_equal(M.pcca_plus(T4, 4), np.eye(4))
    with pytest.raises(ValueError):
        M.pcca_plus(T4, 1)


# --------------------------------------------------------------------- CK and MFPT


def test_ck_passes_for_markov_and_fails_for_hidden_state_process():
    rng = np.random.default_rng(6)
    s = _markov_chain(T4, 300_000, rng)
    df, summary = M.chapman_kolmogorov_test([s], 1, [np.array([0, 1]), np.array([2, 3])], k_max=5)
    assert summary["passed"] and summary["max_abs_deviation"] < 0.01
    assert set(df.columns) >= {"k", "predicted", "estimated", "estimated_se", "abs_deviation"}
    # hidden fast/slow sub-states projected onto two observable states
    H = np.array([[0.98, 0.02, 0.0], [0.02, 0.78, 0.20], [0.0, 0.20, 0.80]])
    h = _markov_chain(H, 300_000, rng)
    obs = np.where(h == 2, 1, 0)
    _, bad = M.chapman_kolmogorov_test([obs], 1, [np.array([0]), np.array([1])], k_max=8)
    assert not bad["passed"] and bad["max_abs_deviation"] > 0.05


def test_mfpt_analytic_two_state_and_deeptime_sets():
    from deeptime.markov.tools.analysis import mfpt as dmfpt

    T2 = np.array([[0.9, 0.1], [0.3, 0.7]])
    pi2 = M.stationary_distribution(T2)
    m = M.mfpt_between_sets(T2, pi2, [np.array([0]), np.array([1])], lag=1, dt_ps=2.0)
    assert np.isclose(m[0, 1], 2.0 / 0.1) and np.isclose(m[1, 0], 2.0 / 0.3)
    pi4 = M.stationary_distribution(T4)
    sets = [np.array([0, 1]), np.array([2, 3])]
    m = M.mfpt_between_sets(T4, pi4, sets, lag=1)
    assert np.isclose(m[0, 1], dmfpt(T4, target=[2, 3], origin=[0, 1]))
    assert np.isclose(m[1, 0], dmfpt(T4, target=[0, 1], origin=[2, 3]))
    assert m[0, 0] == 0.0


def test_basin_overlap_table():
    meta = np.array(["MS1", "MS1", "MS2", "MS2", "MS2"])
    basins = np.array(["State A", "State A", "State B", "State B", "Unassigned"])
    tbl = M.basin_overlap_table(meta, basins)
    assert tbl.loc["MS1", "best_basin"] == "State A"
    assert np.isclose(tbl.loc["MS2", "row_fraction_max"], 2 / 3)


# --------------------------------------------------------------------- workflow


def test_build_and_save_msm(tmp_path):
    rng = np.random.default_rng(7)
    T3 = _reversible(np.array([[0.98, 0.015, 0.005], [0.02, 0.97, 0.01], [0.01, 0.02, 0.97]]))
    centres = np.array([[-3, 0], [3, 0], [0, 4.0]])
    dfs = []
    for r in range(2):
        s = _markov_chain(T3, 4000, rng)
        xy = centres[s] + rng.normal(0, 0.5, size=(4000, 2))
        dfs.append(
            pd.DataFrame(
                {
                    "system": "S",
                    "replicate": f"Rep_{r + 1}",
                    "frame": np.arange(4000),
                    "time_ps": 10.0 * np.arange(4000),
                    "PC1": xy[:, 0],
                    "PC2": xy[:, 1],
                }
            )
        )
    df = pd.concat(dfs, ignore_index=True)
    basins = np.where(df["PC2"] > 2, "State C", np.where(df["PC1"] < 0, "State A", "State B"))
    res = M.build_msm(
        df,
        ["PC1", "PC2"],
        lag=5,
        lags=[1, 2, 5, 10],
        n_clusters=30,
        n_metastable=3,
        basin_states=basins,
    )
    assert res.dt_ps == 10.0
    assert res.n_metastable == 3
    assert res.ck_summary["passed"]
    assert set(res.frame_table["metastable_set"]) == {"MS1", "MS2", "MS3"}
    assert (res.basin_overlap["row_fraction_max"] > 0.95).all()
    # slowest analytic timescale of the 3-state chain (frames -> ps)
    truth = -10.0 / np.log(np.sort(np.linalg.eigvals(T3).real)[::-1][1])
    assert abs(res.msm.timescales_ps[0] - truth) / truth < 0.3
    # stationary probability of the metastable sets vs chain populations
    pi_true = M.stationary_distribution(T3)
    pic = np.sort(res.coarse_stationary["stationary_probability"].to_numpy())
    assert np.max(np.abs(pic - np.sort(pi_true))) < 0.05
    out = M.save_msm(res, tmp_path)
    expected = {
        "count_matrix.csv",
        "transition_matrix.csv",
        "stationary.csv",
        "implied_timescales.csv",
        "ck_test.csv",
        "pcca_memberships.csv",
        "metastable_assignments.csv",
        "mfpt.csv",
        "basin_overlap.csv",
        "msm_summary.json",
        "eigenvalues.csv",
        "cluster_centers.csv",
        "coarse_transition_matrix.csv",
        "coarse_stationary.csv",
        "implied_timescale_changes.csv",
    }
    assert expected <= {p.name for p in out.iterdir()}
    tm = pd.read_csv(out / "transition_matrix.csv", index_col=0)
    assert np.allclose(tm.to_numpy().sum(axis=1), 1.0)
    mf = pd.read_csv(out / "mfpt.csv", index_col=0)
    assert mf.shape == (3, 3)
    res_rs = M.build_msm(
        df, ["PC1", "PC2"], lag=5, lags=[1, 5], clustering="regular-space", dmin=0.8, n_metastable=3
    )
    assert res_rs.clustering.method == "regular-space"
    with pytest.raises(ValueError):
        M.build_msm(df, ["PC1", "PC2"], lag=5, clustering="regular-space")
