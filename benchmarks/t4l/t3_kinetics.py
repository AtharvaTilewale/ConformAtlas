"""T3 of benchmarks/CRITERIA_T4L_HIV.md: TICA and MSM vs deeptime on one shared discretisation.

Run with the timescale-fix code: CA_SRC=<fix/msm-timescale-ordering>/src python benchmarks/t4l/t3_kinetics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_t4l import pooled, save_json  # noqa: E402

LAGS = (10, 50)
GATES = {"tica_rel": 1e-8, "stationary_abs": 1e-7, "T_abs": 1e-7, "timescales_rel": 1e-5}


def main() -> None:
    import deeptime
    from deeptime.clustering import KMeans
    from deeptime.decomposition import TICA
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM

    import conformatlas
    from conformatlas import msm as M
    from conformatlas.pca import compute_cartesian_pca, compute_tica, project_coordinates

    c, lens = pooled("ff99SBstar", "calpha")
    _, vec, _, _, mean = compute_cartesian_pca({"r": c}, weighting="frames", n_components=10)
    X = project_coordinates(c, mean, vec)
    parts = np.split(X, np.cumsum(lens)[:-1])
    km = KMeans(n_clusters=100, init_strategy="kmeans++", fixed_seed=0, max_iter=500).fit_fetch(
        X[:, :3]
    )
    dtrajs = [np.asarray(km.transform(p[:, :3]), dtype=int) for p in parts]
    out: dict = {
        "conformatlas_version": conformatlas.__version__,
        "deeptime": deeptime.__version__,
        "frames_per_replicate": lens,
        "n_clusters_used": int(len(np.unique(np.concatenate(dtrajs)))),
        "gates": GATES,
        "lags": {},
    }
    for lag in LAGS:
        res: dict = {}
        # TICA, single trajectory md1us4 (gate)
        x0 = parts[0]
        ev, *_ = compute_tica({"r": x0}, lag=lag, n_components=5, weighting="frames", epsilon=1e-6)
        dt_ev = np.asarray(
            TICA(lagtime=lag, dim=5, epsilon=1e-6, scaling=None)
            .fit(x0)
            .fetch_model()
            .singular_values
        )[:5]
        rel = float(np.max(np.abs(ev[:5] - dt_ev) / np.abs(dt_ev)))
        res["tica_md1us4"] = {"max_rel_error": rel, "pass": bool(rel <= GATES["tica_rel"])}
        # TICA pooled over 5 trajectories (informational)
        evp, *_ = compute_tica(
            {f"r{i}": p for i, p in enumerate(parts)},
            lag=lag,
            n_components=5,
            weighting="frames",
            epsilon=1e-6,
        )
        dtp = np.asarray(
            TICA(lagtime=lag, dim=5, epsilon=1e-6, scaling=None)
            .fit(parts)
            .fetch_model()
            .singular_values
        )[:5]
        res["tica_pooled_informational"] = {
            "conformatlas": evp[:5].tolist(),
            "deeptime": dtp.tolist(),
            "max_rel_difference": float(np.max(np.abs(evp[:5] - dtp) / np.abs(dtp))),
        }
        # MSM on the 5 discrete trajectories (gate)
        est = M.estimate_msm(dtrajs, lag, n_states=100, reversible=True, dt_ps=1.0, n_eigen=6)
        counts = TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit_fetch(dtrajs)
        dm = MaximumLikelihoodMSM(reversible=True).fit_fetch(counts)
        active = np.asarray(dm.state_symbols())
        same = bool(np.array_equal(np.sort(est.active_set), np.sort(active)))
        pi_err = (
            float(np.max(np.abs(est.stationary - dm.stationary_distribution)))
            if same
            else float("nan")
        )
        t_err = (
            float(np.max(np.abs(est.transition_matrix - dm.transition_matrix)))
            if same
            else float("nan")
        )
        ts_ca, ts_dt = np.asarray(est.timescales_ps)[:5], np.asarray(dm.timescales(5))
        ts_rel = float(np.max(np.abs(ts_ca - ts_dt) / ts_dt))
        res["msm"] = {
            "n_active": [int(est.n_active), int(len(active))],
            "same_active_set": same,
            "max_abs_stationary": pi_err,
            "max_abs_T": t_err,
            "max_rel_timescales": ts_rel,
            "timescales_frames": [ts_ca.tolist(), ts_dt.tolist()],
            "negative_eigenvalues_in_slowest_5": [
                float(v) for v in M.negative_slow_eigenvalues(est.eigenvalues, 5)
            ],
            "pass": bool(
                same
                and pi_err <= GATES["stationary_abs"]
                and t_err <= GATES["T_abs"]
                and ts_rel <= GATES["timescales_rel"]
            ),
        }
        out["lags"][str(lag)] = res
        print(
            lag,
            {
                k: {
                    kk: vv
                    for kk, vv in v.items()
                    if kk not in ("timescales_frames", "conformatlas", "deeptime")
                }
                for k, v in res.items()
            },
            flush=True,
        )
    save_json("t3_kinetics.json", out)


if __name__ == "__main__":
    main()
