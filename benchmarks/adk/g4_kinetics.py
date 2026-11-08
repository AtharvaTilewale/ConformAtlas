"""G4 of benchmarks/CRITERIA.md: TICA and MSM, ConformAtlas vs deeptime on one shared input.

Input: the top 10 Cα principal-component projections (ConformAtlas, frames weighting) of the ADK trajectory.
Discretisation: deeptime k-means (100 clusters, fixed seed) on PC1-PC3, computed once; the same discrete
trajectory is given to both implementations (so ConformAtlas does not cluster its own input).
Time unit: frames (dt_ps = 1 so "ps" in ConformAtlas's output is frames).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import coordinates, save_json  # noqa: E402

LAGS = (10, 50)
N_CLUSTERS = 100
GATES = {"tica_eigenvalues_rel": 1e-8, "stationary_abs": 1e-7, "timescales_rel": 1e-5}


def main() -> None:
    import deeptime
    from deeptime.clustering import KMeans
    from deeptime.decomposition import TICA
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM

    from conformatlas import msm as M
    from conformatlas.pca import compute_cartesian_pca, compute_tica, project_coordinates

    coords = coordinates("calpha")
    _, vec, _, _, mean = compute_cartesian_pca({"r": coords}, weighting="frames", n_components=10)
    X = project_coordinates(coords, mean, vec)
    km = KMeans(
        n_clusters=N_CLUSTERS, init_strategy="kmeans++", fixed_seed=0, max_iter=500
    ).fit_fetch(X[:, :3])
    dtraj = np.asarray(km.transform(X[:, :3]), dtype=int)
    out: dict = {
        "deeptime": deeptime.__version__,
        "n_frames": int(len(X)),
        "n_clusters_used": int(len(np.unique(dtraj))),
        "gates": GATES,
        "lags": {},
    }

    for lag in LAGS:
        res: dict = {}
        # ---- TICA
        ev, _, _, _, _ = compute_tica(
            {"r": X}, lag=lag, n_components=5, weighting="frames", epsilon=1e-6
        )
        model = TICA(lagtime=lag, dim=5, epsilon=1e-6, scaling=None).fit(X).fetch_model()
        dt_ev = np.asarray(model.singular_values)[:5]
        rel = float(np.max(np.abs(ev[:5] - dt_ev) / np.abs(dt_ev)))
        res["tica"] = {
            "conformatlas": ev[:5].tolist(),
            "deeptime": dt_ev.tolist(),
            "max_rel_error": rel,
            "pass": bool(rel <= GATES["tica_eigenvalues_rel"]),
        }

        # ---- MSM on the shared discrete trajectory
        est = M.estimate_msm(
            [dtraj], lag, n_states=N_CLUSTERS, reversible=True, dt_ps=1.0, n_eigen=6
        )
        counts = TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit_fetch(dtraj)
        dmsm = MaximumLikelihoodMSM(reversible=True).fit_fetch(counts)
        dt_active = np.asarray(
            dmsm.state_symbols()
        )  # states of the fitted MSM (largest connected set)
        same_active = bool(np.array_equal(np.sort(est.active_set), np.sort(dt_active)))
        pi_err = (
            float(np.max(np.abs(est.stationary - dmsm.stationary_distribution)))
            if same_active
            else float("nan")
        )
        ts_ca = np.asarray(est.timescales_ps)[:5]
        ts_dt = np.asarray(dmsm.timescales(k=6))[:5] * 1.0
        ts_rel = float(np.max(np.abs(ts_ca - ts_dt) / ts_dt)) if same_active else float("nan")
        t_err = (
            float(np.max(np.abs(est.transition_matrix - dmsm.transition_matrix)))
            if same_active
            else float("nan")
        )
        # diagnosis only: rank ConformAtlas's full spectrum by |lambda| (as deeptime/PyEMMA do)
        full = M.estimate_msm(
            [dtraj], lag, n_states=N_CLUSTERS, reversible=True, dt_ps=1.0, n_eigen=None
        ).eigenvalues
        by_abs = full[np.argsort(-np.abs(full))][1:6]
        ts_abs = -lag / np.log(np.abs(by_abs))
        diag = {
            "ts_rel_error_first_four_signed_order": float(
                np.max(np.abs(ts_ca[:4] - ts_dt[:4]) / ts_dt[:4])
            ),
            "ts_rel_error_all_five_abs_ordered": float(np.max(np.abs(ts_abs - ts_dt) / ts_dt)),
            "negative_eigenvalues_among_top_10_by_magnitude": [
                round(float(v), 6) for v in full[np.argsort(-np.abs(full))][:10] if v < 0
            ],
        }
        res["msm_timescale_ordering_diagnosis_only"] = diag
        res["msm"] = {
            "n_active_conformatlas": int(est.n_active),
            "n_active_deeptime": int(len(dt_active)),
            "same_active_set": same_active,
            "max_abs_stationary_error": pi_err,
            "max_abs_transition_matrix_error": t_err,
            "timescales_frames_conformatlas": ts_ca.tolist(),
            "timescales_frames_deeptime": ts_dt.tolist(),
            "max_rel_timescale_error": ts_rel,
            "pass_stationary": bool(same_active and pi_err <= GATES["stationary_abs"]),
            "pass_timescales": bool(same_active and ts_rel <= GATES["timescales_rel"]),
        }
        out["lags"][str(lag)] = res
        print(
            lag,
            {
                k: (
                    v
                    if k != "tica"
                    else {kk: vv for kk, vv in v.items() if kk in ("max_rel_error", "pass")}
                )
                for k, v in res.items()
            },
            flush=True,
        )
    save_json("g4_kinetics.json", out)


if __name__ == "__main__":
    main()
