"""HIV-1 protease, step 2: ConformAtlas MSM (timescale-fix code, from $CA_SRC) vs PyEMMA and deeptime on the
authors' discretisation. Gates H0-H1; H2-H4 informational. See benchmarks/CRITERIA_T4L_HIV.md.

    CA_SRC=/path/to/fix-branch/src python benchmarks/hiv/h_compare.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("CA_SRC"):
    sys.path.insert(0, os.environ["CA_SRC"])

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
R = HERE / "results"
LAG = 300
K = 5
NS_PER_FRAME = 0.1  # the authors' tau = 30 ns at lag 300 frames
GATES = {"stationary_abs": 1e-6, "T_abs": 1e-6, "timescales_rel": 1e-5}


def compare(est_pi, est_T, est_ts, est_active, ref_pi, ref_T, ref_ts, ref_active) -> dict:
    same = bool(np.array_equal(np.sort(est_active), np.sort(ref_active)))
    out = {"same_active_set": same, "n_active": [int(len(est_active)), int(len(ref_active))]}
    if same:
        out["max_abs_stationary"] = float(np.max(np.abs(est_pi - ref_pi)))
        out["max_abs_T"] = float(np.max(np.abs(est_T - ref_T)))
    out["timescales_frames"] = [list(map(float, est_ts[:K])), list(map(float, ref_ts[:K]))]
    out["max_rel_timescales"] = float(
        np.max(np.abs(np.asarray(est_ts[:K]) - ref_ts[:K]) / ref_ts[:K])
    )
    return out


def main() -> None:
    import deeptime
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM

    import conformatlas
    from conformatlas import msm as M

    z = np.load(R / "pyemma_export.npz")
    meta = json.loads((R / "pyemma_export.json").read_text())
    lengths = z["dtraj_lengths"]
    dtrajs = np.split(z["dtraj_concat"], np.cumsum(lengths)[:-1])
    out: dict = {
        "conformatlas_src": os.environ.get("CA_SRC", "installed"),
        "conformatlas_version": conformatlas.__version__,
        "deeptime": deeptime.__version__,
        "pyemma": meta["pyemma"],
        "gates": GATES,
    }

    # ---- H0
    h0 = {
        "n_dtrajs": int(len(dtrajs)),
        "frames_each": sorted({int(n) for n in lengths}),
        "label_range": [int(z["dtraj_concat"].min()), int(z["dtraj_concat"].max())],
        "centres_shape": list(z["cluster_centres"].shape),
        "saved_lag": meta["saved_model"]["lag"],
        "saved_count_mode": meta["saved_model"]["count_mode"],
    }
    h0["pass"] = bool(
        h0["n_dtrajs"] == 964
        and h0["frames_each"] == [1300]
        and h0["label_range"] == [0, 99]
        and h0["centres_shape"] == [100, 3]
        and h0["saved_lag"] == LAG
    )
    out["H0"] = h0
    print("H0", h0, flush=True)

    # ---- H1
    t0 = time.perf_counter()
    est = M.estimate_msm(dtrajs, LAG, n_states=100, reversible=True, dt_ps=1.0, n_eigen=K + 1)
    t_ca = time.perf_counter() - t0
    t0 = time.perf_counter()
    dm = MaximumLikelihoodMSM(reversible=True).fit_fetch(
        TransitionCountEstimator(lagtime=LAG, count_mode="sliding").fit_fetch(dtrajs)
    )
    t_dt = time.perf_counter() - t0
    ca = (est.stationary, est.transition_matrix, np.asarray(est.timescales_ps), est.active_set)
    h1_py = compare(
        *ca, z["fresh_stationary"], z["fresh_T"], z["fresh_timescales"], z["fresh_active_set"]
    )
    h1_dt = compare(
        *ca,
        dm.stationary_distribution,
        dm.transition_matrix,
        np.asarray(dm.timescales(K)),
        np.asarray(dm.state_symbols()),
    )
    for h in (h1_py, h1_dt):
        h["pass"] = bool(
            h["same_active_set"]
            and h["max_abs_stationary"] <= GATES["stationary_abs"]
            and h["max_abs_T"] <= GATES["T_abs"]
            and h["max_rel_timescales"] <= GATES["timescales_rel"]
        )
    out["H1"] = {
        "vs_pyemma_fresh_sliding": h1_py,
        "vs_deeptime_sliding": h1_dt,
        "negative_eigenvalues_in_slowest_5": [
            float(v) for v in M.negative_slow_eigenvalues(est.eigenvalues, K)
        ],
        "seconds": {
            "conformatlas": t_ca,
            "deeptime": t_dt,
            "pyemma": meta["fresh_model"]["seconds"],
        },
    }
    print("H1 vs PyEMMA", {k: v for k, v in h1_py.items() if k != "timescales_frames"}, flush=True)
    print(
        "H1 vs deeptime", {k: v for k, v in h1_dt.items() if k != "timescales_frames"}, flush=True
    )

    # ---- H2 (informational: the saved model used effective counts)
    h2 = compare(
        *ca, z["saved_stationary"], z["saved_T"], z["saved_timescales"], z["saved_active_set"]
    )
    h2["timescale_ratio_conformatlas_over_saved"] = list(
        map(float, np.asarray(ca[2][:K]) / z["saved_timescales"][:K])
    )
    h2["timescales_ns"] = [
        [t * NS_PER_FRAME for t in h2["timescales_frames"][0]],
        [t * NS_PER_FRAME for t in h2["timescales_frames"][1]],
    ]
    h2["note"] = (
        "saved model: BayesianMSM point estimate from PyEMMA 'effective' counts; differs from sliding counts by construction"
    )
    out["H2_informational"] = h2
    print("H2", {k: v for k, v in h2.items() if k not in ("timescales_frames", "note")}, flush=True)

    # ---- H3 (informational): PCCA+ 5 sets vs the authors' HMM
    hmm_file = R / "pyemma_hmm_export.npz"
    chi = M.pcca_plus(est.transition_matrix, 5)
    crisp = M.crisp_assignment(chi)
    _, pic = M.coarse_grain(est.transition_matrix, est.stationary, chi)
    h3: dict = {
        "pcca_set_sizes": [int(np.sum(crisp == k)) for k in range(5)],
        "pcca_set_stationary": list(map(float, pic)),
    }
    # extra informational reference (added after pre-registration, not a gate): deeptime PCCA+ on the same T
    from deeptime.markov import pcca as dpcca

    dp = dpcca(est.transition_matrix, 5, stationary_distribution=est.stationary)
    dcrisp = np.argmax(dp.memberships, axis=1)
    perms = list(itertools.permutations(range(5)))
    h3["deeptime_pcca"] = {
        "stationary_weighted_crisp_agreement_with_conformatlas": max(
            float(np.sum(est.stationary[np.array(pp)[crisp] == dcrisp])) for pp in perms
        ),
        "max_abs_membership_difference_best_permutation": min(
            float(np.max(np.abs(chi[:, list(pp)] - dp.memberships))) for pp in perms
        ),
        "set_stationary_sorted": sorted(map(float, dp.coarse_grained_stationary_probability)),
    }
    if hmm_file.exists():
        hz = np.load(hmm_file)
        hmm_sets = hz["hmm_set_of_microstate"][est.active_set]
        best, best_perm = -1.0, None
        for perm in itertools.permutations(range(5)):
            agree = float(np.sum(est.stationary[np.array(perm)[crisp] == hmm_sets]))
            if agree > best:
                best, best_perm = agree, perm
        h3.update(
            {
                "stationary_weighted_agreement_best_permutation": best,
                "permutation_pcca_to_hmm": list(best_perm),
                "hmm_pi": list(map(float, hz["hmm_pi"])),
                "pcca_stationary_in_hmm_order": [
                    float(pic[list(best_perm).index(j)]) for j in range(5)
                ],
            }
        )
    else:
        h3["hmm"] = "not available (export needs PyEMMA with bhmm, Python <= 3.9)"
    out["H3_informational"] = h3
    print("H3", h3, flush=True)

    # ---- H4 (informational): implied-timescale scan vs the authors' saved ITS object
    lags = [int(x) for x in z["its_lags"]]
    its = M.implied_timescales_scan(
        dtrajs, lags, n_timescales=K, reversible=True, dt_ps=1.0, n_states=100
    )
    saved_its = z["its_timescales"][:, :K]
    rows = []
    for i, lag in enumerate(lags):
        r = its[its.lag_frames == lag]
        if len(r):
            ca_ts = r[[f"ts_{j + 1}" for j in range(K)]].to_numpy()[0]
            rows.append(
                {
                    "lag": lag,
                    "ratio_ts1": float(ca_ts[0] / saved_its[i, 0]),
                    "max_abs_log_ratio_top5": float(
                        np.nanmax(np.abs(np.log(ca_ts / saved_its[i])))
                    ),
                    "conformatlas_ts1_ns": float(ca_ts[0] * NS_PER_FRAME),
                    "saved_ts1_ns": float(saved_its[i, 0] * NS_PER_FRAME),
                    "n_negative_reported": int(r["n_negative_eigenvalues_reported"].item()),
                }
            )
    out["H4_informational"] = {
        "rows": rows,
        "note": "saved ITS object: PyEMMA timescales_msm with errors='bayes' (effective counts); ratios are informational",
    }
    print("H4 rows", len(rows), flush=True)
    (R / "h_compare.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
