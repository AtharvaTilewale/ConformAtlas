"""HIV-1 protease, step 1 (run in an environment with PyEMMA 2.5): export the authors' discretisation and
models, and a fresh PyEMMA maximum-likelihood MSM with sliding counts, to ``results/pyemma_export.npz``.

Criteria: benchmarks/CRITERIA_T4L_HIV.md (H0-H4). Data: Zenodo 10.5281/zenodo.5006613, unpacked under
$CA_BENCH_DATA/hiv1_protease/data/MD/.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pyemma

DATA = Path(os.environ.get("CA_BENCH_DATA", Path(__file__).resolve().parents[1] / "data"))
OBJ = DATA / "hiv1_protease" / "data" / "MD" / "MSM_obj"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
LAG = 300
K = 5


def export_hmm() -> None:
    """The saved HMM needs the ``bhmm`` package (Python <= 3.9); run with ``--hmm`` in such an environment."""
    hmm = pyemma.load(str(OBJ / "hmm_cl100_tau30_meta5"))
    sets = np.full(100, -1, dtype=np.int64)
    for k, s in enumerate(hmm.metastable_sets):
        sets[np.asarray(s)] = k
    np.savez_compressed(
        OUT / "pyemma_hmm_export.npz",
        hmm_pi=np.asarray(hmm.pi),
        hmm_set_of_microstate=sets,
        hmm_memberships=np.asarray(hmm.metastable_memberships),
        hmm_transition_matrix=np.asarray(hmm.transition_matrix),
    )
    print("HMM sets", [len(s) for s in hmm.metastable_sets], "pi", np.round(hmm.pi, 4).tolist())


def main() -> None:
    cl = pyemma.load(str(OBJ / "cl_obj100_lambda_cyl"))
    dtrajs = [np.asarray(d, dtype=np.int64) for d in cl.dtrajs]
    saved = pyemma.load(str(OBJ / "M_cl100_tau30"))
    its = pyemma.load(str(OBJ / "its_cl100_lambda_cyl"))

    t0 = time.perf_counter()
    fresh = pyemma.msm.estimate_markov_model(
        dtrajs,
        LAG,
        reversible=True,
        count_mode="sliding",
        connectivity="largest",
        mincount_connectivity=0,
    )
    t_fresh = time.perf_counter() - t0

    its_lags = np.asarray(its.lags)
    its_ts = np.asarray(its.timescales)  # (n_lags, n_its) point estimates, frames
    np.savez_compressed(
        OUT / "pyemma_export.npz",
        dtraj_concat=np.concatenate(dtrajs),
        dtraj_lengths=np.array([len(d) for d in dtrajs]),
        cluster_centres=np.asarray(cl.clustercenters),
        saved_active_set=np.asarray(saved.active_set),
        saved_stationary=np.asarray(saved.stationary_distribution),
        saved_T=np.asarray(saved.transition_matrix),
        saved_timescales=np.asarray(saved.timescales(K)),
        fresh_active_set=np.asarray(fresh.active_set),
        fresh_stationary=np.asarray(fresh.stationary_distribution),
        fresh_T=np.asarray(fresh.transition_matrix),
        fresh_timescales=np.asarray(fresh.timescales(K)),
        fresh_eigenvalues=np.asarray(fresh.eigenvalues(K + 1)),
        its_lags=its_lags,
        its_timescales=its_ts,
    )
    meta = {
        "pyemma": pyemma.__version__,
        "n_dtrajs": len(dtrajs),
        "frames_per_dtraj": sorted({len(d) for d in dtrajs}),
        "saved_model": {
            "class": type(saved).__name__,
            "lag": int(saved.lag),
            "reversible": bool(saved.reversible),
            "count_mode": saved.count_mode,
            "connectivity": saved.connectivity,
            "n_active": int(saved.nstates),
        },
        "fresh_model": {
            "call": "pyemma.msm.estimate_markov_model(dtrajs, 300, reversible=True, count_mode='sliding', connectivity='largest', mincount_connectivity=0)",
            "n_active": int(fresh.nstates),
            "seconds": t_fresh,
        },
        "its": {"lags": its_lags.tolist(), "n_timescales": int(its_ts.shape[1])},
    }
    (OUT / "pyemma_export.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    import sys

    export_hmm() if "--hmm" in sys.argv else main()
