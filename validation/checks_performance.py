"""Validation checks for the performance / API track (streaming PCA, HDF5 container,
rebin, MDAnalysis AnalysisBase class, high-level API).

Imported by ``validation/validate.py``::

    from checks_performance import run as run_performance
    run_performance(record, tmp, ROOT, KB)

Every check names its reference (analytical identity or independent
implementation) and tolerance. Stand-alone::

    .venv/bin/python validation/checks_performance.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("CONFORMATLAS_NO_PROGRESS", "1")
SECTION = "Streaming PCA, results container and API"


def _max_rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _vec_cos_err(u, v):
    return float(np.max(np.abs(np.abs(np.sum(u * v, axis=0)) - 1.0)))


def _signed_proj_err(p_test, p_ref, u_test, u_ref):
    sign = np.sign(np.sum(u_test * u_ref, axis=0))
    return float(np.max(np.abs(np.asarray(p_test) * sign - p_ref)) / np.max(np.abs(p_ref)))


def run(record, tmp: Path, ROOT: Path, KB: float) -> None:
    tmp = Path(tmp)
    # ------------------------------------------------------------------ 1
    # Covariance accumulator vs the definition (numpy.cov, ddof=0), chunked + merged
    from conformatlas.streaming import (
        CovarianceAccumulator,
        ReplicateMoments,
        combine_replicate_moments,
        eigendecompose_covariance,
    )

    rng = np.random.default_rng(11)
    x = rng.normal(size=(2001, 9)) @ rng.normal(size=(9, 9)) + 100.0  # correlated, offset
    acc = CovarianceAccumulator(9)
    for i in range(0, len(x), 37):
        acc.update(x[i : i + 37])
    other = CovarianceAccumulator(9)
    other.update(x[:1000])
    rest = CovarianceAccumulator(9)
    rest.update(x[1000:])
    other.merge(rest)
    err = max(
        float(np.max(np.abs(acc.covariance(0) - np.cov(x, rowvar=False, ddof=0)))),
        float(np.max(np.abs(other.covariance(0) - np.cov(x, rowvar=False, ddof=0)))),
        float(np.max(np.abs(acc.mean - x.mean(axis=0)))),
    )
    record(
        SECTION,
        "Chan/Welford accumulator (chunk 37, and two-way merge) vs numpy.cov(ddof=0) on correlated data with offset 100",
        err < 1e-10,
        f"max abs diff {err:.1e}",
        "0",
        "< 1e-10",
        "Chan, Golub & LeVeque (1979) parallel variance update; numpy.cov",
    )

    # ------------------------------------------------------------------ 2
    # Combined replicate moments reproduce compute_cartesian_pca for both weightings
    from conformatlas.pca import compute_cartesian_pca, project_coordinates

    reps = {"a": rng.normal(size=(70, 6, 3)), "b": rng.normal(size=(230, 6, 3)) * 2 + 1}
    for weighting in ("equal-replicate", "frames"):
        ev, vec, _, _, mean = compute_cartesian_pca(reps, weighting=weighting, n_components=18)
        moms = []
        for k, v in reps.items():
            a = CovarianceAccumulator(18)
            a.update(v.reshape(v.shape[0], -1))
            moms.append(ReplicateMoments(k, a.n, a.mean, a.m2, np.arange(a.n), np.zeros(a.n)))
        m_s, c_s = combine_replicate_moments(moms, weighting)
        ev_s, vec_s, _, _ = eigendecompose_covariance(c_s, 18)
        e1, e2, e3 = _max_rel(ev_s, ev), _vec_cos_err(vec_s, vec), float(np.max(np.abs(m_s - mean)))
        record(
            SECTION,
            f"Merged moments ({weighting}) vs compute_cartesian_pca: eigenvalues / eigenvectors / mean",
            e1 < 1e-8 and e2 < 1e-8 and e3 < 1e-10,
            f"rel λ {e1:.1e}, 1−|cos| {e2:.1e}, mean {e3:.1e}",
            "identical",
            "rtol 1e-8",
            "conformatlas.pca.compute_cartesian_pca (validated against MDAnalysis/GROMACS in validate.py)",
        )

    # ------------------------------------------------------------------ 3
    # Streaming PCA on real trajectories vs in-memory path (demo: covariance route; H1N1: dual SVD route)
    from conformatlas.ndx import parse_ndx
    from conformatlas.streaming import streaming_pca
    from conformatlas.trajectory import (
        load_reference_coordinates,
        resolve_selection_indices,
        stream_trajectory_coordinates,
    )

    datasets = {
        "demo WT (2 replicates, 100 frames, D=45)": (
            ROOT / "examples/demo_dataset/WT/topology.pdb",
            [
                ROOT / "examples/demo_dataset/WT/rep1.xtc",
                ROOT / "examples/demo_dataset/WT/rep2.xtc",
            ],
            "Backbone",
            None,
            None,
        ),
        "neuraminidase+ZMR (81 frames, D=1233 > N: dual route)": (
            ROOT / "tests/data/h1n1_ca/neuraminidase_ca.pdb",
            [ROOT / "tests/data/h1n1_ca/neuraminidase_ca.xtc"],
            "C-alpha",
            ROOT / "tests/data/h1n1_ca/index.ndx",
            "ZMR",
        ),
    }
    h1n1_cache = {}
    for name, (top, trajs, group, ndx, lig) in datasets.items():
        ndxg = parse_ndx(ndx) if ndx else None
        idx = resolve_selection_indices(top, group, ndxg, extra_group=lig)
        fidx = resolve_selection_indices(top, group, ndxg)
        ref = load_reference_coordinates(top, atom_indices=fidx, fallback_trajectory=trajs[0])
        coords = {}
        for i, t in enumerate(trajs):
            coords[f"Rep_{i + 1}"], _ = stream_trajectory_coordinates(
                top, t, atom_indices=idx, fit_atom_indices=fidx, reference_fit_coords=ref
            )
        for weighting in ("equal-replicate", "frames"):
            ev, vec, _, _, mean = compute_cartesian_pca(
                coords, weighting=weighting, n_components=10
            )
            proj = np.vstack([project_coordinates(c, mean, vec) for c in coords.values()])
            r = streaming_pca(
                top,
                trajs,
                idx,
                fit_atom_indices=fidx,
                reference_fit_coords=ref,
                weighting=weighting,
                n_components=10,
                chunk_size=16,
                n_workers=1,
                quiet=True,
            )
            e1 = _max_rel(r.eigenvalues, ev)
            e2 = _vec_cos_err(r.eigenvectors, vec)
            e3 = _signed_proj_err(r.projection_array, proj, r.eigenvectors, vec)
            record(
                SECTION,
                f"streaming_pca ({weighting}) vs in-memory path on {name}",
                e1 < 1e-8 and e2 < 1e-8 and e3 < 1e-8,
                f"rel λ {e1:.1e}, 1−|cos| {e2:.1e}, rel proj {e3:.1e}",
                "identical",
                "rtol 1e-8",
                "stream_trajectory_coordinates + compute_cartesian_pca + project_coordinates",
            )
            if "neuraminidase" in name and weighting == "frames":
                h1n1_cache = dict(
                    top=top,
                    trajs=trajs,
                    idx=idx,
                    fidx=fidx,
                    ref=ref,
                    coords=coords,
                    ev=ev,
                    vec=vec,
                    proj=proj,
                )
        # parallel workers + float32 memmap on the multi-replicate demo
        if len(trajs) > 1:
            ev, vec, _, _, mean = compute_cartesian_pca(
                coords, weighting="equal-replicate", n_components=10
            )
            proj = np.vstack([project_coordinates(c, mean, vec) for c in coords.values()])
            r2 = streaming_pca(
                top,
                trajs,
                idx,
                fit_atom_indices=fidx,
                reference_fit_coords=ref,
                n_components=10,
                chunk_size=32,
                n_workers=2,
                precision="float32",
                projection_store=tmp / "proj32.npy",
                quiet=True,
            )
            e1 = _max_rel(r2.eigenvalues, ev)
            e3 = _signed_proj_err(r2.projection_array, proj, r2.eigenvectors, vec)
            record(
                SECTION,
                "streaming_pca with 2 worker processes, float32 .npy memmap projections vs in-memory (demo WT)",
                e1 < 1e-8 and e3 < 1e-4 and r2.projection_array.dtype == np.float32,
                f"rel λ {e1:.1e}, rel proj {e3:.1e} (dtype {r2.projection_array.dtype})",
                "λ identical; projections within float32 rounding",
                "λ rtol 1e-8; projections rel 1e-4",
                "in-memory path; IEEE-754 single precision (ε ≈ 6e-8)",
            )

    # ------------------------------------------------------------------ 4
    # Independent implementation: MDAnalysis.analysis.pca.PCA (ddof = N−1 covariance), no fit
    import MDAnalysis as mda
    from MDAnalysis.analysis import pca as mdapca

    from conformatlas.mda_analysis import ConformAtlasPCA

    u = mda.Universe(str(h1n1_cache["top"]), str(h1n1_cache["trajs"][0]))
    ca = u.select_atoms("name CA")
    ours = ConformAtlasPCA(ca, fit=False, n_components=10).run()
    theirs = mdapca.PCA(u, select="name CA", align=False, mean=None).run()
    n = len(u.trajectory)
    e1 = _max_rel(ours.results.eigenvalues, theirs.results.variance[:10] * (n - 1) / n)
    e2 = _vec_cos_err(ours.results.eigenvectors, theirs.results.p_components[:, :10])
    record(
        SECTION,
        "ConformAtlasPCA (AnalysisBase, fit=False) vs MDAnalysis.analysis.pca.PCA (variance × (N−1)/N) on neuraminidase Cα",
        e1 < 1e-6 and e2 < 1e-8,
        f"rel λ {e1:.1e}, 1−|cos| {e2:.1e}",
        "identical up to covariance normalisation",
        "λ rtol 1e-6; eigenvectors 1e-8",
        "MDAnalysis 2.x analysis.pca.PCA (independent implementation)",
    )

    # ConformAtlasPCA with superposition vs in-memory path, sliced run
    u.trajectory[0]
    ref0 = ca.positions.copy()
    res = ConformAtlasPCA(ca, reference=ref0, n_components=8, chunk_size=10).run(
        start=3, stop=78, step=2
    )
    c, _ = stream_trajectory_coordinates(
        h1n1_cache["top"],
        h1n1_cache["trajs"][0],
        atom_indices=ca.indices,
        fit_atom_indices=ca.indices,
        reference_fit_coords=ref0,
    )
    c = c[3:78:2]
    ev, vec, _, _, mean = compute_cartesian_pca({"r": c}, weighting="frames", n_components=8)
    proj = project_coordinates(c, mean, vec)
    e1, e2 = _max_rel(res.results.eigenvalues, ev), _vec_cos_err(res.results.eigenvectors, vec)
    e3 = _signed_proj_err(res.results.projections, proj, res.results.eigenvectors, vec)
    record(
        SECTION,
        "ConformAtlasPCA.run(start=3, stop=78, step=2) with Kabsch fit vs in-memory path on the same frames",
        e1 < 1e-8 and e2 < 1e-8 and e3 < 1e-8,
        f"rel λ {e1:.1e}, 1−|cos| {e2:.1e}, rel proj {e3:.1e}",
        "identical",
        "rtol 1e-8",
        "stream_trajectory_coordinates + compute_cartesian_pca",
    )

    # ------------------------------------------------------------------ 5
    # HDF5 round trip, results-dir reload and rebin on a full pipeline run (demo WT)
    from conformatlas.models import AnalysisConfig
    from conformatlas.pipeline import run_analysis
    from conformatlas.results_io import HAS_H5PY, load_results_dir, rebin

    out = tmp / "perf_demo"
    cfg = AnalysisConfig(
        topology=ROOT / "examples/demo_dataset/WT/topology.pdb",
        trajectories=[
            ROOT / "examples/demo_dataset/WT/rep1.xtc",
            ROOT / "examples/demo_dataset/WT/rep2.xtc",
        ],
        output_dir=out,
        bins=32,
        smooth_sigma=1.0,
        n_bootstrap=20,
    )
    res = run_analysis(cfg)

    def deep(a, b, problems, path=""):
        if isinstance(a, pd.DataFrame):
            try:
                pd.testing.assert_frame_equal(a, b, check_dtype=False)
            except AssertionError:
                problems.append(path)
        elif isinstance(a, np.ndarray):
            if not (
                isinstance(b, np.ndarray)
                and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
            ):
                problems.append(path)
        elif dataclasses.is_dataclass(a):
            for f in dataclasses.fields(a):
                deep(getattr(a, f.name), getattr(b, f.name), problems, f"{path}.{f.name}")
        elif isinstance(a, dict):
            if set(a) != set(b):
                problems.append(path)
            for k in a:
                if k in b:
                    deep(a[k], b[k], problems, f"{path}[{k!r}]")
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                problems.append(path)
            for i, (p, q) in enumerate(zip(a, b)):
                deep(p, q, problems, f"{path}[{i}]")
        elif isinstance(a, float) and np.isnan(a):
            if not (isinstance(b, float) and np.isnan(b)):
                problems.append(path)
        elif isinstance(a, Path):
            if Path(b) != a:
                problems.append(path)
        elif a != b:
            problems.append(path)

    if HAS_H5PY:
        from conformatlas.results_io import load_results, save_results

        back = load_results(save_results(res, tmp / "perf_demo.h5"))
        problems: list[str] = []
        deep(res, back, problems)
        n_arrays = sum(1 for _ in _walk_arrays(res))
        record(
            SECTION,
            "AnalysisResults.save/load HDF5 round trip: every array, table, dataclass and scalar equal",
            not problems,
            f"{len(problems)} mismatching fields ({n_arrays} arrays/tables compared)",
            "0 mismatches",
            "exact (bitwise arrays, assert_frame_equal)",
            "h5py; pandas.testing.assert_frame_equal",
            note="" if not problems else "; ".join(problems[:5]),
        )
    else:
        record(
            SECTION,
            "AnalysisResults HDF5 round trip",
            False,
            "h5py missing",
            "0 mismatches",
            "exact",
            "h5py",
            note="skipped",
        )

    fromdir = load_results_dir(out)
    ok = (
        np.allclose(
            fromdir.pca.projections.filter(like="PC").to_numpy(),
            res.pca.projections.filter(like="PC").to_numpy(),
        )
        and np.array_equal(fromdir.fel.free_energy, res.fel.free_energy, equal_nan=True)
        and np.array_equal(fromdir.fel.occupancy_mask, res.fel.occupancy_mask)
        and [(s.label, s.basin_id) for s in fromdir.states]
        == [(s.label, s.basin_id) for s in res.states]
        and bool((fromdir.frame_assignments["state"] == res.frame_assignments["state"]).all())
    )
    record(
        SECTION,
        "load_results_dir reconstructs projections, ΔG grid, occupancy, states and frame assignments from CSV/NPY",
        ok,
        "all equal" if ok else "mismatch",
        "equal to in-memory results",
        "exact / allclose (CSV float repr)",
        "AnalysisResults of the same run",
    )

    for bins, sigma in ((24, 0.0), (40, 1.5)):
        fresh = run_analysis(
            dataclasses.replace(
                cfg, output_dir=tmp / f"perf_fresh_{bins}", bins=bins, smooth_sigma=sigma
            )
        )
        worst = 0.0
        ok = True
        for src in (res, fromdir):
            rb = rebin(src, bins, sigma)
            worst = max(worst, float(np.nanmax(np.abs(rb.fel.free_energy - fresh.fel.free_energy))))
            ok &= np.array_equal(rb.fel.free_energy, fresh.fel.free_energy, equal_nan=True)
            ok &= [(s.label, s.population) for s in rb.states] == [
                (s.label, s.population) for s in fresh.states
            ]
            ok &= bool((rb.frame_assignments["state"] == fresh.frame_assignments["state"]).all())
            ok &= np.array_equal(
                rb.extras["basin_map"],
                np.load(tmp / f"perf_fresh_{bins}" / "states" / "basin_map.npy"),
            )
            fb = fresh.extras["barriers"]
            key = ["state_i", "state_j", "saddle_kJ_mol"]
            a = rb.extras["barriers"][key].sort_values(key[:2]).reset_index(drop=True)
            b = fb[key].sort_values(key[:2]).reset_index(drop=True)
            ok &= a.shape == b.shape and (
                a.shape[0] == 0 or np.allclose(a["saddle_kJ_mol"], b["saddle_kJ_mol"])
            )
        record(
            SECTION,
            f"rebin(results, bins={bins}, smooth_sigma={sigma}) (in-memory and dir-loaded) vs fresh pipeline run",
            bool(ok),
            f"max |ΔΔG| {worst:.1e} kJ/mol; states, assignments, basin map, saddles {'equal' if ok else 'differ'}",
            "identical landscape and states",
            "exact",
            "conformatlas.pipeline.run_analysis with the same bins/sigma",
        )

    # ------------------------------------------------------------------ 6
    # High-level API equals the pipeline on the neuraminidase fixture
    from conformatlas import api

    D = ROOT / "tests/data/h1n1_ca"
    ens = api.load_trajectories(
        D / "neuraminidase_ca.pdb",
        [D / "neuraminidase_ca.xtc"],
        group="C-alpha",
        lig_group="ZMR",
        index_file=D / "index.ndx",
    )
    pcs = api.pca(ens, n_components=10)
    land = api.landscape(pcs, temperature=300, bins=12)
    st = api.states(land)
    kin = api.kinetics(st)
    conv = api.convergence(ens, pcs, st)
    full = api.analyze(
        D / "neuraminidase_ca.pdb",
        [D / "neuraminidase_ca.xtc"],
        output_dir=tmp / "perf_api",
        cov_group="C-alpha",
        lsq_group="C-alpha",
        lig_group="ZMR",
        index_file=D / "index.ndx",
        bins=12,
        n_bootstrap=10,
    )
    ok = (
        _max_rel(full.pca.eigenvalues, pcs.eigenvalues) < 1e-12
        and np.array_equal(full.fel.free_energy, land.fel.free_energy, equal_nan=True)
        and [(s.label, s.population) for s in full.states]
        == [(s.label, s.population) for s in st.states]
        and full.extras["kinetics"]["dwell_summary"].equals(kin["dwell_summary"])
        and abs(full.convergence.rmsip_halves - conv.rmsip_halves) < 1e-12
    )
    record(
        SECTION,
        "api.load_trajectories→pca→landscape→states→kinetics→convergence equals api.analyze (pipeline) on neuraminidase",
        ok,
        "all equal" if ok else "mismatch",
        "identical eigenvalues, ΔG grid, states, dwell summary, RMSIP",
        "exact",
        "conformatlas.pipeline.run_analysis",
    )


def _walk_arrays(obj):
    if isinstance(obj, (np.ndarray, pd.DataFrame)):
        yield obj
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            yield from _walk_arrays(getattr(obj, f.name))
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_arrays(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_arrays(v)


if __name__ == "__main__":  # stand-alone driver
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    results = []

    def record(section, name, passed, measured, expected, tolerance, reference, note=""):
        results.append(passed)
        flag = "SKIP" if note == "skipped" else ("PASS" if passed else "FAIL")
        print(f"[{flag}] {section} / {name}: {measured} vs {expected} ({tolerance})", flush=True)

    with tempfile.TemporaryDirectory(prefix="conformatlas_val_") as t:
        run(record, Path(t), ROOT, 0.008314462618)
    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
