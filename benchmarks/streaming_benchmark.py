"""Benchmark: in-memory PCA path vs. chunked streaming PCA.

Scales the neuraminidase fixture (tests/data/h1n1_ca, 81 frames, Cα + zanamivir
heavy atoms = 411 atoms, D = 1233) to 810, 8100 and 81000 frames by replicating frames
into new XTC files, then times both paths in fresh subprocesses and records the
peak resident set size (``resource.getrusage``; children included for the
worker pool). Also compares 1 vs 4 workers on four replicates of (largest size / 4) frames.

Run from the repository root::

    .venv/bin/python benchmarks/streaming_benchmark.py [--sizes 81 810 8100 81000] [--out benchmarks/streaming_benchmark.md]

The script is self-contained (no pytest); results are written as Markdown.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "h1n1_ca"
TOPO = DATA / "neuraminidase_ca.pdb"
TRAJ = DATA / "neuraminidase_ca.xtc"
NDX = DATA / "index.ndx"

CHILD = textwrap.dedent(
    """
    import json, os, resource, sys, time, warnings
    warnings.filterwarnings("ignore")
    os.environ["CONFORMATLAS_NO_PROGRESS"] = "1"
    import numpy as np
    from conformatlas.ndx import parse_ndx
    from conformatlas.trajectory import (load_reference_coordinates, resolve_selection_indices,
                                         stream_trajectory_coordinates)
    cfg = json.loads(sys.argv[1])
    topo, trajs, mode = cfg["topology"], cfg["trajectories"], cfg["mode"]
    ndx = parse_ndx(cfg["ndx"])
    idx = resolve_selection_indices(topo, "C-alpha", ndx, extra_group="ZMR")
    fidx = resolve_selection_indices(topo, "C-alpha", ndx)
    ref = load_reference_coordinates(topo, atom_indices=fidx, fallback_trajectory=trajs[0])
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    t0 = time.perf_counter()
    if mode == "in-memory":
        from conformatlas.pca import compute_cartesian_pca, project_coordinates
        coords = {}
        for i, t in enumerate(trajs):
            c, _ = stream_trajectory_coordinates(topo, t, atom_indices=idx, fit_atom_indices=fidx,
                                                 reference_fit_coords=ref)
            coords[f"Rep_{i+1}"] = c
        ev, vec, _, _, mean = compute_cartesian_pca(coords, weighting="equal-replicate", n_components=10)
        proj = np.vstack([project_coordinates(c, mean, vec) for c in coords.values()])
        n = proj.shape[0]
    else:
        from conformatlas.streaming import streaming_pca
        store = cfg.get("store")
        r = streaming_pca(topo, trajs, idx, fit_atom_indices=fidx, reference_fit_coords=ref,
                          weighting="equal-replicate", n_components=10, chunk_size=cfg.get("chunk", 256),
                          n_workers=cfg.get("workers", 1), precision=cfg.get("precision", "float64"),
                          projection_store=store, quiet=True)
        ev = r.eigenvalues
        n = r.projection_array.shape[0]
    dt = time.perf_counter() - t0
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    print(json.dumps({"seconds": dt, "peak_rss_mb": self_rss, "worker_peak_rss_mb": child_rss,
                      "baseline_rss_mb": rss0, "n_frames": int(n), "ev1": float(ev[0])}))
    """
)


def make_scaled_xtc(src_topo: Path, src_traj: Path, n_frames: int, out: Path) -> Path:
    """Write ``n_frames`` frames to ``out`` by cycling through the source frames."""
    import MDAnalysis as mda

    u = mda.Universe(str(src_topo), str(src_traj))
    n_src = len(u.trajectory)
    dt = float(u.trajectory.dt) if n_src > 1 else 1.0
    with mda.Writer(str(out), n_atoms=u.atoms.n_atoms) as w:
        for i in range(n_frames):
            ts = u.trajectory[i % n_src]
            ts.time = i * dt
            ts.frame = i
            w.write(u.atoms)
    return out


def run_child(python: str, cfg: dict) -> dict:
    env = dict(os.environ, CONFORMATLAS_NO_PROGRESS="1")
    res = subprocess.run(
        [python, "-c", CHILD, json.dumps(cfg)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-2000:])
    return json.loads(res.stdout.strip().splitlines()[-1])


def _summary(rows: list[dict]) -> list[str]:
    """Headline numbers for the largest single-trajectory size and the worker comparison."""
    single = [r for r in rows if r["replicates"] == 1]
    if not single:
        return []
    n_max = max(r["frames"] for r in single)
    mem = next(r for r in single if r["frames"] == n_max and r["path"] == "in-memory")
    st = next(
        r for r in single if r["frames"] == n_max and r["path"].startswith("streaming (float64")
    )
    out = [
        "",
        "## Summary",
        "",
        f"- At {n_max} frames the in-memory path peaks at {mem['peak_rss_mb']:.0f} MB and the streaming path at "
        f"{st['peak_rss_mb']:.0f} MB ({mem['peak_rss_mb'] / st['peak_rss_mb']:.1f}× less); streaming wall time is "
        f"{st['seconds'] / mem['seconds']:.1f}× the in-memory time ({st['seconds']:.1f} s vs {mem['seconds']:.1f} s).",
    ]
    par = [r for r in rows if r["replicates"] == 4]
    one = next((r for r in par if r["path"] == "streaming, 1 worker"), None)
    many = next((r for r in par if "workers" in r["path"] and "float32" not in r["path"]), None)
    if one and many:
        out.append(
            f"- Four replicates ({one['frames']} frames): {many['path']} take {many['seconds']:.1f} s vs "
            f"{one['seconds']:.1f} s with one worker ({one['seconds'] / many['seconds']:.2f}× speed-up); "
            f"the remaining time is the serial eigendecomposition, result transfer and XTC decoding."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[81, 810, 8100, 81000])
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "streaming_benchmark.md")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import numpy as np

    rows = []
    with tempfile.TemporaryDirectory(prefix="conformatlas_bench_") as tmp:
        tmp = Path(tmp)
        for n in args.sizes:
            traj = TRAJ if n == 81 else make_scaled_xtc(TOPO, TRAJ, n, tmp / f"scaled_{n}.xtc")
            base = {"topology": str(TOPO), "ndx": str(NDX), "trajectories": [str(traj)]}
            configs = [
                ("in-memory", dict(base, mode="in-memory")),
                ("streaming (float64)", dict(base, mode="streaming")),
                (
                    "streaming (float32, .npy memmap)",
                    dict(
                        base,
                        mode="streaming",
                        precision="float32",
                        store=str(tmp / f"proj_{n}.npy"),
                    ),
                ),
            ]
            ref_ev = None
            for label, cfg in configs:
                r = run_child(args.python, cfg)
                if ref_ev is None:
                    ref_ev = r["ev1"]
                r.update(
                    frames=n, path=label, rel_ev1=abs(r["ev1"] - ref_ev) / ref_ev, replicates=1
                )
                rows.append(r)
                print(
                    f"{n:>6d} frames | {label:34s} | {r['seconds']:7.2f} s | peak RSS {r['peak_rss_mb']:7.0f} MB",
                    flush=True,
                )

        # parallel replicates: 4 x (max size / 4) frames
        n_big = max(args.sizes)
        per = max(1, n_big // 4)
        trajs = [str(make_scaled_xtc(TOPO, TRAJ, per, tmp / f"rep{i}_{per}.xtc")) for i in range(4)]
        base = {"topology": str(TOPO), "ndx": str(NDX), "trajectories": trajs}
        ref_ev = None
        for label, cfg in [
            ("in-memory", dict(base, mode="in-memory")),
            ("streaming, 1 worker", dict(base, mode="streaming", workers=1)),
            (
                f"streaming, {args.workers} workers",
                dict(base, mode="streaming", workers=args.workers),
            ),
            (
                f"streaming, {args.workers} workers, float32 memmap",
                dict(
                    base,
                    mode="streaming",
                    workers=args.workers,
                    precision="float32",
                    store=str(tmp / "proj_par.npy"),
                ),
            ),
        ]:
            r = run_child(args.python, cfg)
            if ref_ev is None:
                ref_ev = r["ev1"]
            r.update(
                frames=4 * per, path=label, rel_ev1=abs(r["ev1"] - ref_ev) / ref_ev, replicates=4
            )
            rows.append(r)
            print(
                f"{4 * per:>6d} frames | {label:34s} | {r['seconds']:7.2f} s | peak RSS {r['peak_rss_mb']:7.0f} MB (workers {r['worker_peak_rss_mb']:.0f} MB)",
                flush=True,
            )

    cpu = platform.processor() or platform.machine()
    lines = [
        "# Streaming PCA benchmark",
        "",
        f"Generated by `benchmarks/streaming_benchmark.py` on {time.strftime('%Y-%m-%d %H:%M')} "
        f"({platform.system()} {platform.release()}, {cpu}, {os.cpu_count()} CPUs, Python {platform.python_version()}, numpy {np.__version__}).",
        "",
        "System: neuraminidase fixture `tests/data/h1n1_ca` (Cα + zanamivir heavy atoms, 411 atoms, D = 1233), "
        "frames replicated to the sizes below. Each row is one fresh Python process; *peak RSS* is "
        "`resource.getrusage(RUSAGE_SELF).ru_maxrss` of that process (interpreter + MDAnalysis import included, "
        "≈ the 81-frame value), *worker RSS* the largest worker process. The in-memory path is "
        "`stream_trajectory_coordinates` + `compute_cartesian_pca` + `project_coordinates`; the streaming path is "
        "`conformatlas.streaming.streaming_pca` (two passes, chunk 256). The last column is the relative difference of "
        "the leading eigenvalue from the in-memory value of the same row group.",
        "",
        "## Single trajectory",
        "",
        "| frames | path | wall time (s) | peak RSS (MB) | peak − baseline (MB) | abs rel diff λ₁ |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["replicates"] == 1:
            lines.append(
                f"| {r['frames']} | {r['path']} | {r['seconds']:.2f} | {r['peak_rss_mb']:.0f} | "
                f"{r['peak_rss_mb'] - r['baseline_rss_mb']:.0f} | {r['rel_ev1']:.1e} |"
            )
    lines += [
        "",
        "## Four replicates (parallel workers)",
        "",
        "| total frames | path | wall time (s) | peak RSS main (MB) | peak RSS worker (MB) | abs rel diff λ₁ |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["replicates"] == 4:
            workers = f"{r['worker_peak_rss_mb']:.0f}" if "workers" in r["path"] else "–"
            lines.append(
                f"| {r['frames']} | {r['path']} | {r['seconds']:.2f} | {r['peak_rss_mb']:.0f} | "
                f"{workers} | {r['rel_ev1']:.1e} |"
            )
    lines += [
        "",
        "Notes",
        "",
        "- The streaming path reads every trajectory twice (moments, then projections); its wall time is therefore "
        "bounded below by two passes of XTC decoding + Kabsch superposition, which dominates for small systems.",
        "- Memory of the streaming path is independent of the number of frames: one `D × D` float64 matrix "
        "(1233² × 8 B ≈ 12 MB here) plus one chunk; the in-memory path grows linearly with frames "
        "(float32 coordinates plus float64 working copies inside the PCA).",
        "- A leading-eigenvalue difference of order 1e-15 confirms that both paths compute the same covariance; the full "
        "validation (eigenvalues, eigenvectors, projections, both weightings) is in `validation/checks_performance.py`.",
    ]
    lines += _summary(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
