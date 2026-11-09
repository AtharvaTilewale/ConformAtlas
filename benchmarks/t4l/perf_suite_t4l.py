"""T7 performance runs (reported, never gated), same protocol as benchmarks/adk/perf_suite.py:
fresh process per run, 8 BLAS threads, 1800 s timeout; pooled 100 ps data: 4 runs, first discarded, median of 3;
1 ps scaling points: one run each (declared in CRITERIA_T4L_HIV.md).

    python perf_suite_t4l.py --perf-src PATH_TO_PERF_BRANCH_SRC
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "results"
TIMEOUT = 1800


def plan(perf_src: str) -> list[tuple[str, str, str, str, dict, int]]:
    rel = {"CA_SRC": str(REPO / "src")}
    perf = {"CA_SRC": perf_src}
    rows = []
    for sel in ("calpha", "heavy"):
        for label, tool, env in [
            ("ConformAtlas 1.0.2 (released code)", "conformatlas", rel),
            ("ConformAtlas perf branch (Lanczos PCA)", "conformatlas", perf),
            ("ConformAtlas streaming engine (released code)", "conformatlas_streaming", rel),
            ("NumPy dense covariance + eigh", "numpy_dense", rel),
            ("MDAnalysis PCA", "mdanalysis", rel),
            ("ProDy PCA (coordinates read with MDAnalysis)", "prody", rel),
            ("MDTraj + scikit-learn PCA (auto solver)", "sklearn_auto", rel),
            ("MDTraj + scikit-learn PCA (full solver)", "sklearn_full", rel),
            ("GROMACS covar + anaeig", "gromacs", rel),
        ]:
            rows.append((label, tool, "pooled100", sel, env, 4))
    for label, tool, env in [
        ("ConformAtlas 1.0.2 (released code)", "conformatlas", rel),
        ("ConformAtlas perf branch (Lanczos PCA)", "conformatlas", perf),
        ("ConformAtlas streaming engine (released code)", "conformatlas_streaming", rel),
        ("NumPy dense covariance + eigh", "numpy_dense", rel),
        ("MDAnalysis PCA", "mdanalysis", rel),
        ("ProDy PCA (coordinates read with MDAnalysis)", "prody", rel),
        ("MDTraj + scikit-learn PCA (auto solver)", "sklearn_auto", rel),
        ("MDTraj + scikit-learn PCA (full solver)", "sklearn_full", rel),
        ("GROMACS covar + anaeig", "gromacs", rel),
    ]:
        rows.append((label, tool, "md1us4_1ps", "calpha", env, 1))
    for label, tool, env in [
        ("ConformAtlas streaming engine (released code)", "conformatlas_streaming", rel),
        ("GROMACS covar + anaeig", "gromacs", rel),
    ]:
        rows.append((label, tool, "md1us4_1ps", "heavy", env, 1))
    return rows


def one(tool, dataset, sel, extra) -> dict:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "CONFORMATLAS_NO_PROGRESS": "1",
        "PYTHONWARNINGS": "ignore",
        **extra,
    }
    load = os.getloadavg()[0]
    try:
        r = subprocess.run(
            [
                "/usr/bin/time",
                "-f",
                "TIME %e %M",
                sys.executable,
                str(HERE / "perf_run_t4l.py"),
                tool,
                dataset,
                sel,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "load1": load}
    tl = [ln for ln in r.stderr.splitlines() if ln.startswith("TIME ")]
    res = [ln for ln in r.stdout.splitlines() if ln.startswith('{"tool"')]
    if r.returncode != 0 or not tl or not res:
        return {"status": "error", "load1": load, "detail": (r.stderr or r.stdout)[-300:]}
    p = json.loads(res[-1])
    _, w, rss = tl[-1].split()
    peak = p["children_peak_rss_gb"] if tool == "gromacs" else int(rss) / 1e6
    return {
        "status": "ok",
        "load1": load,
        "wall_seconds": float(w),
        "peak_rss_gb": peak,
        "compute_seconds": p["compute_seconds"],
        "top_eigenvalue": p["top_eigenvalue"],
    }


def main(argv: list[str]) -> None:
    perf_src = argv[argv.index("--perf-src") + 1]
    runs, summary = [], []
    for label, tool, dataset, sel, extra, n in plan(perf_src):
        rows = []
        for k in range(n):
            r = one(tool, dataset, sel, extra)
            r.update(
                {
                    "config": label,
                    "dataset": dataset,
                    "selection": sel,
                    "repeat": k,
                    "discarded_warmup": n > 1 and k == 0,
                }
            )
            runs.append(r)
            rows.append(r)
            print(
                label,
                dataset,
                sel,
                k,
                {
                    a: (round(b, 2) if isinstance(b, float) else b)
                    for a, b in r.items()
                    if a in ("status", "compute_seconds", "wall_seconds", "peak_rss_gb", "load1")
                },
                flush=True,
            )
            if r["status"] != "ok":
                break
        good = [r for r in (rows[1:] if n > 1 else rows) if r["status"] == "ok"]
        need = n - 1 if n > 1 else 1
        e = {
            "config": label,
            "dataset": dataset,
            "selection": sel,
            "n_ok_runs": len(good),
            "status": "ok" if len(good) == need else rows[-1]["status"],
        }
        if good:
            e.update(
                {
                    k: statistics.median(r[k] for r in good)
                    for k in ("compute_seconds", "wall_seconds", "peak_rss_gb")
                }
            )
        summary.append(e)
        (OUT / "performance_runs.json").write_text(json.dumps(runs, indent=1))
        (OUT / "performance_summary.json").write_text(json.dumps(summary, indent=1))
    for name, rows, fields in (
        ("performance_runs.csv", runs, sorted({k for r in runs for k in r})),
        (
            "performance_summary.csv",
            summary,
            [
                "config",
                "dataset",
                "selection",
                "status",
                "n_ok_runs",
                "compute_seconds",
                "wall_seconds",
                "peak_rss_gb",
            ],
        ),
    ):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
