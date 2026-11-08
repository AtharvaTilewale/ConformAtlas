"""Performance runs of benchmarks/CRITERIA.md (reported, never gated).

For every (tool, selection): 4 runs in fresh processes, the first discarded, median of the other 3.
Records the compute stage time (perf_run.py), total wall time and peak resident memory
(/usr/bin/time), and the load average before each run. Also writes benchmarks/ENVIRONMENT.md.

    python perf_suite.py [--perf-src PATH_TO_PERF_BRANCH_SRC] [selection ...]
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "results"
GMX = "/home/atharva/.claude-science/conda/envs/gmx-ref/bin/gmx"
THREADS = "8"
REPEATS = 4  # first discarded
TIMEOUT = 1800


def configs(perf_src: str | None) -> list[tuple[str, str, dict]]:
    base = [
        ("ConformAtlas 1.0.2 (released code)", "conformatlas", {"CA_SRC": str(REPO / "src")}),
    ]
    if perf_src:
        base.append(
            ("ConformAtlas perf branch (Lanczos PCA)", "conformatlas", {"CA_SRC": perf_src})
        )
    base += [
        ("NumPy dense covariance + eigh", "numpy_dense", {"CA_SRC": str(REPO / "src")}),
        ("MDAnalysis PCA", "mdanalysis", {"CA_SRC": str(REPO / "src")}),
        ("ProDy PCA", "prody", {"CA_SRC": str(REPO / "src")}),
        (
            "MDTraj + scikit-learn PCA (auto solver)",
            "mdtraj_sklearn_auto",
            {"CA_SRC": str(REPO / "src")},
        ),
        (
            "MDTraj + scikit-learn PCA (full solver)",
            "mdtraj_sklearn_full",
            {"CA_SRC": str(REPO / "src")},
        ),
        ("GROMACS covar + anaeig", "gromacs", {"CA_SRC": str(REPO / "src")}),
    ]
    return base


def one_run(tool: str, sel: str, extra_env: dict) -> dict:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": THREADS,
        "OPENBLAS_NUM_THREADS": THREADS,
        "MKL_NUM_THREADS": THREADS,
        "CONFORMATLAS_NO_PROGRESS": "1",
        "PYTHONWARNINGS": "ignore",
        **extra_env,
    }
    cmd = [
        "/usr/bin/time",
        "-f",
        "TIME %e %M",
        sys.executable,
        str(HERE / "perf_run.py"),
        tool,
        sel,
    ]
    load = os.getloadavg()[0]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "load1": load}
    line = [ln for ln in r.stderr.splitlines() if ln.startswith("TIME ")]
    if r.returncode != 0 or not line:
        return {"status": "error", "load1": load, "detail": (r.stderr or r.stdout)[-300:]}
    result_lines = [ln for ln in r.stdout.splitlines() if ln.startswith('{"tool"')]
    if not result_lines:
        return {"status": "error", "load1": load, "detail": "no result line: " + r.stdout[-200:]}
    payload = json.loads(result_lines[-1])
    _, w, rss = line[-1].split()
    peak = (
        payload["children_peak_rss_gb"] if tool == "gromacs" else int(rss) / 1e6
    )  # gromacs: largest gmx process
    return {
        "status": "ok",
        "load1": load,
        "wall_seconds": float(w),
        "peak_rss_gb": peak,
        "compute_seconds": payload["compute_seconds"],
        "top_eigenvalue": payload["top_eigenvalue"],
    }


def write_environment() -> None:
    def v(mod: str) -> str:
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "?")
        except Exception:
            return "not importable"

    gmx = subprocess.run([GMX, "--version"], capture_output=True, text=True).stdout
    gmx_v = next((ln for ln in gmx.splitlines() if "GROMACS version" in ln), "?").strip()
    cpu = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in open("/proc/cpuinfo")
            if ln.startswith("model name")
        ),
        "?",
    )
    git = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    text = f"""# Benchmark environment

Written by `benchmarks/adk/perf_suite.py` on {time.strftime("%Y-%m-%d %H:%M %Z")}.

| Item | Value |
|---|---|
| Python | {platform.python_version()} |
| numpy / scipy | {v("numpy")} / {v("scipy")} |
| MDAnalysis | {v("MDAnalysis")} |
| ProDy | {v("prody")} |
| MDTraj | {v("mdtraj")} |
| scikit-learn | {v("sklearn")} |
| deeptime | {v("deeptime")} |
| GROMACS | {gmx_v} |
| ConformAtlas repository commit (this branch) | {git} |
| CPU | {cpu} |
| Logical CPUs visible to the benchmark | {os.cpu_count()} (BLAS/OpenMP threads fixed to {THREADS}) |
| Platform | {platform.platform()} |

Notes: other jobs (a large background download) were running on the machine during the performance runs; the
load average before every run is recorded in `results/performance_runs.csv`. Peak memory is the maximum
resident set size reported by `/usr/bin/time` for the process; for GROMACS it is the largest `gmx` process (the Python parent that converts the trajectory is excluded).
"""
    (HERE.parent / "ENVIRONMENT.md").write_text(text)


def main(argv: list[str]) -> None:
    perf_src = None
    if "--perf-src" in argv:
        i = argv.index("--perf-src")
        perf_src = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    sels = argv or ["calpha", "backbone", "heavy"]
    write_environment()
    runs, summary = [], []
    for sel in sels:
        for label, tool, extra in configs(perf_src):
            rows = []
            for k in range(REPEATS):
                r = one_run(tool, sel, extra)
                r.update(
                    {"config": label, "selection": sel, "repeat": k, "discarded_warmup": k == 0}
                )
                runs.append(r)
                rows.append(r)
                print(
                    label,
                    sel,
                    k,
                    {
                        a: (round(b, 2) if isinstance(b, float) else b)
                        for a, b in r.items()
                        if a
                        in ("status", "compute_seconds", "wall_seconds", "peak_rss_gb", "load1")
                    },
                    flush=True,
                )
                if r["status"] != "ok":
                    break
            good = [r for r in rows[1:] if r["status"] == "ok"]
            entry = {
                "config": label,
                "selection": sel,
                "n_ok_runs": len(good),
                "status": "ok" if len(good) == REPEATS - 1 else rows[-1]["status"],
            }
            if good:
                entry.update(
                    {
                        k: statistics.median(r[k] for r in good)
                        for k in ("compute_seconds", "wall_seconds", "peak_rss_gb")
                    }
                )
            summary.append(entry)
            (OUT / "performance_runs.json").write_text(json.dumps(runs, indent=1))
            (OUT / "performance_summary.json").write_text(json.dumps(summary, indent=1))
    keys = sorted({k for r in runs for k in r})
    with open(OUT / "performance_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(runs)
    with open(OUT / "performance_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "selection",
                "status",
                "n_ok_runs",
                "compute_seconds",
                "wall_seconds",
                "peak_rss_gb",
            ],
        )
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main(sys.argv[1:])
