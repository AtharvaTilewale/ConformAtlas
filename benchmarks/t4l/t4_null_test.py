"""T4 of benchmarks/CRITERIA_T4L_HIV.md: null test of `conformatlas compare`.

All 10 splits of the 5 ff99SB*-ILDN replicates into groups of 2 ("A") and 3 ("B"); every true difference is zero.
Gate: fraction of (split, state) population differences whose 95 % bootstrap CI excludes 0 must be <= 0.15.

    python benchmarks/t4l/t4_null_test.py [--work DIR] [--python PYTHON] [--tag SUFFIX]
        [--force-field]
Also used for T5 (``--force-field``): 5 ff99SB*-ILDN vs 3 ff15ipq replicates, informational.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_t4l import TEMPERATURE, replicates, save_json, tpr, xtc  # noqa: E402

GATE = 0.15


def config(name: str, systems: dict[str, tuple[str, list[str]]], out: Path) -> str:
    lines = [
        f"project: {name}",
        f"temperature: {TEMPERATURE}",
        f"output_dir: {out}",
        "pca:",
        "  fit_group: C-alpha",
        "  analysis_group: C-alpha",
        "systems:",
    ]
    for sname, (ff, reps) in systems.items():
        lines += [f"  - name: {sname}", f"    topology: {tpr(ff)}", "    trajectories:"]
        lines += [f"      - {xtc(ff, r)}" for r in reps]
    return "\n".join(lines) + "\n"


def run_compare(
    name: str, systems: dict, work: Path, python: str
) -> tuple[pd.DataFrame | None, dict]:
    out = work / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / "config.yaml"
    cfg.write_text(config(name, systems, out))
    t = time.perf_counter()
    r = subprocess.run(
        [
            python,
            "-m",
            "conformatlas",
            "compare",
            "-c",
            str(cfg),
            "-o",
            str(out),
            "--no-interactive",
            "-q",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "CONFORMATLAS_NO_PROGRESS": "1"},
    )
    info = {"returncode": r.returncode, "seconds": round(time.perf_counter() - t, 1)}
    if r.returncode != 0:
        info["error"] = (r.stderr or r.stdout)[-800:]
        return None, info
    hits = list(out.rglob("state_population_comparison.csv"))
    if not hits:
        info["error"] = "state_population_comparison.csv not found"
        return None, info
    return pd.read_csv(hits[0]), info


def null_test(work: Path, python: str, tag: str = "") -> dict:
    reps = replicates("ff99SBstar")
    rows, runs = [], {}
    for a in itertools.combinations(reps, 2):
        b = [r for r in reps if r not in a]
        name = "split_" + "_".join(x.replace("md1us", "") for x in a)
        df, info = run_compare(
            name, {"A": ("ff99SBstar", list(a)), "B": ("ff99SBstar", b)}, work, python
        )
        runs[name] = {"A": list(a), "B": b, **info}
        if df is None:
            print(name, info, flush=True)
            continue
        # bootstrap columns only (later versions add *_Welch_CI95_* and *_Welch_Diff_Percent)
        lo = [c for c in df.columns if c == "B_minus_A_CI95_Low"]
        hi = [c for c in df.columns if c == "B_minus_A_CI95_High"]
        wp = [c for c in df.columns if c == "B_minus_A_Welch_p"]
        dcol = [c for c in df.columns if c == "B_minus_A_Percent"]
        assert lo and hi and dcol, list(df.columns)
        for _, r in df.iterrows():
            low, high = float(r[lo[0]]), float(r[hi[0]])
            rows.append(
                {
                    "split": name,
                    "state": r["State"],
                    "difference_pp": float(r[dcol[0]]),
                    "ci_low": low,
                    "ci_high": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "welch_p": float(r[wp[0]]) if wp else float("nan"),
                    "A_pop_pct": float(r["A_Population_Percent"]),
                    "B_pop_pct": float(r["B_Population_Percent"]),
                    "A_replicate_sd_pct": float(r.get("A_Replicate_SD_Percent", float("nan"))),
                    "B_replicate_sd_pct": float(r.get("B_Replicate_SD_Percent", float("nan"))),
                    "welch_ci_low": float(r.get("B_minus_A_Welch_CI95_Low", float("nan"))),
                    "welch_ci_high": float(r.get("B_minus_A_Welch_CI95_High", float("nan"))),
                    "bootstrap_level": str(r.get("A_Bootstrap_Level", "frame (1.0.x)")),
                    "A_heterogeneity_ratio": float(
                        r.get("A_Replicate_Heterogeneity_Ratio", float("nan"))
                    ),
                    "B_heterogeneity_ratio": float(
                        r.get("B_Replicate_Heterogeneity_Ratio", float("nan"))
                    ),
                }
            )
        print(name, info, f"{len(df)} states", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(
        Path(__file__).resolve().parent / "results" / f"t4_null_test{tag}_table.csv", index=False
    )
    fpr = float(tab["ci_excludes_zero"].mean()) if len(tab) else float("nan")
    welch = tab["welch_p"].dropna()
    res = {
        "n_splits_run": int(tab["split"].nunique()) if len(tab) else 0,
        "n_split_state_pairs": int(len(tab)),
        "bootstrap_false_positive_rate": fpr,
        "gate": GATE,
        "pass": bool(len(tab) and fpr <= GATE),
        "welch_false_positive_rate": float((welch < 0.05).mean()) if len(welch) else float("nan"),
        "welch_ci_false_positive_rate": float(
            ((tab["welch_ci_low"] > 0) | (tab["welch_ci_high"] < 0))[
                tab["welch_ci_low"].notna()
            ].mean()
        )
        if len(tab) and tab["welch_ci_low"].notna().any()
        else float("nan"),
        "bootstrap_ci_half_width_median_pp": float(((tab["ci_high"] - tab["ci_low"]) / 2).median())
        if len(tab)
        else float("nan"),
        "bootstrap_level": sorted(tab["bootstrap_level"].unique().tolist()) if len(tab) else [],
        "code_version": _code_version(python),
        "states_per_split": tab.groupby("split").size().to_dict() if len(tab) else {},
        "runs": runs,
    }
    print({k: v for k, v in res.items() if k != "runs"}, flush=True)
    return res


def _code_version(python: str) -> str:
    r = subprocess.run(
        [
            python,
            "-c",
            "import conformatlas, pathlib; print(conformatlas.__version__, "
            "pathlib.Path(conformatlas.__file__).parent)",
        ],
        capture_output=True,
        text=True,
    )
    src = r.stdout.split()[-1] if r.stdout else ""
    g = subprocess.run(
        ["git", "-C", src, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    b = subprocess.run(
        ["git", "-C", src, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "-C", src, "status", "--porcelain", "--", "src"], capture_output=True, text=True
    )
    return f"{r.stdout.strip()} @ {b.stdout.strip()} {g.stdout.strip()}" + (
        " (uncommitted changes in src)" if dirty.stdout.strip() else ""
    )


def force_field(work: Path, python: str) -> dict:
    df, info = run_compare(
        "forcefield_ff15ipq_vs_ff99SBstar",
        {
            "ff99SBstar": ("ff99SBstar", replicates("ff99SBstar")),
            "ff15ipq": ("ff15ipq", replicates("ff15ipq")),
        },
        work,
        python,
    )
    res = {"run": info}
    if df is not None:
        res["population_table"] = df.round(4).to_dict(orient="records")
        out = work / "forcefield_ff15ipq_vs_ff99SBstar"
        for fname in (
            "subspace_overlap.csv",
            "fel_overlap.csv",
            "state_free_energy_comparison.csv",
        ):
            hit = list(out.rglob(fname))
            if hit:
                res[fname] = pd.read_csv(hit[0]).round(4).to_dict(orient="records")
    return res


def main(argv: list[str]) -> None:
    work = (
        Path(argv[argv.index("--work") + 1])
        if "--work" in argv
        else Path(os.environ.get("CA_BENCH_WORK", Path(os.environ.get("CA_BENCH_DATA", Path(__file__).resolve().parents[1] / "data")) / "work")) / "t4l_compare_whole"
    )
    python = (
        argv[argv.index("--python") + 1]
        if "--python" in argv
        else str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python")
    )
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else ""
    if "--force-field" in argv:
        res = force_field(work, python)
        res["code_version"] = _code_version(python)
        save_json(f"t5_force_field{tag}.json", res)
        print(
            json.dumps(
                {k: v for k, v in res.items() if k != "population_table"}, indent=1, default=float
            )[:3000]
        )
    else:
        save_json(f"t4_null_test{tag}.json", null_test(work, python, tag))


if __name__ == "__main__":
    main(sys.argv[1:])
