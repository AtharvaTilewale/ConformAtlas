"""T6 of benchmarks/CRITERIA_T4L_HIV.md (informational): replicate agreement.

* Pairwise RMSIP (top 10 Cα modes) between the 8 replicates, each replicate's own PCA after the common Cα fit.
* `conformatlas analyze` (defaults, Cα fit and analysis groups) on every replicate: states, sampling verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_t4l import TEMPERATURE, fitted, replicates, save_json, tpr, xtc  # noqa: E402

WORK = Path("/media/atharva/3EA6170E34C679E4/ca_perf/t4l_analyze_whole")


def rmsip(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    return float(np.sqrt(np.sum((a[:, :k].T @ b[:, :k]) ** 2) / k))


def main() -> None:
    from conformatlas.pca import compute_cartesian_pca

    names, vecs = [], []
    for ff in ("ff99SBstar", "ff15ipq"):
        for rep in replicates(ff):
            _, v, _, _, _ = compute_cartesian_pca(
                {"r": fitted(ff, rep, "calpha")}, weighting="frames", n_components=10
            )
            names.append(f"{ff}/{rep}")
            vecs.append(v)
    mat = [[rmsip(a, b) for b in vecs] for a in vecs]
    out: dict = {"replicates": names, "rmsip_top10": mat}
    within99 = [mat[i][j] for i in range(5) for j in range(i + 1, 5)]
    within15 = [mat[i][j] for i in range(5, 8) for j in range(i + 1, 8)]
    between = [mat[i][j] for i in range(5) for j in range(5, 8)]
    out["rmsip_summary"] = {
        k: {"min": float(min(v)), "median": float(np.median(v)), "max": float(max(v))}
        for k, v in (
            ("within_ff99SBstar", within99),
            ("within_ff15ipq", within15),
            ("between_force_fields", between),
        )
    }
    print("RMSIP", out["rmsip_summary"], flush=True)

    python = str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python")
    out["analyze"] = {}
    for ff in ("ff99SBstar", "ff15ipq"):
        for rep in replicates(ff):
            o = WORK / f"{ff}_{rep}"
            r = subprocess.run(
                [
                    python,
                    "-m",
                    "conformatlas",
                    "analyze",
                    "-s",
                    str(tpr(ff)),
                    "-f",
                    str(xtc(ff, rep)),
                    "-T",
                    str(TEMPERATURE),
                    "-c",
                    "C-alpha",
                    "-l",
                    "C-alpha",
                    "-o",
                    str(o),
                    "-q",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "CONFORMATLAS_NO_PROGRESS": "1"},
            )
            rec: dict = {"returncode": r.returncode}
            if r.returncode == 0 and (o / "summary.json").exists():
                s = json.loads((o / "summary.json").read_text())
                meta = json.loads((o / "run_metadata.json").read_text())
                rec.update(
                    {
                        "states": [
                            (st["label"], st["population_percent"]) for st in s.get("states", [])
                        ],
                        "unassigned_percent": s.get("unassigned_population_percent"),
                        "convergence": s.get("convergence"),
                        "warnings": meta.get("warnings", []),
                    }
                )
            else:
                rec["error"] = (r.stderr or r.stdout)[-500:]
            out["analyze"][f"{ff}/{rep}"] = rec
            print("analyze", ff, rep, {k: v for k, v in rec.items() if k != "warnings"}, flush=True)
    save_json("t6_replicates.json", out)


if __name__ == "__main__":
    main()
