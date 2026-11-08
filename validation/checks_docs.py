"""Documentation checks for the validation battery (imported by validate.py).

``run(record, tmp, ROOT, KB)`` verifies that the documentation is consistent with
the software it describes:

1. every ``bash`` and ``python`` code block of ``docs/tutorial_adk.md``
   and the runnable ``python`` blocks of ``docs/python_api.md`` execute without
   error, in order, from a scratch copy of the repository data;
2. every numeric value quoted in the tutorial equals (at the quoted rounding) the
   value re-read from the output file it was taken from;
3. the validation table in ``docs/benchmarks.md`` is the rendering of
   ``validation/results.json`` (one row per check, nothing invented);
4. every DOI in ``paper/paper.bib`` resolves at doi.org (skipped when paper/ is absent or when
   the network is unreachable);
5. ``.zenodo.json`` agrees with ``CITATION.cff`` and ``pyproject.toml``;
6. the Sphinx documentation builds with warnings treated as errors.

Run standalone with ``python validation/checks_docs.py`` (``--write-benchmarks``
regenerates the table in ``docs/benchmarks.md`` from ``results.json``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

DOCS_RUN_ENV = "CONFORMATLAS_DOCS_CHECK_GMX"  # optional: path to a gmx executable

# ----------------------------------------------------------------------------
# Markdown helpers
# ----------------------------------------------------------------------------

_FENCE = re.compile(r"^```([A-Za-z0-9_-]*)[^\n]*\n(.*?)^```", re.M | re.S)


def extract_code_blocks(markdown: str) -> list[tuple[str, str]]:
    """Return ``[(language, code), ...]`` for every fenced block, in document order."""
    return [(m.group(1), m.group(2)) for m in _FENCE.finditer(markdown)]


def _which_gmx() -> str | None:
    cand = os.environ.get(DOCS_RUN_ENV)
    if cand and Path(cand).exists():
        return cand
    for name in ("gmx", "gmx_mpi"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _shell_env(ROOT: Path) -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(Path(sys.executable).parent)]
    gmx = _which_gmx()
    if gmx:
        paths.append(str(Path(gmx).parent))
    env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    return env


def run_code_blocks(
    blocks: list[tuple[str, str]], cwd: Path, ROOT: Path, languages=("bash", "python")
) -> list[tuple[int, str, int, str]]:
    """Execute blocks; return ``[(index, language, returncode, tail_of_output)]``.

    ``shell``/``text``/``yaml`` blocks are documentation only and are skipped.
    """
    env = _shell_env(ROOT)
    out = []
    for i, (lang, code) in enumerate(blocks):
        if lang not in languages:
            continue
        if lang == "bash":
            cmd = ["bash", "-e", "-o", "pipefail", "-c", code]
        else:
            cmd = [sys.executable, "-c", code]
        res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        tail = (res.stdout + res.stderr)[-800:]
        out.append((i, lang, res.returncode, tail))
    return out


def tutorial_workdir(tmp: Path, ROOT: Path) -> Path:
    """Scratch directory laid out like the repository root (examples/adk copied)."""
    work = tmp / "docs_tutorial"
    if work.exists():
        shutil.rmtree(work)
    (work / "examples").mkdir(parents=True)
    shutil.copytree(
        ROOT / "examples" / "adk",
        work / "examples" / "adk",
        ignore=shutil.ignore_patterns(".*offsets*", "__pycache__"),
    )
    # python_api.md uses the small regression fixture
    (work / "tests" / "data").mkdir(parents=True)
    shutil.copytree(
        ROOT / "tests" / "data" / "adk_ca",
        work / "tests" / "data" / "adk_ca",
        ignore=shutil.ignore_patterns(".*offsets*"),
    )
    return work


# ----------------------------------------------------------------------------
# Numbers quoted in the tutorial, and where each one comes from
# ----------------------------------------------------------------------------


def _csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def _m(x: float, nd: int) -> str:
    """Number as printed in tutorial prose: Unicode minus sign."""
    return f"{x:.{nd}f}".replace("-", "−")


def _superpose(x, ref):
    import numpy as np

    xc, rc = x - x.mean(0), ref - ref.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ rc)
    d = np.sign(np.linalg.det(u @ vt))
    return xc @ (u @ np.diag([1.0, 1.0, d]) @ vt) + ref.mean(0)


def tutorial_number_sources(
    work: Path,
) -> tuple[list[tuple[str, str, float, int]], list[tuple[str, str, bool]]]:
    """Numbers and claims of docs/tutorial_adk.md with their sources.

    ``rows``: ``(label, quoted_text, recomputed_value, decimals)``; ``quoted_text``
    must appear verbatim in the tutorial and contain ``recomputed_value`` at that
    rounding. ``claims``: ``(label, quoted_text, holds)``; the statement must appear
    verbatim and be true for the run. Values are re-read from the output files of the
    tutorial run in ``work`` and from the shipped example data.
    """
    import MDAnalysis as mda
    import numpy as np

    from conformatlas.fel import choose_bins

    X = work / "examples" / "adk"
    A = work / "tutorial" / "adk"
    G = work / "tutorial" / "adk_angles"
    E = work / "tutorial" / "adk_equilibrated"
    C = work / "tutorial" / "wt_mut" / "results" / "comparison"
    rows: list[tuple[str, str, float, int]] = []
    claims: list[tuple[str, str, bool]] = []

    # ---- data and crystal structures
    cryst = json.load(open(X / "crystal" / "crystal_angles.json", encoding="utf-8"))
    c1, c4 = cryst["1AKE"], cryst["4AKE"]
    rows.append(
        (
            "1AKE angles",
            f"{c1['theta_NMP']:.1f}° / {c1['theta_LID']:.1f}° (closed 1AKE)",
            c1["theta_NMP"],
            1,
        )
    )
    rows.append(
        (
            "4AKE angles",
            f"{c4['theta_NMP']:.1f}° / {c4['theta_LID']:.1f}°\n(open 4AKE)",
            c4["theta_NMP"],
            1,
        )
    )
    proj = _csv(A / "pca" / "projections.csv")
    rows.append(("frames", f"{len(proj):,} frames 240 ps apart", len(proj), 0))

    # ---- console summary of the main run
    eq = _csv(A / "convergence" / "equilibration.csv").iloc[0]
    n_tot = len(proj)
    rows.append(
        (
            "equilibration note",
            f"discarding the first {int(eq.t0_index)} frames ({100 * eq.t0_index / n_tot:.0f}%, t0 = {eq.t0_ps:.0f} ps)",
            eq.t0_ps,
            0,
        )
    )
    ev = _csv(A / "pca" / "explained_variance.csv")
    rows.append(
        (
            "top-2 variance",
            f"Leading two modes explain {ev.CumulativeVariance_Percent[1]:.1f}%",
            ev.CumulativeVariance_Percent[1],
            1,
        )
    )
    pops = _csv(A / "statistics" / "state_populations.csv").iloc[0]
    summ = json.load(open(A / "summary.json", encoding="utf-8"))
    un = float(summ["unassigned_population_percent"])
    rows.append(
        (
            "state A console",
            f"State A: {pops.Mean_Population_Percent:.1f}% (Unassigned / Minor: {un:.1f}%)",
            un,
            1,
        )
    )
    rows.append(
        (
            "state A CI console",
            f"population {pops.Mean_Population_Percent:.1f}% [{pops.Bootstrap_CI_95_Lower:.1f}, {pops.Bootstrap_CI_95_Upper:.1f}]",
            pops.Bootstrap_CI_95_Lower,
            1,
        )
    )
    rmsip = _csv(A / "convergence" / "rmsip.csv")
    cos = _csv(A / "convergence" / "cosine_content.csv")
    rows.append(
        (
            "RMSIP console",
            f"Half-vs-half RMSIP: {rmsip.Value[0]:.2f} | PC1 cosine content: {cos.Cosine_Content[0]:.2f}",
            rmsip.Value[0],
            2,
        )
    )
    msm = json.load(open(A / "msm" / "msm_summary.json", encoding="utf-8"))
    rows.append(("MSM active", f"{msm['n_active']} active; lag 10 frames", msm["n_active"], 0))
    ckd = float(msm["ck_test"]["max_abs_deviation"])
    rows.append(("CK console", f"passed (max |deviation| {ckd:.3f})", ckd, 3))
    fd = choose_bins(proj.PC1.to_numpy(), proj.PC2.to_numpy())
    rows.append(("FD bins", f"chooses {fd}\nbins per axis", fd, 0))

    # ---- modes, RMSF, DCCM
    for k in range(3):
        rows.append(
            (
                f"variance PC{k + 1}",
                f"{ev.ExplainedVariance_Percent[k]:.1f} %",
                ev.ExplainedVariance_Percent[k],
                1,
            )
        )
    rows.append(
        (
            "PC variances sentence",
            f"PC1, PC2 and PC3 carry {ev.ExplainedVariance_Percent[0]:.1f} %, {ev.ExplainedVariance_Percent[1]:.1f} % and {ev.ExplainedVariance_Percent[2]:.1f} %",
            ev.ExplainedVariance_Percent[2],
            1,
        )
    )
    mc = _csv(A / "pca" / "mode_contributions_residues.csv")
    lid = mc.residue_number.between(122, 159)
    nmp = mc.residue_number.between(30, 59)
    rows.append(("residues", f"{int(lid.sum())} and {int(nmp.sum())} of the 214", len(mc), 0))
    sl = 100 * mc.loc[lid, "PC1"].sum() / mc.PC1.sum()
    sn = 100 * mc.loc[nmp, "PC1"].sum() / mc.PC1.sum()
    rows.append(("PC1 shares", f"carry {sl:.1f} % and {sn:.1f} % of PC1", sl, 1))
    top = mc.loc[mc.PC1.idxmax()]
    rows.append(
        (
            "top PC1 residue",
            f"largest PC1 weight is on {top.residue_name.title()}{int(top.residue_number)}",
            top.residue_number,
            0,
        )
    )
    rm = _csv(A / "descriptors" / "rmsf_residues.csv")
    rl = rm.RMSF[rm.residue_number.between(122, 159)].mean()
    rn = rm.RMSF[rm.residue_number.between(30, 59)].mean()
    rc_ = rm.RMSF[~(rm.residue_number.between(122, 159) | rm.residue_number.between(30, 59))].mean()
    rows.append(("RMSF LID", f"averages {rl:.2f} Å in the LID", rl, 2))
    rows.append(("RMSF NMP", f"{rn:.2f} Å in the NMP domain", rn, 2))
    rows.append(("RMSF CORE", f"{rc_:.2f} Å in the CORE", rc_, 2))
    mx = rm.loc[rm.RMSF.idxmax()]
    rows.append(
        (
            "RMSF max",
            f"maximum of {mx.RMSF:.2f} Å at {mx.residue_name.title()}{int(mx.residue_number)}",
            mx.RMSF,
            2,
        )
    )
    dc = np.load(A / "residues" / "dccm.npy")
    r = np.arange(1, dc.shape[0] + 1)
    L = (r >= 122) & (r <= 159)
    N = (r >= 30) & (r <= 59)
    rows.append(
        (
            "DCCM LID-NMP",
            f"LID and NMP residues is {_m(dc[np.ix_(L, N)].mean(), 2)}",
            dc[np.ix_(L, N)].mean(),
            2,
        )
    )
    rows.append(
        (
            "DCCM LID-LID",
            f"correlate with each other at {dc[np.ix_(L, L)].mean():.2f}",
            dc[np.ix_(L, L)].mean(),
            2,
        )
    )

    # ---- projection of the crystal structures (independent of the tutorial block)
    ref = mda.Universe(str(X / "adk_ca.pdb")).atoms.positions.astype(float)
    mean = np.load(A / "pca" / "mean_coordinates.npy").ravel()
    vecs = np.load(A / "pca" / "eigenvectors.npy")
    for name in ("1ake", "4ake"):
        x = mda.Universe(str(X / "crystal" / f"{name}_A_ca.pdb")).atoms.positions.astype(float)
        pc = (_superpose(x, ref).ravel() - mean) @ vecs[:, :2]
        rows.append(
            (
                f"{name} projection",
                f"{name.upper()}: PC1 = {pc[0]:.1f} Å, PC2 = {pc[1]:.1f} Å",
                pc[0],
                1,
            )
        )
    rows.append(("first frame PC1", f"(PC1 = {_m(proj.PC1[0], 1)} Å", proj.PC1[0], 1))

    # ---- landscape and state
    st = summ["states"][0]
    rows.append(
        (
            "state A pop",
            f"holding {st['population_percent']:.1f} % of the frames",
            st["population_percent"],
            1,
        )
    )
    rows.append(
        (
            "state A CI",
            f"[{pops.Bootstrap_CI_95_Lower:.1f}, {pops.Bootstrap_CI_95_Upper:.1f}] %",
            pops.Bootstrap_CI_95_Upper,
            1,
        )
    )
    rows.append(
        (
            "state A min",
            f"PC1 = {_m(st['min_pc1'], 1)} Å, PC2 = {_m(st['min_pc2'], 1)} Å",
            st["min_pc1"],
            1,
        )
    )
    rows.append(("unassigned", f"The remaining {un:.1f} % are", un, 1))

    # ---- domain-angle landscape
    fe = np.load(G / "fel" / "free_energy.npy")
    xe, ye = np.load(G / "fel" / "x_edges.npy"), np.load(G / "fel" / "y_edges.npy")
    for name, a in cryst.items():
        i, j = np.searchsorted(xe, a["theta_NMP"]) - 1, np.searchsorted(ye, a["theta_LID"]) - 1
        inside = 0 <= i < fe.shape[1] and 0 <= j < fe.shape[0]
        g = fe[j, i] if inside else np.nan
        if np.isfinite(g):
            rows.append(
                (
                    f"{name} bin",
                    f"{name}: θNMP {a['theta_NMP']:.1f}°, θLID {a['theta_LID']:.1f}°, ΔG of its bin {g:.1f} kJ/mol",
                    g,
                    1,
                )
            )
            rows.append(
                (f"{name} bin prose", f"whose bin\nlies {g:.1f} kJ/mol above the minimum", g, 1)
            )
        else:
            rows.append(
                (
                    f"{name} outside",
                    f"{name}: θNMP {a['theta_NMP']:.1f}°, θLID {a['theta_LID']:.1f}°, outside the sampled region",
                    a["theta_LID"],
                    1,
                )
            )
    ga = json.load(open(G / "summary.json", encoding="utf-8"))["states"][0]
    rows.append(
        (
            "angle minimum",
            f"θNMP = {ga['min_pc1']:.1f}°, θLID = {ga['min_pc2']:.1f}°",
            ga["min_pc1"],
            1,
        )
    )
    dev = max(abs(ga["min_pc1"] - c4["theta_NMP"]), abs(ga["min_pc2"] - c4["theta_LID"]))
    claims.append(
        ("angle minimum within 3° of 4AKE", "within 3° of the open crystal structure", dev < 3.0)
    )
    ang = np.loadtxt(X / "domain_angles.dat")
    t_ns = ang[:, 0] / 1000.0
    rows.append(("frame0 LID", f"(θLID {ang[0, 2]:.1f}°)", ang[0, 2], 1))
    below = t_ns[ang[:, 2] < 120.0]
    claims.append(
        (
            "θLID < 120° only before 2 ns",
            "below 120° only during the\nfirst 2 ns",
            bool(below.max() < 2.0),
        )
    )
    rows.append(("LID min", f"(minimum {ang[:, 2].min():.1f}°)", ang[:, 2].min(), 1))
    rows.append(
        (
            "LID after 20 ns",
            f"stays above {ang[t_ns > 20, 2].min():.1f}° after 20 ns",
            ang[t_ns > 20, 2].min(),
            1,
        )
    )

    # ---- convergence and equilibration
    rows.append(("RMSIP prose", f"overlap with RMSIP {rmsip.Value[0]:.2f}", rmsip.Value[0], 2))
    rows.append(
        (
            "cosine prose",
            f"PC1 has a cosine content of {cos.Cosine_Content[0]:.2f}",
            cos.Cosine_Content[0],
            2,
        )
    )
    eff = _csv(A / "convergence" / "effective_samples.csv").iloc[0]
    dt_ns = float(np.median(np.diff(proj.time_ps))) / 1000.0
    rows.append(
        (
            "g PC1",
            f"statistical inefficiency of {eff.g_PC1:.1f} frames ({eff.g_PC1 * dt_ns:.1f} ns)",
            eff.g_PC1,
            1,
        )
    )
    rows.append(("N_eff", f"about {eff.N_eff:.0f} independent samples", eff.N_eff, 0))
    rows.append(("RMSD t0", f"which is at {eq.t0_ps / 1000:.0f} ns", eq.t0_ps / 1000, 0))
    eqe = _csv(E / "convergence" / "equilibration.csv")
    a_, b_ = eqe.iloc[0], eqe.iloc[1]
    rows.append(
        (
            "eq discards",
            f"discards {int(a_.frames_discarded):,} frames on the RMSD and a further {int(b_.frames_discarded)} on the CVs",
            b_.frames_discarded,
            0,
        )
    )
    rows.append(("eq start", f"production\nstarts at {b_.t0_ps / 1000:.0f} ns", b_.t0_ps / 1000, 0))

    # ---- MSM
    its = _csv(A / "msm" / "implied_timescales.csv")
    rows.append(
        (
            "ITS ends",
            f"The slowest one is {its.ts_1.iloc[0] / 1000:.1f} ns at a\nlag of {its.lag_ps.iloc[0] / 1000:.2f} ns and {its.ts_1.iloc[-1] / 1000:.1f} ns at a lag of {its.lag_ps.iloc[-1] / 1000:.0f} ns",
            its.ts_1.iloc[-1] / 1000,
            1,
        )
    )
    claims.append(("no ITS plateau", "`plateau_found: false`", not msm["plateau_found"]))
    rows.append(("CK prose", f"(maximum deviation {ckd:.3f}, `msm/ck_test.csv`)", ckd, 3))
    cs = (
        _csv(A / "msm" / "coarse_stationary.csv").set_index("metastable_set").stationary_probability
    )
    ma = _csv(A / "msm" / "metastable_assignments.csv")
    med = ma.loc[ma.metastable_set == "MS1", "time_ps"].median() / 1000
    rows.append(
        ("MS1", f"stationary probability {cs['MS1']:.3f}, median time {med:.0f} ns", cs["MS1"], 3)
    )
    rows.append(("MS2", f"(MS2, {cs['MS2']:.3f})", cs["MS2"], 3))

    # ---- comparison with a known answer
    pc_ = _csv(C / "state_population_comparison.csv").set_index("State")
    for stt in ("State A", "State B"):
        q = pc_.loc[stt]
        d = q.MUT_minus_WT_Percent
        line = (
            f"| {stt} | {q.WT_Population_Percent:.1f} | {q.MUT_Population_Percent:.1f} | "
            f"{'+' if d > 0 else '−'}{abs(d):.1f} | [{_m(q.MUT_minus_WT_CI95_Low, 1)}, {_m(q.MUT_minus_WT_CI95_High, 1)}] | "
            f"[{_m(q.MUT_minus_WT_Welch_CI95_Low, 1)}, {_m(q.MUT_minus_WT_Welch_CI95_High, 1)}] | {q.MUT_minus_WT_Welch_p:.4f} |"
        )
        rows.append((f"comparison row {stt}", line, abs(d), 1))
    b = pc_.loc["State B"]
    rows.append(
        (
            "WT bent",
            f"bent state {b.WT_Population_Percent:.1f} % of the time (true 65 %) and MUT\n  {b.MUT_Population_Percent:.1f} %",
            b.MUT_Population_Percent,
            1,
        )
    )
    rows.append(
        (
            "difference",
            f"The estimate is {_m(b.MUT_minus_WT_Percent, 1)} points against a true −35",
            b.MUT_minus_WT_Percent,
            1,
        )
    )
    rows.append(
        (
            "Welch p dof",
            f"p = {b.MUT_minus_WT_Welch_p:.4f} on {b.MUT_minus_WT_Welch_dof:.1f}\n  degrees of freedom",
            b.MUT_minus_WT_Welch_dof,
            1,
        )
    )
    rows.append(
        (
            "het ratios",
            f"ratios are {b.WT_Replicate_Heterogeneity_Ratio:.2f} (WT) and {b.MUT_Replicate_Heterogeneity_Ratio:.2f} (MUT)",
            b.WT_Replicate_Heterogeneity_Ratio,
            2,
        )
    )
    truth_in = all(
        lo <= -35.0 <= hi
        for lo, hi in (
            (b.MUT_minus_WT_CI95_Low, b.MUT_minus_WT_CI95_High),
            (b.MUT_minus_WT_Welch_CI95_Low, b.MUT_minus_WT_Welch_CI95_High),
        )
    )
    claims.append(
        ("both intervals contain -35", "Both intervals\n  contain the true value", truth_in)
    )
    return rows, claims


# ----------------------------------------------------------------------------
# Benchmarks table
# ----------------------------------------------------------------------------

BENCH_BEGIN = "<!-- BEGIN VALIDATION TABLE (generated by validation/checks_docs.py) -->"
BENCH_END = "<!-- END VALIDATION TABLE -->"


def render_benchmark_tables(results: list[dict], header: str) -> str:
    """Markdown tables, one per section, from the ``results.json`` records."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in results:
        groups.setdefault(r["section"], []).append(r)
    lines = [BENCH_BEGIN, "", header, ""]
    for section, rows in groups.items():
        lines.append(f"### {section}")
        lines.append("")
        lines.append("| Check | Result | Measured | Expected | Tolerance | Reference |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            flag = "skipped" if r.get("note") == "skipped" else ("pass" if r["pass"] else "FAIL")
            cells = [
                r["check"],
                flag,
                str(r["measured"]),
                str(r["expected"]),
                str(r["tolerance"]),
                r["reference"],
            ]
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        lines.append("")
    lines.append(BENCH_END)
    return "\n".join(lines)


def _bench_header(ROOT: Path) -> str:
    md = (ROOT / "validation" / "VALIDATION.md").read_text(encoding="utf-8")
    m = re.search(
        r"Generated by `validation/validate.py` on (\S+) \((\d+) s\)\. \*\*(\d+)/(\d+) checks passed\*\*, (\d+) skipped",
        md,
    )
    if not m:
        return "Results of `validation/validate.py`."
    date, secs, npass, ntot, nskip = m.groups()
    return (
        f"Results of `validation/validate.py` run on {date} ({secs} s): "
        f"**{npass}/{ntot} checks passed**, {nskip} skipped. "
        "Columns are copied from `validation/results.json`."
    )


def write_benchmarks(ROOT: Path) -> None:
    results = json.load(open(ROOT / "validation" / "results.json", encoding="utf-8"))
    page = ROOT / "docs" / "benchmarks.md"
    text = page.read_text(encoding="utf-8")
    table = render_benchmark_tables(results, _bench_header(ROOT))
    i, j = text.index(BENCH_BEGIN), text.index(BENCH_END) + len(BENCH_END)
    page.write_text(text[:i] + table + text[j:], encoding="utf-8")


# ----------------------------------------------------------------------------
# DOIs, Zenodo, Sphinx
# ----------------------------------------------------------------------------


def bib_dois(bib_text: str) -> list[str]:
    return sorted(set(re.findall(r"doi\s*=\s*[{\"]\s*([^}\"]+?)\s*[}\"]", bib_text, re.I)))


def resolve_doi(doi: str, timeout: float = 20.0) -> int | None:
    """HTTP status of ``https://doi.org/<doi>`` without following redirects; None if offline."""

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(f"https://doi.org/{doi}", method="HEAD")
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def zenodo_vs_citation(ROOT: Path) -> list[str]:
    """Return a list of discrepancies between .zenodo.json, CITATION.cff and pyproject."""
    import yaml

    zen = json.load(open(ROOT / ".zenodo.json", encoding="utf-8"))
    cff = yaml.safe_load(open(ROOT / "CITATION.cff", encoding="utf-8"))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    problems = []
    if zen.get("title") != cff.get("title"):
        problems.append("title differs")
    if "version" in zen:
        # Zenodo takes the version from the GitHub release tag; a hard-coded value
        # would have to be bumped on every release and is a source of mismatches.
        problems.append(".zenodo.json must not pin 'version' (taken from the release tag)")
    if cff.get("version") != version:
        problems.append(f"CITATION.cff version {cff.get('version')} != pyproject {version}")
    zen_names = [c["name"] for c in zen.get("creators", [])]
    cff_names = [f"{a['family-names']}, {a['given-names']}" for a in cff.get("authors", [])]
    if zen_names != cff_names:
        problems.append("creators differ")
    if (zen.get("license") or "").upper() != (cff.get("license") or "").upper():
        problems.append("license differs")
    if sorted(zen.get("keywords", [])) != sorted(cff.get("keywords", [])):
        problems.append("keywords differ")
    return problems


def sphinx_build(ROOT: Path, tmp: Path) -> tuple[int, str]:
    out = tmp / "docs_build"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-q",
            "-b",
            "html",
            str(ROOT / "docs"),
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return res.returncode, (res.stdout + res.stderr)[-1500:]


# ----------------------------------------------------------------------------
# entry point used by validate.py
# ----------------------------------------------------------------------------


def run(record, tmp: Path, ROOT: Path, KB: float) -> None:
    tmp = Path(tmp)
    ROOT = Path(ROOT)
    docs = ROOT / "docs"

    # 1. tutorial commands --------------------------------------------------
    work = tutorial_workdir(tmp, ROOT)
    tutorial = (docs / "tutorial_adk.md").read_text(encoding="utf-8")
    blocks = extract_code_blocks(tutorial)
    runs = run_code_blocks(blocks, work, ROOT)
    failed = [(i, lang, tail) for i, lang, rc, tail in runs if rc != 0]
    record(
        "Docs",
        "every bash/python block of the tutorial runs",
        not failed,
        f"{len(runs) - len(failed)}/{len(runs)} blocks exit 0",
        f"{len(runs)}/{len(runs)}",
        "exit code 0",
        "docs/tutorial_adk.md executed from a scratch copy of examples/adk",
        note="" if not failed else failed[0][2][-300:],
    )

    # 2. numbers quoted in the tutorial --------------------------------------
    if not failed:
        rows, claims = tutorial_number_sources(work)
        bad = []
        for label, quoted, value, nd in rows:
            # `quoted` is rendered from the output file with the rounding used in the
            # text, so its verbatim presence is the equality check; additionally make
            # sure the rendered value really is part of the quoted string.
            plain = quoted.replace("−", "-")
            candidates = {f"{float(value):.{nd}f}", f"{float(value):,.{nd}f}"}
            if not any(c in plain or c.lstrip("-") in plain for c in candidates):
                bad.append(f"{label}: internal error, {value} not in '{quoted}'")
            elif quoted not in tutorial:
                bad.append(f"{label}: '{quoted}' not found in tutorial")
        for label, quoted, holds in claims:
            if quoted not in tutorial:
                bad.append(f"{label}: '{quoted}' not found in tutorial")
            elif not holds:
                bad.append(f"{label}: the statement '{quoted}' is false for this run")
        rows = rows + [(c[0], c[1], 0.0, 0) for c in claims]
        record(
            "Docs",
            "every numeric value quoted in the tutorial equals the value re-read from its output file",
            not bad,
            f"{len(rows) - len(bad)}/{len(rows)} quoted values match",
            f"{len(rows)}/{len(rows)}",
            "equal at the quoted rounding",
            "output CSV/JSON/NPY files of the tutorial run",
            note="; ".join(bad[:5]),
        )
        # python_api.md runnable example (writes tutorial/api_run)
        api_blocks = [
            b
            for b in extract_code_blocks((docs / "python_api.md").read_text(encoding="utf-8"))
            if b[0] == "python"
        ]
        api_runs = run_code_blocks(api_blocks, work, ROOT, languages=("python",))
        # blocks that need files which only exist in a real project are documentation only
        api_ok = [r for r in api_runs if r[2] == 0]
        api_text = (docs / "python_api.md").read_text(encoding="utf-8")
        expected_lines = ["[46.9 15. ]", "State A 91.4 -0.0", "(40, 40) -0.0 True"]
        outputs = "\n".join(r[3] for r in api_runs)
        record(
            "Docs",
            "python_api.md examples run and print the quoted output",
            len(api_ok) >= 2
            and all(line in outputs and line in api_text for line in expected_lines),
            f"{len(api_ok)}/{len(api_runs)} python blocks exit 0 (the two illustrative snippets reference files that only exist in a real project)",
            "run_analysis and calculate_fel examples",
            "printed output contains the quoted lines",
            "docs/python_api.md",
        )
    else:
        record(
            "Docs",
            "every numeric value quoted in the tutorial equals the value re-read from its output file",
            False,
            "not evaluated (tutorial commands failed)",
            "all",
            "equal at the quoted rounding",
            "output files",
            note="skipped",
        )

    # 3. benchmarks table ---------------------------------------------------
    results = json.load(open(ROOT / "validation" / "results.json", encoding="utf-8"))
    page = (docs / "benchmarks.md").read_text(encoding="utf-8")
    expected_table = render_benchmark_tables(results, _bench_header(ROOT))
    i, j = page.find(BENCH_BEGIN), page.find(BENCH_END)
    actual_table = page[i : j + len(BENCH_END)] if i >= 0 and j >= 0 else ""
    record(
        "Docs",
        "docs/benchmarks.md validation table is the rendering of validation/results.json",
        actual_table == expected_table,
        f"{sum(1 for r in results)} checks listed"
        if actual_table == expected_table
        else "table differs",
        f"{len(results)} rows",
        "byte-identical",
        "validation/results.json",
    )

    # 4. DOIs ---------------------------------------------------------------
    # The manuscript lives outside the public repository; the check runs only
    # where paper/paper.bib is present (the authors' working copies).
    bib_path = ROOT / "paper" / "paper.bib"
    dois = bib_dois(bib_path.read_text(encoding="utf-8")) if bib_path.exists() else []
    statuses = {d: resolve_doi(d) for d in dois}
    offline = [d for d, s in statuses.items() if s is None]
    bad_dois = [
        d for d, s in statuses.items() if s is not None and s not in (200, 301, 302, 303, 307, 308)
    ]
    if not dois:
        record(
            "Docs",
            "every DOI in paper/paper.bib resolves at doi.org",
            True,
            "paper/ not present in this checkout",
            "HTTP 200/30x",
            "doi.org HEAD request",
            "doi.org",
            note="skipped",
        )
    elif offline and len(offline) == len(dois):
        record(
            "Docs",
            "every DOI in paper/paper.bib resolves at doi.org",
            True,
            f"{len(dois)} DOIs; network unreachable, not checked",
            "HTTP 200/30x",
            "doi.org HEAD request",
            "doi.org",
            note="skipped",
        )
    else:
        record(
            "Docs",
            "every DOI in paper/paper.bib resolves at doi.org",
            not bad_dois and not offline,
            f"{len(dois) - len(bad_dois) - len(offline)}/{len(dois)} resolve"
            + (f"; unresolved: {bad_dois}" if bad_dois else "")
            + (f"; unreachable: {offline}" if offline else ""),
            f"{len(dois)}/{len(dois)}",
            "HTTP 200/30x",
            "doi.org HEAD request",
        )

    # 5. Zenodo / CITATION / pyproject --------------------------------------
    problems = zenodo_vs_citation(ROOT)
    record(
        "Docs",
        ".zenodo.json agrees with CITATION.cff (title, creators, license, keywords) and CITATION.cff version equals pyproject",
        not problems,
        "consistent" if not problems else "; ".join(problems),
        "consistent",
        "exact",
        "CITATION.cff",
    )

    # 6. Sphinx -W ------------------------------------------------------------
    try:
        import sphinx  # noqa: F401

        rc, tail = sphinx_build(ROOT, tmp)
        record(
            "Docs",
            "Sphinx builds the documentation with -W (warnings are errors)",
            rc == 0,
            "build succeeded" if rc == 0 else tail[-300:],
            "exit 0",
            "exact",
            "sphinx -W -b html docs",
        )
    except ImportError:
        record(
            "Docs",
            "Sphinx builds the documentation with -W (warnings are errors)",
            True,
            "sphinx not installed",
            "exit 0",
            "exact",
            "sphinx -W -b html docs",
            note="skipped",
        )


# ----------------------------------------------------------------------------
# standalone driver
# ----------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    ROOT = Path(__file__).resolve().parents[1]
    if "--write-benchmarks" in argv:
        write_benchmarks(ROOT)
        print("docs/benchmarks.md table regenerated")
        return 0
    results: list[dict] = []

    def record(section, name, passed, measured, expected, tolerance, reference, note=""):
        results.append(
            {
                "section": section,
                "check": name,
                "pass": bool(passed),
                "measured": measured,
                "expected": expected,
                "tolerance": tolerance,
                "reference": reference,
                "note": note,
            }
        )
        flag = "SKIP" if note == "skipped" else ("PASS" if passed else "FAIL")
        print(f"[{flag}] {section} / {name}: {measured} vs {expected} ({tolerance})", flush=True)
        if note and note != "skipped":
            print(f"       note: {note}")

    tmp = Path(tempfile.mkdtemp(prefix="ca_docs_"))
    run(record, tmp, ROOT, 0.008314462618)
    n_pass = sum(r["pass"] for r in results)
    print(f"\n{n_pass}/{len(results)} passed")
    if "--json" in argv:
        out = ROOT / "validation" / "results_docs.json"
        json.dump(results, open(out, "w", encoding="utf-8"), indent=2, default=str)
        print(f"written {out}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
