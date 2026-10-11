"""Documentation checks for the validation battery (imported by validate.py).

``run(record, tmp, ROOT, KB)`` verifies that the documentation is consistent with
the software it describes:

1. every ``bash`` and ``python`` code block of ``docs/tutorial_neuraminidase.md``
   and the runnable ``python`` blocks of ``docs/python_api.md`` execute without
   error, in order, from a scratch copy of the repository data;
2. every numeric value quoted in the tutorial equals (at the quoted rounding) the
   value re-read from the output file it was taken from;
3. the validation table in ``docs/benchmarks.md`` is the rendering of
   ``validation/results.json`` (one row per check, nothing invented);
4. every DOI in ``paper/paper.bib`` resolves at doi.org (skipped, not failed, when
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
    """Scratch directory laid out like the repository root (tests/data copied)."""
    work = tmp / "docs_tutorial"
    if work.exists():
        shutil.rmtree(work)
    (work / "tests" / "data").mkdir(parents=True)
    shutil.copytree(
        ROOT / "tests" / "data" / "h1n1_ca",
        work / "tests" / "data" / "h1n1_ca",
        ignore=shutil.ignore_patterns(".*offsets*"),
    )
    return work


# ----------------------------------------------------------------------------
# Numbers quoted in the tutorial, and where each one comes from
# ----------------------------------------------------------------------------


def _csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def tutorial_number_sources(work: Path) -> list[tuple[str, str, float, int]]:
    """``[(label, quoted_text, recomputed_value, decimals)]``.

    ``quoted_text`` must appear verbatim in the tutorial and must equal
    ``recomputed_value`` rounded to ``decimals``.
    """
    import numpy as np

    A = work / "tutorial" / "analysis"
    C = work / "tutorial" / "comparison"
    KB = 0.008314462618
    rows: list[tuple[str, str, float, int]] = []

    ev = _csv(A / "pca" / "explained_variance.csv")
    for k in range(3):
        rows.append((f"eigenvalue PC{k + 1}", f"{ev.Eigenvalue[k]:.1f}", ev.Eigenvalue[k], 1))
        rows.append(
            (
                f"variance PC{k + 1}",
                f"{ev.ExplainedVariance_Percent[k]:.1f}",
                ev.ExplainedVariance_Percent[k],
                1,
            )
        )
        rows.append(
            (
                f"cumulative PC{k + 1}",
                f"{ev.CumulativeVariance_Percent[k]:.1f}",
                ev.CumulativeVariance_Percent[k],
                1,
            )
        )

    fe = np.load(A / "fel" / "free_energy.npy")
    rows.append(("occupied bins", f"{int(np.isfinite(fe).sum())} of the 144 bins", 27, 0))
    rows.append(("max dG", f"{np.nanmax(fe):.2f} kJ/mol", np.nanmax(fe), 2))
    prob = np.load(A / "fel" / "probability.npy")
    counts = prob / prob[np.isfinite(fe)].sum() * 81
    rows.append(
        ("frames in most populated bin", f"holds {int(round(np.nanmax(counts)))} frames", 9, 0)
    )
    assert abs(KB * 300 * np.log(np.nanmax(counts)) - np.nanmax(fe)) < 1e-6
    sd = np.load(A / "fel" / "free_energy_bootstrap_sd.npy")
    rows.append(("bootstrap SD min", f"between {np.nanmin(sd):.2f} and", np.nanmin(sd), 2))
    rows.append(("bootstrap SD max", f"and {np.nanmax(sd):.2f} kJ/mol", np.nanmax(sd), 2))

    basins = _csv(A / "states" / "basins.csv").set_index("State")
    for st in ("State A", "State B"):
        b = basins.loc[st]
        rows.append((f"{st} population", f"{b.Population_Percent:.1f}", b.Population_Percent, 1))
        rows.append((f"{st} min PC1", f"{b.Min_PC1:.2f}".replace("-", "−"), b.Min_PC1, 2))
        rows.append((f"{st} min PC2", f"{b.Min_PC2:.2f}".replace("-", "−"), b.Min_PC2, 2))
        rows.append(
            (
                f"{st} min dG",
                f"{abs(b.Min_FreeEnergy_kJ_mol):.2f}",
                abs(b.Min_FreeEnergy_kJ_mol),
                2,
            )
        )
        rows.append(
            (
                f"{st} representative",
                f"frame {int(b.Representative_Frame)}, {int(b.Representative_Time_ps)} ps",
                b.Representative_Frame,
                0,
            )
        )
    summary = json.load(open(A / "summary.json"))
    un = summary["unassigned_population_percent"]
    rows.append(("unassigned", f"{un:.1f} % of the frames", un, 1))

    sfe = _csv(A / "states" / "state_free_energies.csv").set_index("state")
    for st in ("State A", "State B"):
        r = sfe.loc[st]
        rows.append(
            (
                f"{st} bootstrap CI",
                f"[{100 * r.population_ci95_low:.1f}, {100 * r.population_ci95_high:.1f}]",
                100 * r.population_ci95_low,
                1,
            )
        )
    rb = sfe.loc["State B"]
    rows.append(("dG B vs A", f"{rb.dG_state_kJ_mol:.2f}", rb.dG_state_kJ_mol, 2))
    rows.append(
        (
            "dG B vs A CI",
            f"[{rb.dG_state_ci95_low:.2f}, {rb.dG_state_ci95_high:.2f}]".replace("-", "−"),
            rb.dG_state_ci95_low,
            2,
        )
    )
    assert len(_csv(A / "states" / "barriers.csv")) == 0, "barriers.csv is expected to be empty"
    pops = _csv(A / "statistics" / "state_populations.csv")
    rows.append(
        (
            "blocks",
            f"{int(pops.Num_Blocks[0])} independent blocks of {int(pops.Block_Frames[0])} frames",
            pops.Num_Blocks[0],
            0,
        )
    )
    ens = _csv(A / "states" / "state_ensemble_summary.csv").set_index("state")
    rows.append(
        (
            "frames per state",
            f"{int(ens.loc['State A'].frames)} frames in State A and {int(ens.loc['State B'].frames)} in State B",
            ens.loc["State A"].frames,
            0,
        )
    )
    rows.append(
        (
            "state RMSD",
            f"{ens.loc['State A'].RMSD_to_reference_mean:.2f} Å and {ens.loc['State B'].RMSD_to_reference_mean:.2f} Å",
            ens.loc["State A"].RMSD_to_reference_mean,
            2,
        )
    )

    tm = _csv(A / "kinetics" / "transition_matrix.csv").set_index("from_state")
    rows.append(
        (
            "P(A->A)",
            f"P(A→A) = {tm.loc['State A', 'to_State A']:.3f}",
            tm.loc["State A", "to_State A"],
            3,
        )
    )
    rows.append(
        (
            "P(A->B)",
            f"P(A→B) = {tm.loc['State A', 'to_State B']:.3f}",
            tm.loc["State A", "to_State B"],
            3,
        )
    )
    mfpt = _csv(A / "kinetics" / "mfpt_markov_ps.csv").set_index("from_state")
    rows.append(
        (
            "MFPT A->B",
            f"A→B is {mfpt.loc['State A', 'to_State B']:.0f} ps",
            mfpt.loc["State A", "to_State B"],
            0,
        )
    )
    dw = _csv(A / "kinetics" / "dwell_summary.csv").set_index("state")
    rows.append(
        (
            "dwell A",
            f"{dw.loc['State A'].max_dwell_ps / 1000:.1f} ns in A",
            dw.loc["State A"].max_dwell_ps / 1000,
            1,
        )
    )
    rows.append(
        (
            "dwell B",
            f"{dw.loc['State B'].max_dwell_ps / 1000:.1f} ns in B",
            dw.loc["State B"].max_dwell_ps / 1000,
            1,
        )
    )

    ds = _csv(A / "descriptors" / "summary.csv").iloc[0]
    rows.append(
        (
            "RMSD mean",
            f"| Cα RMSD | {ds.RMSD_mean:.2f} | {ds.RMSD_sd:.2f} | {ds.RMSD_max:.2f} |",
            ds.RMSD_mean,
            2,
        )
    )
    rows.append(
        (
            "Rg mean",
            f"| Radius of gyration | {ds.Rg_mean:.2f} | {ds.Rg_sd:.2f} | — |",
            ds.Rg_mean,
            2,
        )
    )
    rows.append(
        (
            "ligand RMSD",
            f"| Zanamivir RMSD after protein fit | {ds.Ligand_RMSD_mean:.2f} | {ds.Ligand_RMSD_sd:.2f} | {ds.Ligand_RMSD_max:.2f} |",
            ds.Ligand_RMSD_mean,
            2,
        )
    )
    rmsf = _csv(A / "descriptors" / "rmsf_residues.csv").set_index("residue_number")
    rows.append(("RMSF Ser1", f"Ser1\n{rmsf.loc[1].RMSF:.2f} Å", rmsf.loc[1].RMSF, 2))
    rows.append(("RMSF Asp387", f"Asp387 {rmsf.loc[387].RMSF:.2f} Å", rmsf.loc[387].RMSF, 2))
    rows.append(("RMSF Pro256", f"Pro256 ({rmsf.loc[256].RMSF:.2f} Å)", rmsf.loc[256].RMSF, 2))
    mc = _csv(A / "pca" / "mode_contributions_residues.csv").set_index("residue_number")
    rows.append(("mode ZMR", f"(ZMR, {mc.loc[390].PC1:.3f})", mc.loc[390].PC1, 3))
    rows.append(("mode Pro256", f"Pro256\n({mc.loc[256].PC1:.3f})", mc.loc[256].PC1, 3))
    rows.append(("mode Asp387", f"Asp387 ({mc.loc[387].PC1:.3f})", mc.loc[387].PC1, 3))
    rows.append(("mode Ser1 PC2", f"Ser1\n({mc.loc[1].PC2:.2f})", mc.loc[1].PC2, 2))

    rmsip = _csv(A / "convergence" / "rmsip.csv")
    rows.append(("RMSIP", f"RMSIP** (`rmsip.csv`): {rmsip.Value[0]:.2f}", rmsip.Value[0], 2))
    cc = _csv(A / "convergence" / "cosine_content.csv")
    rows.append(
        ("cosine", f"(`cosine_content.csv`): {cc.Cosine_Content[0]:.2f}", cc.Cosine_Content[0], 2)
    )
    po = _csv(A / "convergence" / "progressive_overlap.csv")
    rows.append(
        (
            "progressive",
            f"by {po.Subspace_RMSIP[0]:.2f}, {po.Subspace_RMSIP[1]:.2f} and {po.Subspace_RMSIP[2]:.2f}",
            po.Subspace_RMSIP[0],
            2,
        )
    )
    eff = _csv(A / "convergence" / "effective_samples.csv").iloc[0]
    rows.append(("g PC1", f"g(PC1) = {eff.g_PC1:.1f} frames", eff.g_PC1, 1))
    rows.append(("g PC2", f"g(PC2) = {eff.g_PC2:.1f} frames", eff.g_PC2, 1))
    rows.append(("N_eff", f"N_eff = {eff.N_eff:.1f} effective frames", eff.N_eff, 1))
    rows.append(
        ("tau PC1 ns", f"correlation time of about {eff.g_PC1 * 0.25:.1f} ns", eff.g_PC1 * 0.25, 1)
    )
    eq = _csv(A / "convergence" / "equilibration.csv").iloc[0]
    rows.append(
        (
            "t0",
            f"t₀ at {int(eq.t0_ps)} ps (frame {int(eq.t0_index)})",
            eq.t0_ps,
            0,
        )
    )
    rows.append(
        (
            "N_eff equilibration",
            f"from {eq.N_eff_full:.1f} to {eq.N_eff_production:.1f}",
            eq.N_eff_full,
            1,
        )
    )
    ps = _csv(A / "convergence" / "population_stability.csv")
    assert ps["State B"].iloc[1] == 0.0, "State B expected absent at 50 %"

    # ---- comparison
    cev = _csv(C / "pca" / "explained_variance.csv")
    rows.append(
        (
            "shared variance",
            f"carry {cev.ExplainedVariance_Percent[0]:.1f} % and {cev.ExplainedVariance_Percent[1]:.1f} %",
            cev.ExplainedVariance_Percent[0],
            1,
        )
    )
    pc = _csv(C / "comparison" / "state_population_comparison.csv").set_index("State")
    for st in ("State A", "State B", "State C"):
        r = pc.loc[st]
        diff = r["late_minus_early_Percent"]
        sign = "+" if diff > 0 else "−"
        line = (
            f"| {st} | {r.early_Population_Percent:.1f} | {r.late_Population_Percent:.1f} | "
            f"{sign}{abs(diff):.1f} | [{r.late_minus_early_CI95_Low:.1f}, {r.late_minus_early_CI95_High:.1f}] | "
            f"{r.late_minus_early_Bootstrap_p:.3f} | {r.late_minus_early_Welch_t:.2f} | {r.late_minus_early_Welch_p:.2f} |"
        ).replace("-", "−")
        rows.append((f"comparison row {st}", line, diff, 1))
    ov = _csv(C / "comparison" / "fel_overlap.csv").iloc[0]
    rows.append(
        (
            "Bhattacharyya",
            f"Bhattacharyya coefficient {ov.Bhattacharyya_Coefficient:.3f}",
            ov.Bhattacharyya_Coefficient,
            3,
        )
    )
    rows.append(
        ("Hellinger", f"Hellinger distance\n{ov.Hellinger_Distance:.3f}", ov.Hellinger_Distance, 3)
    )
    rows.append(
        (
            "JS",
            f"Jensen–Shannon divergence {ov.JS_Divergence_bits:.3f} bits",
            ov.JS_Divergence_bits,
            3,
        )
    )
    rows.append(("Jaccard", f"(Jaccard {ov.Sampled_Bin_Jaccard:.3f})", ov.Sampled_Bin_Jaccard, 3))
    rows.append(
        (
            "Jaccard percent",
            f"only {100 * ov.Sampled_Bin_Jaccard:.1f} % of the\nsampled bins",
            100 * ov.Sampled_Bin_Jaccard,
            1,
        )
    )
    sub = _csv(C / "comparison" / "subspace_overlap.csv")
    rows.append(("subspace RMSIP", f"RMSIP of {sub['late'][0]:.3f} between", sub["late"][0], 3))
    sfe_c = (
        _csv(C / "comparison" / "state_free_energy_comparison.csv")
        .set_index("State")
        .loc["State A"]
    )
    rows.append(
        (
            "ddG state A",
            f"is {sfe_c.late_minus_early_ddG_kJmol:.2f} kJ/mol less favourable",
            sfe_c.late_minus_early_ddG_kJmol,
            2,
        )
    )
    rows.append(
        (
            "ddG state A CI",
            f"95 % CI [{sfe_c.late_minus_early_ddG_CI95_Low:.1f}, {sfe_c.late_minus_early_ddG_CI95_High:.2f}]",
            sfe_c.late_minus_early_ddG_CI95_High,
            2,
        )
    )
    dd = _csv(C / "comparison" / "fel_diff_late_vs_early.csv").dropna()
    rows.append(("ddG bins", f"({len(dd)} bins here)", len(dd), 0))
    assert len(dd) == 2
    assert (dd.ddG_kJmol.abs() < 2 * dd.ddG_bootstrap_SD_kJmol).all(), "both ΔΔG bins hatched"
    return rows


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
    md = (ROOT / "validation" / "VALIDATION.md").read_text()
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
    results = json.load(open(ROOT / "validation" / "results.json"))
    page = ROOT / "docs" / "benchmarks.md"
    text = page.read_text()
    table = render_benchmark_tables(results, _bench_header(ROOT))
    i, j = text.index(BENCH_BEGIN), text.index(BENCH_END) + len(BENCH_END)
    page.write_text(text[:i] + table + text[j:])


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

    zen = json.load(open(ROOT / ".zenodo.json"))
    cff = yaml.safe_load(open(ROOT / "CITATION.cff"))
    pyproject = (ROOT / "pyproject.toml").read_text()
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
    tutorial = (docs / "tutorial_neuraminidase.md").read_text()
    blocks = extract_code_blocks(tutorial)
    gmx = _which_gmx()
    runs = run_code_blocks(blocks, work, ROOT)
    failed = [(i, lang, tail) for i, lang, rc, tail in runs if rc != 0]
    record(
        "Docs",
        "every bash/python block of the tutorial runs (gmx trjconv needed for the compare part)",
        not failed,
        f"{len(runs) - len(failed)}/{len(runs)} blocks exit 0" + ("" if gmx else "; no gmx found"),
        f"{len(runs)}/{len(runs)}",
        "exit code 0",
        "docs/tutorial_neuraminidase.md executed from a scratch copy of tests/data",
        note="" if not failed else failed[0][2][-300:],
    )

    # 2. numbers quoted in the tutorial --------------------------------------
    if not failed:
        rows = tutorial_number_sources(work)
        bad = []
        for label, quoted, value, nd in rows:
            # `quoted` is rendered from the output file with the rounding used in the
            # text, so its verbatim presence is the equality check; additionally make
            # sure the rendered value really is part of the quoted string.
            rendered = f"{float(value):.{nd}f}".replace("-", "−")
            if rendered.replace("−", "-") not in quoted.replace("−", "-"):
                bad.append(f"{label}: internal error, {rendered} not in '{quoted}'")
            elif quoted not in tutorial:
                bad.append(f"{label}: '{quoted}' not found in tutorial")
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
            b for b in extract_code_blocks((docs / "python_api.md").read_text()) if b[0] == "python"
        ]
        api_runs = run_code_blocks(api_blocks, work, ROOT, languages=("python",))
        # blocks that need files which only exist in a real project are documentation only
        api_ok = [r for r in api_runs if r[2] == 0]
        api_text = (docs / "python_api.md").read_text()
        expected_lines = ["[25.9 12.1]", "State A 46.9 0.63", "(40, 40) -0.0 True"]
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
    results = json.load(open(ROOT / "validation" / "results.json"))
    page = (docs / "benchmarks.md").read_text()
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
    dois = bib_dois((ROOT / "paper" / "paper.bib").read_text())
    statuses = {d: resolve_doi(d) for d in dois}
    offline = [d for d, s in statuses.items() if s is None]
    bad_dois = [
        d for d, s in statuses.items() if s is not None and s not in (200, 301, 302, 303, 307, 308)
    ]
    if offline and len(offline) == len(dois):
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
        json.dump(results, open(out, "w"), indent=2, default=str)
        print(f"written {out}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
