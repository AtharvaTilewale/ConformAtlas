"""Validation checks for the report / figure-control track.

``run(record, tmp, ROOT, KB)`` is imported by validation/validate.py. The checks
run the full ``analyze`` pipeline once on the demo dataset (WT, two replicates)
with ``--fig-format all --path-overlay --colormap viridis`` and then verify, against
the arrays and tables the pipeline wrote and against analytical identities:

* every figure exists in png, svg and pdf; the SVG is vector (``<path>`` elements,
  no embedded ``<image>``); PNG dpi metadata equals the requested dpi; every
  contour fill colour belongs to the requested colormap; doubling ``font_scale``
  doubles the SVG text size;
* the interactive HTML embeds exactly the ``fel/free_energy.npy`` grid (same number
  of bins, identical ΔG values), the hover probabilities equal
  ``fel/probability.npy``, the per-bin state labels agree with
  ``states/basin_map.npy`` and the slider end positions equal the last frame of
  every replicate in ``states/frame_assignments.csv``;
* the 1-D profiles are the marginals of the 2-D histogram (identity with a direct
  ``numpy.histogram`` of CV1/CV2 on the saved edges);
* the report references no external http(s) resource;
* the viewer embeds n_states + 1 structures whose B-factor column equals the
  per-residue RMSF tables (to the %6.2f PDB precision);
* every number quoted in the interpretive sentences equals the corresponding CSV
  value, and the quoted binomial standard error / population-ratio ΔG match their
  analytical formulas;
* the PDF variant contains no script and the static-figure note.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

SECTION = "Report & figures"


def _num_in(text: str, value: float, nd: int) -> bool:
    return f"{value:.{nd}f}" in text


def run(record, tmp: Path, ROOT: Path, KB: float) -> None:
    import matplotlib

    from conformatlas.models import AnalysisConfig
    from conformatlas.pipeline import run_analysis
    from conformatlas.plotting import figure_style, plot_fel_2d, reset_figure_style
    from conformatlas.plotting_interactive import (
        decode_plotly_array,
        extract_plotly_data,
        marginal_profiles,
        plotly_available,
    )
    from conformatlas.report import (
        build_report_context,
        external_resource_urls,
        render_report_html,
        set_report_options,
        weasyprint_available,
    )

    demo = ROOT / "examples" / "demo_dataset" / "WT"
    out = tmp / "report_track"
    cmap = "viridis"
    T = 300.0

    reset_figure_style()
    set_report_options(interactive=True, plotly_js="inline", path_overlay=True)
    cfg = AnalysisConfig(
        topology=demo / "topology.pdb",
        trajectories=[demo / "rep1.xtc", demo / "rep2.xtc"],
        temperature=T,
        output_dir=out,
        n_bootstrap=50,
        seed=0,
        pdf=weasyprint_available(),
    )
    with figure_style(fig_format="all", colormap=cmap, dpi=120, path_overlay=True):
        results = run_analysis(cfg)
    reset_figure_style()
    set_report_options(path_overlay=False)

    # ------------------------------------------------------------ figures
    figs = out / "figures"
    pngs = sorted(p.stem for p in figs.glob("*.png"))
    svgs = sorted(p.stem for p in figs.glob("*.svg"))
    pdfs = sorted(p.stem for p in figs.glob("*.pdf"))
    record(
        SECTION,
        "every figure written in png, svg and pdf",
        pngs == svgs == pdfs and len(pngs) >= 20 and "fel_2d_path" in pngs,
        f"{len(pngs)} png / {len(svgs)} svg / {len(pdfs)} pdf",
        "identical sets, incl. fel_2d_path",
        "exact",
        "file system",
    )
    svg = (figs / "fel_2d.svg").read_text()
    n_path = svg.count("<path")
    record(
        SECTION,
        "SVG landscape is vector",
        n_path > 50 and "<image" not in svg,
        f"{n_path} <path>, {svg.count('<image')} <image>",
        "> 50 paths, 0 images",
        "exact",
        "SVG markup",
    )
    from PIL import Image

    with Image.open(figs / "fel_2d.png") as im:
        dpi = float(im.info["dpi"][0])
    record(
        SECTION,
        "PNG dpi metadata equals --fig-dpi",
        abs(dpi - 120) <= 1,
        dpi,
        120,
        "±1",
        "PNG pHYs chunk",
    )

    palette = {matplotlib.colors.to_hex(matplotlib.colormaps[cmap](i / 255.0)) for i in range(256)}
    fills = {f.lower() for f in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)} | {
        f.lower() for f in re.findall(r"fill:\s*(#[0-9a-fA-F]{6})", svg)
    }
    fills -= {"#ffffff", "#000000", "#f2f2f2", "#333333", "#222222"}
    bad = fills - palette
    record(
        SECTION,
        f"contour fill colours belong to colormap '{cmap}'",
        len(fills) > 5 and not bad,
        f"{len(fills)} fill colours, {len(bad)} outside palette",
        "0 outside palette",
        "exact (8-bit hex)",
        f"matplotlib.colormaps['{cmap}'] sampled at 256 levels",
    )

    sizes = {}
    for scale in (1.0, 2.0):
        with (
            matplotlib.rc_context({"svg.fonttype": "none"}),
            figure_style(fig_format="svg", font_scale=scale),
        ):
            p = plot_fel_2d(results.fel, out_path=tmp / f"fs{scale}.png", save_unannotated=False)
        txt = p.read_text()
        sizes[scale] = max(float(v) for v in re.findall(r"font(?:-size)?:[^;\"]*?([0-9.]+)px", txt))
    ratio = sizes[2.0] / sizes[1.0]
    record(
        SECTION,
        "font_scale 2 doubles SVG text size",
        abs(ratio - 2) < 0.04,
        round(ratio, 3),
        2.0,
        "±2 %",
        "SVG font-size attributes",
    )

    # ------------------------------------------------------------ interactive HTML
    html = (out / "report.html").read_text()
    fe = np.load(out / "fel" / "free_energy.npy")
    prob = np.load(out / "fel" / "probability.npy")
    basin = np.load(out / "states" / "basin_map.npy")
    fa = pd.read_csv(out / "states" / "frame_assignments.csv")
    if plotly_available():
        data = extract_plotly_data(html)
        hm = data["interactive-landscape"][0]
        z = decode_plotly_array(hm["z"])
        same_shape = z.shape == fe.shape
        dz = (
            float(np.nanmax(np.abs(np.nan_to_num(z, nan=0) - np.nan_to_num(fe, nan=0))))
            if same_shape
            else np.inf
        )
        nan_same = same_shape and bool(np.array_equal(np.isnan(z), np.isnan(fe)))
        record(
            SECTION,
            "interactive heatmap has the FEL grid size (bins)",
            same_shape,
            f"{z.shape}",
            f"{fe.shape}",
            "exact",
            "fel/free_energy.npy",
        )
        record(
            SECTION,
            "hover ΔG values equal fel/free_energy.npy",
            same_shape and dz == 0.0 and nan_same,
            f"max |Δ| = {dz:.3g}, NaN pattern identical = {nan_same}",
            "0",
            "exact",
            "fel/free_energy.npy",
        )
        cd = np.asarray(hm["customdata"], dtype=object)
        p_hover = cd[:, :, 0].astype(float)
        dp = float(np.nanmax(np.abs(p_hover - prob)))
        record(
            SECTION,
            "hover probabilities equal fel/probability.npy",
            dp <= 1e-12,
            f"max |Δ| = {dp:.3g}",
            "0",
            "1e-12",
            "fel/probability.npy",
        )
        labels = cd[:, :, 1].astype(str)
        ok_labels = True
        detail = []
        for st in results.states:
            expect = int(((basin == st.basin_id) & np.isfinite(fe)).sum())
            got = int((labels == st.label).sum())
            detail.append(f"{st.label}: {got}/{expect}")
            ok_labels &= got == expect
        ok_labels &= int((labels == "unsampled").sum()) == int((~np.isfinite(fe)).sum())
        record(
            SECTION,
            "hover state labels follow states/basin_map.npy",
            ok_labels,
            "; ".join(detail),
            "bin counts equal",
            "exact",
            "states/basin_map.npy",
        )
        # slider end positions
        steps = None
        m = re.search(r"\"sliders\":\s*\[", html)
        slider_ok = None
        if m is not None:
            import json

            dec = json.JSONDecoder()
            sliders, _ = dec.raw_decode(html, m.end() - 1)
            steps = sliders[0]["steps"]
            last = steps[-1]["args"][0]
            reps = list(dict.fromkeys(fa["replicate"]))
            slider_ok = True
            for i, rep in enumerate(reps):
                row = fa[fa["replicate"] == rep].sort_values("time_ps").iloc[-1]
                slider_ok &= math.isclose(last["x"][i][0], row["CV1"], rel_tol=1e-9, abs_tol=1e-9)
                slider_ok &= math.isclose(last["y"][i][0], row["CV2"], rel_tol=1e-9, abs_tol=1e-9)
        record(
            SECTION,
            "time-slider end positions equal the last frame of each replicate",
            bool(slider_ok),
            f"{len(steps) if steps else 0} steps, end positions match = {slider_ok}",
            "match",
            "1e-9",
            "states/frame_assignments.csv",
        )
        # 1-D profiles = marginals = direct histogram
        prof = marginal_profiles(results.fel)
        xe, ye = np.load(out / "fel" / "x_edges.npy"), np.load(out / "fel" / "y_edges.npy")
        hx, _ = np.histogram(fa["CV1"], bins=xe)
        hy, _ = np.histogram(fa["CV2"], bins=ye)
        err = max(
            float(np.max(np.abs(prof["p_x"] - hx / hx.sum()))),
            float(np.max(np.abs(prof["p_y"] - hy / hy.sum()))),
        )
        prof_html = decode_plotly_array(data["interactive-profiles"][0]["y"])
        same_prof = np.array_equal(
            np.nan_to_num(prof_html, nan=-1), np.nan_to_num(prof["g_x"], nan=-1)
        )
        record(
            SECTION,
            "1-D profiles are the marginals of the 2-D histogram (= numpy.histogram on saved edges)",
            err <= 1e-12 and same_prof,
            f"max |ΔP| = {err:.2g}; HTML profile identical = {same_prof}",
            "0",
            "1e-12",
            "numpy.histogram(CV, bins=fel/x_edges.npy); equal-weight frames",
        )
    else:  # pragma: no cover
        record(
            SECTION,
            "interactive checks",
            False,
            "plotly missing",
            "plotly installed",
            "",
            "",
            note="skipped",
        )

    urls = external_resource_urls(html)
    n_src = len(re.findall(r"<(?:script|link|img)[^>]+(?:src|href)=\"https?://", html))
    record(
        SECTION,
        "report has no external http(s) resource references",
        not urls and n_src == 0,
        f"{len(urls)} resource URLs, {n_src} http src/href attributes",
        "0",
        "exact",
        "regex over src/href/url() attributes (script bodies excluded)",
    )

    # ------------------------------------------------------------ structure viewer
    ctx = build_report_context(results, {p.stem: p for p in figs.glob("*.png")}, extra_sections=[])
    structs = ctx.get("viewer_structures", [])
    n_expect = len(results.states) + (1 if (out / "pca" / "average.pdb").exists() else 0)
    record(
        SECTION,
        "viewer embeds n_states + 1 structures with inlined PDB text",
        len(structs) == n_expect and all("ATOM" in s["pdb"] for s in structs),
        len(structs),
        n_expect,
        "exact",
        "states/*/representative.pdb, pca/average.pdb",
    )
    worst = 0.0
    for st, entry in zip(results.states, structs):
        rmsf = pd.read_csv(out / "states" / st.label.replace(" ", "_") / "rmsf_residues.csv")
        table = {int(r.residue_number): float(r.RMSF) for r in rmsf.itertuples()}
        for line in entry["pdb"].splitlines():
            if line.startswith(("ATOM", "HETATM")):
                worst = max(worst, abs(float(line[60:66]) - table[int(line[22:26])]))
    record(
        SECTION,
        "viewer B-factor column equals per-residue RMSF",
        worst <= 0.005 + 1e-9,
        f"max |Δ| = {worst:.4f} Å",
        "0",
        "0.005 Å (%6.2f PDB precision)",
        "states/State_*/rmsf_residues.csv",
    )
    record(
        SECTION,
        "NGL library inlined (no <script src>)",
        "ngl-viewport" in html and "NGL.Stage" in html and "<script src" not in html,
        f"NGL.Stage present = {'NGL.Stage' in html}",
        "present, inline",
        "exact",
        "templates/assets/ngl.js (MIT)",
    )

    # ------------------------------------------------------------ interpretive sentences vs CSV
    interp = ctx["interp"]
    pop = pd.read_csv(out / "statistics" / "state_populations.csv").set_index("State")
    eff = pd.read_csv(out / "convergence" / "effective_samples.csv")
    rmsip = pd.read_csv(out / "convergence" / "rmsip.csv")
    cos = pd.read_csv(out / "convergence" / "cosine_content.csv")
    sfe = pd.read_csv(out / "states" / "state_free_energies.csv").set_index("state")

    ok_pop = True
    for s in interp["populations"]:
        m = re.match(
            r"The population of (State \w+) is ([0-9.]+) % with a 95 % (\S+) CI of \[([0-9.]+), ([0-9.]+)\]",
            s,
        )
        if not m:
            continue
        lab, mean, kind, lo, hi = (
            m.group(1),
            float(m.group(2)),
            m.group(3),
            float(m.group(4)),
            float(m.group(5)),
        )
        lo_c, hi_c = (
            ("Bootstrap_CI_95_Lower", "Bootstrap_CI_95_Upper")
            if kind == "block-bootstrap"
            else ("CI_95_Lower", "CI_95_Upper")
        )
        ok_pop &= _num_in(f"{mean}", pop.loc[lab, "Mean_Population_Percent"], 1)
        ok_pop &= (
            f"{lo:.1f}" == f"{pop.loc[lab, lo_c]:.1f}"
            and f"{hi:.1f}" == f"{pop.loc[lab, hi_c]:.1f}"
        )
        width = pop.loc[lab, hi_c] - pop.loc[lab, lo_c]
        ok_pop &= f"about {width:.0f} percentage points" in s
    record(
        SECTION,
        "population sentences quote statistics/state_populations.csv (mean, CI, width)",
        ok_pop and len(interp["populations"]) >= len(results.states),
        ok_pop,
        True,
        "quoted precision",
        "statistics/state_populations.csv",
    )

    n_eff_total = float(eff["N_eff"].sum())
    s0 = interp["n_eff"][0]
    ok_neff = f"about {n_eff_total:.0f} of the {len(fa)} frames" in s0
    gmax = {r.Replicate: max(r.g_PC1, r.g_PC2) for r in eff.itertuples()}
    ok_neff &= all(f"{rep}: g = {g:.1f}" in s0 for rep, g in gmax.items())
    record(
        SECTION,
        "N_eff sentence quotes convergence/effective_samples.csv",
        ok_neff,
        s0[:80] + "...",
        f"N_eff = {n_eff_total:.1f}",
        "quoted precision",
        "convergence/effective_samples.csv",
    )
    big = max(results.states, key=lambda s: s.population)
    se_formula = 100 * math.sqrt(big.population * (1 - big.population) / n_eff_total)
    se_quoted = interp["values"]["binomial_se"]["se_percent"]
    record(
        SECTION,
        "quoted binomial SE = 100·sqrt(p(1-p)/N_eff)",
        abs(se_quoted - se_formula) < 1e-9
        and f"about {se_formula:.1f} percentage points" in interp["n_eff"][1],
        round(se_quoted, 4),
        round(se_formula, 4),
        "1e-9",
        "binomial standard error (analytical)",
    )

    r_csv = float(rmsip.loc[rmsip["Scope"] == "all_replicates_pooled", "Value"].iloc[0])
    sc = interp["convergence"][0]
    word = "high" if r_csv >= 0.7 else ("moderate" if r_csv >= 0.5 else "low")
    record(
        SECTION,
        "RMSIP sentence quotes convergence/rmsip.csv with the heuristic wording",
        f"is {r_csv:.2f}" in sc and f"{word} overlap" in sc,
        sc[:70] + "...",
        f"{r_csv:.2f} → {word}",
        "quoted precision",
        "convergence/rmsip.csv; thresholds 0.70 / 0.50",
    )
    c_col = [c for c in cos.columns if c.lower().startswith("cos")]
    c1 = float(cos.iloc[0][c_col[0]]) if c_col else results.convergence.cosine_content[1]
    ok_cos = any(f"cosine content of PC1 is {c1:.2f}" in s for s in interp["convergence"])
    record(
        SECTION,
        "cosine-content sentence quotes convergence/cosine_content.csv",
        ok_cos,
        f"{c1:.2f}",
        f"{results.convergence.cosine_content[1]:.2f}",
        "quoted precision",
        "convergence/cosine_content.csv",
    )

    ranked = sorted(results.states, key=lambda s: s.population, reverse=True)
    dg_formula = -KB * T * math.log(ranked[1].population / ranked[0].population)
    dg_csv = float(sfe.loc[ranked[1].label, "dG_state_kJ_mol"])
    record(
        SECTION,
        "summary ΔG from population ratio = -kT ln(p2/p1) = states/state_free_energies.csv dG_state",
        abs(dg_formula - dg_csv) < 1e-9 and f"{dg_formula:.1f} kJ/mol" in interp["summary"][0],
        round(dg_formula, 6),
        round(dg_csv, 6),
        "1e-9",
        "Boltzmann population ratio (analytical) and bootstrap state table",
    )

    # ------------------------------------------------------------ PDF variant
    pdf_html = render_report_html(ctx, pdf_mode=True)
    record(
        SECTION,
        "PDF variant has no scripts and announces static figures",
        "<script" not in pdf_html
        and "static figures only" in pdf_html
        and "data:image/png;base64" in pdf_html,
        f"scripts = {pdf_html.count('<script')}",
        "0 scripts + note",
        "exact",
        "render_report_html(pdf_mode=True)",
    )
    if weasyprint_available():
        pdf_path = out / "report.pdf"
        head = pdf_path.read_bytes()[:5] if pdf_path.exists() else b""
        record(
            SECTION,
            "report.pdf written by WeasyPrint",
            head == b"%PDF-",
            head.decode(errors="replace"),
            "%PDF-",
            "exact",
            "weasyprint",
        )
    else:  # pragma: no cover
        record(
            SECTION,
            "report.pdf written by WeasyPrint",
            False,
            "weasyprint missing",
            "",
            "",
            "",
            note="skipped",
        )


if __name__ == "__main__":
    import sys
    import tempfile

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    results = []

    def record(section, name, passed, measured, expected, tolerance, reference, note=""):
        results.append(bool(passed) or note == "skipped")
        flag = "SKIP" if note == "skipped" else ("PASS" if passed else "FAIL")
        print(f"[{flag}] {section} / {name}: {measured} vs {expected} ({tolerance})", flush=True)

    run(record, Path(tempfile.mkdtemp(prefix="ca_report_")), ROOT, 0.008314462618)
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)
