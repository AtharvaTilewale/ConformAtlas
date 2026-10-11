"""HTML report generation, interpretive text and metadata export.

The report is a single self-contained HTML file: static figures are embedded as
base64 data URIs, the interactive landscape (Plotly) and the structure viewer
(NGL) carry their JavaScript and data inline, so opening it needs no network
access. ``report.pdf`` (WeasyPrint) uses the static figures only.
"""

from __future__ import annotations

import base64
import logging
import math
import platform
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import pandas as pd
import scipy

import conformatlas
from conformatlas.errors import MissingDependencyError
from conformatlas.models import AnalysisResults, BasinState
from conformatlas.utils import R_GAS_CONSTANT, save_json

logger = logging.getLogger("conformatlas.report")

TEMPLATE_DIR = Path(__file__).parent / "templates"
ASSET_DIR = TEMPLATE_DIR / "assets"
WEASYPRINT_PIP = "pip install weasyprint"

#: RMSIP wording thresholds (same heuristic as convergence.py / the convergence figure).
RMSIP_HIGH = 0.70
RMSIP_MODERATE = 0.50
#: Cosine-content threshold above which the leading PC behaves like random diffusion.
COSINE_DIFFUSIVE = 0.50
#: Below this many effectively independent frames populations are flagged as unreliable.
N_EFF_MIN = 50


# ----------------------------------------------------------------------------
# report options (set by the CLI; the pipeline passes none)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportOptions:
    """Options for the HTML report that are not part of ``AnalysisConfig`` yet.

    ``interactive``: embed the Plotly landscape / profiles (needs plotly, else a
    note is shown); ``plotly_js``: ``'inline'`` (self-contained, ~4.8 MB) or
    ``'cdn'``; ``structure_viewer``: embed the NGL viewer of the state
    representatives; ``path_overlay``: also render the static landscape with the
    trajectory paths (``figures/fel_2d_path.*``).
    """

    interactive: bool = True
    plotly_js: str = "inline"
    structure_viewer: bool = True
    path_overlay: bool = False


_OPTIONS = ReportOptions()


def set_report_options(**kwargs: Any) -> ReportOptions:
    """Update the module-wide :class:`ReportOptions` (``None`` values are ignored)."""
    global _OPTIONS
    kw = {k: v for k, v in kwargs.items() if v is not None}
    if "plotly_js" in kw and kw["plotly_js"] not in ("inline", "cdn"):
        raise ValueError("plotly_js must be 'inline' or 'cdn'")
    _OPTIONS = replace(_OPTIONS, **kw)
    return _OPTIONS


def get_report_options() -> ReportOptions:
    """Return the active :class:`ReportOptions`."""
    return _OPTIONS


# ----------------------------------------------------------------------------
# embedding helpers
# ----------------------------------------------------------------------------

_MIME = {"png": "image/png", "svg": "image/svg+xml", "jpg": "image/jpeg", "jpeg": "image/jpeg"}


def image_to_base64(image_path: str | Path) -> str:
    """Convert an image file to a base64 data URI string.

    PDF figures cannot be shown in ``<img>``; when ``image_path`` is a PDF or
    does not exist, a PNG or SVG sibling with the same stem is used instead.
    """
    p = Path(image_path)
    if not p.exists() or p.suffix.lower() == ".pdf":
        for alt in (p.with_suffix(".png"), p.with_suffix(".svg")):
            if alt.exists():
                p = alt
                break
        else:
            return ""
    ext = p.suffix.lower().replace(".", "")
    mime = _MIME.get(ext)
    if mime is None:
        return ""
    with p.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def ngl_js() -> str:
    """Contents of the bundled NGL library (MIT; templates/assets/ngl.js)."""
    return (ASSET_DIR / "ngl.js").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------
# interpretive sentences
# ----------------------------------------------------------------------------


def _fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}"


def _rmsip_wording(r: float) -> str:
    if r >= RMSIP_HIGH:
        return "high"
    if r >= RMSIP_MODERATE:
        return "moderate"
    return "low"


def binomial_se_percent(p: float, n_eff: float) -> float:
    """Standard error (percentage points) of a population estimate from ``n_eff`` independent frames.

    ``SE = sqrt(p (1 - p) / N_eff)`` with ``p`` a fraction; returned × 100.
    """
    if n_eff <= 0 or not (0.0 <= p <= 1.0):
        return float("nan")
    return 100.0 * math.sqrt(p * (1.0 - p) / n_eff)


def interpretive_sentences(results: AnalysisResults) -> dict[str, Any]:
    """Plain-language sentences derived from the computed numbers.

    Returns ``{"summary": [...], "populations": [...], "n_eff": [...],
    "convergence": [...], "barriers": [...], "not_shown": [...], "warnings": [...],
    "values": {...}}``. Every number quoted in a sentence is also stored in
    ``values`` with its source table so the text can be checked against the CSV
    outputs (validation/checks_report.py does this).
    """
    conv = results.convergence
    T = results.config.temperature
    kT = R_GAS_CONSTANT * T
    states = list(results.states)
    values: dict[str, Any] = {}
    out: dict[str, Any] = {
        k: [] for k in ("summary", "populations", "n_eff", "convergence", "barriers")
    }

    n_frames = int(len(results.frame_assignments))
    n_rep = len(results.trajectories_info)
    n_eff_total = float(sum(conv.n_eff.values())) if conv.n_eff else float("nan")
    values["n_frames"] = n_frames
    values["n_eff_total"] = n_eff_total
    values["n_eff"] = {k: float(v) for k, v in conv.n_eff.items()}

    # ---- populations with CI widths
    pop_df = results.replicate_statistics
    pop_rows: list[dict[str, Any]] = []
    if pop_df is not None and len(pop_df):
        use_boot = (
            "Bootstrap_CI_95_Lower" in pop_df.columns
            and np.isfinite(pd.to_numeric(pop_df["Bootstrap_CI_95_Lower"], errors="coerce")).all()
        )
        lo_col, hi_col = (
            ("Bootstrap_CI_95_Lower", "Bootstrap_CI_95_Upper")
            if use_boot
            else ("CI_95_Lower", "CI_95_Upper")
        )
        kind = "block-bootstrap" if use_boot else "between-replicate"
        for _, r in pop_df.iterrows():
            mean = float(r["Mean_Population_Percent"])
            lo, hi = float(r[lo_col]), float(r[hi_col])
            width = hi - lo
            pop_rows.append(
                {
                    "state": str(r["State"]),
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "width": width,
                    "ci": kind,
                }
            )
            out["populations"].append(
                f"The population of {r['State']} is {_fmt(mean)} % with a 95 % {kind} CI of "
                f"[{_fmt(lo)}, {_fmt(hi)}] — differences in population smaller than about "
                f"{_fmt(width, 0)} percentage points cannot be resolved with this sampling."
            )
        widest = max(pop_rows, key=lambda d: d["width"])
        if widest["width"] > 20:
            out["populations"].append(
                f"The widest interval ({widest['state']}: {_fmt(widest['width'], 0)} points) means the "
                "ranking of states by population is only qualitative; more or longer replicates are "
                "needed before quoting populations to better than tens of percent."
            )
        if kind == "between-replicate" and n_rep < 3:
            out["populations"].append(
                f"With only {n_rep} replicate(s) the between-replicate interval is itself poorly "
                "determined (t-interval on a very small sample); treat it as indicative."
            )
    values["populations"] = pop_rows

    # ---- N_eff and its consequence
    if conv.n_eff:
        g_parts = []
        for rep, g in conv.statistical_inefficiency.items():
            if isinstance(g, dict) and g:
                gmax = max(float(v) for v in g.values())
                g_parts.append(f"{rep}: g = {_fmt(gmax)}")
        g_txt = f" (statistical inefficiency {'; '.join(g_parts)} frames)" if g_parts else ""
        out["n_eff"].append(
            f"Only about {_fmt(n_eff_total, 0)} of the {n_frames} frames are statistically independent"
            f"{g_txt}: consecutive frames are correlated along the landscape coordinates, so the "
            f"information content is that of {_fmt(n_eff_total, 0)} samples, not {n_frames}."
        )
        if states:
            big = max(states, key=lambda s: s.population)
            se = binomial_se_percent(big.population, n_eff_total)
            values["binomial_se"] = {
                "state": big.label,
                "population": big.population,
                "se_percent": se,
            }
            out["n_eff"].append(
                f"Consequence: for {big.label} (population {_fmt(big.population * 100)} %) the binomial "
                f"standard error with N_eff = {_fmt(n_eff_total, 0)} is about {_fmt(se)} percentage points "
                f"(sqrt(p(1-p)/N_eff)); the bootstrap intervals above account for this correlation."
            )
        if n_eff_total < N_EFF_MIN:
            out["n_eff"].append(
                f"Fewer than {N_EFF_MIN} effective samples: populations and free energies are not "
                "statistically reliable regardless of the other diagnostics."
            )

    # ---- RMSIP / cosine content
    r = float(conv.rmsip_halves)
    values["rmsip_halves"] = r
    word = _rmsip_wording(r)
    out["convergence"].append(
        f"The RMSIP between the essential subspaces of the first and second halves of the sampling is "
        f"{_fmt(r, 2)}, i.e. {word} overlap (heuristic: ≥ {RMSIP_HIGH:.2f} high, "
        f"{RMSIP_MODERATE:.2f}–{RMSIP_HIGH:.2f} moderate, < {RMSIP_MODERATE:.2f} low). "
        + (
            "The dominant motions are reproducible between the two halves."
            if word == "high"
            else "The dominant motions are still changing with simulation time: the leading PCs, and "
            "therefore the axes of the landscape, would move if the simulation were extended."
        )
    )
    if conv.rmsip_halves_per_replicate:
        items = sorted(conv.rmsip_halves_per_replicate.items(), key=lambda kv: kv[1])
        values["rmsip_per_replicate"] = {k: float(v) for k, v in items}
        out["convergence"].append(
            f"Per replicate the half-vs-half RMSIP ranges from {_fmt(items[0][1], 2)} ({items[0][0]}) "
            f"to {_fmt(items[-1][1], 2)} ({items[-1][0]})."
        )
    c1 = float(conv.cosine_content.get(1, float("nan")))
    values["cosine_pc1"] = c1
    if np.isfinite(c1):
        out["convergence"].append(
            f"The cosine content of PC1 is {_fmt(c1, 2)}: "
            + (
                f"above {COSINE_DIFFUSIVE:.2f}, the projection looks like a cosine of a random walk, "
                "which indicates the system is drifting rather than sampling a bounded region."
                if c1 >= COSINE_DIFFUSIVE
                else f"below {COSINE_DIFFUSIVE:.2f}, the projection is not consistent with random "
                "diffusion (necessary, not sufficient, for convergence)."
            )
        )

    # ---- barriers
    bar = results.extras.get("barriers")
    bar_rows = []
    if isinstance(bar, pd.DataFrame) and len(bar) and "barrier_from_i_kJ_mol" in bar.columns:
        for _, b in bar.iterrows():
            bi = float(b["barrier_from_i_kJ_mol"])
            bj = float(b["barrier_from_j_kJ_mol"])
            bar_rows.append(
                {
                    "state_i": str(b["state_i"]),
                    "state_j": str(b["state_j"]),
                    "barrier_i": bi,
                    "barrier_j": bj,
                }
            )
            ci = ""
            if "barrier_from_i_ci95_low" in bar.columns and np.isfinite(
                b["barrier_from_i_ci95_low"]
            ):
                ci = (
                    f" (95 % CI [{_fmt(float(b['barrier_from_i_ci95_low']))}, "
                    f"{_fmt(float(b['barrier_from_i_ci95_high']))}])"
                )
            out["barriers"].append(
                f"Leaving {b['state_i']} towards {b['state_j']} crosses a barrier of {_fmt(bi)} kJ/mol"
                f"{ci} = {_fmt(bi / kT)} kT on the 2-D projection; the reverse barrier is {_fmt(bj)} kJ/mol. "
                "Projected barriers are lower bounds of the true barrier (the saddle in the full "
                "coordinate space can be hidden by the projection)."
            )
    values["barriers"] = bar_rows

    # ---- summary line
    major = sum(s.population for s in states) * 100.0
    values["major_population_percent"] = major
    if states:
        ranked = sorted(states, key=lambda s: s.population, reverse=True)
        out["summary"].append(
            f"{len(states)} state(s) hold {_fmt(major)} % of the frames; the most populated is "
            f"{ranked[0].label} ({_fmt(ranked[0].population * 100)} %)."
            + (
                f" ΔG between the two most populated states is "
                f"{_fmt(-kT * math.log(ranked[1].population / ranked[0].population))} kJ/mol "
                f"({_fmt(-math.log(ranked[1].population / ranked[0].population))} kT) from the population ratio."
                if len(ranked) > 1 and ranked[1].population > 0
                else ""
            )
        )

    # ---- what this does NOT show
    cv_mode = results.config.cv
    ev_txt = ""
    if cv_mode == "pc" and results.pca is not None and results.pca.explained_variance is not None:
        pcs = results.config.fel_pcs
        ev = results.pca.explained_variance
        if max(pcs) <= len(ev):
            tot = float(ev[pcs[0] - 1] + ev[pcs[1] - 1])
            values["explained_variance_fel_pcs"] = tot
            ev_txt = (
                f" The two landscape coordinates capture {_fmt(tot)} % of the fluctuation variance; "
                f"the remaining {_fmt(100 - tot)} % is projected away, so two basins can be one state "
                "and one basin can hide several."
            )
    total_ns = (
        sum(max(t.end_time_ps - t.start_time_ps, 0.0) for t in results.trajectories_info) / 1000.0
    )
    values["total_sampling_ns"] = total_ns
    out["not_shown"] = [
        "What this report does not show. (1) States that were never visited in "
        f"{_fmt(total_ns, 2)} ns of total sampling: a free-energy landscape only describes the region "
        "the trajectories explored, and a missing basin is indistinguishable from a high-energy one."
        + ev_txt
        + " (2) Absolute free energies: ΔG is relative to the most populated bin and depends on the bin "
        "width and on the two chosen coordinates. (3) Rates: transition counts and lag-one Markov "
        "mean first-passage times are indicative only; no Markov-state model was validated here. "
        "(4) Convergence: the RMSIP, cosine-content and population-stability diagnostics are necessary "
        "but not sufficient conditions; a trapped simulation can pass all of them. (5) Thermodynamic "
        "relevance of the states: watershed basins are features of the projected histogram, not "
        "necessarily kinetically distinct conformations."
    ]
    out["warnings"] = list(results.warnings)
    out["values"] = values
    return out


_SECTION_KEYS = (
    ("Equilibration and effective sample size", "n_eff"),
    ("State free energies and barriers", "populations"),
    ("Barriers between adjacent states", "barriers"),
    ("State kinetics", "barriers"),
)


def _attach_interpretations(sections: list[dict], interp: dict[str, Any]) -> list[dict]:
    """Add the matching interpretive sentences to the pipeline's extra sections."""
    for sec in sections:
        title = str(sec.get("title", ""))
        for prefix, key in _SECTION_KEYS:
            if title.startswith(prefix) and interp.get(key):
                sec["interpretation"] = list(interp[key])
                break
    return sections


# ----------------------------------------------------------------------------
# interactive + structure viewer contexts
# ----------------------------------------------------------------------------


def _interactive_context(results: AnalysisResults, opts: ReportOptions) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "interactive_enabled": False,
        "interactive_note": "",
        "plotly_js_tag": "",
    }
    if not opts.interactive:
        ctx["interactive_note"] = "Interactive figures were disabled (--no-interactive)."
        return ctx
    from conformatlas.plotting_interactive import (
        PLOTLY_PIP,
        build_landscape_figure,
        build_projection_figure,
        figure_to_div,
        plotly_available,
        plotly_js,
    )

    if not plotly_available():
        ctx["interactive_note"] = (
            f"Interactive figures need the optional package plotly ({PLOTLY_PIP}); static figures are shown."
        )
        return ctx
    basin_map = None
    bm_path = results.output_dir / "states" / "basin_map.npy"
    if bm_path.exists():
        try:
            basin_map = np.load(bm_path)
        except Exception:  # pragma: no cover
            basin_map = None
    from conformatlas.plotting import CV_LABELS

    cv_labels = (CV_LABELS[0], CV_LABELS[1])
    fig = build_landscape_figure(
        results.fel,
        states=results.states,
        frame_assignments=results.frame_assignments,
        basin_map=basin_map,
        cv_labels=cv_labels,
    )
    prof = build_projection_figure(results.fel, cv_labels=cv_labels, states=results.states)
    ctx.update(
        interactive_enabled=True,
        plotly_js_tag=plotly_js(opts.plotly_js),
        interactive_landscape_div=figure_to_div(fig, div_id="interactive-landscape"),
        interactive_profile_div=figure_to_div(prof, div_id="interactive-profiles"),
        plotly_mode=opts.plotly_js,
    )
    return ctx


_ATOM_RE = re.compile(r"^(ATOM  |HETATM)")


def pdb_with_bfactor(pdb_text: str, rmsf: pd.DataFrame | None, default: float = 0.0) -> str:
    """Return ``pdb_text`` with the B-factor column replaced by per-residue RMSF (Å).

    ``rmsf`` needs ``residue_number`` and ``RMSF`` columns (``chain`` optional and
    only used when the PDB has chain identifiers). Residues without a value get
    ``default``. Columns follow the PDB fixed-width format (61-66, ``%6.2f``).
    """
    lookup: dict[tuple[str, int], float] = {}
    by_res: dict[int, float] = {}
    if rmsf is not None and len(rmsf):
        for _, r in rmsf.iterrows():
            resnum = int(r["residue_number"])
            val = float(r["RMSF"])
            chain = str(r["chain"]).strip() if "chain" in rmsf.columns else ""
            lookup[(chain, resnum)] = val
            by_res.setdefault(resnum, val)
    out_lines = []
    for line in pdb_text.splitlines():
        if _ATOM_RE.match(line) and len(line) >= 54:
            line = line.ljust(80)
            chain = line[21].strip()
            try:
                resnum = int(line[22:26])
            except ValueError:
                resnum = None
            val = default
            if resnum is not None:
                val = lookup.get((chain, resnum), by_res.get(resnum, default))
            val = min(max(val, -99.99), 999.99)
            line = f"{line[:60]}{val:6.2f}{line[66:]}".rstrip()
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _read_rmsf(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            df = pd.read_csv(path)
            if {"residue_number", "RMSF"} <= set(df.columns):
                return df
        except Exception:  # pragma: no cover
            return None
    return None


def _structure_viewer_context(results: AnalysisResults, opts: ReportOptions) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "viewer_enabled": False,
        "viewer_structures": [],
        "ngl_js": "",
        "viewer_note": "",
    }
    if not opts.structure_viewer:
        return ctx
    out_dir = results.output_dir
    global_rmsf = _read_rmsf(out_dir / "descriptors" / "rmsf_residues.csv")
    structures: list[dict[str, Any]] = []
    for st in results.states:
        folder = out_dir / "states" / st.label.replace(" ", "_")
        pdb = st.representative_pdb if st.representative_pdb else folder / "representative.pdb"
        pdb = Path(pdb)
        if not pdb.is_absolute() and not pdb.exists():
            pdb = folder / pdb.name
        if not pdb.exists():
            continue
        rmsf = _read_rmsf(folder / "rmsf_residues.csv")
        if rmsf is None:
            rmsf = global_rmsf
        structures.append(
            {
                "id": f"struct-{st.state_id}",
                "label": f"{st.label} representative ({st.population * 100:.1f} %, "
                f"{st.representative_replicate} t = {st.representative_time_ps:.0f} ps)",
                "pdb": pdb_with_bfactor(pdb.read_text(encoding="utf-8", errors="replace"), rmsf),
                "rmsf_source": "state ensemble RMSF"
                if rmsf is not global_rmsf
                else "all-frame RMSF",
                "rmsf_max": float(rmsf["RMSF"].max()) if rmsf is not None and len(rmsf) else 0.0,
            }
        )
    avg = out_dir / "pca" / "average.pdb"
    if avg.exists():
        structures.append(
            {
                "id": "struct-average",
                "label": "Average structure (all frames)",
                "pdb": pdb_with_bfactor(
                    avg.read_text(encoding="utf-8", errors="replace"), global_rmsf
                ),
                "rmsf_source": "all-frame RMSF",
                "rmsf_max": float(global_rmsf["RMSF"].max())
                if global_rmsf is not None and len(global_rmsf)
                else 0.0,
            }
        )
    if not structures:
        ctx["viewer_note"] = "No representative structures were written, so the viewer is omitted."
        return ctx
    try:
        js = ngl_js()
    except OSError as e:  # pragma: no cover - packaging problem
        ctx["viewer_note"] = f"NGL library not found in the installed package ({e})."
        return ctx
    rmsf_max = max(s["rmsf_max"] for s in structures)
    ctx.update(
        viewer_enabled=True,
        viewer_structures=structures,
        ngl_js=js,
        viewer_rmsf_max=max(rmsf_max, 0.01),
    )
    return ctx


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------


def _template():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    return env.get_template("report.html.j2")


def build_report_context(
    results: AnalysisResults,
    figures_dict: dict[str, Path],
    gromacs_version: str = "not used",
    extra_sections: list[dict] | None = None,
    options: ReportOptions | None = None,
) -> dict[str, Any]:
    """Assemble everything the template needs (figures embedded, sentences, interactive
    and viewer fragments). Also renders ``figures/fel_2d_path.*`` when ``path_overlay`` is on."""
    opts = options or _OPTIONS
    out_dir = results.output_dir
    figures = dict(figures_dict)
    if opts.path_overlay:
        from conformatlas.plotting import plot_fel_2d

        try:
            figures["fel_2d_path"] = plot_fel_2d(
                results.fel,
                states=results.states,
                out_path=out_dir / "figures" / "fel_2d_path.png",
                save_unannotated=False,
                path=results.frame_assignments,
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Path-overlay figure failed: {e}")

    embedded_figs = {key: image_to_base64(fig_p) for key, fig_p in figures.items()}

    pop_stats_records = []
    if results.replicate_statistics is not None:
        pop_stats_records = results.replicate_statistics.to_dict(orient="records")

    total_frames = sum(t.n_frames for t in results.trajectories_info)
    major_pop = sum(s.population for s in results.states) * 100.0
    unassigned_pop = results.unassigned_population * 100.0

    interp = interpretive_sentences(results)
    sections = _attach_interpretations(_embed_sections(extra_sections or []), interp)
    state_pdb_paths = {}
    for s in results.states:
        if s.representative_pdb:
            p = Path(s.representative_pdb)
            try:
                p = p.resolve().relative_to(out_dir.resolve())
            except ValueError:
                pass
            state_pdb_paths[s.label] = str(p)

    context = {
        "project_name": results.config.system_name,
        "system_name": results.config.system_name,
        "temperature": results.config.temperature,
        "total_frames": total_frames,
        "num_replicates": len(results.trajectories_info),
        "timestamp": results.run_metadata.get("timestamp", "N/A"),
        "conformatlas_version": conformatlas.__version__,
        "python_version": platform.python_version(),
        "gromacs_version": gromacs_version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "os_info": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "command_line": results.run_metadata.get("command_line", "conformatlas analyze"),
        "states": results.states,
        "state_pdb_paths": state_pdb_paths,
        "major_population_percent": f"{major_pop:.1f}",
        "unassigned_population_percent": f"{unassigned_pop:.1f}",
        "population_stats": pop_stats_records,
        "uncertainty_type": results.run_metadata.get("uncertainty_type", "replicate"),
        "convergence": results.convergence,
        "bins": results.config.bins
        if results.config.bins
        else results.run_metadata.get("bins", "auto"),
        "fig_fel_2d": embedded_figs.get("fel_2d", embedded_figs.get("fel_states", "")),
        "fig_fel_2d_path": embedded_figs.get("fel_2d_path", ""),
        "fig_fel_3d": embedded_figs.get("fel_3d", ""),
        "fig_fel_states": embedded_figs.get("fel_states", embedded_figs.get("fel_2d", "")),
        "fig_basin_map": embedded_figs.get("basin_map", ""),
        "fig_populations": embedded_figs.get("populations", ""),
        "fig_pca_variance": embedded_figs.get("pca_variance", ""),
        "fig_pca_scatter": embedded_figs.get(
            "pca_scatter", embedded_figs.get("pca_projections", "")
        ),
        "fig_pca_density": embedded_figs.get("pca_density", ""),
        "fig_pca_projections": embedded_figs.get(
            "pca_projections", embedded_figs.get("pca_scatter", "")
        ),
        "fig_convergence_progressive": embedded_figs.get("convergence_progressive", ""),
        "fig_convergence_cosine": embedded_figs.get("convergence_cosine", ""),
        "fig_convergence_stability": embedded_figs.get("convergence_stability", ""),
        "fig_convergence": embedded_figs.get("convergence", ""),
        "extra_sections": sections,
        "interp": interp,
        "unassigned_population": unassigned_pop,
        "pdf_mode": False,
    }
    context.update(_interactive_context(results, opts))
    context.update(_structure_viewer_context(results, opts))
    return context


def render_report_html(context: dict[str, Any], pdf_mode: bool = False) -> str:
    """Render the template. ``pdf_mode=True`` drops every script (Plotly, NGL) and
    shows the static figures with a note — this is what WeasyPrint receives."""
    ctx = dict(context)
    if pdf_mode:
        ctx.update(
            pdf_mode=True,
            interactive_enabled=False,
            plotly_js_tag="",
            interactive_note="",
            viewer_enabled=False,
            viewer_note="",
            ngl_js="",
        )
    return _template().render(**ctx)


def generate_html_report(
    results: AnalysisResults,
    figures_dict: dict[str, Path],
    gromacs_version: str = "not used",
    extra_sections: list[dict] | None = None,
    pdf: bool = False,
    options: ReportOptions | None = None,
) -> Path:
    """Compile and render the standalone HTML analysis report (and optionally ``report.pdf``).

    ``options`` defaults to the module-wide :class:`ReportOptions` set by the CLI.
    Besides ``report.html`` this writes ``summary.json``, ``run_metadata.json`` and
    ``report_interpretation.json`` (the interpretive sentences with the numbers they quote).
    """
    out_dir = results.output_dir
    report_path = out_dir / "report.html"
    context = build_report_context(results, figures_dict, gromacs_version, extra_sections, options)
    save_json(context["interp"], out_dir / "report_interpretation.json")

    with report_path.open("w", encoding="utf-8") as f:
        f.write(render_report_html(context))

    if pdf:
        export_pdf(
            report_path, report_path.with_suffix(".pdf"), html=render_report_html(context, True)
        )

    logger.info(f"HTML report generated at {report_path.resolve()}")

    total_frames = context["total_frames"]
    unassigned_pop = context["unassigned_population"]
    save_json(results.run_metadata, out_dir / "run_metadata.json")
    summary_data = {
        "system": results.config.system_name,
        "temperature_K": results.config.temperature,
        "total_frames": total_frames,
        "states": [
            {
                "label": s.label,
                "population_percent": s.population * 100.0,
                "min_pc1": s.min_pc1,
                "min_pc2": s.min_pc2,
                "min_free_energy_kJ_mol": s.min_free_energy,
                "representative_frame": s.representative_frame,
                "representative_time_ps": s.representative_time_ps,
            }
            for s in results.states
        ],
        "unassigned_population_percent": unassigned_pop,
        "convergence": {
            "rmsip_halves": results.convergence.rmsip_halves,
            "pc1_cosine_content": results.convergence.cosine_content.get(1, 0.0),
            "assessment": results.convergence.sampling_assessment,
        },
    }
    save_json(summary_data, out_dir / "summary.json")
    return report_path


def _embed_sections(sections: list[dict]) -> list[dict]:
    """Base64-embed figure paths and stringify table cells of extra report sections."""
    out = []
    for sec in sections:
        item = {
            "title": sec.get("title", ""),
            "paragraphs": list(sec.get("paragraphs", [])),
            "caption": sec.get("caption"),
            "figure": "",
            "table": None,
            "interpretation": list(sec.get("interpretation", [])),
        }
        fig = sec.get("figure")
        if fig and Path(fig).exists():
            item["figure"] = image_to_base64(fig)
        table = sec.get("table")
        if table is not None:
            if hasattr(table, "to_dict"):
                df = table.copy()
                for c in df.columns:
                    if df[c].dtype.kind == "f":
                        df[c] = df[c].map(lambda v: "" if v != v else f"{v:.3g}")
                item["table"] = {
                    "columns": list(map(str, df.columns)),
                    "rows": df.astype(str).to_dict("records"),
                }
            else:
                item["table"] = table
        out.append(item)
    return out


def weasyprint_available() -> bool:
    """True when WeasyPrint can be imported (PDF export)."""
    try:
        import weasyprint  # noqa: F401
    except Exception:
        return False
    return True


def export_pdf(html_path: Path, pdf_path: Path, html: str | None = None) -> Path | None:
    """Render the report to PDF with WeasyPrint.

    ``html`` (optional) is the PDF-specific markup (static figures, no scripts);
    when omitted the file at ``html_path`` is converted. Returns ``None`` and logs
    a warning when WeasyPrint is missing or fails, so the HTML report is never lost.
    """
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as e:
        logger.warning(
            f"PDF export requested but WeasyPrint is unavailable ({e}); skipping. {WEASYPRINT_PIP}"
        )
        return None
    try:
        if html is not None:
            HTML(string=html, base_url=str(Path(html_path).parent)).write_pdf(str(pdf_path))
        else:
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        logger.info(f"PDF report written to {pdf_path}")
        return pdf_path
    except Exception as e:  # pragma: no cover
        logger.warning(f"PDF export failed: {e}")
        return None


def require_pdf_support() -> None:
    """Raise :class:`MissingDependencyError` when ``--pdf`` cannot be honoured."""
    if not weasyprint_available():
        raise MissingDependencyError("weasyprint", WEASYPRINT_PIP, "PDF export (--pdf)")


def external_resource_urls(html: str) -> list[str]:
    """http(s) URLs referenced as *resources* in ``html``: ``src``/``href`` of script, link,
    img, iframe, object, embed, video, audio, source tags and CSS ``url()``/``@import``.

    Anchor (``<a href>``) citation links are not resources and are ignored; so is
    URL text inside script bodies (the inlined libraries contain URL strings that
    are never fetched unless the user asks, e.g. NGL's ``rcsb://`` loader).
    """
    found: list[str] = []
    tag_re = re.compile(
        r"<(script|link|img|iframe|object|embed|video|audio|source)\b[^>]*?\s(?:src|href|data)\s*=\s*"
        r"[\"']?(https?://[^\"'\s>]+)",
        re.I,
    )
    found.extend(m.group(2) for m in tag_re.finditer(html))
    style_blocks = re.findall(r"<style\b[^>]*>(.*?)</style>", html, re.S | re.I)
    style_attrs = re.findall(r"style\s*=\s*\"([^\"]*)\"", html, re.I)
    for css in style_blocks + style_attrs:
        found.extend(re.findall(r"url\(\s*[\"']?(https?://[^\"')\s]+)", css, re.I))
        found.extend(re.findall(r"@import\s+[\"']?(https?://[^\"';\s]+)", css, re.I))
    return found


def state_table_records(states: list[BasinState]) -> list[dict[str, Any]]:
    """Executive-summary rows as plain dicts (used by tests)."""
    return [
        {
            "label": s.label,
            "population_percent": s.population * 100.0,
            "min_cv1": s.min_pc1,
            "min_cv2": s.min_pc2,
            "min_dG": s.min_free_energy,
        }
        for s in states
    ]
