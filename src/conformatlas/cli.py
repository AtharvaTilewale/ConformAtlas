"""Command Line Interface (CLI) for ConformAtlas / ConformAtlas."""

import contextlib
import io
import logging
import sys
from pathlib import Path

import click
import yaml

import conformatlas
from conformatlas.comparison import run_condition_comparison
from conformatlas.config import generate_example_config, parse_comparison_config
from conformatlas.errors import (
    ConformAtlasError,
    MissingDependencyError,
    UnknownGroupError,
    available_groups,
    malformed_config_from_yaml_error,
    selection_error_group,
)
from conformatlas.gromacs import GromacsRunner
from conformatlas.plotting import (
    plot_comparison_populations,
    plot_comparison_shared_pca,
    plot_comparison_system_fel,
    plot_condition_comparison,
    set_figure_style,
)
from conformatlas.utils import (
    BOLD,
    CYAN,
    GREEN,
    NC,
    RED,
    YELLOW,
    print_banner,
    setup_logging,
)

logger = logging.getLogger("conformatlas.cli")

#: Set by ``--quiet``; suppresses the banner and progress output (errors still go to stderr).
_QUIET = False

#: Names of the figure/report options handled by the CLI itself (not AnalysisConfig fields).
_STYLE_OPTION_KEYS = (
    "fig_format",
    "colormap",
    "font_scale",
    "fig_dpi",
    "path_overlay",
    "interactive",
    "plotly_js",
    "quiet",
)


def _set_quiet(flag: bool | None) -> None:
    """Set the quiet state for THIS invocation (``None``/``False`` -> not quiet).

    The state is module-level so that helpers can consult it, but it must not
    leak from one command invocation to the next within a process (test runners
    and library callers invoke ``main`` repeatedly).
    """
    global _QUIET
    _QUIET = bool(flag)


def is_quiet() -> bool:
    """True when ``--quiet`` was given."""
    return _QUIET


def echo(msg: str = "", err: bool = False) -> None:
    """``click.echo`` that honours ``--quiet`` for stdout (stderr is never suppressed)."""
    if err or not _QUIET:
        click.echo(msg, err=err)


def maybe_banner(ctx=None) -> None:
    """Print the ASCII banner only on an interactive terminal and never in quiet mode.

    ``--help`` / ``--version`` never show it (they return before this is called).
    """
    if _QUIET or not sys.stdout.isatty():
        return
    print_banner(ctx)


def _setup_logging(out_dir: Path) -> None:
    """``setup_logging`` that, in quiet mode, keeps only warnings on the console."""
    if not _QUIET:
        setup_logging(out_dir)
        return
    with contextlib.redirect_stdout(io.StringIO()):
        setup_logging(out_dir)
    for h in logging.getLogger("conformatlas").handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setStream(sys.stdout)
            h.setLevel(logging.WARNING)


def _log_failure(msg: str) -> None:
    """Write the traceback to the log file only (the console gets the short message)."""
    lg = logging.getLogger("conformatlas")
    console = [
        h
        for h in lg.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    levels = [h.level for h in console]
    for h in console:
        h.setLevel(logging.CRITICAL + 1)
    try:
        lg.exception(msg)
    finally:
        for h, lv in zip(console, levels):
            h.setLevel(lv)


def fail(message: str, exit_code: int) -> None:
    """Print a short error to stderr (no traceback) and exit."""
    click.echo(f"{RED}Error: {message}{NC}", err=True)
    sys.exit(exit_code)


def apply_style_options(opts: dict) -> dict:
    """Validate and apply the figure/report options of the click parameter dict.

    The options are applied to the module-level style immediately (so that
    invalid values fail before any work starts) and, except ``quiet``, are left
    in the dict so that :func:`conformatlas.config.build_analysis_config` records
    them in the :class:`~conformatlas.models.AnalysisConfig` (``None`` = not
    given = keep the active style). Raises ``click.UsageError`` for invalid values.
    """
    from conformatlas.report import set_report_options

    style = {k: opts.get(k) for k in _STYLE_OPTION_KEYS}
    opts.pop("quiet", None)
    # The group callback (``main``) has already reset the state for this
    # invocation from the group-level ``-q``; OR in the subcommand's flag.
    _set_quiet(bool(style.get("quiet")) or is_quiet())
    if style.get("quiet"):
        opts["progress"] = False
    try:
        set_figure_style(
            fig_format=style.get("fig_format"),
            colormap=style.get("colormap"),
            font_scale=style.get("font_scale"),
            dpi=style.get("fig_dpi"),
            path_overlay=style.get("path_overlay"),
        )
        set_report_options(
            interactive=style.get("interactive"),
            plotly_js=style.get("plotly_js"),
            path_overlay=style.get("path_overlay"),
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    return opts


def check_optional_dependencies(pdf: bool, interactive: bool | None) -> None:
    """Fail early (exit 2, pip hint) when a requested feature needs a missing package.

    ``interactive=None`` means "if available": a missing plotly only disables the
    interactive figures with a notice; ``--interactive`` given explicitly makes it an error.
    """
    from conformatlas.plotting_interactive import PLOTLY_PIP, plotly_available
    from conformatlas.report import require_pdf_support, set_report_options

    if pdf:
        require_pdf_support()
    if not plotly_available():
        if interactive:
            raise MissingDependencyError(
                "plotly", PLOTLY_PIP, "the interactive report figures (--interactive)"
            )
        set_report_options(interactive=False)
        if interactive is None:
            echo(
                f"{YELLOW}NOTE: plotly is not installed; the report will contain static figures only "
                f"({PLOTLY_PIP} to enable the interactive landscape).{NC}"
            )


def check_groups(cfg) -> None:
    """Resolve the fit / analysis / ligand groups on the topology before any heavy work.

    Raises :class:`UnknownGroupError` listing the available group names. Skipped
    silently when MDAnalysis is unavailable or the GROMACS backend was chosen (gmx
    resolves its own groups).
    """
    try:
        from conformatlas.trajectory import HAS_MDANALYSIS, resolve_selection_indices
    except Exception:  # pragma: no cover
        return
    if not HAS_MDANALYSIS or cfg.pca_backend == "gromacs":
        return
    ndx_groups = None
    if cfg.index_file:
        from conformatlas.ndx import parse_ndx

        ndx_groups = parse_ndx(cfg.index_file)
    checks = [(cfg.lsq_group, None), (cfg.cov_group, None)]
    if cfg.lig_group:
        checks.append((cfg.cov_group, cfg.lig_group))
    for group, extra in checks:
        try:
            resolve_selection_indices(cfg.topology, group, ndx_groups, extra_group=extra)
        except ValueError as e:
            bad = selection_error_group(e)
            if bad is None:
                raise
            names, source = available_groups(cfg.topology, cfg.index_file)
            raise UnknownGroupError(bad, names, source) from e
        except Exception as e:  # unreadable topology etc.: let the pipeline report it
            logger.debug(f"Group pre-check skipped: {e}")
            return


def _style_options(f):
    """Figure/report options shared by ``analyze`` and ``compare``."""
    opts = [
        click.option(
            "--fig-format",
            type=str,
            default=None,
            help="Figure format(s): png (default), svg, pdf, all, or a comma list such as png,svg.",
        ),
        click.option(
            "--colormap",
            type=str,
            default=None,
            help="Matplotlib colormap for landscapes / density maps (default turbo).",
        ),
        click.option(
            "--font-scale",
            type=float,
            default=None,
            help="Multiply all figure font sizes (default 1.0; 0.3-4).",
        ),
        click.option("--fig-dpi", type=int, default=None, help="Raster figure dpi (default 300)."),
        click.option(
            "--plotly-js",
            type=click.Choice(["inline", "cdn"]),
            default=None,
            help="Embed plotly.min.js in the report (inline, default, self-contained) or load it from the CDN.",
        ),
        click.option(
            "-q", "--quiet", is_flag=True, default=None, help="No banner or progress output."
        ),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


def print_version(ctx, param, value):
    """Version option callback (no banner)."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"conformatlas, version {conformatlas.__version__}")
    ctx.exit()


@click.group(
    invoke_without_command=True,
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show the version and exit.",
)
@click.option(
    "-s",
    "--structure",
    type=click.Path(),
    help="[Legacy] Structure/topology file (.pdb, .gro, .tpr).",
)
@click.option(
    "-f",
    "--trajectory",
    type=click.Path(),
    multiple=True,
    help="[Legacy] Trajectory file (.xtc). Repeatable for multiple replicates.",
)
@click.option(
    "-T",
    "--temperature",
    type=float,
    default=300.0,
    help="Temperature in Kelvin (default: 300.0).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="Outputs",
    help="Output directory (default: Outputs).",
)
@click.option("-q", "--quiet", is_flag=True, default=False, help="No banner or progress output.")
@click.pass_context
def main(ctx, structure, trajectory, temperature, output, quiet):
    """ConformAtlas: mapping conformational landscapes from molecular simulations."""
    _set_quiet(quiet)
    if ctx.invoked_subcommand is None:
        if structure and trajectory:
            # Legacy invocation routed to analyze
            click.echo(
                f"{YELLOW}Notice: Legacy syntax detected. Routing to 'conformatlas analyze'...{NC}"
            )
            ctx.invoke(
                analyze,
                structure=structure,
                trajectory=trajectory,
                temperature=temperature,
                output=output,
            )
        else:
            click.echo(ctx.get_help())


@main.command(name="analyze")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="YAML file with any of the options below (command-line flags override it).",
)
@click.option(
    "-s",
    "--structure",
    "--topology",
    "structure",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Topology/structure file (.tpr, .gro, .pdb). A .gro/.pdb also serves as the superposition reference.",
)
@click.option(
    "-f",
    "--trajectory",
    "trajectory",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    help="Trajectory file (.xtc/.trr/.dcd). Repeat for replicates: -f rep1.xtc -f rep2.xtc.",
)
@click.option(
    "-T", "--temperature", type=float, default=None, help="Simulation temperature in Kelvin (> 0)."
)
@click.option(
    "-o", "--output", type=click.Path(file_okay=False), default=None, help="Output directory."
)
@click.option(
    "-b", "--begin", "begin_ps", type=float, default=None, help="First time to read (ps)."
)
@click.option("-e", "--end", "end_ps", type=float, default=None, help="Last time to read (ps).")
@click.option(
    "--equilibration",
    type=click.Choice(["none", "auto"]),
    default=None,
    help="auto: discard the initial transient detected per replicate (Chodera max-N_eff criterion). Default none (reported only).",
)
@click.option(
    "-n",
    "--index",
    "index_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="GROMACS index file; group names given to -l/-c/-g are looked up here first.",
)
@click.option(
    "-l", "--lsq-group", type=str, default=None, help="Least-squares fit group (default Backbone)."
)
@click.option(
    "-c",
    "--cov-group",
    type=str,
    default=None,
    help="Analysis (covariance) group (default Backbone).",
)
@click.option(
    "-g",
    "--lig",
    "lig_group",
    type=str,
    default=None,
    help="Ligand residue/group name (e.g. ZMR, ATP): added to the analysis group and given its own RMSD.",
)
@click.option(
    "--pca-type",
    type=click.Choice(["cartesian", "dihedral", "tica"]),
    default=None,
    help="Decomposition: Cartesian PCA (default), backbone dihedral PCA, or TICA on fitted coordinates.",
)
@click.option("--tica-lag", type=int, default=None, help="TICA lag time in frames (default 10).")
@click.option("--n-components", type=int, default=None, help="Components to retain (default 10).")
@click.option(
    "--pca-backend",
    type=click.Choice(["auto", "python", "gromacs"]),
    default=None,
    help="auto/python: in-memory MDAnalysis route (Å); gromacs: gmx covar/anaeig (nm).",
)
@click.option(
    "--cv",
    type=click.Choice(["pc", "rmsd-rg", "file"]),
    default=None,
    help="Collective variables of the landscape: two components (default), RMSD x Rg, or columns of --cv-file.",
)
@click.option(
    "--fel-pcs",
    "fel_pcs",
    type=str,
    default=None,
    help="Components for the landscape, e.g. 1,2 (default) or 1,3.",
)
@click.option(
    "--cv-file",
    "cv_files",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    help="External CV file per trajectory (.xvg/.dat/.csv; time + columns), same order as -f.",
)
@click.option(
    "--cv-cols",
    "cv_columns",
    type=str,
    default=None,
    help="0-based columns of --cv-file holding CV1,CV2 (default 1,2).",
)
@click.option(
    "--cv-labels", type=str, default=None, help="Axis names for --cv file, e.g. 'd1 (nm),d2 (nm)'."
)
@click.option(
    "--weights-file",
    "weights_files",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    help="Per-frame statistical weights per trajectory (biased sampling reweighting), same order as -f.",
)
@click.option(
    "--bins",
    type=str,
    default=None,
    help="FEL bins per axis, or 'auto' (Freedman-Diaconis; default).",
)
@click.option(
    "--smooth-sigma", type=float, default=None, help="Gaussian smoothing SD in bins (0 = raw)."
)
@click.option(
    "--pca-weighting",
    type=click.Choice(["equal-replicate", "frames"]),
    default=None,
    help="Weighting of replicates in PCA (default equal-replicate).",
)
@click.option("--stride", type=int, default=None, help="Frame stride (default 1).")
@click.option(
    "--min-state-population",
    type=float,
    default=None,
    help="Minimum fraction for a major state (default 0.05).",
)
@click.option(
    "--max-states", type=int, default=None, help="Maximum number of major states (default 10)."
)
@click.option(
    "--min-barrier",
    type=float,
    default=None,
    help="Minimum barrier (kJ/mol) between basins; shallower ones are merged (default 1 kT).",
)
@click.option(
    "--bootstrap",
    "n_bootstrap",
    type=int,
    default=None,
    help="Block-bootstrap resamples for ΔG, populations and barriers (default 200; 0 disables).",
)
@click.option("--seed", type=int, default=None, help="Random seed for the bootstrap (default 0).")
@click.option(
    "--n-representatives", type=int, default=None, help="Structures written per state (default 1)."
)
@click.option(
    "--representative",
    "representative_method",
    type=click.Choice(["minimum", "medoid"]),
    default=None,
    help="Representative choice: closest to basin minimum (default) or medoid in PC space.",
)
@click.option(
    "--fel-backend",
    type=click.Choice(["python", "gromacs"]),
    default=None,
    help="FEL backend (python default).",
)
@click.option("--gmx-command", type=str, default=None, help="Path to a GROMACS executable.")
@click.option(
    "--fit/--no-fit",
    "fit",
    default=None,
    help="Superpose frames on --lsq-group before PCA (default on). --no-fit only for pre-fitted trajectories.",
)
@click.option(
    "--mass-weighted-fit", is_flag=True, default=None, help="Mass-weight the superposition."
)
@click.option(
    "--pdf", is_flag=True, default=None, help="Also export report.pdf (requires weasyprint)."
)
@click.option(
    "--path-overlay",
    is_flag=True,
    default=None,
    help="Also draw each replicate's time-coloured path on the static 2-D landscape (figures/fel_2d_path).",
)
@click.option(
    "--interactive/--no-interactive",
    "interactive",
    default=None,
    help="Embed the Plotly landscape and the NGL structure viewer in report.html (default: when plotly is installed).",
)
# ---- Markov state model
@click.option(
    "--msm/--no-msm",
    "msm",
    default=None,
    help="Build a Markov state model on the leading components.",
)
@click.option("--msm-lag", type=int, default=None, help="MSM lag time in frames (default 10).")
@click.option(
    "--msm-lags",
    type=str,
    default=None,
    help="Comma-separated lags (frames) for the implied-timescale scan (default: geometric 1..5*lag).",
)
@click.option(
    "--msm-clusters",
    "msm_n_clusters",
    type=int,
    default=None,
    help="k-means microstates (default 100).",
)
@click.option(
    "--msm-clustering",
    type=click.Choice(["kmeans", "regular-space"]),
    default=None,
    help="Microstate clustering (default kmeans).",
)
@click.option(
    "--msm-dmin",
    type=float,
    default=None,
    help="Regular-space clustering radius (component units).",
)
@click.option(
    "--msm-components",
    "msm_n_components",
    type=int,
    default=None,
    help="Leading components used for clustering (default 3; the two landscape components come first).",
)
@click.option(
    "--msm-states",
    "msm_n_metastable",
    type=int,
    default=None,
    help="Number of PCCA+ metastable sets (default: number of landscape states, at least 2).",
)
@click.option(
    "--msm-nonreversible",
    "msm_reversible",
    flag_value=False,
    default=None,
    help="Row-normalised estimator instead of the reversible maximum-likelihood estimator.",
)
@click.option(
    "--msm-ck-k",
    type=int,
    default=None,
    help="Chapman-Kolmogorov multiples of the lag (default 5).",
)
# ---- statistics
@click.option(
    "--bootstrap-mode",
    type=click.Choice(["fixed", "resegment"]),
    default=None,
    help="'resegment' re-runs the watershed on every resample (state-count distribution); default fixed.",
)
@click.option(
    "--equilibration-on",
    type=click.Choice(["rmsd", "cv"]),
    default=None,
    help="Observable for equilibration detection: fit-group RMSD (default) or both landscape CVs (latest t0).",
)
# ---- engine / performance
@click.option(
    "--engine",
    "--pca-engine",
    "engine",
    type=click.Choice(["auto", "memory", "streaming"]),
    default=None,
    help="auto (default): in-memory PCA unless the memory estimate exceeds 2 GB; streaming: chunked two-pass PCA.",
)
@click.option(
    "--precision",
    type=click.Choice(["float64", "float32"]),
    default=None,
    help="Storage dtype of the projections with the streaming engine (default float64).",
)
@click.option(
    "--n-workers",
    type=int,
    default=None,
    help="Worker processes for the streaming engine (one per trajectory).",
)
@click.option(
    "--chunk-size", type=int, default=None, help="Frames per chunk (streaming engine, default 256)."
)
@click.option(
    "--from",
    "from_results",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Re-analyse an existing output directory (rebin, re-segment, bootstrap, kinetics, MSM) without reading trajectories.",
)
@click.option(
    "--save-hdf5/--no-save-hdf5",
    "save_hdf5",
    default=None,
    help="Also write results.h5, a single HDF5 container of every table and array (requires h5py).",
)
@click.option(
    "--no-progress", "progress", flag_value=False, default=None, help="Disable progress bars."
)
# ---- residue-level analyses and dihedral sets
@click.option(
    "--residue-analysis/--no-residue-analysis",
    "residue_analysis",
    default=None,
    help="DCCM, per-residue state differences and (with --lig) ligand-contact persistence (residues/).",
)
@click.option(
    "--ligand-selection",
    type=str,
    default=None,
    help="MDAnalysis selection of the ligand for the contact analysis (default: the atoms of --lig).",
)
@click.option(
    "--contact-cutoff",
    type=float,
    default=None,
    help="Heavy-atom contact cutoff in Å (default 4.5).",
)
@click.option(
    "--dihedral-set",
    type=click.Choice(["backbone", "backbone+chi", "chi", "nucleic"]),
    default=None,
    help="Torsions used by --pca-type dihedral (default backbone phi/psi).",
)
@_style_options
@click.pass_context
def analyze(ctx, config_file, **options):
    """PCA / free-energy-landscape analysis of one system (single or multi-replicate)."""
    from conformatlas.config import build_analysis_config
    from conformatlas.pipeline import run_analysis

    options = apply_style_options(options)
    try:
        cfg = build_analysis_config(options, config_file=config_file)
    except yaml.YAMLError as e:
        fail(str(malformed_config_from_yaml_error(config_file, e)), 2)
    except FileNotFoundError as e:
        fail(
            f"{e}\n  Check the path (relative paths in a YAML are resolved against the YAML's directory).",
            2,
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    try:
        check_optional_dependencies(bool(cfg.pdf), cfg.interactive)
        if cfg.save_hdf5:
            from conformatlas.results_io import HAS_H5PY

            if not HAS_H5PY:
                raise MissingDependencyError(
                    "h5py", "pip install h5py", "the HDF5 results container (--save-hdf5)"
                )
        if cfg.from_results is None:
            check_groups(cfg)
    except ConformAtlasError as e:
        fail(str(e), e.exit_code)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir)
    maybe_banner(ctx)
    echo(f"{CYAN}ConformAtlas {conformatlas.__version__} — analysis pipeline{NC}")
    if cfg.from_results is not None:
        echo(f"  • Stored results: {cfg.from_results}")
    else:
        echo(f"  • Topology:     {cfg.topology}")
        echo(f"  • Trajectories: {len(cfg.trajectories)} file(s)")
    echo(f"  • Temperature:  {cfg.temperature:.1f} K")
    echo(f"  • Output:       {out_dir.resolve()}")

    def progress(msg: str) -> None:
        if msg.startswith("Step") or msg.startswith("Report"):
            echo(f"\n{CYAN}{msg}{NC}")
        elif "WARNING" in msg or "NOTE" in msg:
            echo(f"{YELLOW}{msg}{NC}")
        else:
            echo(msg)

    try:
        results = run_analysis(cfg, echo=progress)
    except ConformAtlasError as e:
        _log_failure("Analysis failed")
        fail(str(e), e.exit_code)
    except ValueError as e:
        _log_failure("Analysis failed")
        bad = selection_error_group(e)
        if bad is not None:
            names, source = available_groups(cfg.topology, cfg.index_file)
            fail(str(UnknownGroupError(bad, names, source)), 2)
        fail(str(e), 1)
    except MemoryError:
        _log_failure("Analysis failed")
        fail("Out of memory. Use --stride, a smaller group (-c), or -b/-e to reduce the frames.", 1)
    except Exception as e:
        _log_failure("Analysis failed")
        fail(f"{e}\n  Details are in the log file in {out_dir.resolve()}.", 1)
    echo(f"\n{GREEN}✔ Analysis complete: {results.output_dir / 'report.html'}{NC}")
    if results.warnings:
        echo(f"{YELLOW}{len(results.warnings)} warning(s) recorded in run_metadata.json{NC}")


@main.command(name="compare")
@click.option(
    "-c",
    "--config",
    "config_file",
    type=click.Path(exists=True),
    required=True,
    help="YAML configuration file for comparison.",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Override output directory.",
)
@click.option(
    "--interactive/--no-interactive",
    "interactive",
    default=None,
    help="Write comparison_interactive.html (side-by-side Plotly landscapes; default when plotly is installed).",
)
@click.option(
    "--bootstrap-mode",
    type=click.Choice(["fixed", "resegment"]),
    default=None,
    help="'resegment' re-runs the watershed on every pooled-landscape resample; default fixed (YAML: bootstrap_mode).",
)
@_style_options
@click.pass_context
def compare(ctx, config_file, output_dir, **options):
    """Compare WT and Mutant systems in a shared PCA coordinate space."""
    apply_style_options(options)
    try:
        comp_config = parse_comparison_config(config_file)
    except yaml.YAMLError as e:
        fail(str(malformed_config_from_yaml_error(config_file, e)), 2)
    except FileNotFoundError as e:
        fail(str(e), 2)
    except (ValueError, KeyError, TypeError) as e:
        raise click.UsageError(f"Invalid comparison configuration: {e}") from e
    if output_dir:
        comp_config.output_dir = Path(output_dir)
    for key in (
        "fig_format",
        "colormap",
        "font_scale",
        "fig_dpi",
        "interactive",
        "plotly_js",
        "bootstrap_mode",
    ):
        if options.get(key) is not None:
            setattr(comp_config, key, options[key])
    try:
        check_optional_dependencies(False, comp_config.interactive)
        comp_config.__post_init__()
    except ConformAtlasError as e:
        fail(str(e), e.exit_code)
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    _setup_logging(comp_config.output_dir)
    maybe_banner(ctx)
    echo(f"{CYAN}Starting Condition Comparison for project: {comp_config.project}{NC}")

    try:
        res = run_condition_comparison(comp_config)
    except ConformAtlasError as e:
        _log_failure("Comparison failed")
        fail(str(e), e.exit_code)
    except ValueError as e:
        _log_failure("Comparison failed")
        bad = selection_error_group(e)
        if bad is not None:
            first_top = comp_config.systems[0].topology if comp_config.systems else None
            names, source = available_groups(first_top, comp_config.index_file)
            fail(str(UnknownGroupError(bad, names, source)), 2)
        fail(str(e), 1)
    except Exception as e:
        _log_failure("Comparison failed")
        fail(f"{e}\n  Details are in the log file in {comp_config.output_dir.resolve()}.", 1)
    fig_dir = comp_config.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_comparison_shared_pca(res, fig_dir / "comparison_shared_pca.png")
    plot_comparison_populations(res, fig_dir / "comparison_populations.png")
    for s_name in res.get("system_fels", {}):
        plot_comparison_system_fel(res, s_name, fig_dir / f"comparison_fel_{s_name}.png")
    plot_condition_comparison(res, fig_dir / "condition_comparison.png")
    from conformatlas.plotting_extra import plot_ddg_map

    ref_name = res["reference_system"]
    global_fel = res["global_fel"]
    for s_name, dd in res.get("ddg_maps", {}).items():
        plot_ddg_map(
            global_fel.x_centers,
            global_fel.y_centers,
            dd["ddg"],
            ref_name,
            s_name,
            fig_dir / f"ddg_map_{s_name}_vs_{ref_name}.png",
            sd=dd.get("sd"),
        )
    from conformatlas.report import get_report_options

    if get_report_options().interactive:
        from conformatlas.plotting_interactive import (
            build_comparison_figure,
            plotly_available,
            write_html,
        )

        if plotly_available():
            # states are defined on the pooled landscape, so they are shown on every panel
            fels = dict(res.get("system_fels", {}))
            if fels:
                shared_states = list(res.get("states", []))
                fig = build_comparison_figure(
                    fels, states_by_system={name: shared_states for name in fels}
                )
                write_html(
                    fig,
                    comp_config.output_dir / "comparison_interactive.html",
                    f"ConformAtlas comparison — {comp_config.project}",
                    plotly_js_mode=get_report_options().plotly_js,
                )
    ov = res.get("landscape_overlap")
    if ov is not None and len(ov):
        for _, r in ov.iterrows():
            echo(
                f"  Landscape overlap {r['System_A']} vs {r['System_B']}: "
                f"Bhattacharyya = {r['Bhattacharyya_Coefficient']:.3f}, "
                f"JS divergence = {r['JS_Divergence_bits']:.3f} bits"
            )
    sub = res.get("subspace_overlap")
    if sub is not None:
        names = list(sub.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                echo(f"  Essential-subspace RMSIP {names[i]} vs {names[j]}: {sub.iloc[i, j]:.3f}")
    echo(f"{GREEN}✔ Comparison complete! Results in: {comp_config.output_dir.resolve()}{NC}")


@main.command(name="doctor")
def doctor():
    """Check environment, dependencies, and external software versions."""
    click.echo(f"{BOLD}ConformAtlas System Diagnostics{NC}\n" + "=" * 40)
    click.echo(f"ConformAtlas Version: {conformatlas.__version__}")
    click.echo(f"Python Version:     {sys.version.split()[0]} ({sys.executable})")

    # GROMACS
    click.echo("\nChecking GROMACS installation:")
    try:
        runner = GromacsRunner()
        click.echo(f"  {GREEN}✔ Found GROMACS:{NC} {runner.gmx_path} ({runner.version})")
    except Exception as e:
        click.echo(f"  {RED}✘ GROMACS not detected:{NC} {e}")

    # Python dependencies
    deps = [
        ("numpy", True),
        ("scipy", True),
        ("pandas", True),
        ("matplotlib", True),
        ("click", True),
        ("jinja2", True),
        ("yaml", True),
        ("skimage", True),
        ("MDAnalysis", True),
        ("weasyprint", False),
        ("plotly", False),
        ("h5py", False),
        ("rich", False),
    ]
    purposes = {
        "MDAnalysis": "trajectory reading, PCA backend, descriptors, residue analyses",
        "weasyprint": "report.pdf (--pdf)",
        "plotly": "interactive report figures (--interactive)",
        "h5py": "HDF5 results container (--save-hdf5)",
        "rich": "progress bars on a terminal",
    }

    import importlib.metadata

    click.echo("\nChecking Python libraries:")
    for mod_name, required in deps:
        req_str = "Required" if required else "Optional"
        try:
            __import__(mod_name)
            # Try to get distribution version via importlib.metadata
            pkg_name = "scikit-image" if mod_name == "skimage" else mod_name
            try:
                ver = importlib.metadata.version(pkg_name)
            except Exception:
                m = sys.modules.get(mod_name)
                ver = getattr(m, "__version__", "installed")
            extra = f" — {purposes[mod_name]}" if mod_name in purposes else ""
            click.echo(f"  {GREEN}✔ {mod_name:<12}{NC} [{req_str}]: version {ver}{extra}")
        except ImportError:
            status = f"{RED}✘ Missing" if required else f"{YELLOW}○ Not installed"
            extra = f" — {purposes[mod_name]}" if mod_name in purposes else ""
            click.echo(f"  {status} {mod_name:<12}{NC} [{req_str}]{extra}")


# ----------------------------------------------------------------------------
# reweight: biased simulations -> per-frame weights for `analyze --weights-file`
# ----------------------------------------------------------------------------


def _read_times_file(path: str | Path) -> "np.ndarray":  # noqa: F821 - numpy imported lazily
    """First numeric column of a .xvg/.dat/.csv file (comment lines skipped)."""
    import numpy as np

    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.strip()
            if not t or t[0] in "#@;":
                continue
            try:
                rows.append(float(t.replace(",", " ").split()[0]))
            except ValueError:
                continue
    if not rows:
        raise click.UsageError(f"No numeric data in {path}")
    return np.asarray(rows, dtype=float)


@main.group(name="reweight")
def reweight():
    """Per-frame weights from biased sampling (metadynamics, umbrella sampling).

    The weight files are consumed unchanged by ``conformatlas analyze --weights-file``
    (one per trajectory; frame counts must match the analysed frames, hence
    ``--frame-times`` or a COLVAR stride equal to the trajectory output stride).
    """


@reweight.command(name="metad")
@click.option(
    "--hills",
    "hills_file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="PLUMED HILLS file.",
)
@click.option(
    "--colvar",
    "colvar_files",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    required=True,
    help="PLUMED COLVAR file(s), one per trajectory (repeatable).",
)
@click.option(
    "--cv",
    "cvs",
    type=str,
    multiple=True,
    required=True,
    help="Biased CV column name(s) in the COLVAR file.",
)
@click.option("-T", "--temperature", type=float, required=True, help="Temperature in Kelvin.")
@click.option(
    "--method",
    type=click.Choice(["tiwary-parrinello", "final-bias"]),
    default="tiwary-parrinello",
    show_default=True,
    help="Time-dependent c(t) reweighting (Tiwary & Parrinello 2015) or static end-of-run bias.",
)
@click.option(
    "--skip-time",
    type=float,
    default=0.0,
    show_default=True,
    help="Frames before this time (ps) get weight 0.",
)
@click.option(
    "--grid-bins",
    type=int,
    default=200,
    show_default=True,
    help="Grid points per CV for the c(t) integrals.",
)
@click.option(
    "--hill-shape",
    type=click.Choice(["plumed", "gaussian"]),
    default="plumed",
    show_default=True,
    help="PLUMED >= 2.5 stretched Gaussian or a plain Gaussian (HILLS from PLUMED < 2.5).",
)
@click.option(
    "--frame-times",
    "frame_times_files",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    help="File(s) with the trajectory frame times (ps), one per COLVAR, to pick the matching COLVAR rows.",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False),
    default="reweight",
    show_default=True,
)
@click.pass_context
def reweight_metad(
    ctx,
    hills_file,
    colvar_files,
    cvs,
    temperature,
    method,
    skip_time,
    grid_bins,
    hill_shape,
    frame_times_files,
    output_dir,
):
    """Metadynamics reweighting from HILLS + COLVAR."""
    import numpy as np

    from conformatlas.reweighting import (
        default_grid,
        hills_bias,
        match_times,
        metad_weights,
        pmf_1d,
        read_colvar,
        read_hills,
        write_weights_file,
    )
    from conformatlas.utils import save_json

    if temperature <= 0:
        raise click.UsageError("Temperature must be positive Kelvin.")
    if frame_times_files and len(frame_times_files) != len(colvar_files):
        raise click.UsageError("--frame-times must be given once per --colvar.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    maybe_banner(ctx)
    try:
        hills = read_hills(hills_file)
    except Exception as e:
        fail(f"Cannot read HILLS file {hills_file}: {e}", 2)
    echo(f"{CYAN}Metadynamics reweighting ({method}){NC}")
    echo(
        f"  • {len(hills.time)} hills on {', '.join(hills.cv_names)}; bias factor "
        + (f"{float(np.max(hills.biasf)):.1f}" if np.any(hills.biasf > 1) else "none (untempered)")
    )
    summary = {
        "method": method,
        "temperature_K": temperature,
        "hills": str(Path(hills_file).resolve()),
        "cvs": list(cvs),
        "skip_time_ps": skip_time,
        "grid_bins": grid_bins,
        "hill_shape": hill_shape,
        "files": [],
    }
    all_vals, all_w = [], []
    for k, cpath in enumerate(colvar_files):
        try:
            vals, t, _tab = read_colvar(cpath, list(cvs))
        except Exception as e:
            fail(f"Cannot read COLVAR file {cpath}: {e}", 2)
        if t is None:
            fail(f"{cpath}: no 'time' column; metadynamics reweighting needs frame times.", 2)
        try:
            res = metad_weights(
                vals,
                t,
                hills,
                temperature,
                method=method,
                skip_time=skip_time,
                bins=grid_bins,
                hill_shape=hill_shape,
            )
        except Exception as e:
            _log_failure("Reweighting failed")
            fail(str(e), 1)
        name = Path(cpath).stem
        times_out, w_out = res.times, res.weights
        if frame_times_files:
            ft = _read_times_file(frame_times_files[k])
            try:
                idx = match_times(ft, t)
            except Exception as e:
                fail(f"{frame_times_files[k]}: {e}", 2)
            times_out, w_out = ft, res.weights[idx]
            kept = w_out > 0
            if kept.any():
                w_out = np.where(kept, w_out / w_out[kept].mean(), 0.0)
        wpath = write_weights_file(
            out / f"weights_{name}.dat",
            times_out,
            w_out,
            header=f"{method} metadynamics weights from {Path(hills_file).name} / {Path(cpath).name}; "
            f"T = {temperature} K; skip_time = {skip_time} ps (mean 1 over kept frames)",
        )
        n_kept = int(np.sum(w_out > 0))
        w_kept = w_out[w_out > 0]
        n_eff_w = float(w_kept.sum() ** 2 / np.sum(w_kept**2)) if n_kept else 0.0
        echo(
            f"  • {Path(cpath).name}: {n_kept} frames weighted (Kish N_eff = {n_eff_w:.0f}) -> {wpath}"
        )
        summary["files"].append(
            {
                "colvar": str(Path(cpath).resolve()),
                "weights": str(wpath),
                "frames_kept": n_kept,
                "kish_n_eff": n_eff_w,
                "bias_factor": res.bias_factor,
            }
        )
        if res.hill_times is not None and res.rct_history is not None and k == 0:
            import pandas as pd

            pd.DataFrame({"hill_time_ps": res.hill_times, "c_t_kJ_mol": res.rct_history}).to_csv(
                out / "rct.csv", index=False
            )
        all_vals.append(vals[res.kept, 0])
        all_w.append(res.weights[res.kept])
    # final bias on the default grid (first CV when 1-D)
    try:
        axes = default_grid(hills, bins=grid_bins)
        if len(axes) == 1:
            import pandas as pd

            pts = axes[0][:, None]
            v = hills_bias(hills, pts, hill_shape=hill_shape)
            pd.DataFrame({hills.cv_names[0]: axes[0], "bias_kJ_mol": v}).to_csv(
                out / "bias_final.csv", index=False
            )
    except Exception as e:  # pragma: no cover - diagnostic output only
        logger.warning(f"bias_final.csv not written: {e}")
    if all_vals:
        v = np.concatenate(all_vals)
        w = np.concatenate(all_w)
        if len(v) and w.sum() > 0:
            pmf_1d(v, w, temperature, bins=min(100, max(10, len(v) // 20))).to_csv(
                out / "pmf.csv", index=False
            )
            summary["pmf"] = str(out / "pmf.csv")
    save_json(summary, out / "reweight_summary.json")
    echo(f"{GREEN}✔ Weights written to {out.resolve()}{NC}")


@reweight.command(name="umbrella")
@click.option(
    "--colvar",
    "colvar_files",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    required=True,
    help="One COLVAR file per umbrella window (repeatable, in the order of --center).",
)
@click.option("--cv", "cv", type=str, required=True, help="Restrained CV column name.")
@click.option(
    "--center",
    "centers",
    type=float,
    multiple=True,
    required=True,
    help="Restraint centre per window.",
)
@click.option(
    "--kappa",
    "kappas",
    type=float,
    multiple=True,
    required=True,
    help="Force constant kappa (kJ/mol per CV unit^2, V = 1/2 kappa (x-x0)^2); one value or one per window.",
)
@click.option("-T", "--temperature", type=float, required=True, help="Temperature in Kelvin.")
@click.option("--method", type=click.Choice(["mbar", "wham"]), default="mbar", show_default=True)
@click.option("--bins", type=int, default=100, show_default=True, help="PMF bins.")
@click.option(
    "--range",
    "value_range",
    type=(float, float),
    default=None,
    help="PMF range lo hi (default: data range).",
)
@click.option(
    "--skip-time",
    type=float,
    default=0.0,
    show_default=True,
    help="Discard COLVAR rows before this time (ps).",
)
@click.option(
    "--period",
    type=float,
    default=None,
    help="Period of a periodic CV (e.g. 6.283185 for radians).",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False),
    default="reweight",
    show_default=True,
)
@click.pass_context
def reweight_umbrella(
    ctx,
    colvar_files,
    cv,
    centers,
    kappas,
    temperature,
    method,
    bins,
    value_range,
    skip_time,
    period,
    output_dir,
):
    """Umbrella-sampling weights and PMF by MBAR or WHAM."""

    from conformatlas.reweighting import (
        mbar,
        umbrella_windows_from_colvars,
        wham,
        write_umbrella_outputs,
    )
    from conformatlas.utils import save_json

    if temperature <= 0:
        raise click.UsageError("Temperature must be positive Kelvin.")
    if len(centers) != len(colvar_files):
        raise click.UsageError("--center must be given once per --colvar.")
    if len(kappas) not in (1, len(colvar_files)):
        raise click.UsageError("--kappa must be given once or once per window.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    maybe_banner(ctx)
    kap = list(kappas) if len(kappas) > 1 else float(kappas[0])
    try:
        windows, times = umbrella_windows_from_colvars(
            list(colvar_files), cv, list(centers), kap, skip_time=skip_time, period=period
        )
    except Exception as e:
        fail(f"Cannot build umbrella windows: {e}", 2)
    echo(f"{CYAN}Umbrella sampling: {len(windows)} windows, {method.upper()}{NC}")
    try:
        if method == "mbar":
            res = mbar(windows, temperature, bins=bins, value_range=value_range)
        else:
            res = wham(windows, temperature, bins=bins, value_range=value_range)
    except Exception as e:
        _log_failure("Reweighting failed")
        fail(str(e), 1)
    names = [Path(c).stem for c in colvar_files]
    written = write_umbrella_outputs(res, out, times=times, window_names=names)
    echo(f"  • converged: {res.converged} after {res.n_iter} iterations")
    for k, (n, f) in enumerate(zip(names, res.window_free_energies)):
        echo(
            f"  • {n}: centre {centers[k]:.4g}, f = {f:.3f} kJ/mol, {len(windows[k].samples)} samples"
        )
    summary = {
        "method": method,
        "temperature_K": temperature,
        "cv": cv,
        "centers": list(centers),
        "kappas": list(kappas),
        "period": period,
        "skip_time_ps": skip_time,
        "bins": bins,
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
        "window_free_energies_kJ_mol": [float(v) for v in res.window_free_energies],
        "files": {k: str(v) for k, v in written.items()},
    }
    save_json(summary, out / "reweight_summary.json")
    echo(f"{GREEN}✔ Weights and PMF written to {out.resolve()}{NC}")


@main.command(name="init-config")
@click.option(
    "-t",
    "--type",
    "config_type",
    type=click.Choice(["compare", "analyze"]),
    default="compare",
    show_default=True,
    help="Configuration template type.",
)
@click.option(
    "-o",
    "--output",
    "out_file",
    type=click.Path(),
    default=None,
    help="File to save template to. If omitted, prints to console.",
)
def init_config(config_type, out_file):
    """Generate a starter YAML configuration template."""
    content = generate_example_config(config_type)
    if out_file:
        p = Path(out_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"{GREEN}✔ Created {config_type} config template at {p.resolve()}{NC}")
    else:
        click.echo(content)


@main.command(name="example")
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(),
    default="example_project",
    show_default=True,
    help="Output directory to generate example files.",
)
def example(output_dir):
    """Generate a lightweight example dataset and comparison YAML configuration."""
    from conformatlas.examples_gen import generate_mini_example_dataset

    p = Path(output_dir)
    try:
        generate_mini_example_dataset(p)
    except ConformAtlasError as e:
        fail(str(e), e.exit_code)
    click.echo(f"{GREEN}✔ Example dataset created in: {p.resolve()}{NC}")
    click.echo("To test single analysis run:")
    click.echo(
        f"  conformatlas analyze -s {p}/WT/topology.pdb -f {p}/WT/rep1.xtc -T 300 -o {p}/results_wt"
    )
    click.echo("To test comparison run:")
    click.echo(f"  conformatlas compare --config {p}/comparison.yaml -o {p}/results_comparison")


if __name__ == "__main__":
    main()
