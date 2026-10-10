"""Command Line Interface (CLI) for ConformAtlas / ConformAtlas."""

import logging
import sys
from pathlib import Path

import click

import conformatlas
from conformatlas.comparison import run_condition_comparison
from conformatlas.config import generate_example_config, parse_comparison_config
from conformatlas.gromacs import GromacsRunner
from conformatlas.plotting import (
    plot_comparison_populations,
    plot_comparison_shared_pca,
    plot_comparison_system_fel,
    plot_condition_comparison,
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


class ConformAtlasCommand(click.Command):
    """Command subclass that prints the ConformAtlas banner before showing help."""

    def format_help(self, ctx, formatter):
        print_banner(ctx)
        super().format_help(ctx, formatter)


class ConformAtlasGroup(click.Group):
    """Group subclass that prints the ConformAtlas banner before showing help and commands."""

    command_class = ConformAtlasCommand

    def format_help(self, ctx, formatter):
        print_banner(ctx)
        super().format_help(ctx, formatter)


def print_version(ctx, param, value):
    """Version option callback that displays banner and version."""
    if not value or ctx.resilient_parsing:
        return
    print_banner(ctx)
    click.echo(f"conformatlas, version {conformatlas.__version__}")
    ctx.exit()


@click.group(
    cls=ConformAtlasGroup,
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
@click.pass_context
def main(ctx, structure, trajectory, temperature, output):
    """ConformAtlas: MD Ensemble, PCA, and Free Energy Landscape Analysis."""
    print_banner(ctx)
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


@main.command(name="analyze", cls=ConformAtlasCommand)
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
    help="auto: MDAnalysis when installed, else gmx covar/anaeig.",
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
@click.pass_context
def analyze(ctx, config_file, **options):
    """PCA / free-energy-landscape analysis of one system (single or multi-replicate)."""
    from conformatlas.config import build_analysis_config
    from conformatlas.pipeline import run_analysis

    try:
        cfg = build_analysis_config(options, config_file=config_file)
    except (ValueError, FileNotFoundError) as e:
        raise click.UsageError(str(e)) from e

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    print_banner(ctx)
    click.echo(f"{CYAN}ConformAtlas {conformatlas.__version__} — analysis pipeline{NC}")
    click.echo(f"  • Topology:     {cfg.topology}")
    click.echo(f"  • Trajectories: {len(cfg.trajectories)} file(s)")
    click.echo(f"  • Temperature:  {cfg.temperature:.1f} K")
    click.echo(f"  • Output:       {out_dir.resolve()}")

    def echo(msg: str) -> None:
        if msg.startswith("Step") or msg.startswith("Report"):
            click.echo(f"\n{CYAN}{msg}{NC}")
        elif "WARNING" in msg or "NOTE" in msg:
            click.echo(f"{YELLOW}{msg}{NC}")
        else:
            click.echo(msg)

    try:
        results = run_analysis(cfg, echo=echo)
    except Exception as e:
        logging.getLogger("conformatlas").exception("Analysis failed")
        click.echo(f"{RED}Error: {e}{NC}", err=True)
        sys.exit(1)
    click.echo(f"\n{GREEN}✔ Analysis complete: {results.output_dir / 'report.html'}{NC}")
    if results.warnings:
        click.echo(f"{YELLOW}{len(results.warnings)} warning(s) recorded in run_metadata.json{NC}")


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
def compare(config_file, output_dir):
    """Compare WT and Mutant systems in a shared PCA coordinate space."""
    comp_config = parse_comparison_config(config_file)
    if output_dir:
        comp_config.output_dir = Path(output_dir)

    setup_logging(comp_config.output_dir)
    click.echo(f"{CYAN}Starting Condition Comparison for project: {comp_config.project}{NC}")

    res = run_condition_comparison(comp_config)
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
    ov = res.get("landscape_overlap")
    if ov is not None and len(ov):
        for _, r in ov.iterrows():
            click.echo(
                f"  Landscape overlap {r['System_A']} vs {r['System_B']}: "
                f"Bhattacharyya = {r['Bhattacharyya_Coefficient']:.3f}, "
                f"JS divergence = {r['JS_Divergence_bits']:.3f} bits"
            )
    sub = res.get("subspace_overlap")
    if sub is not None:
        names = list(sub.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                click.echo(
                    f"  Essential-subspace RMSIP {names[i]} vs {names[j]}: {sub.iloc[i, j]:.3f}"
                )
    click.echo(f"{GREEN}✔ Comparison complete! Results in: {comp_config.output_dir.resolve()}{NC}")


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
        ("MDAnalysis", False),
        ("weasyprint", False),
    ]

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
            click.echo(f"  {GREEN}✔ {mod_name:<12}{NC} [{req_str}]: version {ver}")
        except ImportError:
            status = f"{RED}✘ Missing" if required else f"{YELLOW}○ Not installed"
            click.echo(f"  {status:<12}{NC} [{req_str}]")


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
    generate_mini_example_dataset(p)
    click.echo(f"{GREEN}✔ Example dataset created in: {p.resolve()}{NC}")
    click.echo("To test single analysis run:")
    click.echo(
        f"  conformatlas analyze -s {p}/WT/topology.pdb -f {p}/WT/rep1.xtc -T 300 -o {p}/results_wt"
    )
    click.echo("To test comparison run:")
    click.echo(f"  conformatlas compare --config {p}/comparison.yaml -o {p}/results_comparison")


if __name__ == "__main__":
    main()
