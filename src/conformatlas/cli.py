"""Command Line Interface (CLI) for ConformAtlas / ConformAtlas."""

import datetime
import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

import conformatlas
from conformatlas.basins import detect_energy_basins, save_basin_results
from conformatlas.comparison import run_condition_comparison
from conformatlas.config import generate_example_config, parse_comparison_config
from conformatlas.convergence import (
    compute_convergence_diagnostics,
    save_convergence_results,
)
from conformatlas.fel import calculate_fel, save_fel_grid
from conformatlas.gromacs import GromacsRunner
from conformatlas.models import AnalysisConfig, AnalysisResults, TrajectoryInfo
from conformatlas.pca import (
    PCAResults,
    compute_cartesian_pca,
    project_coordinates,
    run_gromacs_pca,
    save_pca_results,
)
from conformatlas.plotting import (
    plot_basin_map,
    plot_comparison_populations,
    plot_comparison_shared_pca,
    plot_comparison_system_fel,
    plot_condition_comparison,
    plot_convergence_cosine,
    plot_convergence_diagnostics,
    plot_convergence_progressive,
    plot_convergence_stability,
    plot_eigenvalues_and_variance,
    plot_fel,
    plot_fel_2d,
    plot_fel_3d,
    plot_pca_density,
    plot_pca_projections,
    plot_pca_scatter,
    plot_state_populations,
)
from conformatlas.report import generate_html_report
from conformatlas.representatives import extract_representative_structures
from conformatlas.trajectory import (
    HAS_MDANALYSIS,
    inspect_trajectory,
    stream_trajectory_coordinates,
)
from conformatlas.uncertainty import (
    compute_replicate_statistics,
    save_uncertainty_results,
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


@main.command(name="analyze")
@click.option(
    "-s",
    "--structure",
    "--topology",
    "structure",
    type=click.Path(exists=True),
    required=True,
    help="Topology/structure file (.tpr, .gro, .pdb).",
)
@click.option(
    "-f",
    "--trajectory",
    "trajectory",
    type=click.Path(exists=True),
    multiple=True,
    required=True,
    help="Trajectory file (.xtc). Pass multiple times for replicates: -f rep1.xtc -f rep2.xtc.",
)
@click.option(
    "-T",
    "--temperature",
    type=float,
    default=300.0,
    show_default=True,
    help="Simulation temperature in Kelvin (must be > 0).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="Outputs",
    show_default=True,
    help="Output directory for results and figures.",
)
@click.option(
    "-l",
    "--lsq-group",
    default="Backbone",
    show_default=True,
    help="Least squares alignment group (e.g. Backbone, C-alpha, Protein).",
)
@click.option(
    "-c",
    "--cov-group",
    default="Backbone",
    show_default=True,
    help="Covariance / PCA analysis group.",
)
@click.option(
    "-g",
    "--lig",
    "lig_group",
    default=None,
    help="Optional ligand group name (e.g. LIG, ATP).",
)
@click.option(
    "--bins",
    type=int,
    default=64,
    show_default=True,
    help="Number of 2D FEL histogram bins along PC1 and PC2.",
)
@click.option(
    "--smooth-sigma",
    type=float,
    default=0.0,
    show_default=True,
    help="Gaussian smoothing standard deviation (0.0 for raw density).",
)
@click.option(
    "--pca-weighting",
    type=click.Choice(["equal-replicate", "frames"]),
    default="equal-replicate",
    show_default=True,
    help="PCA weighting strategy for multi-replicate trajectories.",
)
@click.option(
    "--stride",
    type=int,
    default=1,
    show_default=True,
    help="Frame stride for streaming trajectory processing.",
)
@click.option(
    "--min-state-population",
    type=float,
    default=0.05,
    show_default=True,
    help="Minimum fraction threshold for major conformational states (0.05 = 5%).",
)
@click.option(
    "--max-states",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of major states to segment.",
)
@click.option(
    "--fel-backend",
    type=click.Choice(["python", "gromacs"]),
    default="python",
    show_default=True,
    help="Free Energy Landscape calculation backend.",
)
@click.option(
    "--gmx-command",
    type=str,
    default=None,
    help="Path to custom GROMACS executable.",
)
def analyze(
    structure,
    trajectory,
    temperature,
    output,
    lsq_group,
    cov_group,
    lig_group,
    bins,
    smooth_sigma,
    pca_weighting,
    stride,
    min_state_population,
    max_states,
    fel_backend,
    gmx_command,
):
    """Perform PCA and Free Energy Landscape analysis on single or multi-replicate MD trajectories."""
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    click.echo(f"{CYAN}Initializing ConformAtlas Analysis Pipeline{NC}")
    click.echo(f"  • Topology:    {structure}")
    click.echo(f"  • Trajectories: {len(trajectory)} file(s)")
    click.echo(f"  • Temperature: {temperature:.1f} K")
    click.echo(f"  • Output Dir:  {out_dir.resolve()}")

    # Initialize GROMACS runner if available
    gmx_runner = None
    gmx_version = "None"
    try:
        gmx_runner = GromacsRunner(gmx_command)
        gmx_version = gmx_runner.version
        click.echo(f"{GREEN}✔ GROMACS found: {gmx_runner.gmx_path} ({gmx_version}){NC}")
    except Exception as e:
        click.echo(f"{YELLOW}Notice: GROMACS not available directly: {e}{NC}")
        if fel_backend == "gromacs":
            click.echo(
                f"{RED}Error: GROMACS backend requested but GROMACS executable is unavailable.{NC}"
            )
            sys.exit(1)

    # 1. Inspect trajectories
    traj_infos: list[TrajectoryInfo] = []
    traj_map: dict = {}
    for idx, t_path in enumerate(trajectory):
        rep_id = f"Rep_{idx + 1}"
        t_path = Path(t_path)
        traj_map[rep_id] = t_path
        info = inspect_trajectory(structure, t_path, rep_id, "System", gmx_runner=gmx_runner)
        traj_infos.append(info)
        click.echo(
            f"  • {rep_id}: {info.n_frames} frames ({info.start_time_ps:.1f} to {info.end_time_ps:.1f} ps)"
        )

    # 2. PCA calculation
    click.echo(f"\n{CYAN}Step 1: Computing Principal Component Analysis...{NC}")
    coords_dict = {}
    system_times = {}

    if HAS_MDANALYSIS:
        for idx, t_path in enumerate(trajectory):
            rep_id = f"Rep_{idx + 1}"
            coords, times = stream_trajectory_coordinates(
                structure, t_path, group=cov_group, stride=stride
            )
            coords_dict[rep_id] = coords
            system_times[rep_id] = times

        evals, evecs, exp_var, cum_var, mean_struct = compute_cartesian_pca(
            coords_dict, weighting=pca_weighting, n_components=10
        )

        proj_dfs = []
        for rep_id, coords in coords_dict.items():
            proj_mat = project_coordinates(coords, mean_struct, evecs)
            t_vals = system_times[rep_id]
            df = pd.DataFrame(proj_mat, columns=[f"PC{i + 1}" for i in range(proj_mat.shape[1])])
            df.insert(0, "system", "System")
            df.insert(1, "replicate", rep_id)
            df.insert(2, "frame", np.arange(len(t_vals)))
            df.insert(3, "time_ps", t_vals)
            proj_dfs.append(df)
        projections = pd.concat(proj_dfs, ignore_index=True)

        pca_results = PCAResults(
            eigenvalues=evals,
            eigenvectors=evecs,
            explained_variance=exp_var,
            cumulative_variance=cum_var,
            projections=projections,
            mean_coords=mean_struct,
        )
    else:
        if gmx_runner is None:
            raise RuntimeError("Either MDAnalysis or GROMACS must be installed.")
        pca_results = run_gromacs_pca(
            gmx_runner,
            Path(structure),
            [Path(p) for p in trajectory],
            lsq_group=lsq_group,
            cov_group=cov_group,
            output_dir=out_dir,
            n_components=10,
        )

    save_pca_results(pca_results, out_dir)
    top2_var = (
        pca_results.cumulative_variance[1]
        if len(pca_results.cumulative_variance) > 1
        else (
            pca_results.cumulative_variance[0] if len(pca_results.cumulative_variance) > 0 else 0.0
        )
    )
    click.echo(f"{GREEN}✔ PCA completed: Dominant modes explain {top2_var:.1f}% of variance.{NC}")

    # 3. FEL calculation
    click.echo(
        f"\n{CYAN}Step 2: Calculating Free Energy Landscape (T = {temperature:.1f} K)...{NC}"
    )
    pc1 = pca_results.projections["PC1"].to_numpy()
    pc2 = pca_results.projections["PC2"].to_numpy()

    fel_grid = calculate_fel(
        pc1,
        pc2,
        temperature=temperature,
        bins=bins,
        smooth_sigma=smooth_sigma,
    )
    save_fel_grid(fel_grid, out_dir)
    click.echo(f"{GREEN}✔ FEL calculated on {bins}x{bins} grid.{NC}")

    # 4. Basin detection & states
    click.echo(f"\n{CYAN}Step 3: Detecting Conformational Basins & Energy Minima...{NC}")
    states, unassigned_pop, frame_assignments, basin_map = detect_energy_basins(
        fel_grid,
        pca_results.projections,
        min_state_population=min_state_population,
        max_states=max_states,
    )
    save_basin_results(states, frame_assignments, basin_map, out_dir)

    for st in states:
        click.echo(
            f"  • {BOLD}{st.label}{NC}: {st.population * 100:.1f}% pop | "
            f"Minima at PC1={st.min_pc1:.2f}, PC2={st.min_pc2:.2f} (ΔG = {st.min_free_energy:.2f} kJ/mol) | "
            f"Rep frame {st.representative_frame} ({st.representative_time_ps:.1f} ps)"
        )
    if unassigned_pop > 0:
        click.echo(f"  • Minor / Transitional Conformations: {unassigned_pop * 100:.1f}%")

    # 5. Extract representatives
    click.echo(f"\n{CYAN}Step 4: Extracting Representative PDB Structures...{NC}")
    extract_representative_structures(
        states,
        {"System": Path(structure)},
        traj_map,
        output_dir=out_dir,
        gmx_runner=gmx_runner,
        system_name="System",
    )
    click.echo(f"{GREEN}✔ Representative PDBs saved to states/ directory.{NC}")

    # 6. Convergence diagnostics
    click.echo(f"\n{CYAN}Step 5: Assessing Trajectory Sampling & Convergence...{NC}")
    if coords_dict:
        conv = compute_convergence_diagnostics(
            coords_dict, pca_results.projections, frame_assignments, n_modes=10
        )
    else:
        # Fallback dummy for non-coords mode
        conv = compute_convergence_diagnostics(
            {"Rep_1": np.zeros((len(projections), 1, 3))},
            pca_results.projections,
            frame_assignments,
            n_modes=2,
        )
    save_convergence_results(conv, out_dir)
    click.echo(f"  • Subspace Overlap (RMSIP 1st vs 2nd half): {conv.rmsip_halves:.2f}")
    click.echo(f"  • PC1 Cosine Content: {conv.cosine_content.get(1, 0.0):.2f}")
    click.echo(f"  • Sampling Assessment: {BOLD}{conv.sampling_assessment}{NC}")

    # 7. Uncertainty
    click.echo(f"\n{CYAN}Step 6: Calculating Replicate / Block Uncertainty...{NC}")
    summary_pop_df, detailed_pop_df, unc_type = compute_replicate_statistics(frame_assignments)
    save_uncertainty_results(summary_pop_df, detailed_pop_df, out_dir)
    click.echo(f"{GREEN}✔ Uncertainty estimated ({unc_type}).{NC}")

    # 8. Publication plots
    click.echo(f"\n{CYAN}Step 7: Generating Publication Figures...{NC}")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Standalone separate publication figures
    fig_pca_var = plot_eigenvalues_and_variance(pca_results, fig_dir / "explained_variance.png")
    fig_pca_scatter = plot_pca_scatter(pca_results.projections, fig_dir / "pca_scatter.png")
    fig_pca_density = plot_pca_density(pca_results.projections, fig_dir / "pca_density.png")
    fig_fel_2d = plot_fel_2d(fel_grid, states=states, out_path=fig_dir / "fel_2d.png")
    fig_fel_3d = plot_fel_3d(fel_grid, out_path=fig_dir / "fel_3d.png")
    fig_basin = plot_basin_map(fel_grid, basin_map, states, fig_dir / "basin_map.png")
    fig_pop = plot_state_populations(
        summary_pop_df, detailed_pop_df, fig_dir / "state_populations.png"
    )
    fig_conv_prog = plot_convergence_progressive(conv, fig_dir / "convergence_progressive.png")
    fig_conv_cosine = plot_convergence_cosine(conv, fig_dir / "convergence_cosine.png")
    fig_conv_stability = plot_convergence_stability(conv, fig_dir / "convergence_stability.png")

    # Multi-panel overview figures (legacy compatibility)
    plot_fel(fel_grid, states=states, out_path=fig_dir / "fel_2d_3d.png")
    plot_pca_projections(pca_results.projections, fig_dir / "pca.png")
    fig_conv = plot_convergence_diagnostics(conv, fig_dir / "convergence.png")

    click.echo(f"{GREEN}✔ Figures saved to {fig_dir.resolve()}{NC}")

    # 9. HTML Report & Metadata
    click.echo(f"\n{CYAN}Step 8: Compiling Comprehensive HTML Report...{NC}")
    run_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "conformatlas_version": conformatlas.__version__,
        "temperature_K": temperature,
        "bins": bins,
        "smooth_sigma": smooth_sigma,
        "stride": stride,
        "pca_weighting": pca_weighting,
        "topology": str(Path(structure).resolve()),
        "trajectories": [str(Path(p).resolve()) for p in trajectory],
        "uncertainty_type": unc_type,
    }

    results = AnalysisResults(
        config=AnalysisConfig(
            topology=Path(structure),
            trajectories=[Path(p) for p in trajectory],
            temperature=temperature,
            output_dir=out_dir,
            bins=bins,
            smooth_sigma=smooth_sigma,
            stride=stride,
        ),
        trajectories_info=traj_infos,
        pca=pca_results,
        fel=fel_grid,
        states=states,
        unassigned_population=unassigned_pop,
        frame_assignments=frame_assignments,
        convergence=conv,
        replicate_statistics=summary_pop_df,
        output_dir=out_dir,
        run_metadata=run_meta,
    )

    figures_map = {
        "fel_2d": fig_fel_2d,
        "fel_3d": fig_fel_3d,
        "fel_states": fig_fel_2d,
        "basin_map": fig_basin,
        "populations": fig_pop,
        "pca_variance": fig_pca_var,
        "pca_scatter": fig_pca_scatter,
        "pca_density": fig_pca_density,
        "pca_projections": fig_pca_scatter,
        "convergence_progressive": fig_conv_prog,
        "convergence_cosine": fig_conv_cosine,
        "convergence_stability": fig_conv_stability,
        "convergence": fig_conv,
    }

    report_p = generate_html_report(results, figures_map, gromacs_version=gmx_version)
    click.echo(f"{GREEN}✔ Analysis Complete! Report generated: {report_p.resolve()}{NC}")


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
