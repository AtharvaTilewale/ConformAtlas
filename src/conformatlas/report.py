"""HTML report generation and metadata export."""

import base64
import logging
import platform
from pathlib import Path

import jinja2
import numpy as np
import scipy

import conformatlas
from conformatlas.models import AnalysisResults
from conformatlas.utils import save_json

logger = logging.getLogger("conformatlas.report")


def image_to_base64(image_path: str | Path) -> str:
    """Convert an image file to a base64 data URI string."""
    p = Path(image_path)
    if not p.exists():
        return ""
    with p.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = p.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else f"image/{ext}"
    return f"data:{mime};base64,{encoded}"


def generate_html_report(
    results: AnalysisResults,
    figures_dict: dict[str, Path],
    gromacs_version: str = "GROMACS 2024.4",
) -> Path:
    """Compile and render the standalone HTML analysis report."""
    out_dir = results.output_dir
    report_path = out_dir / "report.html"

    # Template loader
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    # Embed figures as base64 for self-contained portability
    embedded_figs = {}
    for key, fig_p in figures_dict.items():
        embedded_figs[key] = image_to_base64(fig_p)

    # Replicate population statistics records
    pop_stats_records = []
    if results.replicate_statistics is not None:
        pop_stats_records = results.replicate_statistics.to_dict(orient="records")

    total_frames = sum(t.n_frames for t in results.trajectories_info)
    major_pop = sum(s.population for s in results.states) * 100.0
    unassigned_pop = results.unassigned_population * 100.0

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
        "major_population_percent": f"{major_pop:.1f}",
        "unassigned_population_percent": f"{unassigned_pop:.1f}",
        "population_stats": pop_stats_records,
        "uncertainty_type": results.run_metadata.get("uncertainty_type", "replicate"),
        "convergence": results.convergence,
        "bins": results.config.bins,
        "fig_fel_2d": embedded_figs.get("fel_2d", embedded_figs.get("fel_states", "")),
        "fig_fel_3d": embedded_figs.get("fel_3d", ""),
        "fig_fel_states": embedded_figs.get("fel_states", embedded_figs.get("fel_2d", "")),
        "fig_basin_map": embedded_figs.get("basin_map", ""),
        "fig_populations": embedded_figs.get("populations", ""),
        "fig_pca_variance": embedded_figs.get("pca_variance", ""),
        "fig_pca_scatter": embedded_figs.get("pca_scatter", embedded_figs.get("pca_projections", "")),
        "fig_pca_density": embedded_figs.get("pca_density", ""),
        "fig_pca_projections": embedded_figs.get("pca_projections", embedded_figs.get("pca_scatter", "")),
        "fig_convergence_progressive": embedded_figs.get("convergence_progressive", ""),
        "fig_convergence_cosine": embedded_figs.get("convergence_cosine", ""),
        "fig_convergence_stability": embedded_figs.get("convergence_stability", ""),
        "fig_convergence": embedded_figs.get("convergence", ""),
    }

    html_content = template.render(**context)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML report generated at {report_path.resolve()}")

    # Export machine-readable metadata
    save_json(results.run_metadata, out_dir / "run_metadata.json")

    # Summary JSON
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
