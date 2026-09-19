"""Principal Component Analysis (PCA) engine with multi-replicate and shared basis support."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from conformatlas.gromacs import GromacsRunner
from conformatlas.models import PCAResults

logger = logging.getLogger("conformatlas.pca")


def compute_cartesian_pca(
    coords_dict: dict[str, np.ndarray],
    weighting: str = "equal-replicate",
    n_components: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Cartesian PCA on coordinates from one or more replicates/systems.

    Parameters
    ----------
    coords_dict : Dict[str, np.ndarray]
        Mapping from replicate_key to coordinate array of shape (n_frames, n_atoms, 3).
    weighting : str
        'equal-replicate' (each replicate receives equal total weight 1/K)
        or 'frames' (each frame receives equal weight 1/N).
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (eigenvalues, eigenvectors, explained_variance, cumulative_variance, mean_structure)
        where eigenvectors has shape (3*n_atoms, n_components).
    """
    replicate_keys = list(coords_dict.keys())
    if not replicate_keys:
        raise ValueError("Cannot compute PCA with empty coordinates dictionary.")

    # Flatten coordinates: (n_frames, 3 * n_atoms)
    flattened_reps: dict[str, np.ndarray] = {}
    n_dims = None
    for k in replicate_keys:
        arr = coords_dict[k]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Replicate {k} coordinates must have shape (n_frames, n_atoms, 3).")
        flat = arr.reshape(arr.shape[0], -1).astype(np.float64)
        if n_dims is None:
            n_dims = flat.shape[1]
        elif flat.shape[1] != n_dims:
            raise ValueError(
                f"Coordinate dimension mismatch in replicate {k}: {flat.shape[1]} vs {n_dims}"
            )
        flattened_reps[k] = flat

    total_frames = sum(flat.shape[0] for flat in flattened_reps.values())
    n_reps = len(replicate_keys)

    # Compute mean structure
    if weighting == "equal-replicate":
        mean_struct = np.zeros(n_dims, dtype=np.float64)
        for k in replicate_keys:
            mean_struct += np.mean(flattened_reps[k], axis=0) / n_reps
    else:  # 'frames'
        all_frames = np.vstack(list(flattened_reps.values()))
        mean_struct = np.mean(all_frames, axis=0)

    # Compute covariance matrix C = sum_k w_k * (Y_k^T Y_k)
    cov_matrix = np.zeros((n_dims, n_dims), dtype=np.float64)

    for k in replicate_keys:
        centered = flattened_reps[k] - mean_struct
        n_k = centered.shape[0]
        if n_k == 0:
            continue
        if weighting == "equal-replicate":
            w = 1.0 / (n_reps * n_k)
        else:
            w = 1.0 / total_frames
        cov_matrix += w * (centered.T @ centered)

    # Symmetrize to prevent numerical asymmetry
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

    # Eigen-decomposition: eigh returns ascending order
    evals, evecs = np.linalg.eigh(cov_matrix)

    # Reverse to descending order
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    # Handle negative numerical eigenvalues near zero
    evals = np.maximum(evals, 0.0)

    total_variance = np.sum(evals)
    if total_variance > 0:
        explained_var = 100.0 * (evals / total_variance)
    else:
        explained_var = np.zeros_like(evals)

    cum_var = np.cumsum(explained_var)

    # Truncate to n_components
    k_comp = min(n_components, len(evals))
    return (
        evals[:k_comp],
        evecs[:, :k_comp],
        explained_var[:k_comp],
        cum_var[:k_comp],
        mean_struct,
    )


def project_coordinates(
    coords: np.ndarray,
    mean_struct: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Project coordinate array (n_frames, n_atoms, 3) onto principal components."""
    flat = coords.reshape(coords.shape[0], -1).astype(np.float64)
    centered = flat - mean_struct
    # Projections: (n_frames, n_components)
    return centered @ eigenvectors


def run_gromacs_pca(
    gmx_runner: GromacsRunner,
    topology_path: Path,
    trajectory_paths: list[Path],
    lsq_group: str = "Backbone",
    cov_group: str = "Backbone",
    output_dir: Path = Path("Outputs"),
    n_components: int = 10,
) -> PCAResults:
    """Execute PCA via GROMACS covar and anaeig."""
    pca_dir = output_dir / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    eigenval_xvg = pca_dir / "eigenval.xvg"
    eigenvec_trr = pca_dir / "eigenvec.trr"
    average_pdb = pca_dir / "average.pdb"

    # Step 1: covar on the first/pooled trajectory
    input_groups = f"{lsq_group}\n{cov_group}\n"
    primary_traj = trajectory_paths[0]

    logger.info("Executing GROMACS covar...")
    gmx_runner.run(
        "covar",
        [
            "-s",
            str(topology_path.resolve()),
            "-f",
            str(primary_traj.resolve()),
            "-o",
            str(eigenval_xvg.resolve()),
            "-v",
            str(eigenvec_trr.resolve()),
            "-av",
            str(average_pdb.resolve()),
        ],
        input_text=input_groups,
        cwd=pca_dir,
    )

    # Parse eigenvalues
    eigenvalues = []
    with eigenval_xvg.open("r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    eigenvalues.append(float(parts[1]))
                except ValueError:
                    pass

    evals_arr = np.array(eigenvalues, dtype=np.float64)
    tot_var = np.sum(evals_arr) if len(evals_arr) > 0 else 1.0
    exp_var = (evals_arr / tot_var) * 100.0 if tot_var > 0 else np.zeros_like(evals_arr)
    cum_var = np.cumsum(exp_var)

    # Step 2: anaeig for each trajectory and project PC1 to PCn
    proj_dfs = []
    for rep_idx, traj_path in enumerate(trajectory_paths):
        rep_id = f"Rep_{rep_idx + 1}"
        proj_xvg = pca_dir / f"proj_{rep_id}.xvg"
        gmx_runner.run(
            "anaeig",
            [
                "-s",
                str(topology_path.resolve()),
                "-f",
                str(traj_path.resolve()),
                "-v",
                str(eigenvec_trr.resolve()),
                "-first",
                "1",
                "-last",
                str(min(n_components, len(evals_arr))),
                "-proj",
                str(proj_xvg.resolve()),
            ],
            input_text=input_groups,
            cwd=pca_dir,
        )

        times = []
        pc_rows = []
        with proj_xvg.open("r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line.startswith("@") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    pc_rows.append([float(p) for p in parts[1:]])

        pc_matrix = np.array(pc_rows, dtype=np.float64)
        rep_df = pd.DataFrame(pc_matrix, columns=[f"PC{i + 1}" for i in range(pc_matrix.shape[1])])
        rep_df.insert(0, "system", "System")
        rep_df.insert(1, "replicate", rep_id)
        rep_df.insert(2, "frame", np.arange(len(times)))
        rep_df.insert(3, "time_ps", times)
        proj_dfs.append(rep_df)

    projections_df = pd.concat(proj_dfs, ignore_index=True)

    return PCAResults(
        eigenvalues=evals_arr[:n_components],
        eigenvectors=np.empty((0, 0)),
        explained_variance=exp_var[:n_components],
        cumulative_variance=cum_var[:n_components],
        projections=projections_df,
    )


def save_pca_results(pca_res: PCAResults, output_dir: str | Path) -> Path:
    """Save PCA tables (eigenvalues, explained variance, projections) to output directory."""
    pca_dir = Path(output_dir) / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    # Eigenvalues & variance table
    ev_df = pd.DataFrame(
        {
            "Mode": np.arange(1, len(pca_res.eigenvalues) + 1),
            "Eigenvalue": pca_res.eigenvalues,
            "ExplainedVariance_Percent": pca_res.explained_variance,
            "CumulativeVariance_Percent": pca_res.cumulative_variance,
        }
    )
    ev_df.to_csv(pca_dir / "explained_variance.csv", index=False)
    ev_df[["Mode", "Eigenvalue"]].to_csv(pca_dir / "eigenvalues.csv", index=False)

    # Projections
    pca_res.projections.to_csv(pca_dir / "projections.csv", index=False)
    return pca_dir
