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
        arr = np.asarray(coords_dict[k])
        if arr.ndim == 3 and arr.shape[2] != 3:
            raise ValueError(f"Replicate {k} coordinates must have shape (n_frames, n_atoms, 3).")
        if arr.ndim not in (2, 3):
            raise ValueError(
                f"Replicate {k} must be (n_frames, n_atoms, 3) coordinates or (n_frames, n_features)."
            )
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

    def _weight(n_k: int) -> float:
        return 1.0 / (n_reps * n_k) if weighting == "equal-replicate" else 1.0 / total_frames

    if n_dims > total_frames:
        # Dual (SVD) route: with fewer frames than coordinates the covariance has
        # rank <= total_frames, and the SVD of the sqrt(w)-scaled centred data
        # matrix gives the same eigenpairs at O(N^2 D) instead of O(D^3) cost.
        blocks = []
        for k in replicate_keys:
            centered = flattened_reps[k] - mean_struct
            if centered.shape[0] == 0:
                continue
            blocks.append(np.sqrt(_weight(centered.shape[0])) * centered)
        data = np.vstack(blocks)
        _, sing, vt = np.linalg.svd(data, full_matrices=False)
        evals = np.maximum(sing**2, 0.0)
        evecs = vt.T
    else:
        cov_matrix = np.zeros((n_dims, n_dims), dtype=np.float64)
        for k in replicate_keys:
            centered = flattened_reps[k] - mean_struct
            n_k = centered.shape[0]
            if n_k == 0:
                continue
            cov_matrix += _weight(n_k) * (centered.T @ centered)
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        evals, evecs = np.linalg.eigh(cov_matrix)
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
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


def compute_tica(
    coords_dict: dict[str, np.ndarray],
    lag: int = 10,
    n_components: int = 10,
    weighting: str = "equal-replicate",
    epsilon: float = 1e-6,
    max_reduced_dims: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-lagged independent component analysis (Molgedey & Schuster 1994;
    Perez-Hernandez et al. 2013) on per-replicate coordinate or feature arrays.

    Solves the generalised eigenproblem C(tau) v = lambda C(0) v with the
    symmetrised time-lagged covariance, after whitening C(0) and discarding
    directions with variance below ``epsilon`` x the largest. Eigenvalues are the
    autocorrelations at lag ``lag`` (frames); implied timescales are
    -lag / ln(lambda).

    Returns (eigenvalues, eigenvectors (n_dims, k), kinetic_variance_percent,
    cumulative_kinetic_variance_percent, mean) so the output is interchangeable
    with ``compute_cartesian_pca``.
    """
    keys = list(coords_dict.keys())
    if not keys:
        raise ValueError("Cannot compute TICA with empty coordinates dictionary.")
    flat = {
        k: np.asarray(v).reshape(np.asarray(v).shape[0], -1).astype(np.float64)
        for k, v in coords_dict.items()
    }
    n_dims = next(iter(flat.values())).shape[1]
    n_reps = len(keys)
    total_pairs = sum(max(0, f.shape[0] - lag) for f in flat.values())
    if total_pairs <= 0:
        raise ValueError(f"Lag {lag} frames is too long for the trajectories supplied.")

    # TICA over-fits when the number of lagged pairs is not much larger than the
    # dimension (eigenvalues saturate at 1). Reduce to the leading principal
    # components first in that regime and map the result back afterwards.
    reduce_to = min(n_dims, max(2, total_pairs // 10), max_reduced_dims)
    basis = None
    if n_dims > reduce_to:
        _, basis, _, _, pca_mean = compute_cartesian_pca(
            flat, weighting=weighting, n_components=reduce_to
        )
        flat = {k: (f - pca_mean) @ basis for k, f in flat.items()}
        n_dims = reduce_to

    mean = np.zeros(n_dims)
    for k in keys:
        f = flat[k]
        m = f.shape[0] - lag
        if m <= 0:
            continue
        w = 1.0 / n_reps if weighting == "equal-replicate" else m / total_pairs
        mean += w * 0.5 * (f[:m].mean(axis=0) + f[lag:].mean(axis=0))

    c0 = np.zeros((n_dims, n_dims))
    ct = np.zeros((n_dims, n_dims))
    for k in keys:
        f = flat[k] - mean
        m = f.shape[0] - lag
        if m <= 0:
            continue
        w = (1.0 / (n_reps * m)) if weighting == "equal-replicate" else 1.0 / total_pairs
        a, b = f[:m], f[lag:]
        c0 += w * 0.5 * (a.T @ a + b.T @ b)
        ct += w * 0.5 * (a.T @ b + b.T @ a)

    s0, u0 = np.linalg.eigh(c0)
    keep = s0 > epsilon * s0.max()
    whiten = u0[:, keep] / np.sqrt(s0[keep])
    ct_w = whiten.T @ ct @ whiten
    ct_w = 0.5 * (ct_w + ct_w.T)
    lam, v = np.linalg.eigh(ct_w)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    v = v[:, order]
    evecs = whiten @ v
    if basis is not None:
        evecs = basis @ evecs
        mean = pca_mean + basis @ mean
    evecs = evecs / np.linalg.norm(evecs, axis=0, keepdims=True)

    k = min(n_components, len(lam))
    kin = lam**2
    kin_pct = 100.0 * kin / kin.sum() if kin.sum() > 0 else np.zeros_like(kin)
    return lam[:k], evecs[:, :k], kin_pct[:k], np.cumsum(kin_pct)[:k], mean


def implied_timescales(eigenvalues: np.ndarray, lag_frames: int, dt_ps: float) -> np.ndarray:
    """-tau / ln(lambda) in ps; infinite for lambda >= 1, NaN for lambda <= 0."""
    lam = np.asarray(eigenvalues, dtype=np.float64)
    out = np.full_like(lam, np.nan)
    ok = (lam > 0) & (lam < 1)
    out[ok] = -lag_frames * dt_ps / np.log(lam[ok])
    out[lam >= 1] = np.inf
    return out


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

        times, pc_matrix = _read_anaeig_projection(proj_xvg)
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


def _read_anaeig_projection(proj_xvg: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``gmx anaeig -proj`` output.

    anaeig writes one data set per eigenvector, separated by ``&`` lines, each
    set being (time, projection) rows. Older single-set files with several
    columns per row are accepted too.
    """
    sets: list[list[list[float]]] = [[]]
    with proj_xvg.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#@":
                continue
            if line.startswith("&"):
                sets.append([])
                continue
            parts = line.split()
            if len(parts) >= 2:
                sets[-1].append([float(p) for p in parts])
    sets = [np.array(s_, dtype=np.float64) for s_ in sets if s_]
    if not sets:
        raise ValueError(f"No projection data in {proj_xvg}")
    if len(sets) == 1 and sets[0].shape[1] > 2:
        return sets[0][:, 0], sets[0][:, 1:]
    n = min(len(s_) for s_ in sets)
    times = sets[0][:n, 0]
    pc_matrix = np.column_stack([s_[:n, 1] for s_ in sets])
    return times, pc_matrix


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
