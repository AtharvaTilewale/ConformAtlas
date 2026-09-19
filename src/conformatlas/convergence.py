"""PCA convergence and sampling diagnostics (RMSIP, progressive overlap, cosine content)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from conformatlas.models import ConvergenceResults
from conformatlas.pca import compute_cartesian_pca

logger = logging.getLogger("conformatlas.convergence")


def calculate_subspace_rmsip(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate Root Mean Square Inner Product (RMSIP) between two subspaces.

    RMSIP = sqrt( (1 / s) * sum_{i=1}^s sum_{j=1}^s (u_i . v_j)^2 )
          = sqrt( (1 / s) * || U^T V ||_F^2 )

    Parameters
    ----------
    u : np.ndarray
        Eigenvectors matrix of shape (d, s).
    v : np.ndarray
        Eigenvectors matrix of shape (d, s).

    Returns
    -------
    float
        RMSIP similarity score bounded between 0.0 (orthogonal) and 1.0 (identical).
    """
    if u.shape != v.shape:
        raise ValueError(f"Subspace shape mismatch: {u.shape} vs {v.shape}")
    s = u.shape[1]
    if s == 0:
        return 0.0

    # Inner products matrix M = U^T V
    m = u.T @ v
    # Frobenius norm squared divided by s
    frobenius_sq = np.sum(m**2)
    val = np.sqrt(frobenius_sq / s)
    return float(np.clip(val, 0.0, 1.0))


def calculate_cosine_content(projections_1d: np.ndarray, period_k: int = 1) -> float:
    """Calculate the cosine content of a 1D principal component projection.

    c_k = (2 / N) * [ sum_{t=0}^{N-1} p(t) * cos(k * pi * t / (N-1)) ]^2 / sum_{t=0}^{N-1} p(t)^2

    A cosine content near 1.0 indicates that trajectory motion along this PC
    resembles random diffusion / a random walk, indicating insufficient sampling.

    Parameters
    ----------
    projections_1d : np.ndarray
        1D array of projections along a PC.
    period_k : int
        Cosine period harmonic (default: 1).

    Returns
    -------
    float
        Cosine content value between 0.0 and 1.0.
    """
    p = np.asarray(projections_1d, dtype=np.float64)
    n = len(p)
    if n < 3:
        return 0.0

    t = np.arange(n, dtype=np.float64)
    cos_wave = np.cos(period_k * np.pi * t / (n - 1.0))

    denom = np.sum(p**2)
    if denom == 0.0:
        return 0.0

    numer = (np.sum(p * cos_wave)) ** 2
    c = (2.0 / n) * (numer / denom)
    return float(np.clip(c, 0.0, 1.0))


def compute_convergence_diagnostics(
    coords_dict: dict[str, np.ndarray],  # rep_id -> (n_frames, n_atoms, 3)
    projections_df: pd.DataFrame,
    frame_assignments_df: pd.DataFrame,
    n_modes: int = 10,
) -> ConvergenceResults:
    """Run comprehensive convergence analysis across trajectory coordinates and projections."""
    # 1. First-half vs Second-half RMSIP
    all_reps = list(coords_dict.keys())
    primary_coords = coords_dict[all_reps[0]]
    n_frames = len(primary_coords)

    if n_frames >= 4:
        half_idx = n_frames // 2
        first_half = primary_coords[:half_idx]
        second_half = primary_coords[half_idx:]

        _, evecs_1, _, _, _ = compute_cartesian_pca({"half1": first_half}, n_components=n_modes)
        _, evecs_2, _, _, _ = compute_cartesian_pca({"half2": second_half}, n_components=n_modes)
        s_modes = min(evecs_1.shape[1], evecs_2.shape[1], n_modes)
        rmsip_halves = calculate_subspace_rmsip(evecs_1[:, :s_modes], evecs_2[:, :s_modes])
    else:
        rmsip_halves = 1.0

    # 2. Progressive overlap at 25%, 50%, 75%, 100%
    progressive_overlap = {}
    fractions = [25, 50, 75, 100]
    _, evecs_full, _, _, _ = compute_cartesian_pca(coords_dict, n_components=n_modes)
    s_full = min(evecs_full.shape[1], n_modes)

    for frac in fractions:
        if frac == 100:
            progressive_overlap[100] = 1.0
        else:
            sub_dict = {}
            for k, c in coords_dict.items():
                end_f = max(2, int(len(c) * (frac / 100.0)))
                sub_dict[k] = c[:end_f]
            try:
                _, evecs_frac, _, _, _ = compute_cartesian_pca(sub_dict, n_components=s_full)
                s_cur = min(evecs_frac.shape[1], s_full)
                progressive_overlap[frac] = calculate_subspace_rmsip(
                    evecs_frac[:, :s_cur], evecs_full[:, :s_cur]
                )
            except Exception as e:
                logger.debug(f"Progressive PCA at {frac}% failed: {e}")
                progressive_overlap[frac] = 0.0

    # 3. Cosine content for first 5 PCs
    cosine_dict = {}
    for i in range(1, min(6, 11)):
        col = f"PC{i}"
        if col in projections_df.columns:
            cosine_dict[i] = calculate_cosine_content(projections_df[col].to_numpy(), period_k=1)

    # 4. State population stability over cumulative time
    pop_stability_records = []
    major_states = [s for s in frame_assignments_df["state"].unique() if s != "Unassigned"]

    for frac in fractions:
        end_idx = max(1, int(len(frame_assignments_df) * (frac / 100.0)))
        subset = frame_assignments_df.iloc[:end_idx]
        tot_sub = len(subset)
        row = {"Fraction_Percent": frac, "Frames": tot_sub}
        for st in major_states:
            st_cnt = np.sum(subset["state"] == st)
            row[st] = (st_cnt / tot_sub) * 100.0
        pop_stability_records.append(row)

    pop_stability_df = pd.DataFrame(pop_stability_records)

    # 5. Pairwise RMSIP if multiple replicates
    pairwise_df = None
    if len(coords_dict) > 1:
        rep_names = list(coords_dict.keys())
        matrix = np.eye(len(rep_names))
        rep_evecs = {}
        for r_name in rep_names:
            _, ev, _, _, _ = compute_cartesian_pca(
                {r_name: coords_dict[r_name]}, n_components=n_modes
            )
            rep_evecs[r_name] = ev

        for i, r1 in enumerate(rep_names):
            for j, r2 in enumerate(rep_names):
                if i < j:
                    s_ij = min(rep_evecs[r1].shape[1], rep_evecs[r2].shape[1], n_modes)
                    score = calculate_subspace_rmsip(
                        rep_evecs[r1][:, :s_ij], rep_evecs[r2][:, :s_ij]
                    )
                    matrix[i, j] = score
                    matrix[j, i] = score

        pairwise_df = pd.DataFrame(matrix, index=rep_names, columns=rep_names)

    # 6. Overall Sampling Assessment (Heuristic)
    notes = []
    pc1_cosine = cosine_dict.get(1, 0.0)

    if rmsip_halves >= 0.70 and pc1_cosine < 0.50:
        assessment = "Good evidence of stability"
        notes.append(
            f"High subspace overlap between trajectory halves (RMSIP = {rmsip_halves:.2f} >= 0.70)."
        )
        notes.append(
            f"Low PC1 cosine content ({pc1_cosine:.2f} < 0.50), indicating motion is not simple diffusive drift."
        )
    elif rmsip_halves >= 0.50 and pc1_cosine < 0.70:
        assessment = "Mixed evidence"
        notes.append(
            f"Moderate subspace overlap between trajectory halves (RMSIP = {rmsip_halves:.2f})."
        )
        notes.append(f"Moderate PC1 cosine content ({pc1_cosine:.2f}).")
    else:
        assessment = "Insufficient evidence"
        notes.append(
            f"Low subspace overlap between trajectory halves (RMSIP = {rmsip_halves:.2f} < 0.50)."
        )
        if pc1_cosine >= 0.70:
            notes.append(
                f"High PC1 cosine content ({pc1_cosine:.2f} >= 0.70) suggests diffusive behavior along PC1."
            )

    return ConvergenceResults(
        rmsip_halves=rmsip_halves,
        progressive_overlap=progressive_overlap,
        cosine_content=cosine_dict,
        population_stability=pop_stability_df,
        pairwise_rmsip=pairwise_df,
        sampling_assessment=assessment,
        assessment_notes=notes,
    )


def save_convergence_results(conv: ConvergenceResults, output_dir: str | Path) -> Path:
    """Save convergence metrics to convergence/ directory."""
    conv_dir = Path(output_dir) / "convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)

    # rmsip.csv
    pd.DataFrame(
        [
            {
                "Metric": "First_vs_Second_Half_RMSIP",
                "Value": conv.rmsip_halves,
                "Sampling_Assessment": conv.sampling_assessment,
            }
        ]
    ).to_csv(conv_dir / "rmsip.csv", index=False)

    # cosine_content.csv
    cosine_df = pd.DataFrame(
        [{"PC": f"PC{pc}", "Cosine_Content": val} for pc, val in conv.cosine_content.items()]
    )
    cosine_df.to_csv(conv_dir / "cosine_content.csv", index=False)

    # progressive_overlap.csv
    prog_df = pd.DataFrame(
        [
            {"Fraction_Percent": frac, "Subspace_RMSIP": val}
            for frac, val in conv.progressive_overlap.items()
        ]
    )
    prog_df.to_csv(conv_dir / "progressive_overlap.csv", index=False)

    # population_stability.csv
    if conv.population_stability is not None:
        conv.population_stability.to_csv(conv_dir / "population_stability.csv", index=False)

    # pairwise_rmsip.csv
    if conv.pairwise_rmsip is not None:
        conv.pairwise_rmsip.to_csv(conv_dir / "pairwise_rmsip.csv")

    return conv_dir
