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


def describe_rmsip(value: float) -> str:
    """Qualitative descriptor for a subspace overlap (RMSIP) value."""
    if value >= 0.70:
        return "high"
    if value >= 0.50:
        return "moderate"
    return "low"


def describe_cosine_content(value: float) -> str:
    """Qualitative descriptor for a cosine content value (Hess 2002 heuristics)."""
    if value >= 0.70:
        return "high"
    if value >= 0.50:
        return "moderate"
    return "low"


def _cumulative_fraction_per_replicate(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """First ``fraction`` of the frames of EVERY replicate, pooled.

    Frames are stored replicate after replicate, so slicing the pooled table
    directly would make the 25 % point consist of replicate 1 only. Slicing each
    replicate separately gives a true cumulative-simulation-time fraction.
    """
    if "replicate" not in df.columns:
        end_idx = max(1, int(len(df) * fraction))
        return df.iloc[:end_idx]
    parts = []
    for _, rep_df in df.groupby("replicate", sort=False):
        end_idx = max(1, int(len(rep_df) * fraction))
        parts.append(rep_df.iloc[:end_idx])
    return pd.concat(parts)


def compute_convergence_diagnostics(
    coords_dict: dict[str, np.ndarray],  # rep_id -> (n_frames, n_atoms, 3)
    projections_df: pd.DataFrame,
    frame_assignments_df: pd.DataFrame,
    n_modes: int = 10,
    fractions: tuple[int, ...] = (25, 50, 75, 100),
) -> ConvergenceResults:
    """Run sampling / convergence diagnostics across all replicates.

    * ``rmsip_halves`` compares the essential subspace of the pooled first halves
      of every replicate with that of the pooled second halves. Per-replicate
      values are reported in ``rmsip_halves_per_replicate``.
    * ``progressive_overlap`` compares the subspace of the first f % of every
      replicate with the full-data subspace, for each f in ``fractions``. The
      100 % entry is computed like the others and equals 1 by construction; it is
      kept as the end point of the curve.
    * ``population_stability`` evaluates state populations on the first f % of
      every replicate (true cumulative simulation time, not table order).
    """
    all_reps = list(coords_dict.keys())
    if not all_reps:
        raise ValueError("coords_dict must contain at least one replicate.")

    # ---- Half-vs-half subspace overlap -------------------------------------
    first_halves: dict[str, np.ndarray] = {}
    second_halves: dict[str, np.ndarray] = {}
    per_rep_halves: dict[str, float] = {}
    for rep, coords in coords_dict.items():
        n = len(coords)
        if n < 4:
            logger.warning(f"Replicate {rep} has {n} frames; half-vs-half RMSIP skipped.")
            continue
        h = n // 2
        first_halves[rep] = coords[:h]
        second_halves[rep] = coords[h:]
        _, ev1, _, _, _ = compute_cartesian_pca({rep: coords[:h]}, n_components=n_modes)
        _, ev2, _, _, _ = compute_cartesian_pca({rep: coords[h:]}, n_components=n_modes)
        k = min(ev1.shape[1], ev2.shape[1], n_modes)
        per_rep_halves[rep] = calculate_subspace_rmsip(ev1[:, :k], ev2[:, :k])

    if first_halves:
        _, ev_first, _, _, _ = compute_cartesian_pca(first_halves, n_components=n_modes)
        _, ev_second, _, _, _ = compute_cartesian_pca(second_halves, n_components=n_modes)
        k = min(ev_first.shape[1], ev_second.shape[1], n_modes)
        rmsip_halves = calculate_subspace_rmsip(ev_first[:, :k], ev_second[:, :k])
    else:
        rmsip_halves = float("nan")

    # ---- Progressive subspace overlap --------------------------------------
    progressive_overlap: dict[int, float] = {}
    _, evecs_full, _, _, _ = compute_cartesian_pca(coords_dict, n_components=n_modes)
    s_full = min(evecs_full.shape[1], n_modes)
    for frac in fractions:
        sub_dict = {}
        for k_rep, c in coords_dict.items():
            end_f = max(2, int(round(len(c) * (frac / 100.0))))
            sub_dict[k_rep] = c[:end_f]
        try:
            _, evecs_frac, _, _, _ = compute_cartesian_pca(sub_dict, n_components=s_full)
            s_cur = min(evecs_frac.shape[1], s_full)
            progressive_overlap[int(frac)] = calculate_subspace_rmsip(
                evecs_frac[:, :s_cur], evecs_full[:, :s_cur]
            )
        except Exception as e:
            logger.warning(f"Progressive overlap at {frac}% could not be computed: {e}")
            progressive_overlap[int(frac)] = float("nan")

    # ---- Cosine content of the leading projections --------------------------
    cosine_dict: dict[int, float] = {}
    for i in range(1, 6):
        col = f"PC{i}"
        if col in projections_df.columns:
            cosine_dict[i] = calculate_cosine_content(projections_df[col].to_numpy(), period_k=1)

    # ---- State population stability (cumulative time per replicate) ---------
    pop_stability_records = []
    major_states = [st for st in frame_assignments_df["state"].unique() if st != "Unassigned"]
    for frac in fractions:
        subset = _cumulative_fraction_per_replicate(frame_assignments_df, frac / 100.0)
        tot_sub = len(subset)
        row = {"Fraction_Percent": int(frac), "Frames": tot_sub}
        for st in major_states:
            st_cnt = int(np.sum(subset["state"] == st))
            row[st] = (st_cnt / tot_sub) * 100.0 if tot_sub else float("nan")
        pop_stability_records.append(row)
    pop_stability_df = pd.DataFrame(pop_stability_records)

    # ---- Pairwise replicate subspace overlap --------------------------------
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

    # ---- Heuristic assessment ----------------------------------------------
    notes = []
    pc1_cosine = cosine_dict.get(1, 0.0)
    rmsip_word = describe_rmsip(rmsip_halves) if np.isfinite(rmsip_halves) else "undetermined"
    cos_word = describe_cosine_content(pc1_cosine)

    if np.isfinite(rmsip_halves) and rmsip_halves >= 0.70 and pc1_cosine < 0.50:
        assessment = "Good evidence of stability"
    elif np.isfinite(rmsip_halves) and rmsip_halves >= 0.50 and pc1_cosine < 0.70:
        assessment = "Mixed evidence"
    else:
        assessment = "Insufficient evidence"

    if np.isfinite(rmsip_halves):
        notes.append(
            f"{rmsip_word.capitalize()} subspace overlap between the first and second "
            f"halves of the replicates (RMSIP = {rmsip_halves:.2f}; "
            f"heuristic: >= 0.70 high, 0.50-0.70 moderate, < 0.50 low)."
        )
    else:
        notes.append("Half-vs-half RMSIP could not be evaluated (too few frames).")
    if len(per_rep_halves) > 1:
        worst = min(per_rep_halves, key=per_rep_halves.get)
        notes.append(
            f"Per-replicate half-vs-half RMSIP ranges from {per_rep_halves[worst]:.2f} ({worst}) "
            f"to {max(per_rep_halves.values()):.2f}."
        )
    if cos_word == "low":
        notes.append(
            f"Low PC1 cosine content ({pc1_cosine:.2f} < 0.50): the dominant motion is not "
            "consistent with random diffusion."
        )
    elif cos_word == "moderate":
        notes.append(
            f"Moderate PC1 cosine content ({pc1_cosine:.2f}): PC1 partly resembles random "
            "diffusion; the leading mode may not be converged."
        )
    else:
        notes.append(
            f"High PC1 cosine content ({pc1_cosine:.2f} >= 0.70): PC1 resembles random "
            "diffusion, which indicates insufficient sampling along the dominant mode."
        )

    return ConvergenceResults(
        rmsip_halves=rmsip_halves,
        progressive_overlap=progressive_overlap,
        cosine_content=cosine_dict,
        population_stability=pop_stability_df,
        pairwise_rmsip=pairwise_df,
        sampling_assessment=assessment,
        assessment_notes=notes,
        rmsip_halves_per_replicate=per_rep_halves,
    )


def save_convergence_results(conv: ConvergenceResults, output_dir: str | Path) -> Path:
    """Save convergence metrics to convergence/ directory."""
    conv_dir = Path(output_dir) / "convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)

    # rmsip.csv: pooled half-vs-half value first, then one row per replicate
    rmsip_rows = [
        {
            "Metric": "First_vs_Second_Half_RMSIP",
            "Scope": "all_replicates_pooled",
            "Value": conv.rmsip_halves,
            "Sampling_Assessment": conv.sampling_assessment,
        }
    ]
    for rep, val in conv.rmsip_halves_per_replicate.items():
        rmsip_rows.append(
            {
                "Metric": "First_vs_Second_Half_RMSIP",
                "Scope": rep,
                "Value": val,
                "Sampling_Assessment": "",
            }
        )
    pd.DataFrame(rmsip_rows).to_csv(conv_dir / "rmsip.csv", index=False)

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
