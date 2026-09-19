"""Statistical uncertainty estimation (replicate SD, SEM, 95% CI, block bootstrap)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("conformatlas.uncertainty")


def compute_replicate_statistics(
    frame_assignments_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compute state population statistics and uncertainty across replicates.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, str]
        (state_summary_df, detailed_replicates_df, uncertainty_type)
        where uncertainty_type is either 'between-replicate' or 'intra-trajectory-block'.
    """
    reps = frame_assignments_df["replicate"].unique().tolist()
    states = [s for s in frame_assignments_df["state"].unique() if s != "Unassigned"]

    if len(reps) > 1:
        # Multiple independent replicates available
        uncertainty_type = "between-replicate"
        rep_pop_records = []

        for rep in reps:
            rep_df = frame_assignments_df[frame_assignments_df["replicate"] == rep]
            n_rep_frames = len(rep_df)
            row = {"Replicate": rep, "Total_Frames": n_rep_frames}
            for st in states:
                st_count = np.sum(rep_df["state"] == st)
                row[st] = (st_count / n_rep_frames) * 100.0 if n_rep_frames > 0 else 0.0
            rep_pop_records.append(row)

        detailed_df = pd.DataFrame(rep_pop_records)

        summary_records = []
        k = len(reps)
        t_crit = stats.t.ppf(0.975, df=k - 1) if k > 1 else 1.96

        for st in states:
            vals = detailed_df[st].to_numpy()
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if k > 1 else 0.0
            sem_val = float(sd_val / np.sqrt(k)) if k > 1 else 0.0
            ci95_margin = float(t_crit * sem_val)

            summary_records.append(
                {
                    "State": st,
                    "Mean_Population_Percent": mean_val,
                    "SD": sd_val,
                    "SEM": sem_val,
                    "CI_95_Lower": max(0.0, mean_val - ci95_margin),
                    "CI_95_Upper": min(100.0, mean_val + ci95_margin),
                    "Num_Replicates": k,
                    "Uncertainty_Type": "between-replicate",
                }
            )

        summary_df = pd.DataFrame(summary_records)
        if summary_df.empty:
            summary_df = pd.DataFrame(
                columns=[
                    "State",
                    "Mean_Population_Percent",
                    "SD",
                    "SEM",
                    "CI_95_Lower",
                    "CI_95_Upper",
                    "Num_Replicates",
                    "Uncertainty_Type",
                ]
            )
        return summary_df, detailed_df, uncertainty_type

    else:
        # Single replicate: perform intra-trajectory block analysis (e.g. 5 blocks)
        uncertainty_type = "intra-trajectory-block"
        n_blocks = 5
        total_frames = len(frame_assignments_df)
        block_size = max(1, total_frames // n_blocks)

        block_pop_records = []
        for b_idx in range(n_blocks):
            start = b_idx * block_size
            end = (b_idx + 1) * block_size if b_idx < n_blocks - 1 else total_frames
            block_df = frame_assignments_df.iloc[start:end]
            n_b_frames = len(block_df)

            row = {"Block": f"Block_{b_idx + 1}", "Total_Frames": n_b_frames}
            for st in states:
                st_count = np.sum(block_df["state"] == st)
                row[st] = (st_count / n_b_frames) * 100.0 if n_b_frames > 0 else 0.0
            block_pop_records.append(row)

        detailed_df = pd.DataFrame(block_pop_records)

        summary_records = []
        k = n_blocks
        t_crit = stats.t.ppf(0.975, df=k - 1)

        for st in states:
            vals = detailed_df[st].to_numpy()
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1))
            sem_val = float(sd_val / np.sqrt(k))
            ci95_margin = float(t_crit * sem_val)

            summary_records.append(
                {
                    "State": st,
                    "Mean_Population_Percent": mean_val,
                    "Block_SD": sd_val,
                    "Block_SEM": sem_val,
                    "CI_95_Lower": max(0.0, mean_val - ci95_margin),
                    "CI_95_Upper": min(100.0, mean_val + ci95_margin),
                    "Num_Blocks": k,
                    "Uncertainty_Type": "intra-trajectory-block",
                }
            )

        summary_df = pd.DataFrame(summary_records)
        if summary_df.empty:
            summary_df = pd.DataFrame(
                columns=[
                    "State",
                    "Mean_Population_Percent",
                    "Block_SD",
                    "Block_SEM",
                    "CI_95_Lower",
                    "CI_95_Upper",
                    "Num_Blocks",
                    "Uncertainty_Type",
                ]
            )
        return summary_df, detailed_df, uncertainty_type


def save_uncertainty_results(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Save population and uncertainty tables to statistics/ directory."""
    stat_dir = Path(output_dir) / "statistics"
    stat_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(stat_dir / "state_populations.csv", index=False)
    detailed_df.to_csv(stat_dir / "replicate_statistics.csv", index=False)

    return stat_dir
