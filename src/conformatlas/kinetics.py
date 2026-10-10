"""State kinetics from frame assignments: transition counts, dwell times, MFPT.

All quantities are derived from the discrete state sequence of each replicate.
Frames outside the major states ("Unassigned") are skipped: a transition is counted
between consecutive *major-state* frames of the same replicate, and the number of
skipped frames is reported. Transitions are never counted across replicate
boundaries. Nothing here assumes Markovianity except the mean-first-passage-time
estimate, which is labelled as such.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("conformatlas.kinetics")


def _state_sequences(frame_assignments: pd.DataFrame, labels: list[str]):
    """Yield (replicate, states array, times array) with unassigned frames removed."""
    keep = frame_assignments[frame_assignments["state"].isin(labels)]
    for rep, df in keep.groupby("replicate", sort=False):
        df = df.sort_values("time_ps") if "time_ps" in df.columns else df
        yield rep, df["state"].to_numpy(), df["time_ps"].to_numpy(dtype=float)


def compute_state_kinetics(
    frame_assignments: pd.DataFrame,
    state_labels: list[str],
    dt_ps: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Transition counts, transition probabilities, dwell times and MFPT.

    Parameters
    ----------
    frame_assignments : DataFrame with columns replicate, time_ps, state.
    state_labels : ordered list of major-state labels.
    dt_ps : frame spacing in ps; inferred from the time column when None.

    Returns
    -------
    dict with DataFrames: ``transition_counts`` (pooled), ``transition_counts_per_replicate``,
    ``transition_matrix`` (row-normalised, lag one frame), ``dwell_times`` (one row per
    visit), ``dwell_summary`` (per state), ``mfpt`` (Markov estimate, ps) and
    ``summary`` (per-replicate frame counts and skipped frames).
    """
    labels = list(state_labels)
    idx = {s: i for i, s in enumerate(labels)}
    k = len(labels)

    if dt_ps is None:
        t = frame_assignments["time_ps"].to_numpy(dtype=float)
        diffs = np.diff(np.unique(t))
        dt_ps = float(np.median(diffs)) if len(diffs) else 1.0

    pooled = np.zeros((k, k), dtype=np.int64)
    per_rep_rows = []
    dwell_rows = []
    summary_rows = []

    for rep, seq, times in _state_sequences(frame_assignments, labels):
        n_rep_total = int((frame_assignments["replicate"] == rep).sum())
        summary_rows.append(
            {
                "replicate": rep,
                "frames_total": n_rep_total,
                "frames_in_major_states": len(seq),
                "frames_skipped_unassigned": n_rep_total - len(seq),
            }
        )
        if len(seq) == 0:
            continue
        counts = np.zeros((k, k), dtype=np.int64)
        for a, b in zip(seq[:-1], seq[1:]):
            counts[idx[a], idx[b]] += 1
        pooled += counts
        for i, a in enumerate(labels):
            row = {"replicate": rep, "from_state": a}
            for j, b in enumerate(labels):
                row[f"to_{b}"] = int(counts[i, j])
            per_rep_rows.append(row)

        # Dwell times: lengths of consecutive runs (in frames) times dt.
        start = 0
        for i in range(1, len(seq) + 1):
            if i == len(seq) or seq[i] != seq[start]:
                n_frames = i - start
                dwell_rows.append(
                    {
                        "replicate": rep,
                        "state": seq[start],
                        "start_time_ps": float(times[start]),
                        "end_time_ps": float(times[i - 1]),
                        "frames": n_frames,
                        "dwell_ps": n_frames * dt_ps,
                        # The first and last runs are censored (true dwell unknown).
                        "censored": bool(start == 0 or i == len(seq)),
                    }
                )
                start = i

    counts_df = pd.DataFrame(pooled, index=labels, columns=labels)
    counts_df.index.name = "from_state"
    counts_df.columns = [f"to_{c}" for c in labels]

    row_sums = pooled.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        tpm = np.where(row_sums > 0, pooled / np.maximum(row_sums, 1), np.nan)
    tpm_df = pd.DataFrame(tpm, index=labels, columns=[f"to_{c}" for c in labels])
    tpm_df.index.name = "from_state"

    dwell_df = pd.DataFrame(dwell_rows)
    if dwell_df.empty:
        dwell_df = pd.DataFrame(
            columns=[
                "replicate",
                "state",
                "start_time_ps",
                "end_time_ps",
                "frames",
                "dwell_ps",
                "censored",
            ]
        )

    summ_rows = []
    for s in labels:
        d = dwell_df[dwell_df["state"] == s]
        d_unc = d[~d["censored"].astype(bool)] if len(d) else d
        n_out = int(pooled[idx[s]].sum() - pooled[idx[s], idx[s]])
        n_in = int(pooled[:, idx[s]].sum() - pooled[idx[s], idx[s]])
        total_time = float(d["dwell_ps"].sum()) if len(d) else 0.0
        summ_rows.append(
            {
                "state": s,
                "visits": int(len(d)),
                "visits_uncensored": int(len(d_unc)),
                "mean_dwell_ps": float(d_unc["dwell_ps"].mean()) if len(d_unc) else np.nan,
                "median_dwell_ps": float(d_unc["dwell_ps"].median()) if len(d_unc) else np.nan,
                "max_dwell_ps": float(d["dwell_ps"].max()) if len(d) else np.nan,
                "total_time_ps": total_time,
                "transitions_out": n_out,
                "transitions_in": n_in,
                # Mean lifetime from the exit rate (all visits, incl. censored).
                "lifetime_from_rate_ps": (total_time / n_out) if n_out > 0 else np.inf,
            }
        )
    dwell_summary = pd.DataFrame(summ_rows)

    mfpt_df = _mean_first_passage_times(tpm, labels, dt_ps)

    return {
        "transition_counts": counts_df,
        "transition_counts_per_replicate": pd.DataFrame(per_rep_rows),
        "transition_matrix": tpm_df,
        "dwell_times": dwell_df,
        "dwell_summary": dwell_summary,
        "mfpt": mfpt_df,
        "summary": pd.DataFrame(summary_rows),
        "dt_ps": pd.DataFrame([{"dt_ps": dt_ps}]),
    }


def _mean_first_passage_times(tpm: np.ndarray, labels: list[str], dt_ps: float) -> pd.DataFrame:
    """MFPT (ps) between states assuming a Markov chain at lag one frame.

    For target j, solve (I - P_{-j}) m = 1 over the non-target states, where
    P_{-j} is the transition matrix with row and column j removed. Rows with no
    outgoing data give NaN.
    """
    k = len(labels)
    out = np.full((k, k), np.nan)
    if k == 0:
        return pd.DataFrame(out, index=labels, columns=labels)
    p = np.nan_to_num(tpm, nan=0.0)
    for j in range(k):
        keep = [i for i in range(k) if i != j]
        if not keep:
            out[j, j] = 0.0
            continue
        sub = p[np.ix_(keep, keep)]
        a = np.eye(len(keep)) - sub
        try:
            m = np.linalg.solve(a, np.ones(len(keep)))
            for r, i in enumerate(keep):
                out[i, j] = m[r] * dt_ps if np.isfinite(m[r]) and m[r] > 0 else np.nan
        except np.linalg.LinAlgError:
            pass
        out[j, j] = 0.0
    df = pd.DataFrame(out, index=labels, columns=[f"to_{c}" for c in labels])
    df.index.name = "from_state"
    return df


def save_kinetics(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    kin_dir = Path(output_dir) / "kinetics"
    kin_dir.mkdir(parents=True, exist_ok=True)
    results["transition_counts"].to_csv(kin_dir / "transition_counts.csv")
    results["transition_counts_per_replicate"].to_csv(
        kin_dir / "transition_counts_per_replicate.csv", index=False
    )
    results["transition_matrix"].to_csv(kin_dir / "transition_matrix.csv")
    results["dwell_times"].to_csv(kin_dir / "dwell_times.csv", index=False)
    results["dwell_summary"].to_csv(kin_dir / "dwell_summary.csv", index=False)
    results["mfpt"].to_csv(kin_dir / "mfpt_markov_ps.csv")
    results["summary"].to_csv(kin_dir / "replicate_summary.csv", index=False)
    return kin_dir
