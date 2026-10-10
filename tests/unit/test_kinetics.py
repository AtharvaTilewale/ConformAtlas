"""State kinetics on a hand-built state sequence."""

import numpy as np
import pandas as pd

from conformatlas.kinetics import compute_state_kinetics, save_kinetics


def _assignments():
    # A A A B B A U A B B B ; dt = 10 ps ; one replicate
    seq = (
        ["State A"] * 3
        + ["State B"] * 2
        + ["State A"]
        + ["Unassigned"]
        + ["State A"]
        + ["State B"] * 3
    )
    return pd.DataFrame(
        {
            "replicate": ["Rep_1"] * len(seq),
            "frame": np.arange(len(seq)),
            "time_ps": 10.0 * np.arange(len(seq)),
            "state": seq,
        }
    )


def test_transition_counts_and_dwell_times(tmp_path):
    res = compute_state_kinetics(_assignments(), ["State A", "State B"])
    counts = res["transition_counts"]
    # Transitions after dropping the unassigned frame: AAABBAABBB
    # A->A: 3, A->B: 2, B->A: 1, B->B: 3
    assert counts.loc["State A", "to_State A"] == 3
    assert counts.loc["State A", "to_State B"] == 2
    assert counts.loc["State B", "to_State A"] == 1
    assert counts.loc["State B", "to_State B"] == 3

    tpm = res["transition_matrix"]
    np.testing.assert_allclose(tpm.to_numpy().sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(tpm.loc["State A", "to_State B"], 2 / 5)

    dwell = res["dwell_times"]
    # visits: A(3), B(2), A(2 incl. frame after the skipped one), B(3)
    assert len(dwell) == 4
    summary = res["dwell_summary"].set_index("state")
    assert summary.loc["State A", "visits"] == 2
    assert summary.loc["State B", "visits"] == 2
    assert summary.loc["State B", "max_dwell_ps"] == 30.0

    mfpt = res["mfpt"]
    assert np.all(np.isfinite(mfpt.to_numpy(dtype=float)[~np.eye(2, dtype=bool)]))
    assert res["summary"]["frames_skipped_unassigned"].iloc[0] == 1

    out = save_kinetics(res, tmp_path)
    assert (out / "transition_counts.csv").exists()
    assert (out / "dwell_summary.csv").exists()
    assert (out / "mfpt_markov_ps.csv").exists()


def test_two_replicates_are_not_concatenated():
    df = _assignments()
    df2 = df.copy()
    df2["replicate"] = "Rep_2"
    both = pd.concat([df, df2], ignore_index=True)
    res = compute_state_kinetics(both, ["State A", "State B"])
    assert res["transition_counts"].to_numpy().sum() == 2 * 9
    per_rep = res["transition_counts_per_replicate"]
    assert set(per_rep["replicate"]) == {"Rep_1", "Rep_2"}
