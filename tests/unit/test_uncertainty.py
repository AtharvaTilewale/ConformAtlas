"""Unit tests for replicate uncertainty and statistical estimators."""

import numpy as np
import pandas as pd
import pytest

from conformatlas.uncertainty import compute_replicate_statistics


def test_multiple_replicates_uncertainty():
    """Verify mean, SD, SEM, and CI95 calculation across 3 independent replicates."""
    records = []
    # 3 replicates with known state populations
    # State A: Rep1: 60%, Rep2: 70%, Rep3: 65% -> Mean = 65.0%, SD = 5.0%, SEM = 5.0 / sqrt(3) ≈ 2.8868%
    for f in range(100):
        records.append({"replicate": "Rep_1", "state": "State A" if f < 60 else "State B"})
    for f in range(100):
        records.append({"replicate": "Rep_2", "state": "State A" if f < 70 else "State B"})
    for f in range(100):
        records.append({"replicate": "Rep_3", "state": "State A" if f < 65 else "State B"})

    df = pd.DataFrame(records)
    summary_df, detailed_df, unc_type = compute_replicate_statistics(df)

    assert unc_type == "between-replicate"
    assert len(summary_df) == 2

    row_a = summary_df[summary_df["State"] == "State A"].iloc[0]
    assert pytest.approx(row_a["Mean_Population_Percent"], abs=1e-5) == 65.0
    assert pytest.approx(row_a["SD"], abs=1e-5) == 5.0
    expected_sem = 5.0 / np.sqrt(3)
    assert pytest.approx(row_a["SEM"], abs=1e-5) == expected_sem
    assert row_a["CI_95_Lower"] < 65.0 < row_a["CI_95_Upper"]


def test_single_replicate_block_uncertainty():
    """Verify that a single replicate is flagged as intra-trajectory block analysis."""
    records = []
    for f in range(500):
        records.append({"replicate": "SingleRep", "state": "State A" if f < 350 else "State B"})

    df = pd.DataFrame(records)
    summary_df, detailed_df, unc_type = compute_replicate_statistics(df)

    assert unc_type == "intra-trajectory-block"
    assert "Block_SD" in summary_df.columns
    assert "Block_SEM" in summary_df.columns
    assert summary_df.iloc[0]["Uncertainty_Type"] == "intra-trajectory-block"
