"""Unit tests for convergence diagnostics across replicates."""

import numpy as np
import pandas as pd

from conformatlas.convergence import compute_convergence_diagnostics


def _coords(rng, n_frames, n_atoms=8):
    return rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float32)


def test_population_stability_uses_cumulative_time_per_replicate():
    rng = np.random.default_rng(0)
    coords = {"Rep_1": _coords(rng, 40), "Rep_2": _coords(rng, 40)}
    # Rep_1 is entirely State A, Rep_2 entirely State B. A correct cumulative
    # fraction contains both replicates at every point (50 % / 50 %); slicing the
    # concatenated table would report 100 % State A at 25 % and 50 %.
    rows = []
    for rep, state in (("Rep_1", "State A"), ("Rep_2", "State B")):
        for f in range(40):
            rows.append({"system": "S", "replicate": rep, "frame": f, "time_ps": f, "state": state})
    assignments = pd.DataFrame(rows)
    projections = assignments.assign(PC1=rng.normal(size=80), PC2=rng.normal(size=80))

    conv = compute_convergence_diagnostics(coords, projections, assignments, n_modes=5)
    stab = conv.population_stability.set_index("Fraction_Percent")
    for frac in (25, 50, 75, 100):
        assert abs(stab.loc[frac, "State A"] - 50.0) < 1e-9
        assert abs(stab.loc[frac, "State B"] - 50.0) < 1e-9
        assert stab.loc[frac, "Frames"] == 2 * max(1, int(40 * frac / 100))


def test_half_rmsip_pools_all_replicates_and_reports_each():
    rng = np.random.default_rng(1)
    coords = {"Rep_1": _coords(rng, 30), "Rep_2": _coords(rng, 30), "Rep_3": _coords(rng, 30)}
    assignments = pd.DataFrame(
        {
            "system": "S",
            "replicate": np.repeat(list(coords), 30),
            "frame": np.tile(np.arange(30), 3),
            "time_ps": np.tile(np.arange(30), 3),
            "state": "State A",
        }
    )
    projections = assignments.assign(PC1=rng.normal(size=90), PC2=rng.normal(size=90))
    conv = compute_convergence_diagnostics(coords, projections, assignments, n_modes=5)

    assert set(conv.rmsip_halves_per_replicate) == set(coords)
    assert 0.0 <= conv.rmsip_halves <= 1.0
    assert all(0.0 <= v <= 1.0 for v in conv.rmsip_halves_per_replicate.values())
    # Progressive overlap is computed for every fraction; 100 % is identity.
    assert set(conv.progressive_overlap) == {25, 50, 75, 100}
    assert abs(conv.progressive_overlap[100] - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in conv.progressive_overlap.values())


def test_assessment_wording_matches_values():
    rng = np.random.default_rng(2)
    coords = {"Rep_1": _coords(rng, 40)}
    assignments = pd.DataFrame(
        {
            "system": "S",
            "replicate": "Rep_1",
            "frame": np.arange(40),
            "time_ps": np.arange(40),
            "state": "State A",
        }
    )
    # A perfectly linear PC1 has cosine content close to 1 (diffusive signature).
    projections = assignments.assign(PC1=np.linspace(-1, 1, 40), PC2=rng.normal(size=40))
    conv = compute_convergence_diagnostics(coords, projections, assignments, n_modes=5)
    notes = " ".join(conv.assessment_notes)
    assert conv.cosine_content[1] > 0.7
    assert "High PC1 cosine content" in notes
    assert "Moderate PC1 cosine content" not in notes
