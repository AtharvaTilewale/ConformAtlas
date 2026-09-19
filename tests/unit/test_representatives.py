"""Unit tests for representative structure extraction and provenance mapping."""

import json

from conformatlas.models import BasinState
from conformatlas.representatives import extract_representative_structures


def test_representative_provenance_and_metadata(tmp_path):
    """Confirm representative frame provenance mapping and metadata serialization."""
    out_dir = tmp_path / "test_rep_output"

    state_a = BasinState(
        state_id=1,
        label="State A",
        population=0.75,
        min_pc1=1.2,
        min_pc2=-0.5,
        min_free_energy=0.0,
        representative_replicate="Rep_1",
        representative_frame=42,
        representative_time_ps=420.0,
        representative_distance=0.015,
    )

    # Empty dummy topologies / trajectories for metadata test
    dummy_top = tmp_path / "top.pdb"
    dummy_top.write_text("HEADER DUMMY\n")
    dummy_traj = tmp_path / "traj.xtc"
    dummy_traj.write_text("DUMMY")

    _ = extract_representative_structures(
        [state_a],
        {"System": dummy_top},
        {"Rep_1": dummy_traj},
        output_dir=out_dir,
        system_name="System",
    )

    state_dir = out_dir / "states" / "State_A"
    assert state_dir.exists()
    meta_path = state_dir / "metadata.json"
    assert meta_path.exists()

    with meta_path.open() as f:
        meta = json.load(f)

    assert meta["state"] == "State A"
    assert meta["replicate"] == "Rep_1"
    assert meta["frame_index"] == 42
    assert meta["time_ps"] == 420.0
    assert meta["population_percent"] == 75.0
    assert meta["distance_from_basin_minimum"] == 0.015
