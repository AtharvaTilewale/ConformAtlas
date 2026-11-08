"""Regression tests for defects found in the dev-branch audit (each with a known answer)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conformatlas.config import parse_comparison_config
from conformatlas.convergence import calculate_cosine_content, compute_convergence_diagnostics
from conformatlas.models import AnalysisConfig
from conformatlas.pca import compute_cartesian_pca
from conformatlas.pipeline import run_analysis
from conformatlas.streaming import eigendecompose_covariance
from conformatlas.trajectory import HAS_MDANALYSIS, map_group_to_mda_selection
from conformatlas.uncertainty import compute_replicate_statistics

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "demo_dataset" / "WT"
ADK = ROOT / "examples" / "adk"
needs_mda = pytest.mark.skipif(not HAS_MDANALYSIS, reason="MDAnalysis is required")


# --------------------------------------------------------------------- populations
def test_weighted_populations_with_several_replicates():
    fa = pd.DataFrame(
        {
            "replicate": ["Rep_1"] * 4 + ["Rep_2"] * 4,
            "state": ["State A", "State A", "State B", "State B"] * 2,
        }
    )
    w = np.array([9.0, 9.0, 1.0, 1.0] * 2)
    summary, detailed, kind = compute_replicate_statistics(fa, weights=w)
    assert kind == "between-replicate"
    assert detailed["State A"].tolist() == pytest.approx([90.0, 90.0])
    a = summary.set_index("State").loc["State A", "Mean_Population_Percent"]
    assert a == pytest.approx(90.0)
    # unweighted call unchanged
    summary_u, _, _ = compute_replicate_statistics(fa)
    assert summary_u.set_index("State").loc["State A", "Mean_Population_Percent"] == pytest.approx(
        50.0
    )


def test_single_replicate_blocks_never_empty():
    fa = pd.DataFrame({"replicate": ["Rep_1"] * 4, "state": ["State A"] * 4})
    summary, detailed, kind = compute_replicate_statistics(fa)
    assert kind == "intra-trajectory-block"
    assert (detailed["Total_Frames"] > 0).all()
    assert summary["Mean_Population_Percent"].iloc[0] == pytest.approx(100.0)


# --------------------------------------------------------------------- convergence
def _proj(series_by_rep: dict[str, np.ndarray]) -> pd.DataFrame:
    parts = []
    for rep, x in series_by_rep.items():
        n = len(x)
        parts.append(
            pd.DataFrame(
                {
                    "replicate": rep,
                    "frame": np.arange(n),
                    "time_ps": np.arange(n) * 10.0,
                    "PC1": x,
                    "PC2": x[::-1],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_cosine_content_is_computed_per_replicate():
    rng = np.random.default_rng(0)
    n = 2000
    # two stationary replicates in different places: concatenated they look like a step
    proj = _proj({"Rep_1": rng.normal(-3, 1, n), "Rep_2": rng.normal(3, 1, n)})
    concatenated = calculate_cosine_content(proj["PC1"].to_numpy())
    assert concatenated > 0.5  # the artefact the per-replicate evaluation removes
    fa = proj.assign(state="State A")
    conv = compute_convergence_diagnostics(None, proj, fa)
    per = conv.cosine_content_per_replicate
    assert set(per) == {"Rep_1", "Rep_2"}
    assert conv.cosine_content[1] == pytest.approx(max(per["Rep_1"]["PC1"], per["Rep_2"]["PC1"]))
    assert conv.cosine_content[1] < 0.05
    # a single diffusive replicate keeps its high value
    t = np.linspace(0, 1, 1000)
    one = _proj({"Rep_1": np.cos(np.pi * t)})
    conv1 = compute_convergence_diagnostics(None, one, one.assign(state="State A"))
    assert conv1.cosine_content[1] > 0.99


def test_n_eff_uses_the_landscape_coordinates():
    rng = np.random.default_rng(1)
    n = 3000
    fast = rng.normal(size=n)
    slow = np.zeros(n)
    for i in range(1, n):  # AR(1), phi = 0.99: g = (1 + phi) / (1 - phi) = 199
        slow[i] = 0.99 * slow[i - 1] + rng.normal()
    proj = _proj({"Rep_1": fast})
    cv = proj.assign(CV1=slow, CV2=slow)
    fa = cv.assign(state="State A")
    on_pcs = compute_convergence_diagnostics(None, proj, fa)
    on_cvs = compute_convergence_diagnostics(
        None, proj, fa, cv_source=cv, cv_columns=("CV1", "CV2")
    )
    assert on_pcs.statistical_inefficiency["Rep_1"]["PC1"] < 3
    assert on_cvs.statistical_inefficiency["Rep_1"]["CV1"] > 50
    assert on_cvs.n_eff["Rep_1"] < on_pcs.n_eff["Rep_1"] / 10
    labelled = compute_convergence_diagnostics(
        None, proj, fa, cv_source=proj, cv_columns=("PC1", "PC2"), cv_labels=("PC1", "PC2")
    )
    assert set(labelled.statistical_inefficiency["Rep_1"]) == {"PC1", "PC2"}


def test_population_stability_is_weighted():
    fa = pd.DataFrame(
        {
            "replicate": ["Rep_1"] * 4,
            "frame": range(4),
            "time_ps": [0.0, 10, 20, 30],
            "state": ["State A", "State B"] * 2,
        }
    )
    proj = fa.assign(PC1=[0.0, 1, 2, 3], PC2=[1.0, 0, 1, 0])
    conv = compute_convergence_diagnostics(None, proj, fa, weights=np.array([3.0, 1.0, 3.0, 1.0]))
    last = conv.population_stability.iloc[-1]
    assert last["State A"] == pytest.approx(75.0)


# --------------------------------------------------------------------- PCA signs
def test_every_pca_route_uses_the_same_sign_convention():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(150, 40, 3)) * np.linspace(0.2, 2.0, 40)[None, :, None]
    reps = {"Rep_1": a[:80], "Rep_2": a[80:]}
    # dense eigh route (D = 120 < N = 150)
    _, ev_dense, _, _, mean = compute_cartesian_pca(reps, n_components=5)
    flat = np.concatenate([r.reshape(len(r), -1) for r in reps.values()])
    # streaming dense route on the equal-replicate covariance
    cov = np.zeros((120, 120))
    for r in reps.values():
        c = r.reshape(len(r), -1) - mean.reshape(-1)
        cov += 0.5 * (c.T @ c) / len(r)
    _, ev_stream, _, _ = eigendecompose_covariance(cov, 5)
    # dual SVD route (fewer frames than coordinates)
    _, ev_svd, _, _, _ = compute_cartesian_pca({"Rep_1": a[:30]}, n_components=5)
    for ev in (ev_dense, ev_stream, ev_svd):
        idx = np.argmax(np.abs(ev), axis=0)
        assert (ev[idx, np.arange(ev.shape[1])] > 0).all()
    np.testing.assert_allclose(ev_dense, ev_stream, atol=1e-8)
    assert flat.shape == (150, 120)


# --------------------------------------------------------------------- selections
@needs_mda
def test_c_alpha_group_excludes_calcium_ions(tmp_path):
    import MDAnalysis as mda

    pdb = tmp_path / "ca.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
        "HETATM    4 CA    CA B 101      10.000  10.000  10.000  1.00  0.00          CA\n"
        "END\n",
        encoding="utf-8",
    )
    u = mda.Universe(str(pdb))
    sel = u.select_atoms(map_group_to_mda_selection("C-alpha"))
    assert [(a.resname, a.name) for a in sel] == [("ALA", "CA")]


# --------------------------------------------------------------------- compare config
def _write_cfg(tmp_path, extra: str) -> Path:
    p = tmp_path / "cmp.yaml"
    p.write_text(
        "project: t\nsystems:\n  - name: WT\n    topology: top.pdb\n    trajectories: [a.xtc]\n"
        + extra,
        encoding="utf-8",
    )
    return p


def test_compare_config_reads_pca_weighting(tmp_path):
    assert (
        parse_comparison_config(_write_cfg(tmp_path, "pca_weighting: frames\n")).pca_weighting
        == "frames"
    )
    cfg = parse_comparison_config(_write_cfg(tmp_path, "pca:\n  weighting: frames\n"))
    assert cfg.pca_weighting == "frames"
    assert parse_comparison_config(_write_cfg(tmp_path, "")).pca_weighting == "equal-replicate"


@pytest.mark.parametrize(
    "extra, match",
    [
        ("smoth_sigma: 1.0\n", "Unknown key 'smoth_sigma'"),
        ("pca:\n  refrence: combined\n", "Unknown key 'refrence'"),
        ("pca:\n  reference: reference_system\n", "pca.reference must be"),
        ("pca_weighting: balanced\n", "pca_weighting must be"),
    ],
)
def test_compare_config_rejects_unknown_or_invalid_keys(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        parse_comparison_config(_write_cfg(tmp_path, extra))


def test_compare_config_accepts_top_level_reference_system(tmp_path):
    cfg = parse_comparison_config(_write_cfg(tmp_path, "reference_system: WT\n"))
    assert cfg.reference_system == "WT"


# --------------------------------------------------------------------- pipeline
def _demo_cfg(tmp_path, **kw) -> AnalysisConfig:
    base = dict(
        topology=DEMO / "topology.pdb",
        trajectories=[DEMO / "rep1.xtc", DEMO / "rep2.xtc"],
        output_dir=tmp_path / "out",
        bins=20,
        n_bootstrap=0,
        progress=False,
    )
    base.update(kw)
    return AnalysisConfig(**base)


@needs_mda
def test_msm_is_skipped_for_weighted_frames(tmp_path):
    w1, w2 = tmp_path / "w1.dat", tmp_path / "w2.dat"
    w1.write_text("\n".join("1.0" for _ in range(100)), encoding="utf-8")
    w2.write_text("\n".join("2.0" for _ in range(100)), encoding="utf-8")
    res = run_analysis(_demo_cfg(tmp_path, weights_files=[w1, w2], msm=True, msm_lag=2))
    assert "msm" not in res.extras
    assert not (tmp_path / "out" / "msm").exists()
    assert any("MSM skipped" in w for w in res.warnings)


@needs_mda
def test_fel_backend_gromacs_is_deprecated_and_recorded_as_python(tmp_path):
    res = run_analysis(_demo_cfg(tmp_path, fel_backend="gromacs"))
    assert any("--fel-backend gromacs is deprecated" in w for w in res.warnings)
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["fel_backend"] == "python"


@pytest.mark.skipif(shutil.which("gmx") is None, reason="GROMACS is required")
@pytest.mark.parametrize("kw", [{"stride": 2}, {"begin_ps": 100.0}, {"equilibration": "auto"}])
def test_gromacs_backend_rejects_unapplied_frame_selection(tmp_path, kw):
    with pytest.raises(ValueError, match="cannot be used with --pca-backend gromacs"):
        run_analysis(_demo_cfg(tmp_path, pca_backend="gromacs", **kw))


@needs_mda
def test_streaming_equilibration_keeps_the_time_window_of_every_replicate(tmp_path):
    """-b must survive the streaming re-read for replicates without a transient."""
    import MDAnalysis as mda

    u = mda.Universe(str(ADK / "adk_ca.pdb"), str(ADK / "adk_ca.xtc"))
    rng = np.random.default_rng(3)
    paths = []
    # Rep_1 stationary (one structure + independent noise: no transient, t0 = 0);
    # Rep_2 starts in the opening transient of the ADK run (frames 0-299)
    p1 = tmp_path / "r1.xtc"
    base = u.trajectory[3000].positions.copy()
    with mda.Writer(str(p1), n_atoms=u.atoms.n_atoms) as w:
        for k in range(300):
            u.atoms.positions = base + rng.normal(0.0, 0.3, base.shape)
            u.trajectory.ts.time = 240.0 * k
            w.write(u.atoms)
    paths.append(p1)
    p2 = tmp_path / "r2.xtc"
    with mda.Writer(str(p2), n_atoms=u.atoms.n_atoms) as w:
        for k, ts in enumerate(u.trajectory[0:300]):
            ts.time = 240.0 * k
            w.write(u.atoms)
    paths.append(p2)
    common = dict(
        topology=ADK / "adk_ca.pdb",
        trajectories=paths,
        index_file=ADK / "index.ndx",
        lsq_group="C-alpha",
        cov_group="C-alpha",
        begin_ps=2400.0,
        equilibration="auto",
        bins=20,
        n_bootstrap=0,
        progress=False,
    )
    times = {}
    for engine in ("memory", "streaming"):
        res = run_analysis(AnalysisConfig(output_dir=tmp_path / engine, engine=engine, **common))
        times[engine] = res.pca.projections.groupby("replicate")["time_ps"].agg(["min", "count"])
    eq = pd.read_csv(tmp_path / "streaming" / "convergence" / "equilibration.csv")
    t0 = dict(zip(eq["Replicate"], eq["t0_index"]))
    assert t0["Rep_1"] == 0 and t0["Rep_2"] > 0  # the case the old begin map got wrong
    pd.testing.assert_frame_equal(times["memory"], times["streaming"])
    assert (times["streaming"]["min"] >= 2400.0).all()
