"""Interpretive sentences, PDB B-factor colouring, resource scan and error helpers."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from conformatlas.errors import (
    MalformedConfigError,
    MissingDependencyError,
    UnknownGroupError,
    available_groups,
    malformed_config_from_yaml_error,
    selection_error_group,
)
from conformatlas.models import (
    AnalysisConfig,
    AnalysisResults,
    BasinState,
    ConvergenceResults,
    FELGrid,
    PCAResults,
    TrajectoryInfo,
)
from conformatlas.report import (
    binomial_se_percent,
    external_resource_urls,
    interpretive_sentences,
    pdb_with_bfactor,
)
from conformatlas.utils import R_GAS_CONSTANT

T = 310.0


def _results(tmp_path, with_boot=True, n_eff=(11.0, 15.0), rmsip=0.6, cos=0.01):
    n = 16
    x = np.linspace(-3, 3, n + 1)
    xc = 0.5 * (x[1:] + x[:-1])
    X, Y = np.meshgrid(xc, xc)
    p = np.exp(-0.5 * (X**2 + Y**2))
    p /= p.sum()
    g = -R_GAS_CONSTANT * T * np.log(p / p.max())
    fel = FELGrid(x, x, xc, xc, p, g, T)
    states = [
        BasinState(1, "State A", 0.695, 0.0, 0.0, 0.0, "Rep_1", 3, 30.0, 0.1, basin_id=1),
        BasinState(2, "State B", 0.295, 1.5, -0.8, 2.38, "Rep_1", 7, 70.0, 0.1, basin_id=2),
    ]
    fa = pd.DataFrame(
        {
            "replicate": ["Rep_1"] * 100 + ["Rep_2"] * 100,
            "time_ps": list(np.arange(100) * 10.0) * 2,
            "CV1": np.zeros(200),
            "CV2": np.zeros(200),
            "state": ["State A"] * 139 + ["State B"] * 59 + ["unassigned"] * 2,
        }
    )
    pop = pd.DataFrame(
        {
            "State": ["State B", "State A"],
            "Mean_Population_Percent": [29.5, 69.5],
            "SD": [0.7, 0.7],
            "SEM": [0.5, 0.5],
            "CI_95_Lower": [23.1, 63.1],
            "CI_95_Upper": [35.9, 75.9],
            "Num_Replicates": [2, 2],
            "Uncertainty_Type": ["between-replicate"] * 2,
        }
    )
    if with_boot:
        pop["Bootstrap_CI_95_Lower"] = [15.1125, 54.3375]
        pop["Bootstrap_CI_95_Upper"] = [44.5, 83.8875]
    conv = ConvergenceResults(
        rmsip_halves=rmsip,
        progressive_overlap={25: 0.7, 100: 1.0},
        cosine_content={1: cos, 2: 0.002},
        sampling_assessment="Insufficient evidence",
        rmsip_halves_per_replicate={"Rep_1": 0.57, "Rep_2": 0.59},
        statistical_inefficiency={
            "Rep_1": {"PC1": 9.08, "PC2": 1.5},
            "Rep_2": {"PC1": 6.7, "PC2": 1.9},
        },
        n_eff={"Rep_1": n_eff[0], "Rep_2": n_eff[1]},
    )
    pca = PCAResults(
        eigenvalues=np.array([5.0, 3.5, 1.5]),
        eigenvectors=np.eye(3),
        explained_variance=np.array([50.0, 35.0, 15.0]),
        cumulative_variance=np.array([50.0, 85.0, 100.0]),
        projections=pd.DataFrame(),
    )
    barriers = pd.DataFrame(
        [
            {
                "state_i": "State A",
                "state_j": "State B",
                "saddle_kJ_mol": 4.0,
                "barrier_from_i_kJ_mol": 4.0,
                "barrier_from_i_ci95_low": 2.5,
                "barrier_from_i_ci95_high": 5.5,
                "barrier_from_j_kJ_mol": 1.62,
            }
        ]
    )
    cfg = AnalysisConfig(topology=tmp_path / "top.pdb", trajectories=[], temperature=T)
    return AnalysisResults(
        config=cfg,
        trajectories_info=[
            TrajectoryInfo(tmp_path / "a.xtc", "Rep_1", "S", 100, 0.0, 990.0, 10.0),
            TrajectoryInfo(tmp_path / "b.xtc", "Rep_2", "S", 100, 0.0, 990.0, 10.0),
        ],
        pca=pca,
        fel=fel,
        states=states,
        unassigned_population=0.01,
        frame_assignments=fa,
        convergence=conv,
        replicate_statistics=pop,
        output_dir=tmp_path,
        extras={"barriers": barriers},
        warnings=["Rep_2: equilibration detection suggests discarding the first 53 frames"],
    )


def _nums(text):
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]


def test_population_sentences_quote_bootstrap_ci(tmp_path):
    res = _results(tmp_path)
    out = interpretive_sentences(res)
    sa = [s for s in out["populations"] if s.startswith("The population of State A")][0]
    assert "69.5 %" in sa and "[54.3, 83.9]" in sa and "block-bootstrap" in sa
    width = 83.8875 - 54.3375
    assert f"about {width:.0f} percentage points" in sa
    vals = out["values"]["populations"]
    assert {v["state"]: v["width"] for v in vals}["State A"] == pytest.approx(width)
    assert any("only qualitative" in s for s in out["populations"])


def test_population_sentences_fall_back_to_replicate_ci(tmp_path):
    out = interpretive_sentences(_results(tmp_path, with_boot=False))
    sa = [s for s in out["populations"] if "State A" in s][0]
    assert "between-replicate" in sa and "[63.1, 75.9]" in sa
    assert any("only 2 replicate" in s for s in out["populations"])


def test_n_eff_sentences_and_binomial_se(tmp_path):
    res = _results(tmp_path)
    out = interpretive_sentences(res)
    assert out["n_eff"][0].startswith("Only about 26 of the 200 frames")
    assert "Rep_1: g = 9.1" in out["n_eff"][0]
    se = 100 * math.sqrt(0.695 * 0.305 / 26.0)
    assert f"about {se:.1f} percentage points" in out["n_eff"][1]
    assert out["values"]["binomial_se"]["se_percent"] == pytest.approx(se)
    assert any("Fewer than 50 effective samples" in s for s in out["n_eff"])
    big = interpretive_sentences(_results(tmp_path, n_eff=(400.0, 400.0)))
    assert not any("Fewer than 50" in s for s in big["n_eff"])
    assert binomial_se_percent(0.5, 100) == pytest.approx(5.0)
    assert math.isnan(binomial_se_percent(0.5, 0))


@pytest.mark.parametrize(
    "rmsip,word", [(0.85, "high overlap"), (0.6, "moderate overlap"), (0.3, "low overlap")]
)
def test_rmsip_wording(tmp_path, rmsip, word):
    out = interpretive_sentences(_results(tmp_path, rmsip=rmsip))
    s = out["convergence"][0]
    assert f"is {rmsip:.2f}" in s and word in s
    if word == "high overlap":
        assert "reproducible" in s
    else:
        assert "still changing" in s


def test_cosine_and_per_replicate_rmsip(tmp_path):
    out = interpretive_sentences(_results(tmp_path, cos=0.7))
    assert any("cosine content of PC1 is 0.70" in s and "drifting" in s for s in out["convergence"])
    assert any("from 0.57 (Rep_1) to 0.59 (Rep_2)" in s for s in out["convergence"])


def test_barrier_sentence_in_kT(tmp_path):
    out = interpretive_sentences(_results(tmp_path))
    s = out["barriers"][0]
    kT = R_GAS_CONSTANT * T
    assert "barrier of 4.0 kJ/mol (95 % CI [2.5, 5.5])" in s
    assert f"= {4.0 / kT:.1f} kT" in s and "reverse barrier is 1.6 kJ/mol" in s
    assert "lower bounds" in s


def test_summary_and_not_shown(tmp_path):
    res = _results(tmp_path)
    out = interpretive_sentences(res)
    kT = R_GAS_CONSTANT * T
    dg = -kT * math.log(0.295 / 0.695)
    assert f"{dg:.1f} kJ/mol" in out["summary"][0] and "99.0 %" in out["summary"][0]
    ns = out["not_shown"][0]
    assert "1.98 ns" in ns and "85.0 %" in ns and "15.0 %" in ns
    assert "does not show" in ns and "Absolute free energies" in ns
    assert out["warnings"] == res.warnings
    assert out["values"]["explained_variance_fel_pcs"] == pytest.approx(85.0)


def test_pdb_with_bfactor_writes_rmsf_column():
    pdb = (
        "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C\n"
        "ATOM      3  CA  GLY A   2      12.000   7.000  -5.000  1.00  0.00           C\n"
        "HETATM    4  O   HOH B   9       1.000   1.000   1.000  1.00  0.00           O\n"
        "END\n"
    )
    rmsf = pd.DataFrame({"chain": ["A", "A"], "residue_number": [1, 2], "RMSF": [0.7447, 1.2]})
    out = pdb_with_bfactor(pdb, rmsf)
    lines = out.splitlines()
    assert [ln[60:66] for ln in lines[:4]] == ["  0.74", "  0.74", "  1.20", "  0.00"]
    # coordinates and element columns untouched
    assert lines[0][30:54] == pdb.splitlines()[0][30:54] and lines[0][76:78] == " N"
    assert lines[-1] == "END"
    # no RMSF table -> zeros, structure preserved
    assert pdb_with_bfactor(pdb, None).splitlines()[1][60:66] == "  0.00"


def test_external_resource_urls():
    html = (
        '<html><head><script src="https://cdn.plot.ly/plotly.min.js"></script>'
        '<link rel="stylesheet" href="https://fonts.example/x.css">'
        "<style>body{background:url(https://img.example/a.png)}</style></head>"
        '<body><a href="https://doi.org/10.1/abc">citation</a>'
        '<img src="data:image/png;base64,AAAA">'
        "<script>var u = 'https://inside.script/not-a-resource';</script></body></html>"
    )
    urls = external_resource_urls(html)
    assert urls == [
        "https://cdn.plot.ly/plotly.min.js",
        "https://fonts.example/x.css",
        "https://img.example/a.png",
    ]
    assert external_resource_urls("<a href='https://doi.org/x'>c</a><img src='data:...'>") == []


def test_error_helpers(tmp_path):
    try:
        yaml.safe_load("a: 1\nb: [1, 2\nc: 3\n")
    except yaml.YAMLError as e:
        err = malformed_config_from_yaml_error(tmp_path / "c.yaml", e)
    assert isinstance(err, MalformedConfigError) and err.exit_code == 2
    assert err.line is not None and "line" in str(err) and "c.yaml" in str(err)
    dep = MissingDependencyError("weasyprint", "pip install weasyprint", "PDF export")
    assert "pip install weasyprint" in str(dep) and dep.exit_code == 2
    assert selection_error_group(ValueError("Selection 'Foo' matched 0 atoms in x.pdb.")) == "Foo"
    assert (
        selection_error_group(
            ValueError("Selection 'C-alpha' (query: 'name CA') matched 0 atoms in x")
        )
        == "C-alpha"
    )
    assert selection_error_group(ValueError("other")) is None
    ndx = tmp_path / "i.ndx"
    ndx.write_text("[ C-alpha ]\n1 2 3\n[ ZMR ]\n4 5\n", encoding="utf-8")
    names, source = available_groups(None, ndx)
    assert names[:2] == ["C-alpha", "ZMR"] and "Backbone" in names and "i.ndx" in source
    ug = UnknownGroupError("Foo", names, source)
    assert "Foo" in str(ug) and "ZMR" in str(ug)
