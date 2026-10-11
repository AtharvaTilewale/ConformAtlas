"""Tests for the documentation consistency helpers in validation/checks_docs.py
and for the static consistency of the shipped documentation/metadata files."""

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))

import checks_docs  # noqa: E402


def test_extract_code_blocks_languages_and_order():
    md = "text\n```bash\necho a\n```\nmore\n```python\nprint(1)\n```\n```text\nout\n```\n"
    blocks = checks_docs.extract_code_blocks(md)
    assert [b[0] for b in blocks] == ["bash", "python", "text"]
    assert blocks[0][1] == "echo a\n" and blocks[1][1] == "print(1)\n"


def test_run_code_blocks_reports_exit_codes(tmp_path):
    blocks = [("bash", "echo hi\n"), ("python", "import sys; sys.exit(3)\n"), ("text", "skip")]
    res = checks_docs.run_code_blocks(blocks, tmp_path, ROOT)
    assert [(i, lang, rc) for i, lang, rc, _ in res] == [(0, "bash", 0), (1, "python", 3)]


def test_bib_dois_parses_braces_and_quotes():
    bib = '@article{a, doi = {10.1000/x1}}\n@misc{b, DOI="10.1000/y2"}\n@book{c, title={no doi}}'
    assert checks_docs.bib_dois(bib) == ["10.1000/x1", "10.1000/y2"]


@pytest.mark.skipif(
    not (ROOT / "paper" / "paper.bib").exists(),
    reason="the manuscript (paper/) is kept out of the public repository",
)
def test_paper_bib_entries_all_have_dois_and_are_cited():
    bib = (ROOT / "paper" / "paper.bib").read_text()
    keys = re.findall(r"^@\w+\{([^,]+),", bib, re.M)
    dois = checks_docs.bib_dois(bib)
    assert len(keys) == len(dois), "every bib entry must carry a DOI"
    paper = (ROOT / "paper" / "paper.md").read_text()
    uncited = [k for k in keys if f"@{k}" not in paper]
    assert not uncited, f"bib entries not cited in paper.md: {uncited}"
    cited = set(re.findall(r"@([A-Za-z]+\d{4}[a-z]?)", paper))
    assert cited <= set(keys), f"citations without bib entry: {cited - set(keys)}"


def test_benchmark_table_matches_results_json():
    results = json.load(open(ROOT / "validation" / "results.json"))
    page = (ROOT / "docs" / "benchmarks.md").read_text()
    table = checks_docs.render_benchmark_tables(results, checks_docs._bench_header(ROOT))
    assert table in page
    # one data row per check, grouped under one heading per section
    # header row ("| Check |") per section plus one data row per check
    assert table.count("\n| ") - len({r["section"] for r in results}) == len(results)


def test_zenodo_consistent_with_citation_cff():
    assert checks_docs.zenodo_vs_citation(ROOT) == []
    zen = json.load(open(ROOT / ".zenodo.json"))
    assert "version" not in zen  # taken from the GitHub release tag
    cff = yaml.safe_load(open(ROOT / "CITATION.cff"))
    assert (
        zen["creators"][0]["name"]
        == "Tilewale, Atharva"
        == (f"{cff['authors'][0]['family-names']}, {cff['authors'][0]['given-names']}")
    )


def test_conda_recipe_dependencies_match_pyproject():
    pyproject = (ROOT / "pyproject.toml").read_text()
    deps = re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.M | re.S).group(1)
    names = {
        re.match(r'\s*"([A-Za-z0-9_-]+)', line).group(1).lower()
        for line in deps.strip().splitlines()
    }
    recipe = (ROOT / "conda-recipe" / "meta.yaml").read_text()
    run_block = recipe.split("  run:")[1].split("  run_constrained:")[0]
    recipe_names = {
        re.match(r"\s*-\s*([A-Za-z0-9_-]+)", line).group(1).lower()
        for line in run_block.strip().splitlines()
    }
    # conda-forge names matplotlib-base for the headless build
    recipe_names = {"matplotlib" if n == "matplotlib-base" else n for n in recipe_names}
    assert names <= recipe_names
    assert "python" in recipe_names


def test_tutorial_figures_exist_and_are_referenced():
    docs = ROOT / "docs"
    tutorial = (docs / "tutorial_neuraminidase.md").read_text()
    refs = re.findall(r"\]\((_static/tutorial/[^)]+)\)", tutorial)
    assert len(refs) >= 10
    for ref in refs:
        assert (docs / ref).exists(), ref
    for png in (docs / "_static" / "tutorial").glob("*.png"):
        assert f"_static/tutorial/{png.name}" in tutorial, f"unreferenced figure {png.name}"


def test_index_toctree_lists_new_pages():
    index = (ROOT / "docs" / "index.rst").read_text()
    for page in ("tutorial_neuraminidase", "faq", "benchmarks", "python_api", "contributing"):
        assert re.search(rf"^\s+{page}\s*$", index, re.M), page


def test_readthedocs_config_points_at_sphinx_conf():
    cfg = yaml.safe_load(open(ROOT / ".readthedocs.yaml"))
    assert cfg["version"] == 2
    assert cfg["sphinx"]["configuration"] == "docs/conf.py"
    assert cfg["sphinx"]["fail_on_warning"] is True
    pip_installs = [i for i in cfg["python"]["install"] if i.get("method") == "pip"]
    assert pip_installs and "docs" in pip_installs[0]["extra_requirements"]


def test_ci_has_codecov_upload_and_docs_job():
    ci = yaml.safe_load(open(ROOT / ".github" / "workflows" / "ci.yml"))
    steps = ci["jobs"]["test"]["steps"]
    assert any("codecov/codecov-action" in (s.get("uses") or "") for s in steps)
    assert any("--cov-report=xml" in (s.get("run") or "") for s in steps)
    docs_steps = ci["jobs"]["docs"]["steps"]
    assert any("sphinx -W" in (s.get("run") or "") for s in docs_steps)


@pytest.mark.parametrize(
    "path",
    [
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/scientific_question.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
    ],
)
def test_issue_templates_are_valid_yaml(path):
    data = yaml.safe_load(open(ROOT / path))
    assert isinstance(data, dict)
    if "body" in data:
        assert data["name"] and data["body"]


def test_code_of_conduct_is_contributor_covenant_2_1():
    text = (ROOT / "CODE_OF_CONDUCT.md").read_text()
    assert "Contributor Covenant" in text and "version 2.1" in text
    assert "https://www.contributor-covenant.org/version/2/1/code_of_conduct.html" in text
