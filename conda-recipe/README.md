# conda-forge recipe

`recipe.yaml` is the recipe submitted to conda-forge (v1 format, built with
rattler-build; the older `meta.yaml` format is deprecated for new recipes). It
builds a pure-Python `noarch: python` package from the PyPI source distribution.

- `source.sha256` is the hash of the published `conformatlas-<version>.tar.gz`
  on PyPI. It changes with every release.
- Run dependencies are the core requirements from `pyproject.toml`, including
  MDAnalysis. WeasyPrint, Plotly, h5py and rich are optional and appear only
  under `run_constraints` (a version constraint, not an install).
- `license_file` lists the project licence and the licence of the vendored NGL
  viewer (`ngl.js`, MIT).
- The sdist ships no test suite, so the recipe tests are `pip check` and the
  import of the main modules on the oldest and the newest Python available on
  conda-forge, plus a smoke test of the installed CLI (`--version`, `--help`,
  `doctor`, then `example` and `analyze` on the package's own synthetic dataset).

## Submission status

Submitted as https://github.com/conda-forge/staged-recipes/pull/35139. After the
merge, the feedstock `conda-forge/conformatlas-feedstock` is created and the
maintainers listed under `extra: recipe-maintainers` are invited to it.

## Releasing a new version

1. Publish the new version on PyPI (a GitHub release triggers `publish.yml`).
2. After the merge of the first submission, the conda-forge bot opens a pull
   request in the feedstock that updates `version` and `sha256`. Check that the
   dependency list still matches `pyproject.toml`, wait for CI and merge.
3. Keep `recipe.yaml` in this directory in step with the feedstock; the test
   `test_conda_recipe_dependencies_match_pyproject` checks the dependency list
   against `pyproject.toml`.

Rebuilding without a new version only requires incrementing `build: number`.

## Local checks

```bash
conda create -n condabuild -c conda-forge rattler-build conda-smithy
conda activate condabuild
rattler-build build --recipe conda-recipe/recipe.yaml -m conda-recipe/variants.yaml -c conda-forge
# lint: copy recipe.yaml to a folder recipes/conformatlas/, then
conda smithy recipe-lint recipes/conformatlas
```
