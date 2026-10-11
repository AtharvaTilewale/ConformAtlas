# conda-forge recipe

`meta.yaml` follows the conda-forge pattern for a pure-Python (`noarch: python`)
package built from the PyPI source distribution. State of the recipe for 1.0.0:

- `sha256` is the hash of the published `conformatlas-1.0.0.tar.gz` on PyPI
  (`4cd9a903…1101f6`).
- `conda smithy recipe-lint` reports no problems, and `conda build` against the
  conda-forge channel builds the package and passes the recipe tests
  (`pip check`, `conformatlas --version/--help/doctor`, and an end-to-end
  `example` + `analyze` run on the package's own synthetic dataset). The sdist
  ships no test suite, so the recipe does not run `pytest`.
- Run dependencies are the core requirements from `pyproject.toml`, including
  MDAnalysis. WeasyPrint, Plotly, h5py and rich are optional and appear only
  under `run_constrained` (a version constraint, not an install).
- `license_file` lists the project licence and the licence of the vendored NGL
  viewer (`ngl.js`, MIT).

## Submitting (first time)

1. Add every maintainer's GitHub username under `extra: recipe-maintainers` in
   `meta.yaml` (each person must agree to maintain the feedstock).
2. Fork https://github.com/conda-forge/staged-recipes, create a branch, copy
   `meta.yaml` and `conda_build_config.yaml` to `recipes/conformatlas/`, and open
   a pull request using the template checklist.
3. When CI is green, comment `@conda-forge/help-python, ready for review!` and
   address the reviewer's comments.
4. After the merge, conda-forge creates `conda-forge/conformatlas-feedstock`
   and invites the maintainers (accept the invitation). The package is built and
   uploaded to the `conda-forge` channel automatically; it is installable with
   `conda install -c conda-forge conformatlas` after the channel index refreshes.

## Later releases

After each PyPI release, the conda-forge bot opens a pull request in the
feedstock that updates `version` and `sha256`; check that the dependency list
still matches `pyproject.toml`, wait for CI, and merge. Rebuilding without a new
version only requires incrementing `build: number`.

## Local checks

```bash
conda create -n condabuild -c conda-forge conda-build conda-smithy
conda activate condabuild
conda smithy recipe-lint conda-recipe        # needs the layout recipes/conformatlas/
conda build conda-recipe -c conda-forge --override-channels
```
