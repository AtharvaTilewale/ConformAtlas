# conda-forge recipe draft

`meta.yaml` follows the conda-forge pattern for a pure-Python (`noarch: python`)
package built from the PyPI source distribution. To submit it:

1. Release `conformatlas` on PyPI (tag `v1.0.0`; `.github/workflows/publish.yml`
   publishes on a GitHub release).
2. Replace the `sha256` value in `meta.yaml` with the hash of the uploaded sdist
   (`pip download conformatlas==1.0.0 --no-binary :all: --no-deps && sha256sum conformatlas-1.0.0.tar.gz`).
3. Fork https://github.com/conda-forge/staged-recipes, copy `meta.yaml` and
   `conda_build_config.yaml` to `recipes/conformatlas/`, and open a pull request.
   Lint locally with `conda smithy recipe-lint recipes/conformatlas` and build with
   `conda build recipes/conformatlas` (or `python build-locally.py` in staged-recipes).

WeasyPrint, Plotly, h5py and rich are optional at run time and are declared under
`run_constrained`; MDAnalysis is a run dependency.
