.. ConformAtlas documentation master file

ConformAtlas Documentation
==========================

*Mapping conformational landscapes from molecular simulations*

.. image:: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10 | 3.11 | 3.12
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT
.. image:: https://github.com/AtharvaTilewale/ConformAtlas/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/AtharvaTilewale/ConformAtlas/actions/workflows/ci.yml
   :alt: CI Status

**ConformAtlas** turns molecular dynamics trajectories into a quantitative map of conformational space: principal components (or dihedral PCA / TICA), a free-energy landscape with bootstrap uncertainties, its metastable states, barriers and kinetics, optional Markov state models, condition comparisons in a shared PCA space, and a convergence assessment. It runs as one command (``conformatlas analyze``) and as a Python library.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   cli
   tutorial_adk
   faq

.. toctree::
   :maxdepth: 2
   :caption: Scientific Methods & Theory

   methods
   interpretation
   convergence
   multiple_replicates
   comparison
   basin_analysis
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: History

   outputs
   CURRENT_ARCHITECTURE

.. toctree::
   :maxdepth: 2
   :caption: Python API Reference

   python_api
   api

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing

Citing ConformAtlas
===================

Please cite the Zenodo archive, `doi:10.5281/zenodo.22847857 <https://doi.org/10.5281/zenodo.22847857>`_.
This concept DOI covers every release and resolves to the latest one; each
release also has its own version DOI, listed on the Zenodo record, for citing
the exact version you used. A machine-readable citation is in ``CITATION.cff``.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
