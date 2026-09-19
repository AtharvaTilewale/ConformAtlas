"""Sphinx configuration file for ConformAtlas documentation on ReadTheDocs."""

import os
import sys
from datetime import datetime

# Insert source code directory to sys.path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

import conformatlas

project = "ConformAtlas"
author = "Atharva Tilewale"
copyright = f"2024-{datetime.now().year}, {author} and ConformAtlas Contributors"
version = conformatlas.__version__
release = conformatlas.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# MyST parser configuration for Markdown files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 3

master_doc = "index"

# HTML output styling
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_context = {
    "display_github": True,
    "github_user": "AtharvaTilewale",
    "github_repo": "ConformAtlas",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
