"""Sphinx configuration for the HydroEXA documentation tree.

One tree documents three things:

* the design notes in ``docs/`` itself (Markdown, via MyST),
* the Python companion package ``PyHydroEXA`` (autodoc / ``sphinx.ext.apidoc``),
* the C++ solver in ``include/`` and ``src/`` (Doxygen XML -> Breathe).

Nothing here lists solver files or Python modules by name.  The C++ side is fed
by the Doxygen ``INPUT`` directories and the Python side by walking
``PyHydroEXA/src/pyhydroexa`` at build time, so files can be moved, split or
merged without touching this configuration.  The intended direction is for
per-test analysis code to migrate *into* ``pyhydroexa`` (see the roadmap in
``index.md``), and those pages will appear automatically once it does.

The build environment is ``docs/pyproject.toml``: a uv project of its own,
excluded from the repository workspace, holding the Sphinx stack plus
``numpy``, ``scipy`` and ``matplotlib``.  Those three are what ``pyhydroexa``
imports at module level; everything else it uses is imported inside functions,
which is what lets this build run without GDAL or ``yt`` installed.  Adding a
module-level heavy import to the package therefore changes the docs
environment too -- the autodoc pass imports modules, it does not read them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
PY_SRC = REPO / "PyHydroEXA" / "src"
PY_PKG = PY_SRC / "pyhydroexa"
DOXY_XML = DOCS / "doxygen" / "xml"

sys.path.insert(0, str(PY_SRC))

# --------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------
project = "HydroEXA"
author = "HydroEXA contributors"
copyright = "2026, HydroEXA contributors"
version = release = "0.1"
language = "en"

# --------------------------------------------------------------------------
# Extensions
# --------------------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
]

# Breathe is registered only when the Doxygen XML tree exists, so the Python
# and prose halves of the docs still build on a machine without doxygen.
has_doxygen_xml = DOXY_XML.is_dir()
if has_doxygen_xml:
    extensions.append("breathe")
    breathe_projects = {"HydroEXA": str(DOXY_XML)}
    breathe_default_project = "HydroEXA"
    # Scaffolding mode: the index lists compounds only.  Rendering every member
    # signature of a GPU/AMReX codebase makes Sphinx's C++ domain choke on
    # device-lambda and template declarations; curated ``doxygenclass`` blocks
    # can add members per type once the doc-comment pass lands.
    breathe_default_members = ()
    breathe_show_include = False
    breathe_domain_by_extension = {"h": "cpp", "H": "cpp", "hpp": "cpp", "cpp": "cpp"}

# --------------------------------------------------------------------------
# Python API (automatic: recursive autosummary over the package)
# --------------------------------------------------------------------------
# One stub page per module, generated at build time into api/generated/.  The
# list of modules is never written down: `:recursive:` walks pyhydroexa, so
# moving or adding a module (including the planned analysis/ and plotting/
# split) documents itself.
autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# The documentation must build without the full geospatial/IO stack installed
# (a laptop prose edit, a docs-only CI job).  Mock only modules that are
# genuinely absent here; anything installed is documented for real.
_CANDIDATE_DEPS = [
    "fiona",
    "geopandas",
    "h5py",
    "imageio",
    "matplotlib",
    "numpy",
    "polars",
    "pyogrio",
    "rasterio",
    "scipy",
    "shapely",
    "yt",
]
autodoc_mock_imports = []
for _mod in _CANDIDATE_DEPS:
    try:
        _missing = importlib.util.find_spec(_mod) is None
    except (ImportError, ValueError):
        _missing = True
    if _missing:
        autodoc_mock_imports.append(_mod)
if autodoc_mock_imports:
    print(f"[docs] mocking unavailable modules: {', '.join(autodoc_mock_imports)}")

# --------------------------------------------------------------------------
# Markdown / math
# --------------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
html_theme = "furo"
html_title = "HydroEXA"
html_static_path = ["_static"] if (DOCS / "_static").is_dir() else []
html_last_updated_fmt = "%Y-%m-%d"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
exclude_patterns = [
    "_build",
    # The docs environment lives in docs/.venv, inside this source directory.
    # Sphinx scans the tree recursively, so without this it tries to build the
    # autosummary templates shipped inside site-packages as if they were pages.
    ".venv",
    "doxygen/xml",
    "doxygen/html",
    "Thumbs.db",
    ".DS_Store",
]

# --------------------------------------------------------------------------
# Cross references
# --------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

nitpicky = False
# Doxygen emits "include file not found" for AMReX/HDF5 headers it is not asked
# to parse; those are noise, not documentation defects.
suppress_warnings = ["myst.header"]


def setup(app):
    """Expose build-time flags so ``ifconfig`` can branch on them.

    ``docs/api/cpp.rst`` guards the Doxygen directives with
    ``.. ifconfig:: has_doxygen_xml``; ``ifconfig`` only sees values registered
    as Sphinx config values, not module-level names.
    """
    app.add_config_value("has_doxygen_xml", has_doxygen_xml, "env")
    return {"parallel_read_safe": True, "parallel_write_safe": True}

