# --------------------------------------------------------------------------
# HydroEXA documentation build
#
#   make -C docs html       Doxygen XML -> Sphinx HTML (docs/_build/html)
#   make -C docs doxygen    Doxygen XML only
#   make -C docs linkcheck check outbound links
#   make -C docs clean      remove every generated artifact
#
# Override the tools if needed, e.g.
#   make -C docs html DOXYGEN=/opt/doxygen/bin/doxygen
# --------------------------------------------------------------------------

SHELL      := bash
SOURCEDIR  := .
BUILDDIR   := _build
DOXYFILE   := doxygen/Doxyfile

DOXYGEN    ?= doxygen
# The Sphinx environment is docs/pyproject.toml -- a uv project of its own,
# outside the repository workspace, so it holds the Sphinx stack and the three
# libraries autodoc imports, and nothing else.  `--locked` fails when the lock
# does not match pyproject.toml instead of silently resolving something else.
SPHINXBUILD ?= uv run --locked sphinx-build
# The doc-comment pass is complete, so warnings are now failures: an
# undocumented C++ entity stops Doxygen (WARN_AS_ERROR in the Doxyfile) and a
# warning from Sphinx stops this build.  `--keep-going` keeps reporting the
# rest of the tree instead of stopping at the first one.
SPHINXOPTS ?= -W --keep-going

.PHONY: help sync doxygen html strict linkcheck clean

help:
	@echo "targets: sync | doxygen | html | strict | linkcheck | clean"

sync:
	uv sync --locked

doxygen:
	@command -v $(DOXYGEN) >/dev/null 2>&1 || { \
		echo "doxygen not found -- skipping C++ API pages (set DOXYGEN=...)"; \
		exit 0; }; \
	mkdir -p doxygen && cd doxygen && $(DOXYGEN) Doxyfile

html: doxygen
	$(SPHINXBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"

# Historic alias for the same strict build `html` now performs.
strict: html

linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck"

clean:
	rm -rf "$(BUILDDIR)" doxygen/xml doxygen/html api/generated
