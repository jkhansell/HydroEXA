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
SPHINXBUILD ?= uv run sphinx-build
# Advisory while the doc-comment pass is pending: collect all problems, fail
# nothing.  `make strict` flips to -W once the headers are documented.
SPHINXOPTS ?= --keep-going

.PHONY: help doxygen html strict linkcheck clean

help:
	@echo "targets: doxygen | html | strict | linkcheck | clean"

doxygen:
	@command -v $(DOXYGEN) >/dev/null 2>&1 || { \
		echo "doxygen not found -- skipping C++ API pages (set DOXYGEN=...)"; \
		exit 0; }
	@mkdir -p doxygen
	cd doxygen && $(DOXYGEN) Doxyfile

html: doxygen
	$(SPHINXBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"

strict: SPHINXOPTS = -W --keep-going
strict: html

linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck"

clean:
	rm -rf "$(BUILDDIR)" doxygen/xml doxygen/html api/generated
