"""Software-engineering audit of the HydroEXA source tree.

This package is *meta*: it measures the codebase rather than the fluids.  It
exists because maintenance friction is a measurable quantity, and because the
two things that make a solver hard to change are structural, not stylistic:

* **code mass** — where the SLOC actually lives (and whether it is spread
  evenly or pinned in a few god-files), measured as comment-aware SLOC plus a
  Shannon entropy over the per-file distribution;
* **dependency structure** — who includes whom.  Fan-in measures edit
  amplification (how many files a change can break), fan-out measures
  fragility, cycles defeat both reasoning and incremental builds, and the
  condensation depth measures how far a change can ripple.

Everything here is standard library only, so it runs anywhere the repository
is checked out — no AMReX, no numpy, no plotfiles.

Typical use::

    from pyhydroexa.audit import TreeReport

    report = TreeReport.build(".")
    print(report.markdown())

Or from the command line::

    python -m pyhydroexa.audit --markdown docs/architecture.md
"""

from __future__ import annotations

from .entropy import (
    dependency_entropy,
    effective_count,
    hhi,
    instability,
    mass_entropy,
    normalized_entropy,
    shannon,
)
from .graph import DependencyGraph, build_graph
from .report import Finding, TreeReport
from .sloc import FileStats, count_sloc, iter_source_files

__all__ = [
    "DependencyGraph",
    "FileStats",
    "Finding",
    "TreeReport",
    "build_graph",
    "count_sloc",
    "dependency_entropy",
    "effective_count",
    "hhi",
    "instability",
    "iter_source_files",
    "mass_entropy",
    "main",
    "normalized_entropy",
    "shannon",
]


def main(argv: list[str] | None = None) -> int:
    """Console entry point -- delegates to :mod:`pyhydroexa.audit.__main__`.

    Args:
        argv: Arguments to parse; defaults to :data:`sys.argv`.

    Returns:
        Process exit code.
    """
    from .__main__ import main as _main

    return _main(argv)
