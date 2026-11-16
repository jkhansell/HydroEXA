"""PyHydroEXA -- Python tools for HydroEXA simulation data.

The package is split by responsibility, so importing one layer does not
drag in the others:

``preprocessing``
    rasters, DEMs and user fields -> one HydroEXA HDF5 input file.
``io``
    read plotfiles: metadata, field data, AMR block geometry (``yt``).
``analysis``
    reduce data to numbers: profiles along arbitrary lines.
``plotting``
    draw data: field maps, AMR overlays, profile plots, animations.
``audit``
    measure the source tree itself: SLOC entropy and coupling.

Only the CLI is imported here, and only because every console script
hangs off it.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for the ``pyhydroexa`` console script."""
    from .cli import main as _cli_main

    _cli_main()
