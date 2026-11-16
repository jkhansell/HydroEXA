"""Drawing HydroEXA data: field maps, AMR overlays, profiles, animations.

Everything here depends on :mod:`pyhydroexa.io` for the data and on
matplotlib for the drawing, which is the whole point of the split: the
reader layer stays usable on a headless machine with no plotting stack.

Example usage
-------------
>>> from pyhydroexa.io import Plotfile, PlotfileSeries
>>> from pyhydroexa.plotting import plot_field_with_amr_overlay
>>> plot_field_with_amr_overlay(Plotfile("plt00010"), field="h",
...                             output="depth.png")
>>> from pyhydroexa.plotting import generate_animation
>>> generate_animation(PlotfileSeries("plt*"), "h", "evolution.gif")
"""

from .animation import generate_animation
from .plots import (
    plot_field,
    plot_field_with_amr_overlay,
    plot_multi_field,
    plot_profiles,
)

__all__ = [
    "generate_animation",
    "plot_field",
    "plot_field_with_amr_overlay",
    "plot_multi_field",
    "plot_profiles",
]
