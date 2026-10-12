"""Public API for the pyhydroexa.postprocessing package.

Example usage
-------------
>>> from pyhydroexa.postprocessing import Plotfile, PlotfileSeries
>>> pf = Plotfile("plt00010")
>>> print(pf.field_names)
>>> from pyhydroexa.postprocessing import plot_field_with_amr_overlay
>>> plot_field_with_amr_overlay(pf, field="h", output="depth.png")
>>> from pyhydroexa.postprocessing import extract_profile
>>> x, profiles = extract_profile(pf, field="h", axis="x", positions=[0.25, 0.5, 0.75])
"""

from .fields import (
    DEFAULT_ALIASES,
    resolve_field,
    extract_field,
    get_field_coordinates,
)
from .plotfile import Plotfile, PlotfileSeries, _is_plotfile
from .amr import (
    MeshBlock,
    get_meshblock_rectangles,
    get_available_levels,
    filter_blocks_by_region,
)
from .profiles import (
    LineProfile,
    extract_profile,
    extract_horizontal_cut,
    extract_vertical_cut,
)
from .plots import (
    plot_field,
    plot_field_with_amr_overlay,
    plot_profiles,
    plot_multi_field,
)
from .animation import generate_animation

__all__ = [
    # Fields
    "DEFAULT_ALIASES",
    "resolve_field",
    "extract_field",
    "get_field_coordinates",
    # Plotfile
    "Plotfile",
    "PlotfileSeries",
    "_is_plotfile",
    # AMR
    "MeshBlock",
    "get_meshblock_rectangles",
    "get_available_levels",
    "filter_blocks_by_region",
    # Profiles
    "LineProfile",
    "extract_profile",
    "extract_horizontal_cut",
    "extract_vertical_cut",
    # Plots
    "plot_field",
    "plot_field_with_amr_overlay",
    "plot_profiles",
    "plot_multi_field",
    # Animation
    "generate_animation",
]
