"""Reading HydroEXA / AMReX plotfiles.

This layer only *reads*: it wraps ``yt`` and exposes plotfile metadata,
field data, AMR block geometry.  Nothing here imports matplotlib, so a
script that only extracts numbers pays nothing for a plotting stack.

Example usage
-------------
>>> from pyhydroexa.io import Plotfile, PlotfileSeries
>>> pf = Plotfile("plt00010")
>>> print(pf.field_names)
>>> depth = pf.extract_field("h")
>>> for frame in PlotfileSeries("plt*"):
...     print(frame.time)

Field names are resolved through an alias table, so ``"depth"`` and
``"h"`` both work: see :func:`resolve_field` and :data:`DEFAULT_ALIASES`.
"""

from .amr import (
    MeshBlock,
    filter_blocks_by_region,
    get_available_levels,
    get_meshblock_rectangles,
)
from .fields import (
    DEFAULT_ALIASES,
    extract_field,
    get_field_coordinates,
    resolve_field,
)
from .plotfile import Plotfile, PlotfileSeries, _is_plotfile

__all__ = [
    # Fields
    "DEFAULT_ALIASES",
    "resolve_field",
    "extract_field",
    "get_field_coordinates",
    # Plotfile
    "Plotfile",
    "PlotfileSeries",
    # AMR geometry
    "MeshBlock",
    "get_meshblock_rectangles",
    "get_available_levels",
    "filter_blocks_by_region",
]
