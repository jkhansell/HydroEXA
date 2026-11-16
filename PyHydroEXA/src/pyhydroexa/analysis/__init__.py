"""Reusable analysis of simulation output.

The dividing line against :mod:`pyhydroexa.plotting` is whether a result
is a number or a picture.  A profile is a number set: it is cheap, it is
comparable, it can be written to a CSV or asserted against an analytic
solution.  Plotting it is a separate, optional step.

The dividing line against :mod:`pyhydroexa.io` is whether data is being
read or reduced: readers know how a plotfile is laid out, analysis knows
what the numbers mean.

Example usage
-------------
>>> from pyhydroexa.io import Plotfile
>>> from pyhydroexa.analysis import extract_profile
>>> pf = Plotfile("plt00010")
>>> x, profiles = extract_profile(pf, "h", axis="x", positions=[0.5])
>>> profiles[0].values.max()
"""

from .profiles import (
    LineProfile,
    extract_horizontal_cut,
    extract_profile,
    extract_vertical_cut,
)

__all__ = [
    "LineProfile",
    "extract_profile",
    "extract_horizontal_cut",
    "extract_vertical_cut",
]
