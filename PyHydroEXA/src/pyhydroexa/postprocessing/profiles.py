"""Generalized line / profile extraction from 2-D fields.

Supports horizontal and vertical cuts at user-specified physical
coordinates, as well as evenly-spaced cuts across a domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import yt
    HAS_YT = True
except ImportError:
    yt = None  # type: ignore
    HAS_YT = False


@dataclass
class LineProfile:
    """A 1-D profile extracted along a line.

    Attributes
    ----------
    coordinate :
        The fixed coordinate perpendicular to the cut (e.g. y for x-cut).
    values :
        Extracted 1-D values along the cut.
    axis_coords :
        Physical coordinates along the cut direction.
    label :
        Optional human-readable label.
    """

    coordinate: float
    values: np.ndarray
    axis_coords: np.ndarray
    label: str = ""


def extract_profile(
    ds: "yt.Dataset",
    field_name: str,
    axis: str,
    positions: Optional[List[float]] = None,
    ny_lines: int = 7,
    level: int = 0,
    aliases: Optional[dict] = None,
) -> Tuple[np.ndarray, List[LineProfile]]:
    """Extract 1-D profiles from a 2-D field.

    Parameters
    ----------
    ds :
        A ``yt`` AMReX/BoxLib dataset.
    field_name :
        Logical field name (resolved via :mod:`fields`).
    axis :
        Extraction axis: ``"x"`` (horizontal cuts at fixed y) or
        ``"y"`` (vertical cuts at fixed x).
    positions :
        Physical coordinates for the cuts.  If ``None``, *ny_lines*
        evenly-spaced positions are used.
    ny_lines :
        Number of evenly-spaced cuts when *positions* is ``None``.
    level :
        AMR level for data extraction.
    aliases :
        Custom field aliases passed to :func:`fields.extract_field`.

    Returns
    -------
    axis_coords : np.ndarray
        Physical coordinates along the cut direction.
    profiles : list[LineProfile]
        One :class:`LineProfile` per cut.

    Raises
    ------
    ValueError
        If *axis* is not ``"x"`` or ``"y"``, or if the field data
        is not 2-D.
    """
    axis = axis.lower().strip()
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

    from .fields import extract_field, get_field_coordinates

    data = extract_field(ds, field_name, level=level, aliases=aliases)
    x, y, x_lo, x_hi, y_lo, y_hi = get_field_coordinates(ds, level=level)

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2-D data for '{field_name}', got shape {data.shape}"
        )

    ny, nx = data.shape

    if positions is None:
        if axis == "x":
            positions = np.linspace(float(y[0]), float(y[-1]), ny_lines).tolist()
        else:
            positions = np.linspace(float(x[0]), float(x[-1]), ny_lines).tolist()

    profiles: List[LineProfile] = []

    for pos in positions:
        if axis == "x":
            # Horizontal cut: fixed y, extract along x.
            row_idx = int(np.argmin(np.abs(y - pos)))
            row_idx = max(0, min(row_idx, ny - 1))
            vals = data[row_idx, :]
            coord = float(y[row_idx])
            label = f"y = {coord:.6g}"
            profiles.append(LineProfile(
                coordinate=coord,
                values=vals,
                axis_coords=x,
                label=label,
            ))
        else:
            # Vertical cut: fixed x, extract along y.
            col_idx = int(np.argmin(np.abs(x - pos)))
            col_idx = max(0, min(col_idx, nx - 1))
            vals = data[:, col_idx]
            coord = float(x[col_idx])
            label = f"x = {coord:.6g}"
            profiles.append(LineProfile(
                coordinate=coord,
                values=vals,
                axis_coords=y,
                label=label,
            ))

    return x if axis == "x" else y, profiles


def extract_horizontal_cut(
    ds: "yt.Dataset",
    field_name: str,
    y: float,
    level: int = 0,
    aliases: Optional[dict] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Extract a single horizontal (x-direction) profile at fixed *y*.

    Returns
    -------
    y_val : float
        Actual y coordinate used.
    x_coords : np.ndarray
        Cell-centred x coordinates.
    values : np.ndarray
        Field values along the cut.
    """
    x_coords, profiles = extract_profile(
        ds, field_name, axis="x", positions=[y], level=level, aliases=aliases
    )
    return profiles[0].coordinate, x_coords, profiles[0].values


def extract_vertical_cut(
    ds: "yt.Dataset",
    field_name: str,
    x: float,
    level: int = 0,
    aliases: Optional[dict] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Extract a single vertical (y-direction) profile at fixed *x*.

    Returns
    -------
    x_val : float
        Actual x coordinate used.
    y_coords : np.ndarray
        Cell-centred y coordinates.
    values : np.ndarray
        Field values along the cut.
    """
    y_coords, profiles = extract_profile(
        ds, field_name, axis="y", positions=[x], level=level, aliases=aliases
    )
    return profiles[0].coordinate, y_coords, profiles[0].values
