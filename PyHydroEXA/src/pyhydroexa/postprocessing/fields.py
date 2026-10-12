"""Generalized field discovery, alias resolution, and array extraction.

HydroEXA plotfiles expose fields through ``yt``, which wraps AMReX/BoxLib
data.  Field names vary between simulations (``h_fluid``, ``h``,
``water_depth``, etc.).  This module provides a configurable alias system
and a single entry point for extracting any field as a NumPy array from a
``yt`` covering grid.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yt
    HAS_YT = True
except ImportError:
    HAS_YT = False

# ---------------------------------------------------------------------------
# Default aliases
# ---------------------------------------------------------------------------

DEFAULT_ALIASES: Dict[str, List[str]] = {
    "h": ["h_fluid", "h", "water_depth", "Depth"],
    "hu": ["hu", "hu_momentum", "x_momentum", "MomentumX"],
    "hv": ["hv", "hv_momentum", "y_momentum", "MomentumY"],
    "z": ["z_bathymetry", "z", "bathymetry", "BedElevation", "elev"],
}


def resolve_field(
    ds: "yt.Dataset",
    field_name: str,
    aliases: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, ...]:
    """Resolve a logical field name to an actual ``yt`` field tuple.

    The resolver tries these strategies in order:

    1. Exact match among known aliases for *field_name*.
    2. Substring match: any field in ``ds.field_list`` whose name contains
       one of the alias strings.

    Parameters
    ----------
    ds :
        A ``yt`` dataset (any AMReX/BoxLib plotfile).
    field_name :
        Logical name (e.g. ``"h"``, ``"hu"``, ``"z"``).
    aliases :
        Custom alias mapping.  Defaults to :data:`DEFAULT_ALIASES`.

    Returns
    -------
    tuple
        A valid ``yt`` field tuple such as ``("boxlib", "h_fluid")``.

    Raises
    ------
    ValueError
        If no matching field is found in the dataset.
    RuntimeError
        If ``yt`` is not installed.
    """
    if not HAS_YT:
        raise RuntimeError("yt is required for field resolution.")

    if aliases is None:
        aliases = DEFAULT_ALIASES

    candidate_aliases = aliases.get(field_name, [field_name])

    # Strategy 1 — exact match with ("boxlib", alias) or (alias,) forms.
    for alias in candidate_aliases:
        for key in [("boxlib", alias), (field_name, alias), (alias,)]:
            if key in ds.field_list:
                return key

    # Strategy 2 — substring match against all field names.
    for field in ds.field_list:
        name = field[1] if len(field) > 1 else str(field)
        if any(alias in name for alias in candidate_aliases):
            return field

    # Strategy 3 — bare field name check.
    bare = (field_name,)
    if bare in ds.field_list:
        return bare

    available = sorted(f[1] if len(f) > 1 else str(f) for f in ds.field_list)
    raise ValueError(
        f"Could not resolve field '{field_name}' (aliases: {candidate_aliases}). "
        f"Available fields: {available}"
    )


def extract_field(
    ds: "yt.Dataset",
    field_name: str,
    level: int = 0,
    aliases: Optional[Dict[str, List[str]]] = None,
    field_aliases: Optional[Dict[str, List[str]]] = None,
) -> np.ndarray:
    """Extract a scalar field from a ``yt`` dataset as a NumPy array.

    Loads data from the covering grid at the specified AMR level and
    returns a 2-D array with physical coordinates.  The result is
    transposed to match standard NumPy / image conventions (row = y,
    column = x).

    Parameters
    ----------
    ds :
        A ``yt`` dataset.
    field_name :
        Logical field name to resolve and extract.
    level :
        AMR level from which to extract data (default ``0``, the
        finest uniform covering grid).
    aliases :
        Deprecated — use *field_aliases* instead.
    field_aliases :
        Custom alias mapping passed to :func:`resolve_field`.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(ny, nx)``.

    Raises
    ------
    ValueError
        If the extracted data is not 2-D.
    """
    if aliases is not None and field_aliases is None:
        field_aliases = aliases

    field_tuple = resolve_field(ds, field_name, aliases=field_aliases)

    left_edge = np.asarray(ds.domain_left_edge.to_ndarray(), dtype=np.float64)
    dims = np.asarray(ds.domain_dimensions, dtype=int)

    covering = yt.covering_grid(
        ds, level=level, left_edge=left_edge, dims=dims
    )
    data = np.asarray(covering[field_tuple].to_ndarray(), dtype=np.float64)

    # yt covering-grid data may be 3-D for 2-D simulations (singleton
    # in the z dimension).  Squeeze and transpose to (ny, nx).
    if data.ndim == 3:
        data = data.squeeze()
    if data.ndim == 3:
        # Edge case: 3-D simulation.
        data = data.transpose(2, 1, 0)  # (nz, ny, nx)
        return data

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2-D data for '{field_name}', got shape {data.shape}"
        )

    return data.T  # (ny, nx)


def get_field_coordinates(
    ds: "yt.Dataset",
    level: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return physical coordinates matching an extracted field grid.

    Parameters
    ----------
    ds :
        A ``yt`` dataset.
    level :
        AMR level (must match the level used for extraction).

    Returns
    -------
    x : np.ndarray
        Cell-centred x coordinates, shape ``(nx,)``.
    y : np.ndarray
        Cell-centred y coordinates, shape ``(ny,)``.
    x_lo : float
        Domain lower x bound.
    x_hi : float
        Domain upper x bound.
    (Also y_lo, y_hi are returned as part of the tuple below.)
    """
    left_edge = np.asarray(ds.domain_left_edge.to_ndarray(), dtype=np.float64)
    right_edge = np.asarray(ds.domain_right_edge.to_ndarray(), dtype=np.float64)
    dims = np.asarray(ds.domain_dimensions, dtype=int)

    covering = yt.covering_grid(
        ds, level=level, left_edge=left_edge, dims=dims
    )

    nx = int(dims[0])
    ny = int(dims[1]) if ds.dimensionality >= 2 else 1

    dx = (right_edge[0] - left_edge[0]) / nx
    x = left_edge[0] + (np.arange(nx) + 0.5) * dx

    if ds.dimensionality >= 2:
        dy = (right_edge[1] - left_edge[1]) / ny
        y = left_edge[1] + (np.arange(ny) + 0.5) * dy
    else:
        dy = 0.0
        y = np.array([left_edge[1] + 0.5 * dy])

    return x, y, float(left_edge[0]), float(right_edge[0]), float(left_edge[1]), float(right_edge[1])
