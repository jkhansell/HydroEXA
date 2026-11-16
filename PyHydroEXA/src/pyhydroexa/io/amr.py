"""AMR grid / MeshBlock utilities.

Extracts per-block geometry from a ``yt`` dataset so that AMR
boundaries can be overlaid on plots or analysed independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import yt
    HAS_YT = True
except ImportError:
    yt = None  # type: ignore
    HAS_YT = False


@dataclass
class MeshBlock:
    """Geometry of a single AMR MeshBlock (grid cell).

    Attributes
    ----------
    x0, y0 :
        Lower-left corner (physical coordinates).
    width, height :
        Extent in x and y.
    level :
        AMR refinement level (integer >= 0).
    index :
        Global grid index within the dataset.
    """

    x0: float
    y0: float
    width: float
    height: float
    level: int
    index: int = 0


def get_meshblock_rectangles(
    ds: "yt.Dataset",
    level_filter: Optional[Sequence[int]] = None,
) -> List[MeshBlock]:
    """Extract MeshBlock rectangles from a ``yt`` dataset.

    Parameters
    ----------
    ds :
        A ``yt`` AMReX/BoxLib dataset.
    level_filter :
        Optional subset of levels to include (e.g. ``(1, 2, 3)``).
        Defaults to all levels.

    Returns
    -------
    list[MeshBlock]
    """
    if not HAS_YT:
        raise RuntimeError("yt is required for AMR utilities.")

    lefts = np.asarray(ds.index.grid_left_edge, dtype=np.float64)
    dims = np.asarray(ds.index.grid_dimensions, dtype=int)
    levels = np.asarray(ds.index.grid_levels, dtype=int).flatten()

    dx_table = ds.index.level_dds

    blocks: List[MeshBlock] = []
    for i in range(len(lefts)):
        lvl = int(levels[i])
        if level_filter is not None and lvl not in level_filter:
            continue
        dx = dx_table[lvl]
        blocks.append(MeshBlock(
            x0=float(lefts[i, 0]),
            y0=float(lefts[i, 1]) if ds.dimensionality >= 2 else 0.0,
            width=float(dims[i, 0] * dx[0]),
            height=float(dims[i, 1] * dx[1]) if ds.dimensionality >= 2 else 0.0,
            level=lvl,
            index=int(i),
        ))

    return blocks


def get_available_levels(
    ds: "yt.Dataset",
) -> np.ndarray:
    """Return sorted unique AMR levels in the dataset.

    Parameters
    ----------
    ds :
        A ``yt`` AMReX/BoxLib dataset.

    Returns
    -------
    np.ndarray
        Sorted array of unique level integers.
    """
    if not HAS_YT:
        raise RuntimeError("yt is required for AMR utilities.")
    return np.sort(np.unique(ds.index.grid_levels.flatten()))


def filter_blocks_by_region(
    blocks: List[MeshBlock],
    x_lo: float,
    y_lo: float,
    x_hi: float,
    y_hi: float,
) -> List[MeshBlock]:
    """Return blocks that overlap a rectangular region.

    Parameters
    ----------
    blocks :
        List of :class:`MeshBlock` objects.
    x_lo, y_lo, x_hi, y_hi :
        Bounding box (physical coordinates).

    Returns
    -------
    list[MeshBlock]
    """
    filtered: List[MeshBlock] = []
    for b in blocks:
        block_x_hi = b.x0 + b.width
        block_y_hi = b.y0 + b.height
        if (
            b.x0 < x_hi and block_x_hi > x_lo
            and b.y0 < y_hi and block_y_hi > y_lo
        ):
            filtered.append(b)
    return filtered
