"""Physical-coordinate field resampling.

Maps source fields (at arbitrary resolution and domain) onto the target
grid defined by the :class:`GridResolution`. All resampling is done in
physical (x, y) coordinates, not array indices.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from .fields import SpatialField
from .geometry import CELL_EPS, GridResolution


@dataclass(frozen=True)
class ResampledField:
    """A field resampled onto the target grid.

    Attributes:
        data: Resampled 2D array in (ny, nx) order.
        source_name: Original field name.
        source_crs: CRS of the source field, if any.
    """

    data: npt.NDArray[np.float64]
    source_name: str
    source_crs: str | None = None


def resample_to_grid(
    source: SpatialField,
    target: GridResolution,
    x_ll_target: float,
    y_ll_target: float,
) -> ResampledField:
    """Resample a source field onto the target grid.

    Resampling strategy:
        1. Compute the source field's physical extent.
        2. Determine the overlap region with the target grid.
        3. Map source array indices to target array indices via physical
           coordinates.
        4. Use nearest-neighbour interpolation (robust for fields with
           NODATA sentinel values).

    For fields with geographic CRS, the source must be reprojected to
    match the target's CRS before calling this function.

    Args:
        source: The source field to resample.
        target: The target grid resolution.
        x_ll_target: Physical x-coordinate of the target grid's lower-left.
        y_ll_target: Physical y-coordinate of the target grid's lower-left.

    Returns:
        A :class:`ResampledField` on the target grid.

    Raises:
        ValueError: If the source and target domains do not overlap.
    """
    # Compute target physical extent
    x_hi_target = x_ll_target + target.nx * target.dx
    y_hi_target = y_ll_target + target.ny * target.dy

    # Compute source physical extent
    x_hi_source = source.x_ll + source.nx * source.dx
    y_hi_source = source.y_ll + source.ny * source.dy

    # Determine overlap region
    x_lo_overlap = max(x_ll_target, source.x_ll)
    y_lo_overlap = max(y_ll_target, source.y_ll)
    x_hi_overlap = min(x_hi_target, x_hi_source)
    y_hi_overlap = min(y_hi_target, y_hi_source)

    if x_lo_overlap >= x_hi_overlap or y_lo_overlap >= y_hi_overlap:
        raise ValueError(
            f"Source field '{source.name}' and target grid do not overlap. "
            f"Source: [{source.x_ll}, {x_hi_source}] x [{source.y_ll}, {y_hi_source}]. "
            f"Target: [{x_ll_target}, {x_hi_target}] x [{y_ll_target}, {y_hi_target}]."
        )

    # Target cells that overlap with the source.  The lower bound is the cell
    # containing the overlap start; the upper bound is a ceiling, so a cell the
    # overlap only partly covers is still included.  The epsilon keeps an
    # overlap that ends exactly on a cell boundary from claiming the next cell.
    col_lo = max(0, int((x_lo_overlap - x_ll_target) / target.dx))
    row_lo = max(0, int((y_lo_overlap - y_ll_target) / target.dy))
    col_hi = min(
        target.nx,
        max(col_lo, math.ceil((x_hi_overlap - x_ll_target) / target.dx - CELL_EPS)),
    )
    row_hi = min(
        target.ny,
        max(row_lo, math.ceil((y_hi_overlap - y_ll_target) / target.dy - CELL_EPS)),
    )

    # The result is a full target-grid array: a source that covers only part of
    # the domain must still produce a field that can be dropped onto that grid
    # without knowing where it came from.  Cells outside the source stay NODATA.
    nodata = source.nodata
    out_data = np.full((target.ny, target.nx), nodata, dtype=np.float64)

    # Map source cells to target cells, sampling at the target cell's lower-left
    # corner so that a cell's value is the source value that physically contains
    # that point.
    for row in range(row_hi - row_lo):
        y_phys = y_ll_target + (row_lo + row) * target.dy
        # Source row index for this physical y
        src_row = int((y_phys - source.y_ll) / source.dy)
        src_row = max(0, min(source.ny - 1, src_row))

        for col in range(col_hi - col_lo):
            x_phys = x_ll_target + (col_lo + col) * target.dx
            # Source col index for this physical x
            src_col = int((x_phys - source.x_ll) / source.dx)
            src_col = max(0, min(source.nx - 1, src_col))

            val = source.data[src_row, src_col]
            if val != nodata:
                out_data[row_lo + row, col_lo + col] = val

    return ResampledField(
        data=out_data,
        source_name=source.name,
        source_crs=source.crs,
    )


def resample_fields_to_grid(
    fields: list[SpatialField],
    target: GridResolution,
    x_ll_target: float,
    y_ll_target: float,
) -> list[ResampledField]:
    """Resample multiple fields onto the target grid.

    Each field is resampled independently and returned as its own
    :class:`ResampledField` on the full target grid, so a caller that knows
    which field is which keeps that knowledge.  Overlaying several results is a
    separate step (:func:`combine_resampled_fields`), which is first-wins:
    a later field only fills cells the earlier ones left as NODATA.

    Args:
        fields: List of source fields.
        target: Target grid resolution.
        x_ll_target: Physical x-coordinate of target lower-left.
        y_ll_target: Physical y-coordinate of target lower-left.

    Returns:
        List of :class:`ResampledField` on the target grid.
    """
    resampled = []
    for field in fields:
        try:
            result = resample_to_grid(field, target, x_ll_target, y_ll_target)
            resampled.append(result)
        except ValueError as e:
            warnings.warn(f"Skipping field '{field.name}': {e}", stacklevel=2)
    return resampled


def combine_resampled_fields(
    resampled: list[ResampledField],
    nodata: float = -9999.0,
) -> npt.NDArray[np.float64]:
    """Combine multiple resampled fields into a single 2D array.

    Fields are overlaid in order. For each cell, the first non-NODATA
    value is used (earlier fields take precedence).

    Args:
        resampled: List of resampled fields (all same shape).
        nodata: NODATA sentinel value.

    Returns:
        Combined 2D array in (ny, nx) order.
    """
    if not resampled:
        return np.array([], dtype=np.float64).reshape(0, 0)

    # All fields should have the same shape
    shape = resampled[0].data.shape
    combined = np.full(shape, nodata, dtype=np.float64)
    filled = np.zeros(shape, dtype=bool)

    for field in resampled:
        if field.data.shape != shape:
            warnings.warn(
                f"Field '{field.source_name}' has different shape after "
                f"resampling — skipping.",
                stacklevel=2,
            )
            continue
        # `filled` is what makes this first-wins rather than last-wins: a
        # later field may only contribute where nothing has contributed yet.
        mask = (field.data != nodata) & ~filled
        combined[mask] = field.data[mask]
        filled |= mask

    return combined
