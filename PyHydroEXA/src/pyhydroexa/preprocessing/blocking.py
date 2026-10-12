"""AMReX blocking factor constraint handling.

AMReX requires that each grid level's dimensions be divisible by the
blocking factor (default 16). This module provides utilities for
computing valid grid sizes and adjusting domains to satisfy the constraint.
"""

from __future__ import annotations

from math import gcd

from .geometry import GridResolution


# ---------------------------------------------------------------------------
# Blocking factor utilities
# ---------------------------------------------------------------------------

# Default blocking factors used by AMReX
DEFAULT_BLOCKING_FACTOR = 16

# Valid blocking factors (powers of 2, typical range)
VALID_BLOCKING_FACTORS = (8, 16, 32, 64, 128)


def is_valid_blocking_factor(bf: int) -> bool:
    """Check if a blocking factor is valid for AMReX.

    AMReX requires blocking factors that are powers of 2.
    """
    return bf > 0 and (bf & (bf - 1)) == 0


def nearest_valid_blocking_factor(
    nx: int,
    ny: int,
    preferred: int = DEFAULT_BLOCKING_FACTOR,
) -> int:
    """Find the nearest valid blocking factor for given grid dimensions.

    Returns the largest power-of-2 blocking factor that divides both
    nx and ny, up to the preferred value.
    """
    if not is_valid_blocking_factor(preferred):
        preferred = _largest_power_of_2(preferred)

    g = gcd(nx, ny)
    # Find largest power of 2 that divides both
    candidate = _largest_power_of_2(g)
    return min(candidate, preferred)


def _largest_power_of_2(n: int) -> int:
    """Return the largest power of 2 <= n."""
    if n <= 0:
        return 1
    return 1 << (n.bit_length() - 1)


# ---------------------------------------------------------------------------
# Grid sizing
# ---------------------------------------------------------------------------


def compute_grid_size(
    physical_size: float,
    cell_size: float,
    blocking_factor: int,
) -> int:
    """Compute a valid grid size for a given physical extent.

    Rounds up to the nearest multiple of blocking_factor.

    Args:
        physical_size: Physical extent in the dimension.
        cell_size: Cell width/height.
        blocking_factor: AMReX blocking factor.

    Returns:
        Grid size (number of cells) that is a multiple of blocking_factor.
    """
    raw_size = round(physical_size / cell_size)
    if raw_size <= 0:
        raw_size = blocking_factor
    # Round up to next multiple of blocking_factor
    return ((raw_size + blocking_factor - 1) // blocking_factor) * blocking_factor


def compute_physical_size(
    grid_size: int,
    cell_size: float,
) -> float:
    """Compute the physical extent for a given grid size."""
    return grid_size * cell_size


# ---------------------------------------------------------------------------
# Domain adjustment
# ---------------------------------------------------------------------------


def adjust_domain_for_blocking(
    x_lo: float,
    y_lo: float,
    x_hi: float,
    y_hi: float,
    cell_size: float,
    blocking_factor: int,
) -> tuple[float, float, float, float]:
    """Adjust domain boundaries to satisfy blocking factor constraint.

    Expands the domain (only the hi boundaries) to ensure the resulting
    grid dimensions are multiples of blocking_factor.

    Args:
        x_lo: Left x-boundary.
        y_lo: Bottom y-boundary.
        x_hi: Right x-boundary.
        y_hi: Top y-boundary.
        cell_size: Cell width/height (square cells).
        blocking_factor: AMReX blocking factor.

    Returns:
        Adjusted (x_lo, y_lo, x_hi, y_hi).
    """
    nx = round((x_hi - x_lo) / cell_size)
    ny = round((y_hi - y_lo) / cell_size)

    adj_nx = ((nx + blocking_factor - 1) // blocking_factor) * blocking_factor
    adj_ny = ((ny + blocking_factor - 1) // blocking_factor) * blocking_factor

    adj_x_hi = x_lo + adj_nx * cell_size
    adj_y_hi = y_lo + adj_ny * cell_size

    return (x_lo, y_lo, adj_x_hi, adj_y_hi)


# ---------------------------------------------------------------------------
# Terrain refinement
# ---------------------------------------------------------------------------


def terrain_base_resolution(
    fine_nx: int,
    fine_ny: int,
    terrain_ref_lev: int,
) -> tuple[int, int]:
    """Compute the base grid resolution from a fine-grid resolution.

    The base grid is coarser by a factor of 2^terrain_ref_lev.

    Args:
        fine_nx: Fine-grid (terrain dataset) nx.
        fine_ny: Fine-grid (terrain dataset) ny.
        terrain_ref_lev: AMR terrain refinement level.

    Returns:
        (base_nx, base_ny) — the base grid resolution.
    """
    scale = 1 << terrain_ref_lev
    base_nx = fine_nx // scale
    base_ny = fine_ny // scale
    return base_nx, base_ny


def terrain_dataset_resolution(
    base_nx: int,
    base_ny: int,
    terrain_ref_lev: int,
) -> tuple[int, int]:
    """Compute the required terrain dataset resolution from the base grid.

    The terrain dataset must be at resolution 2^terrain_ref_lev times
    the base grid resolution.

    Args:
        base_nx: Base grid nx.
        base_ny: Base grid ny.
        terrain_ref_lev: AMR terrain refinement level.

    Returns:
        (terrain_nx, terrain_ny) — the terrain dataset resolution.
    """
    scale = 1 << terrain_ref_lev
    return base_nx * scale, base_ny * scale
