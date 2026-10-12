"""Domain, cell size, and grid resolution.

Computes the base grid resolution from domain and cell size, enforcing
the square-cell invariant (dx == dy) and AMReX blocking factor constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PreprocessConfig


@dataclass(frozen=True)
class GridResolution:
    """Computed grid resolution for the base grid.

    Attributes:
        nx: Number of cells in x (after applying refinement_scale).
        ny: Number of cells in y.
        dx: Cell width (== dy, enforced).
        dy: Cell height (== dx, enforced).
        refinement_scale: 2^terrain_ref_lev.
        blocking_factor: AMReX blocking factor.
        adj_x_hi: Adjusted x_hi after blocking-factor rounding (optional).
        adj_y_hi: Adjusted y_hi after blocking-factor rounding (optional).
    """

    nx: int
    ny: int
    dx: float
    dy: float
    refinement_scale: int
    blocking_factor: int
    adj_x_hi: float | None = None
    adj_y_hi: float | None = None

    def __post_init__(self) -> None:
        # Square-cell invariant
        if abs(self.dx - self.dy) > 1e-10 * max(abs(self.dx), abs(self.dy)):
            raise ValueError(
                f"Square-cell invariant violated: dx={self.dx}, dy={self.dy}"
            )
        # Blocking factor constraint
        if self.nx % self.blocking_factor != 0:
            raise ValueError(
                f"nx={self.nx} is not divisible by blocking_factor={self.blocking_factor}"
            )
        if self.ny % self.blocking_factor != 0:
            raise ValueError(
                f"ny={self.ny} is not divisible by blocking_factor={self.blocking_factor}"
            )

    @property
    def x_size(self) -> float:
        return self.nx * self.dx

    @property
    def y_size(self) -> float:
        return self.ny * self.dy


def compute_resolution(
    config: PreprocessConfig,
) -> GridResolution:
    """Compute the base grid resolution from a :class:`PreprocessConfig`.

    The resolution is derived from the physical domain and cell size:
        nx = (x_hi - x_lo) / cell_size
        ny = (y_hi - y_lo) / cell_size

    If the domain is not explicitly provided, it is inferred from the
    input file or the physical fields.

    **Square-cell invariant**: a single scalar ``cell_size`` is used for
    both dimensions.  The grid resolution (nx, ny) is adjusted to match
    the physical aspect ratio so that ``dx == dy`` is preserved exactly.
    This mirrors the approach used in the HydroEXA test scripts
    (e.g. ``tests/dambreakAMR/make_test_h5.py``).

    The resolution is also adjusted to satisfy the blocking factor
    constraint (nx % bf == 0, ny % bf == 0).

    Args:
        config: The preprocessing configuration.

    Returns:
        A :class:`GridResolution`.

    Raises:
        ValueError: If the resolution cannot be computed or constraints
            cannot be satisfied.
    """
    # Determine cell size
    cell_size = config.cell_size
    if cell_size is None:
        raise ValueError(
            "cell_size must be provided either explicitly or inferred from "
            "physical field metadata. Set it in the config or ensure all "
            "source fields have dx/dy attributes."
        )

    # Determine domain
    domain = config.domain
    if domain is None:
        # Try to infer from physical fields
        domain = _infer_domain(config)

    x_lo, y_lo, x_hi, y_hi = domain
    x_size = x_hi - x_lo
    y_size = y_hi - y_lo

    if x_size <= 0 or y_size <= 0:
        raise ValueError(
            f"Domain extent must be positive: x_size={x_size}, y_size={y_size}"
        )

    # Compute raw resolution
    raw_nx = round(x_size / cell_size)
    raw_ny = round(y_size / cell_size)

    if raw_nx <= 0 or raw_ny <= 0:
        raise ValueError(
            f"Computed resolution is non-positive: nx={raw_nx}, ny={raw_ny}"
        )

    # Enforce square cells by adjusting resolution to match the physical
    # aspect ratio.  A single cell_size is used for both dimensions, so
    # nx / ny must equal x_size / y_size (within floating-point tolerance).
    nx, ny = _adjust_for_square_cells(raw_nx, raw_ny, x_size, y_size)

    # Adjust to blocking factor constraint (round up)
    nx, ny = _adjust_for_blocking_factor(nx, ny, config.blocking_factor)

    # Recompute the physical extent with the adjusted resolution
    # (the domain may be slightly larger than requested)
    adj_x_hi = x_lo + nx * cell_size
    adj_y_hi = y_lo + ny * cell_size

    return GridResolution(
        nx=nx,
        ny=ny,
        dx=cell_size,
        dy=cell_size,
        refinement_scale=config.refinement_scale,
        blocking_factor=config.blocking_factor,
        adj_x_hi=adj_x_hi,
        adj_y_hi=adj_y_hi,
    )


def _infer_domain(config: PreprocessConfig) -> tuple[float, float, float, float]:
    """Infer the physical domain from input file or physical fields.

    Strategy:
        1. If input file is an existing HDF5, read its spatial metadata.
        2. If physical fields are provided, use their bounding box.
        3. Otherwise, raise an error.
    """
    # Strategy 1: Read from existing HDF5 input file
    if config.input_file.suffix in (".h5", ".hdf5") and config.input_file.exists():
        domain = _read_hdf5_domain(config.input_file)
        if domain is not None:
            return domain

    # Strategy 2: Use physical field bounding boxes
    if config.physical_fields:
        domain = _compute_field_bbox(config)
        if domain is not None:
            return domain

    raise ValueError(
        "Cannot infer domain. Provide an explicit domain tuple "
        "(x_lo, y_lo, x_hi, y_hi) in the config, or ensure the input "
        "file is an existing HDF5 with spatial metadata, or provide "
        "physical fields with geospatial coordinates."
    )


def _read_hdf5_domain(path: str) -> tuple[float, float, float, float] | None:
    """Read domain from an HDF5 file's spatial metadata attributes."""
    try:
        import h5py
    except ImportError:
        return None

    try:
        with h5py.File(path, "r") as f:
            # Check for terrain dataset with spatial attributes
            if "terrain" in f:
                terrain = f["terrain"]
                dx = float(terrain.attrs.get("dx", 0))
                dy = float(terrain.attrs.get("dy", dx))
                x_ll = float(terrain.attrs.get("x_ll", 0))
                y_ll = float(terrain.attrs.get("y_ll", 0))
                nx = terrain.shape[1]
                ny = terrain.shape[0]
                return (x_ll, y_ll, x_ll + nx * dx, y_ll + ny * dy)
    except (KeyError, ValueError, OSError):
        pass

    return None


def _compute_field_bbox(
    config: PreprocessConfig,
) -> tuple[float, float, float, float] | None:
    """Compute bounding box from physical field metadata."""
    from .fields import read_field

    x_min = float("inf")
    y_min = float("inf")
    x_max = float("-inf")
    y_max = float("-inf")
    found = False

    for name, src in config.physical_fields.items():
        try:
            field = read_field(src, name=name)
            if field.crs is not None and config.crs is not None:
                # Skip fields with different CRS — user must reproject
                continue
            x_min = min(x_min, field.x_ll)
            y_min = min(y_min, field.y_ll)
            x_max = max(x_max, field.x_ur)
            y_max = max(y_max, field.y_ur)
            found = True
        except Exception:
            # Skip fields that can't be read
            continue

    if not found:
        return None

    return (x_min, y_min, x_max, y_max)


def _adjust_for_square_cells(
    raw_nx: int,
    raw_ny: int,
    x_size: float,
    y_size: float,
) -> tuple[int, int]:
    """Adjust nx and ny so that a single cell_size produces the correct aspect ratio.

    The physical aspect ratio is ``x_size / y_size``.  We want ``nx / ny``
    to match this ratio while keeping cells square (dx == dy).

    Strategy (mirrors the test script):
        1. Compute the physical aspect ratio.
        2. Start from the raw ny, then set nx = round(aspect_ratio * ny).
        3. If the resulting nx differs from raw_nx by more than 1,
           swap roles (start from raw nx, compute ny).
        4. Fall back to the raw values if neither approach works.

    Args:
        raw_nx: Raw nx from (x_size / cell_size).
        raw_ny: Raw ny from (y_size / cell_size).
        x_size: Physical width of the domain.
        y_size: Physical height of the domain.

    Returns:
        Adjusted (nx, ny) that match the aspect ratio.
    """
    aspect_ratio = x_size / y_size

    # Approach 1: start from raw_ny, compute nx
    candidate_nx_1 = round(aspect_ratio * raw_ny)
    diff_1 = abs(candidate_nx_1 - raw_nx)

    # Approach 2: start from raw_nx, compute ny
    candidate_ny_2 = round(raw_nx / aspect_ratio)
    diff_2 = abs(candidate_ny_2 - raw_ny)

    if diff_1 <= diff_2:
        return candidate_nx_1, raw_ny
    else:
        return raw_nx, candidate_ny_2


def _adjust_for_blocking_factor(
    nx: int,
    ny: int,
    blocking_factor: int,
) -> tuple[int, int]:
    """Adjust resolution to satisfy blocking factor constraint.

    Rounds up to the nearest multiple of blocking_factor.
    """
    adj_nx = ((nx + blocking_factor - 1) // blocking_factor) * blocking_factor
    adj_ny = ((ny + blocking_factor - 1) // blocking_factor) * blocking_factor
    return adj_nx, adj_ny
