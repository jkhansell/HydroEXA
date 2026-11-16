"""HDF5 file writer.

Writes terrain and fluid datasets in the exact schema expected by the
HydroEXA C++ solver's :class:`IOHandler::ReadHDF5HyperslabComponents`:

    - Dataset "terrain": 3-D, shape (2, ny, nx) -- (z, roughness)
    - Dataset "fluid": 3-D, shape (3, ny, nx) -- (h, hu, hv)
    - Attributes per dataset: "dx", "dy", "x_ll", "y_ll"
    - NODATA sentinel: -9999.0

The component is the *leading* axis, which is what the C++ reader assumes
(``HDF5Reader::Dataset`` takes the last two dimensions as spatial) and what
every ``tests/*/*.h5`` in this repository contains.  A 2-D single-component
dataset is also readable, but a multi-component one must not be flattened
along x: the solver would read a component boundary as a spatial boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from .fields import SpatialField
from .geometry import GridResolution


# ---------------------------------------------------------------------------
# Dataset constants (must match C++ IOHandler)
# ---------------------------------------------------------------------------

TERRAIN_DATASET = "terrain"
FLUID_DATASET = "fluid"

TERRAIN_NCOMP = 2  # (z, roughness)
FLUID_NCOMP = 3    # (h, hu, hv)

# Attribute names (must match C++ IOHandler::ReadHDF5Metadata)
ATTR_DX = "dx"
ATTR_DY = "dy"
ATTR_X_LL = "x_ll"
ATTR_Y_LL = "y_ll"


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


def write_hdf5(
    output_path: str | Path,
    terrain_data: npt.NDArray[np.float64],
    fluid_data: npt.NDArray[np.float64],
    dx: float,
    dy: float,
    x_ll: float,
    y_ll: float,
) -> None:
    """Write terrain and fluid datasets to an HDF5 file.

    The output file matches the exact schema expected by the HydroEXA
    C++ solver. Both datasets share the same spatial attributes.

    Args:
        output_path: Path for the output HDF5 file.
        terrain_data: Terrain array of shape (2, ny, nx): component 0 is
            z (bed elevation), component 1 is Manning's roughness.
        fluid_data: Fluid array of shape (3, ny, nx): (h, hu, hv).
        dx: Cell width.
        dy: Cell height.
        x_ll: Physical x-coordinate of lower-left corner.
        y_ll: Physical y-coordinate of lower-left corner.

    Raises:
        ValueError: If array shapes or dimensions are invalid.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 output")

    output_path = Path(output_path)

    # Validate dimensions: the component axis is the leading one, because
    # that is the axis the C++ reader treats as non-spatial.
    if terrain_data.ndim != 3 or terrain_data.shape[0] != TERRAIN_NCOMP:
        raise ValueError(
            f"terrain_data must have shape ({TERRAIN_NCOMP}, ny, nx), "
            f"got {terrain_data.shape}"
        )
    if fluid_data.ndim != 3 or fluid_data.shape[0] != FLUID_NCOMP:
        raise ValueError(
            f"fluid_data must have shape ({FLUID_NCOMP}, ny, nx), "
            f"got {fluid_data.shape}"
        )

    ny_T, nx_T = terrain_data.shape[1:]
    ny_F, nx_F = fluid_data.shape[1:]

    if ny_T != ny_F or nx_T != nx_F:
        raise ValueError(
            f"Terrain ({ny_T}x{nx_T}) and fluid ({ny_F}x{nx_F}) dimensions "
            f"must match"
        )

    # Ensure square cells
    if abs(dx - dy) > 1e-10 * max(abs(dx), abs(dy)):
        raise ValueError(
            f"Square-cell invariant violated: dx={dx}, dy={dy}"
        )

    # Write
    with h5py.File(output_path, "w") as f:
        # Terrain dataset
        terrain = f.create_dataset(
            TERRAIN_DATASET,
            data=terrain_data,
            dtype=np.float64,
        )
        terrain.attrs[ATTR_DX] = dx
        terrain.attrs[ATTR_DY] = dy
        terrain.attrs[ATTR_X_LL] = x_ll
        terrain.attrs[ATTR_Y_LL] = y_ll

        # Fluid dataset
        fluid = f.create_dataset(
            FLUID_DATASET,
            data=fluid_data,
            dtype=np.float64,
        )
        fluid.attrs[ATTR_DX] = dx
        fluid.attrs[ATTR_DY] = dy
        fluid.attrs[ATTR_X_LL] = x_ll
        fluid.attrs[ATTR_Y_LL] = y_ll


# ---------------------------------------------------------------------------
# Field assembly helpers
# ---------------------------------------------------------------------------


def assemble_terrain(
    z_field: SpatialField,
    roughness_field: SpatialField | None = None,
    default_roughness: float = 0.01,
) -> npt.NDArray[np.float64]:
    """Assemble the terrain dataset array from source fields.

    The terrain dataset is 3-D with the component on the leading axis:
        terrain[0] -- z (bathymetry / bed elevation)
        terrain[1] -- roughness (Manning's n)

    Args:
        z_field: Bathymetry field.
        roughness_field: Roughness field (Manning's n).
            If None, uses default_roughness everywhere.
        default_roughness: Default Manning's n value.

    Returns:
        Terrain array of shape (2, ny, nx).
    """
    ny = z_field.ny
    nx = z_field.nx
    nodata = z_field.nodata

    terrain = np.full((TERRAIN_NCOMP, ny, nx), nodata, dtype=np.float64)

    # Component 0: z.  Masked so that a NODATA cell stays NODATA in every
    # component; otherwise the sentinel would be overwritten with a real bed
    # elevation of -9999 and the solver could not tell the two apart.
    z_mask = z_field.is_valid
    terrain[0][z_mask] = z_field.data[z_mask]

    # Component 1: roughness
    if roughness_field is not None:
        r_mask = roughness_field.is_valid
        terrain[1][r_mask] = roughness_field.data[r_mask]
    else:
        terrain[1][z_mask] = default_roughness

    return terrain


def assemble_fluid(
    h_field: SpatialField,
    hu_field: SpatialField | None = None,
    hv_field: SpatialField | None = None,
    default_hu: float = 0.0,
    default_hv: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Assemble the fluid dataset array from source fields.

    The fluid dataset is 3-D with the component on the leading axis:
        fluid[0] -- h (water depth, h = eta - z)
        fluid[1] -- hu (x-momentum)
        fluid[2] -- hv (y-momentum)

    Args:
        h_field: Water depth field.
        hu_field: x-momentum field (optional).
        hv_field: y-momentum field (optional).
        default_hu: Default x-momentum value.
        default_hv: Default y-momentum value.

    Returns:
        Fluid array of shape (3, ny, nx).
    """
    ny = h_field.ny
    nx = h_field.nx
    nodata = h_field.nodata

    fluid = np.full((FLUID_NCOMP, ny, nx), nodata, dtype=np.float64)

    # Component 0: h
    h_mask = h_field.is_valid
    fluid[0][h_mask] = h_field.data[h_mask]

    # Component 1: hu
    if hu_field is not None:
        hu_mask = hu_field.is_valid
        fluid[1][hu_mask] = hu_field.data[hu_mask]
    else:
        fluid[1][h_mask] = default_hu

    # Component 2: hv
    if hv_field is not None:
        hv_mask = hv_field.is_valid
        fluid[2][hv_mask] = hv_field.data[hv_mask]
    else:
        fluid[2][h_mask] = default_hv

    return fluid


def assemble_from_water_level(
    z_field: SpatialField,
    eta_field: SpatialField,
    hu_field: SpatialField | None = None,
    hv_field: SpatialField | None = None,
    default_hu: float = 0.0,
    default_hv: float = 0.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Assemble terrain and fluid arrays from bed elevation and water level.

    Computes water depth as h = eta - z (only where both are valid).

    Args:
        z_field: Bed elevation (bathymetry).
        eta_field: Water level (eta).
        hu_field: x-momentum (optional).
        hv_field: y-momentum (optional).
        default_hu: Default x-momentum.
        default_hv: Default y-momentum.

    Returns:
        (terrain_array, fluid_array) ready for HDF5 writing.
    """
    # Compute water depth: h = eta - z
    h_data = np.full_like(z_field.data, z_field.nodata, dtype=np.float64)

    # Valid where both z and eta are valid
    z_valid = z_field.is_valid
    eta_valid = eta_field.is_valid
    both_valid = z_valid & eta_valid

    h_data[both_valid] = eta_field.data[both_valid] - z_field.data[both_valid]

    # Create h field with same geometry as z_field
    h_field = SpatialField(
        data=h_data,
        x_ll=z_field.x_ll,
        y_ll=z_field.y_ll,
        dx=z_field.dx,
        dy=z_field.dy,
        nodata=z_field.nodata,
        name="h (computed)",
        crs=z_field.crs,
    )

    terrain = assemble_terrain(z_field)
    fluid = assemble_fluid(h_field, hu_field, hv_field, default_hu, default_hv)

    return terrain, fluid
