"""Spatial field representation.

Provides :class:`SpatialField` as a unified interface for reading and
manipulating 2D spatial data from various formats (GeoTIFF, NumPy,
xarray datasets).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SpatialField:
    """A 2D spatial field with optional geospatial metadata.

    Attributes:
        data: 2D array in (ny, nx) order (row-major, matching HDF5).
        x_ll: Physical x-coordinate of the lower-left corner.
        y_ll: Physical y-coordinate of the lower-left corner.
        dx: Cell width in x.
        dy: Cell height in y.
        nodata: NODATA sentinel value (default -9999.0).
        name: Human-readable field name.
        crs: CRS string, if geographic.
    """

    data: npt.NDArray[np.float64]
    x_ll: float
    y_ll: float
    dx: float
    dy: float
    nodata: float = -9999.0
    name: str = "field"
    crs: str | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"SpatialField data must be 2D, got {self.data.ndim}D")
        if self.dx <= 0:
            raise ValueError(f"dx must be positive, got {self.dx}")
        if self.dy <= 0:
            raise ValueError(f"dy must be positive, got {self.dy}")
        # Enforce square cells
        if abs(self.dx - self.dy) > 1e-10 * max(abs(self.dx), abs(self.dy)):
            raise ValueError(
                f"Square-cell invariant violated: dx={self.dx}, dy={self.dy}. "
                "All cells must be square (dx == dy)."
            )

    @property
    def nx(self) -> int:
        """Number of cells in x."""
        return self.data.shape[1]

    @property
    def ny(self) -> int:
        """Number of cells in y."""
        return self.data.shape[0]

    @property
    def x_size(self) -> float:
        """Physical width of the domain."""
        return self.nx * self.dx

    @property
    def y_size(self) -> float:
        """Physical height of the domain."""
        return self.ny * self.dy

    @property
    def x_ur(self) -> float:
        """x-coordinate of the upper-right corner."""
        return self.x_ll + self.x_size

    @property
    def y_ur(self) -> float:
        """y-coordinate of the upper-right corner."""
        return self.y_ll + self.y_size

    @property
    def is_valid(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of non-NODATA cells."""
        return self.data != self.nodata

    @property
    def is_nodata(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of NODATA cells."""
        return self.data == self.nodata

    def clip_to_domain(
        self,
        x_lo: float,
        y_lo: float,
        x_hi: float,
        y_hi: float,
    ) -> SpatialField:
        """Clip the field to a physical domain rectangle.

        Args:
            x_lo: Left x-boundary.
            y_lo: Bottom y-boundary.
            x_hi: Right x-boundary.
            y_hi: Top y-boundary.

        Returns:
            A new :class:`SpatialField` clipped to the domain.
        """
        # Convert physical coordinates to array indices
        col_lo = max(0, int((x_lo - self.x_ll) / self.dx))
        row_lo = max(0, int((y_lo - self.y_ll) / self.dy))
        col_hi = min(self.nx, int((x_hi - self.x_ll) / self.dx) + 1)
        row_hi = min(self.ny, int((y_hi - self.y_ll) / self.dy) + 1)

        clipped_data = self.data[row_lo:row_hi, col_lo:col_hi].copy()

        new_x_ll = self.x_ll + col_lo * self.dx
        new_y_ll = self.y_ll + row_lo * self.dy

        return SpatialField(
            data=clipped_data,
            x_ll=new_x_ll,
            y_ll=new_y_ll,
            dx=self.dx,
            dy=self.dy,
            nodata=self.nodata,
            name=f"{self.name}[{x_lo}:{x_hi},{y_lo}:{y_hi}]",
            crs=self.crs,
        )


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_field(
    path: str | Path,
    name: str | None = None,
    nodata: float | None = None,
    crs: str | None = None,
) -> SpatialField:
    """Read a 2D spatial field from a file.

    Supported formats:
        - GeoTIFF (.tif, .tiff): reads raster, uses geotransform for coords.
        - NetCDF (.nc): reads first 2D variable.
        - NumPy (.npy, .npz): reads array, assumes physical coords from
          accompanying .geojson or uses user-supplied domain.
        - HDF5 (.h5): reads first 2D dataset.

    Args:
        path: Path to the source file.
        name: Field name (defaults to stem of path).
        nodata: NODATA value (overrides file metadata).
        crs: CRS string override.

    Returns:
        A :class:`SpatialField`.

    Raises:
        ValueError: If the file cannot be read or is not 2D.
    """
    path = Path(path)
    ext = path.suffix.lower()
    field_name = name or path.stem

    if ext in (".tif", ".tiff"):
        return _read_geotiff(path, field_name, nodata, crs)
    elif ext == ".nc":
        return _read_netcdf(path, field_name, nodata, crs)
    elif ext in (".npy", ".npz"):
        return _read_numpy(path, field_name, nodata, crs)
    elif ext == ".h5" or ext == ".hdf5":
        return _read_hdf5_field(path, field_name, nodata, crs)
    else:
        raise ValueError(f"Unsupported field format: {ext}")


def _read_geotiff(
    path: Path,
    name: str,
    nodata: float | None,
    crs: str | None,
) -> SpatialField:
    """Read a GeoTIFF file as a SpatialField."""
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for GeoTIFF files")

    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(
                f"Expected 1 band in GeoTIFF {path}, got {src.count}"
            )
        data = src.read(1, masked=False).astype(np.float64)

        # Get nodata from file or use default
        file_nodata = nodata if nodata is not None else src.nodata
        if file_nodata is None:
            file_nodata = -9999.0

        # Get geotransform (affine transform from pixel to world coords)
        transform = src.transform
        dx = transform.a
        dy = -transform.e  # negative because rasterio y goes up, array y goes down

        # Get bounds
        x_ll = transform.x
        y_ll = transform.y + data.shape[0] * dy  # top-left y + height*dy = bottom y

        # CRS
        field_crs = crs
        if field_crs is None and src.crs is not None:
            field_crs = src.crs.to_string()

        return SpatialField(
            data=data,
            x_ll=float(x_ll),
            y_ll=float(y_ll),
            dx=float(abs(dx)),
            dy=float(abs(dy)),
            nodata=float(file_nodata),
            name=name,
            crs=field_crs,
        )


def _read_netcdf(
    path: Path,
    name: str,
    nodata: float | None,
    crs: str | None,
) -> SpatialField:
    """Read a NetCDF file as a SpatialField."""
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required for NetCDF files")

    ds = xr.open_dataset(path)
    # Find the first 2D data variable
    data_var = None
    for var_name in ds.data_vars:
        if ds[var_name].ndim == 2:
            data_var = ds[var_name]
            break

    if data_var is None:
        raise ValueError(f"No 2D data variable found in {path}")

    data = data_var.values.astype(np.float64)

    # Try to get coordinates
    x_coord = None
    y_coord = None
    for dim in data_var.dims:
        if dim in ds:
            coord = ds[dim]
            if coord.ndim == 1:
                if x_coord is None and "x" in dim.lower():
                    x_coord = coord.values
                elif y_coord is None and "y" in dim.lower():
                    y_coord = coord.values

    if x_coord is not None and y_coord is not None:
        dx = float(np.diff(x_coord).mean())
        dy = float(np.diff(y_coord).mean())
        x_ll = float(x_coord[0])
        y_ll = float(y_coord[0])
    else:
        # Fallback: assume uniform grid starting at (0, 0)
        ny, nx = data.shape
        dx = 1.0
        dy = 1.0
        x_ll = 0.0
        y_ll = 0.0
        warnings.warn(
            f"Could not determine coordinates in {path}. "
            f"Assuming uniform grid starting at (0, 0) with dx=dy=1.0",
            stacklevel=4,
        )

    # Get nodata
    file_nodata = nodata
    if file_nodata is None:
        # Check _FillValue or missing_value attributes
        for attr_name in ("_FillValue", "missing_value", "nodata"):
            if hasattr(data_var, attr_name):
                file_nodata = float(getattr(data_var, attr_name))
                break
    if file_nodata is None:
        file_nodata = -9999.0

    # CRS
    field_crs = crs
    if field_crs is None:
        for attr_name in ("crs", "crs_wkt", "spatial_ref"):
            if hasattr(ds, attr_name):
                field_crs = str(getattr(ds, attr_name))
                break

    return SpatialField(
        data=data,
        x_ll=x_ll,
        y_ll=y_ll,
        dx=dx,
        dy=dy,
        nodata=float(file_nodata),
        name=name,
        crs=field_crs,
    )


def _read_numpy(
    path: Path,
    name: str,
    nodata: float | None,
    crs: str | None,
) -> SpatialField:
    """Read a NumPy file as a SpatialField.

    For .npy files, the array is read directly. Physical coordinates
    must be supplied via domain override in the config, or the field
    will assume a unit grid starting at (0, 0).

    For .npz files, the first array is used.
    """
    if path.suffix.lower() == ".npy":
        data = np.load(path).astype(np.float64)
    else:
        # .npz — read first array
        with np.load(path) as npz:
            arr_name = next(iter(npz.files))
            data = npz[arr_name].astype(np.float64)

    if data.ndim != 2:
        raise ValueError(f"NumPy array must be 2D, got {data.ndim}D")

    # NODATA value
    file_nodata = nodata if nodata is not None else -9999.0

    # For plain NumPy files, we cannot know physical coords.
    # Return a placeholder — the pipeline will need domain/cell_size overrides.
    return SpatialField(
        data=data,
        x_ll=0.0,
        y_ll=0.0,
        dx=1.0,
        dy=1.0,
        nodata=file_nodata,
        name=name,
        crs=crs,
    )


def _read_hdf5_field(
    path: Path,
    name: str,
    nodata: float | None,
    crs: str | None,
) -> SpatialField:
    """Read a 2D dataset from an HDF5 file as a SpatialField.

    Reads the first 2D dataset found. Physical coordinates must come
    from attributes or config overrides.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 files")

    with h5py.File(path, "r") as f:
        # Find first 2D dataset
        dataset_name = None
        dataset = None
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                dataset_name = key
                dataset = obj
                break

        if dataset is None:
            raise ValueError(f"No 2D dataset found in {path}")

        data = np.array(dataset, dtype=np.float64)

        # Try to read attributes
        dx = float(dataset.attrs.get("dx", 1.0))
        dy = float(dataset.attrs.get("dy", dx))
        x_ll = float(dataset.attrs.get("x_ll", 0.0))
        y_ll = float(dataset.attrs.get("y_ll", 0.0))

    file_nodata = nodata if nodata is not None else -9999.0

    return SpatialField(
        data=data,
        x_ll=x_ll,
        y_ll=y_ll,
        dx=dx,
        dy=dy,
        nodata=file_nodata,
        name=name,
        crs=crs,
    )
