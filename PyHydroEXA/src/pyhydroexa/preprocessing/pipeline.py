"""Main preprocessing pipeline.

Orchestrates the full conversion from HydroEXA input files and user-
provided physical fields into a single HDF5 file consumable by the
HydroEXA C++ solver.

Usage:
    from pyhydroexa.preprocessing import preprocess

    preprocess(
        input_file="inputs.dam_break",
        output_file="input.h5",
        fields={
            "z": "bathymetry.tif",
            "roughness": "mannings_n.tif",
            "h": "water_depth.tif",
            "hu": "x_momentum.tif",
            "hv": "y_momentum.tif",
        },
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PreprocessConfig, validate_config
from .fields import SpatialField, read_field
from .geometry import GridResolution, compute_resolution
from .hdf5 import (
    assemble_fluid,
    assemble_terrain,
    assemble_from_water_level,
    write_hdf5,
)
from .input import parse_input_file
from .resampling import combine_resampled_fields, resample_fields_to_grid


def preprocess(
    input_file: str | Path,
    output_file: str | Path,
    fields: dict[str, str] | None = None,
    crs: str | None = None,
    domain: tuple[float, float, float, float] | None = None,
    cell_size: float | None = None,
    default_roughness: float = 0.01,
    default_hu: float = 0.0,
    default_hv: float = 0.0,
    verbose: bool = False,
) -> PreprocessConfig:
    """Run the full preprocessing pipeline.

    Steps:
        1. Parse the HydroEXA input file to extract AMR parameters.
        2. Build a :class:`PreprocessConfig`.
        3. Validate the config.
        4. Compute the base grid :class:`GridResolution`.
        5. Read and resample all physical fields onto the target grid.
        6. Assemble terrain and fluid arrays.
        7. Write the output HDF5 file.

    Args:
        input_file: Path to the HydroEXA input file (.in).
        output_file: Path for the output HDF5 file.
        fields: Mapping of field name → source file path.
            Supported names: z, roughness, h, hu, hv, eta.
            If "eta" is provided, h is computed as eta - z.
            If "h" is provided directly, it is used as-is.
        crs: CRS string for geographic data (e.g. "EPSG:4326").
        domain: Explicit (x_lo, y_lo, x_hi, y_hi) override.
        cell_size: Explicit cell size (meters).
        default_roughness: Default Manning's n (if roughness not provided).
        default_hu: Default x-momentum (if hu not provided).
        default_hv: Default y-momentum (if hv not provided).
        verbose: Print progress messages.

    Returns:
        The :class:`PreprocessConfig` used.

    Raises:
        ValueError: If validation fails or processing encounters an error.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    # Step 1: Parse input file
    if verbose:
        print(f"Parsing input file: {input_file}")
    parsed = parse_input_file(input_file)

    # Step 2: Build config
    config = PreprocessConfig.from_input_file(
        input_path=input_file,
        output_path=output_file,
        fields=fields,
        crs=crs,
        domain=domain,
        cell_size=cell_size,
    )

    # Step 3: Validate
    errors = validate_config(config)
    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Step 4: Compute resolution
    if verbose:
        print(f"Computing grid resolution (blocking_factor={config.blocking_factor})...")
    resolution = compute_resolution(config)

    if verbose:
        print(
            f"  Base grid: {resolution.nx} x {resolution.ny} "
            f"(dx=dy={resolution.dx:.4g} m)"
        )
        # Use adjusted domain bounds when available
        if resolution.adj_x_hi is not None and resolution.adj_y_hi is not None:
            x_lo, y_lo = (config.domain[:2] if config.domain else (0.0, 0.0))
            x_hi, y_hi = resolution.adj_x_hi, resolution.adj_y_hi
        else:
            x_lo, y_lo = 0.0, 0.0
            x_hi = resolution.nx * resolution.dx
            y_hi = resolution.ny * resolution.dy
        print(
            f"  Domain: [{x_lo:.4g}, {y_lo:.4g}] "
            f"to [{x_hi:.4g}, {y_hi:.4g}]"
        )

    # Step 5: Read physical fields
    fields_dict = fields or {}
    spatial_fields: dict[str, SpatialField] = {}

    if verbose:
        print("Reading physical fields...")

    for name, src in fields_dict.items():
        if verbose:
            print(f"  Reading {name}: {src}")
        try:
            field = read_field(src, name=name, crs=crs)
            spatial_fields[name] = field
        except Exception as e:
            raise ValueError(f"Failed to read field '{name}' from {src}: {e}") from e

    # Step 6: Resample fields to target grid
    if verbose:
        print("Resampling fields to target grid...")

    x_ll = resolution.dx * 0  # Will be set from domain or field bounds
    y_ll = resolution.dy * 0

    # Determine target origin from domain or field bounds
    if config.domain is not None:
        x_ll, y_ll, _, _ = config.domain
    elif spatial_fields:
        # Use the minimum bounds across all fields
        x_ll = min(f.x_ll for f in spatial_fields.values())
        y_ll = min(f.y_ll for f in spatial_fields.values())
    else:
        raise ValueError("Cannot determine target origin: no domain or fields provided")

    # Clip all fields to the target domain
    # Use adjusted bounds from GridResolution (blocking-factor-rounded)
    if resolution.adj_x_hi is not None and resolution.adj_y_hi is not None:
        x_hi = resolution.adj_x_hi
        y_hi = resolution.adj_y_hi
    else:
        x_hi = x_ll + resolution.nx * resolution.dx
        y_hi = y_ll + resolution.ny * resolution.dy

    target_domain = (x_ll, y_ll, x_hi, y_hi)

    for name, field in spatial_fields.items():
        spatial_fields[name] = field.clip_to_domain(*target_domain)

    # Resample
    all_fields = list(spatial_fields.values())
    resampled = resample_fields_to_grid(all_fields, resolution, x_ll, y_ll)

    # Combine into a single array per field (in case multiple sources overlap)
    combined_data = combine_resampled_fields(resampled)

    # Reconstruct SpatialFields on the target grid
    target_fields: dict[str, SpatialField] = {}
    for name, field in spatial_fields.items():
        # Find the resampled data for this field
        field_resampled = [r for r in resampled if r.source_name == name]
        if field_resampled:
            data = field_resampled[0].data
        else:
            data = np.full((resolution.ny, resolution.nx), field.nodata, dtype=np.float64)

        target_fields[name] = SpatialField(
            data=data,
            x_ll=x_ll,
            y_ll=y_ll,
            dx=resolution.dx,
            dy=resolution.dy,
            nodata=field.nodata,
            name=name,
            crs=field.crs,
        )

    # Step 7: Assemble terrain and fluid arrays
    if verbose:
        print("Assembling terrain and fluid arrays...")

    terrain_arr: Any = None
    fluid_arr: Any = None

    # Determine which fields are available
    has_z = "z" in target_fields
    has_eta = "eta" in target_fields
    has_h = "h" in target_fields
    has_roughness = "roughness" in target_fields
    has_hu = "hu" in target_fields
    has_hv = "hv" in target_fields

    # Assemble terrain
    if has_z:
        z_field = target_fields["z"]
        roughness_field = target_fields.get("roughness")
        terrain_arr = assemble_terrain(
            z_field, roughness_field, default_roughness
        )
    elif has_eta and not has_h:
        # Need z to compute h from eta — fail if neither z nor h available
        raise ValueError(
            "Either 'z' (bathymetry) or 'h' (water depth) must be provided. "
            "If using 'eta' (water level), 'z' is required to compute h = eta - z."
        )
    else:
        # No terrain data — fill with NODATA
        ny, nx = resolution.ny, resolution.nx
        terrain_arr = np.full((ny, nx * 2), -9999.0, dtype=np.float64)

    # Assemble fluid
    if has_h:
        h_field = target_fields["h"]
        hu_field = target_fields.get("hu")
        hv_field = target_fields.get("hv")
        fluid_arr = assemble_fluid(h_field, hu_field, hv_field, default_hu, default_hv)
    elif has_eta and has_z:
        eta_field = target_fields["eta"]
        z_field = target_fields["z"]
        hu_field = target_fields.get("hu")
        hv_field = target_fields.get("hv")
        terrain_arr, fluid_arr = assemble_from_water_level(
            z_field, eta_field, hu_field, hv_field, default_hu, default_hv
        )
    else:
        # No fluid data — fill with NODATA
        ny, nx = resolution.ny, resolution.nx
        fluid_arr = np.full((ny, nx * 3), -9999.0, dtype=np.float64)

    # Step 8: Write HDF5
    if verbose:
        print(f"Writing HDF5 output: {output_file}")

    write_hdf5(
        output_path=output_file,
        terrain_data=terrain_arr,
        fluid_data=fluid_arr,
        dx=resolution.dx,
        dy=resolution.dy,
        x_ll=x_ll,
        y_ll=y_ll,
    )

    if verbose:
        print(f"Done. Output written to {output_file}")

    return config
