"""Preprocessing submodule — convert HydroEXA inputs to HDF5.

Public API:
    preprocess() — main pipeline function

Configuration:
    PreprocessConfig — typed configuration dataclass
    validate_config() — config validation

Input parsing:
    parse_input_file() — parse HydroEXA .in files
    ParsedInput — parsed input dataclass

Spatial fields:
    SpatialField — 2D field with geospatial metadata
    read_field() — read from GeoTIFF/NetCDF/NumPy/HDF5

Grid resolution:
    GridResolution — computed base grid parameters
    compute_resolution() — derive resolution from config

HDF5 output:
    write_hdf5() — write terrain + fluid datasets
    assemble_terrain() — build terrain array from z + roughness
    assemble_fluid() — build fluid array from h + hu + hv
    assemble_from_water_level() — compute h from eta - z

Blocking factor:
    is_valid_blocking_factor() — power-of-2 check
    compute_grid_size() — round up to blocking factor multiple
    adjust_domain_for_blocking() — expand domain to satisfy BF
    terrain_base_resolution() — base grid from fine grid
    terrain_dataset_resolution() — fine grid from base grid
"""

from .blocking import (
    DEFAULT_BLOCKING_FACTOR,
    VALID_BLOCKING_FACTORS,
    adjust_domain_for_blocking,
    compute_grid_size,
    compute_physical_size,
    is_valid_blocking_factor,
    nearest_valid_blocking_factor,
    terrain_base_resolution,
    terrain_dataset_resolution,
)
from .config import PreprocessConfig, validate_config
from .fields import SpatialField, read_field
from .geometry import GridResolution, compute_resolution
from .hdf5 import (
    FLUID_DATASET,
    TERRAIN_DATASET,
    assemble_fluid,
    assemble_from_water_level,
    assemble_terrain,
    write_hdf5,
)
from .input import ParsedInput, parse_input_file
from .pipeline import preprocess
from .resampling import (
    ResampledField,
    combine_resampled_fields,
    resample_fields_to_grid,
    resample_to_grid,
)

__all__ = [
    # Pipeline
    "preprocess",
    # Config
    "PreprocessConfig",
    "validate_config",
    # Input parsing
    "ParsedInput",
    "parse_input_file",
    # Fields
    "SpatialField",
    "read_field",
    # Geometry
    "GridResolution",
    "compute_resolution",
    # HDF5
    "TERRAIN_DATASET",
    "FLUID_DATASET",
    "assemble_terrain",
    "assemble_fluid",
    "assemble_from_water_level",
    "write_hdf5",
    # Resampling
    "ResampledField",
    "resample_to_grid",
    "resample_fields_to_grid",
    "combine_resampled_fields",
    # Blocking
    "DEFAULT_BLOCKING_FACTOR",
    "VALID_BLOCKING_FACTORS",
    "is_valid_blocking_factor",
    "nearest_valid_blocking_factor",
    "compute_grid_size",
    "compute_physical_size",
    "adjust_domain_for_blocking",
    "terrain_base_resolution",
    "terrain_dataset_resolution",
]
