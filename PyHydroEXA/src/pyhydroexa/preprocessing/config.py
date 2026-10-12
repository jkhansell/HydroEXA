"""Typed configuration for the preprocessing pipeline.

Extracts and validates the subset of HydroEXA input parameters that
preprocessing needs, plus user-supplied field configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .input import ParsedInput


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for a single preprocessing run.

    Attributes:
        input_file: Path to the HydroEXA input file (.in).
        output_file: Path for the output HDF5 file.
        physical_fields: Mapping of field name → source file path.
            Keys are the field names used in the output.
            Values are paths to GeoTIFF, NetCDF, or NumPy files.
        crs: CRS string (e.g. EPSG:4326) for geographic data.
            If None, data is assumed to be in physical (m) units.
        terrain_ref_lev: AMR terrain refinement level from the input file.
        blocking_factor: AMReX blocking factor from the input file.
        max_level: Maximum AMR level from the input file.
        domain: Explicit domain override (x_lo, y_lo, x_hi, y_hi).
            If None, domain is inferred from the input file or fields.
        cell_size: Explicit cell size override. If None, computed from
            the input file's domain and blocking_factor.
        default_nodata: NODATA sentinel value (default -9999.0).
    """

    input_file: Path
    output_file: Path
    physical_fields: dict[str, str] = field(default_factory=dict)
    crs: str | None = None
    terrain_ref_lev: int = 0
    blocking_factor: int = 16
    max_level: int = 0
    domain: tuple[float, float, float, float] | None = None
    cell_size: float | None = None
    default_nodata: float = -9999.0

    # --- Derived properties ---

    @property
    def refinement_scale(self) -> int:
        """Grid refinement scale: 2^terrain_ref_lev."""
        return 1 << self.terrain_ref_lev

    @property
    def base_nx(self) -> int | None:
        """Base grid nx, if domain and cell_size are known."""
        if self.domain and self.cell_size:
            x_lo, y_lo, x_hi, y_hi = self.domain
            nx = round((x_hi - x_lo) / self.cell_size)
            return nx
        return None

    @property
    def base_ny(self) -> int | None:
        """Base grid ny, if domain and cell_size are known."""
        if self.domain and self.cell_size:
            x_lo, y_lo, x_hi, y_hi = self.domain
            ny = round((y_hi - y_lo) / self.cell_size)
            return ny
        return None

    @classmethod
    def from_input_file(
        cls,
        input_path: str | Path,
        output_path: str | Path,
        fields: dict[str, str] | None = None,
        crs: str | None = None,
        domain: tuple[float, float, float, float] | None = None,
        cell_size: float | None = None,
    ) -> PreprocessConfig:
        """Build config from a parsed HydroEXA input file.

        Args:
            input_path: Path to the .in file.
            output_path: Path for the output HDF5 file.
            fields: Mapping of field name → source file.
            crs: CRS string for geographic data.
            domain: Explicit (x_lo, y_lo, x_hi, y_hi) override.
            cell_size: Explicit cell size override.

        Returns:
            A :class:`PreprocessConfig` with values from the input file.
        """
        parsed = ParsedInput.__new__(ParsedInput)
        # We need to parse the file first
        from .input import parse_input_file
        parsed = parse_input_file(input_path)

        return cls(
            input_file=Path(input_path),
            output_file=Path(output_path),
            physical_fields=fields or {},
            crs=crs,
            terrain_ref_lev=parsed.terrain_ref_lev,
            blocking_factor=parsed.blocking_factor,
            max_level=parsed.max_level,
            domain=domain,
            cell_size=cell_size,
        )


def validate_config(config: PreprocessConfig) -> list[str]:
    """Validate a :class:`PreprocessConfig` and return a list of errors.

    Checks:
        - Input file exists.
        - Output directory is writable.
        - All physical field source files exist.
        - Blocking factor is a power of 2.
        - Cell size is positive (if provided).
        - Domain is valid (if provided).
        - Square-cell invariant: dx == dy (enforced by single cell_size).

    Returns:
        List of error strings (empty if valid).
    """
    errors: list[str] = []

    if not config.input_file.exists():
        errors.append(f"Input file not found: {config.input_file}")

    output_dir = config.output_file.parent
    if not output_dir.exists():
        errors.append(f"Output directory does not exist: {output_dir}")

    for name, src in config.physical_fields.items():
        if not Path(src).exists():
            errors.append(f"Physical field source not found: {name} → {src}")

    if config.blocking_factor <= 0:
        errors.append(f"Blocking factor must be positive: {config.blocking_factor}")
    elif (config.blocking_factor & (config.blocking_factor - 1)) != 0:
        errors.append(
            f"Blocking factor must be a power of 2: {config.blocking_factor}"
        )

    if config.cell_size is not None and config.cell_size <= 0:
        errors.append(f"Cell size must be positive: {config.cell_size}")

    if config.domain is not None:
        x_lo, y_lo, x_hi, y_hi = config.domain
        if x_hi <= x_lo:
            errors.append(f"Invalid domain x range: {x_lo} > {x_hi}")
        if y_hi <= y_lo:
            errors.append(f"Invalid domain y range: {y_lo} > {y_hi}")

    return errors
