"""Tests for blocking factor and grid resolution utilities."""

from __future__ import annotations

import pytest

from pyhydroexa.preprocessing.blocking import (
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
from pyhydroexa.preprocessing.config import PreprocessConfig
from pyhydroexa.preprocessing.geometry import GridResolution, compute_resolution
from pathlib import Path


class TestIsValidBlockingFactor:
    """Test blocking factor validation."""

    def test_power_of_2(self) -> None:
        assert is_valid_blocking_factor(8) is True
        assert is_valid_blocking_factor(16) is True
        assert is_valid_blocking_factor(32) is True
        assert is_valid_blocking_factor(64) is True

    def test_not_power_of_2(self) -> None:
        assert is_valid_blocking_factor(6) is False
        assert is_valid_blocking_factor(12) is False
        assert is_valid_blocking_factor(10) is False

    def test_edge_cases(self) -> None:
        assert is_valid_blocking_factor(0) is False
        assert is_valid_blocking_factor(-1) is False
        assert is_valid_blocking_factor(1) is True


class TestComputeGridSize:
    """Test grid size computation."""

    def test_exact_division(self) -> None:
        size = compute_grid_size(160.0, 10.0, 16)
        assert size == 16

    def test_rounds_up(self) -> None:
        size = compute_grid_size(165.0, 10.0, 16)
        assert size == 176  # round to 16.5 → 17 cells → round up to 112...

    def test_small_domain(self) -> None:
        size = compute_grid_size(5.0, 10.0, 16)
        assert size == 16  # minimum is blocking_factor


class TestAdjustDomainForBlocking:
    """Test domain adjustment."""

    def test_exact_fit(self) -> None:
        result = adjust_domain_for_blocking(0, 0, 160, 160, 10, 16)
        assert result == (0, 0, 160, 160)

    def test_expands_hi(self) -> None:
        result = adjust_domain_for_blocking(0, 0, 165, 165, 10, 16)
        # 16.5 cells → 17 cells → round up to 112...
        # Actually: round(165/10) = 17, ceil(17/16)*16 = 32
        # Wait: ((17 + 15) // 16) * 16 = (32 // 16) * 16 = 2 * 16 = 32
        # So x_hi = 0 + 32 * 10 = 320
        assert result[2] == 320
        assert result[3] == 320


class TestTerrainResolution:
    """Test terrain refinement resolution computation."""

    def test_base_from_fine(self) -> None:
        nx, ny = terrain_base_resolution(256, 256, 2)
        assert nx == 64
        assert ny == 64

    def test_fine_from_base(self) -> None:
        nx, ny = terrain_dataset_resolution(64, 64, 2)
        assert nx == 256
        assert ny == 256

    def test_no_refinement(self) -> None:
        nx, ny = terrain_base_resolution(100, 100, 0)
        assert nx == 100
        assert ny == 100


class TestGridResolution:
    """Test GridResolution dataclass validation."""

    def test_valid_resolution(self) -> None:
        r = GridResolution(
            nx=160, ny=160, dx=10.0, dy=10.0,
            refinement_scale=1, blocking_factor=16,
        )
        assert r.nx == 160
        assert r.x_size == 1600.0

    def test_square_cell_violation(self) -> None:
        with pytest.raises(ValueError, match="square-cell"):
            GridResolution(
                nx=160, ny=160, dx=10.0, dy=11.0,
                refinement_scale=1, blocking_factor=16,
            )

    def test_blocking_factor_violation(self) -> None:
        with pytest.raises(ValueError, match="blocking_factor"):
            GridResolution(
                nx=161, ny=160, dx=10.0, dy=10.0,
                refinement_scale=1, blocking_factor=16,
            )


class TestComputeResolution:
    """Test resolution computation from config."""

    def test_compute_with_domain_and_cell_size(
        self, tmp_path: Path
    ) -> None:
        inp = tmp_path / "inputs"
        inp.write_text("amr.blocking_factor 16\n")
        config = PreprocessConfig(
            input_file=inp,
            output_file=tmp_path / "out.h5",
            domain=(0.0, 0.0, 160.0, 160.0),
            cell_size=10.0,
            blocking_factor=16,
        )
        resolution = compute_resolution(config)
        assert resolution.nx == 16
        assert resolution.ny == 16
        assert resolution.dx == 10.0
        assert resolution.dy == 10.0

    def test_compute_requires_cell_size(
        self, tmp_path: Path
    ) -> None:
        inp = tmp_path / "inputs"
        inp.write_text("amr.blocking_factor 16\n")
        config = PreprocessConfig(
            input_file=inp,
            output_file=tmp_path / "out.h5",
            blocking_factor=16,
        )
        with pytest.raises(ValueError, match="cell_size"):
            compute_resolution(config)

    def test_compute_adjusts_for_blocking(
        self, tmp_path: Path
    ) -> None:
        inp = tmp_path / "inputs"
        inp.write_text("amr.blocking_factor 16\n")
        # 165 / 10 = 16.5 → round to 17 → ceil to 32
        config = PreprocessConfig(
            input_file=inp,
            output_file=tmp_path / "out.h5",
            domain=(0.0, 0.0, 165.0, 165.0),
            cell_size=10.0,
            blocking_factor=16,
        )
        resolution = compute_resolution(config)
        assert resolution.nx == 32
        assert resolution.ny == 32
