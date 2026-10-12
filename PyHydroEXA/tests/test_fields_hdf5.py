"""Tests for spatial fields, HDF5 output, and resampling."""

from __future__ import annotations

import numpy as np
import pytest

from pyhydroexa.preprocessing.fields import SpatialField, read_field
from pyhydroexa.preprocessing.hdf5 import (
    FLUID_DATASET,
    TERRAIN_DATASET,
    assemble_fluid,
    assemble_from_water_level,
    assemble_terrain,
    write_hdf5,
)
from pyhydroexa.preprocessing.resampling import (
    ResampledField,
    combine_resampled_fields,
    resample_fields_to_grid,
    resample_to_grid,
)


class TestSpatialField:
    """Test SpatialField dataclass."""

    def test_basic(self) -> None:
        data = np.ones((10, 10))
        f = SpatialField(data=data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        assert f.nx == 10
        assert f.ny == 10
        assert f.x_size == 10.0
        assert f.y_size == 10.0
        assert f.x_ur == 10.0
        assert f.y_ur == 10.0

    def test_nodata_mask(self) -> None:
        data = np.ones((5, 5))
        data[0, 0] = -9999.0
        f = SpatialField(data=data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        assert f.is_nodata[0, 0] is True
        assert f.is_valid[0, 0] is False
        assert f.is_valid[1, 1] is True

    def test_square_cell_violation(self) -> None:
        data = np.ones((10, 10))
        with pytest.raises(ValueError, match="square-cell"):
            SpatialField(data=data, x_ll=0.0, y_ll=0.0, dx=10.0, dy=11.0)

    def test_non_2d_raises(self) -> None:
        data = np.ones((5, 5, 5))
        with pytest.raises(ValueError, match="2D"):
            SpatialField(data=data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)

    def test_clip_to_domain(self) -> None:
        data = np.ones((100, 100))
        f = SpatialField(data=data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        clipped = f.clip_to_domain(25.0, 25.0, 75.0, 75.0)
        assert clipped.nx == 50
        assert clipped.ny == 50
        assert clipped.x_ll == 25.0
        assert clipped.y_ll == 25.0


class TestAssembleTerrain:
    """Test terrain array assembly."""

    def test_z_only(self) -> None:
        z_data = np.full((10, 10), 5.0)
        z = SpatialField(data=z_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        terrain = assemble_terrain(z)
        assert terrain.shape == (10, 20)  # (ny, nx * 2)
        assert np.allclose(terrain[:, 0], 5.0)  # z component
        assert np.allclose(terrain[:, 1], 0.01)  # default roughness

    def test_z_and_roughness(self) -> None:
        z_data = np.full((10, 10), 5.0)
        r_data = np.full((10, 10), 0.03)
        z = SpatialField(data=z_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        r = SpatialField(data=r_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        terrain = assemble_terrain(z, r)
        assert np.allclose(terrain[:, 0], 5.0)
        assert np.allclose(terrain[:, 1], 0.03)

    def test_nodata_preserved(self) -> None:
        z_data = np.full((10, 10), -9999.0)
        z = SpatialField(data=z_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        terrain = assemble_terrain(z)
        assert np.all(terrain == -9999.0)


class TestAssembleFluid:
    """Test fluid array assembly."""

    def test_h_only(self) -> None:
        h_data = np.full((10, 10), 2.0)
        h = SpatialField(data=h_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        fluid = assemble_fluid(h)
        assert fluid.shape == (10, 30)  # (ny, nx * 3)
        assert np.allclose(fluid[:, 0], 2.0)  # h component
        assert np.allclose(fluid[:, 1], 0.0)  # default hu
        assert np.allclose(fluid[:, 2], 0.0)  # default hv

    def test_h_hu_hv(self) -> None:
        h_data = np.full((10, 10), 2.0)
        hu_data = np.full((10, 10), 1.5)
        hv_data = np.full((10, 10), 0.5)
        h = SpatialField(data=h_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        hu = SpatialField(data=hu_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        hv = SpatialField(data=hv_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        fluid = assemble_fluid(h, hu, hv)
        assert np.allclose(fluid[:, 0], 2.0)
        assert np.allclose(fluid[:, 1], 1.5)
        assert np.allclose(fluid[:, 2], 0.5)


class TestAssembleFromWaterLevel:
    """Test computing h from eta - z."""

    def test_computation(self) -> None:
        z_data = np.full((10, 10), 0.0)
        eta_data = np.full((10, 10), 5.0)
        z = SpatialField(data=z_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        eta = SpatialField(data=eta_data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)
        terrain, fluid = assemble_from_water_level(z, eta)
        # h = 5.0 - 0.0 = 5.0
        assert np.allclose(fluid[:, 0], 5.0)


class TestWriteHDF5:
    """Test HDF5 file writing."""

    def test_write_and_read(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        import h5py

        out = tmp_path / "test.h5"
        ny, nx = 16, 32
        terrain = np.full((ny, nx * 2), -9999.0, dtype=np.float64)
        fluid = np.full((ny, nx * 3), -9999.0, dtype=np.float64)

        write_hdf5(out, terrain, fluid, dx=10.0, dy=10.0, x_ll=0.0, y_ll=0.0)

        with h5py.File(out, "r") as f:
            assert TERRAIN_DATASET in f
            assert FLUID_DATASET in f
            assert f[TERRAIN_DATASET].shape == (ny, nx * 2)
            assert f[FLUID_DATASET].shape == (ny, nx * 3)
            assert float(f[TERRAIN_DATASET].attrs["dx"]) == 10.0
            assert float(f[TERRAIN_DATASET].attrs["dy"]) == 10.0
            assert float(f[TERRAIN_DATASET].attrs["x_ll"]) == 0.0
            assert float(f[TERRAIN_DATASET].attrs["y_ll"]) == 0.0

    def test_square_cell_enforced(self) -> None:
        terrain = np.zeros((10, 20))
        fluid = np.zeros((10, 30))
        with pytest.raises(ValueError, match="square-cell"):
            write_hdf5("out.h5", terrain, fluid, dx=10.0, dy=11.0, x_ll=0.0, y_ll=0.0)


class TestResampleToGrid:
    """Test physical-coordinate resampling."""

    def test_exact_match(self) -> None:
        data = np.arange(100).reshape(10, 10).astype(np.float64)
        source = SpatialField(data=data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)

        from pyhydroexa.preprocessing.geometry import GridResolution
        target = GridResolution(
            nx=10, ny=10, dx=1.0, dy=1.0,
            refinement_scale=1, blocking_factor=16,
        )
        result = resample_to_grid(source, target, 0.0, 0.0)
        assert result.data.shape == (10, 10)
        assert np.allclose(result.data, data)

    def test_no_overlap_raises(self) -> None:
        data = np.ones((10, 10))
        source = SpatialField(data=data, x_ll=0.0, y_ll=0.0, dx=1.0, dy=1.0)

        from pyhydroexa.preprocessing.geometry import GridResolution
        target = GridResolution(
            nx=10, ny=10, dx=1.0, dy=1.0,
            refinement_scale=1, blocking_factor=16,
        )
        with pytest.raises(ValueError, match="do not overlap"):
            resample_to_grid(source, target, 100.0, 100.0)


class TestCombineResampledFields:
    """Test combining multiple resampled fields."""

    def test_combine_non_nodata(self) -> None:
        d1 = np.array([[1.0, -9999.0], [3.0, 4.0]])
        d2 = np.array([[-9999.0, 2.0], [5.0, 6.0]])
        r1 = ResampledField(data=d1, source_name="a")
        r2 = ResampledField(data=d2, source_name="b")
        combined = combine_resampled_fields([r1, r2])
        assert np.allclose(combined, [[1.0, 2.0], [3.0, 4.0]])

    def test_empty_list(self) -> None:
        combined = combine_resampled_fields([])
        assert combined.shape == (0, 0)
