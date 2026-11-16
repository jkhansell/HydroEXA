"""Tests for :mod:`pyhydroexa.io` -- plotfile reading, fields, AMR geometry.

No real plotfiles: :func:`conftest.make_mock_ds` supplies a fake ``yt``
dataset whose covering grid mimics the real ``(nx, ny, nz)`` layout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conftest import make_mock_ds

from pyhydroexa.io.amr import (
    MeshBlock,
    filter_blocks_by_region,
    get_available_levels,
    get_meshblock_rectangles,
)
from pyhydroexa.io.fields import (
    DEFAULT_ALIASES,
    extract_field,
    get_field_coordinates,
    resolve_field,
)
from pyhydroexa.io.plotfile import (
    Plotfile,
    PlotfileSeries,
    _discover_plotfiles,
    _is_plotfile,
    _natural_sort_key,
)


class TestNaturalSort:
    """Plotfile names must sort numerically, not lexicographically."""

    def test_basic(self):
        assert _natural_sort_key("plt00009") < _natural_sort_key("plt00010")
        assert not (_natural_sort_key("plt00010") < _natural_sort_key("plt00009"))

    def test_mixed(self):
        names = [
            "plt00100", "plt00001", "plt00010", "plt00002", "plt00020",
        ]
        assert sorted(names, key=_natural_sort_key) == [
            "plt00001", "plt00002", "plt00010", "plt00020", "plt00100",
        ]


class TestFieldAliasResolution:
    def test_exact_match(self):
        ds = make_mock_ds()
        assert resolve_field(ds, "h") == ("boxlib", "h_fluid")

    def test_custom_alias(self):
        ds = make_mock_ds()
        # A custom map replaces the defaults rather than extending them,
        # so the alias list has to name the real field.
        aliases = {"h": ["water_depth", "h_fluid"]}
        assert resolve_field(ds, "h", aliases=aliases) == ("boxlib", "h_fluid")

    def test_custom_alias_without_match_raises(self):
        ds = make_mock_ds()
        with pytest.raises(ValueError, match="Could not resolve field"):
            resolve_field(ds, "h", aliases={"h": ["water_depth"]})

    def test_unresolved_field(self):
        ds = make_mock_ds()
        with pytest.raises(ValueError, match="Could not resolve field"):
            resolve_field(ds, "nonexistent_field")

    def test_hu_alias(self):
        ds = make_mock_ds()
        assert resolve_field(ds, "hu") == ("boxlib", "hu_momentum")

    def test_hv_alias(self):
        ds = make_mock_ds()
        assert resolve_field(ds, "hv") == ("boxlib", "hv_momentum")

    def test_z_alias(self):
        ds = make_mock_ds()
        assert resolve_field(ds, "z") == ("boxlib", "z_bathymetry")

    def test_default_alias_table_covers_conserved_variables(self):
        assert set(DEFAULT_ALIASES) == {"h", "hu", "hv", "z"}


class TestDomainExtraction:
    def test_domain_bounds(self):
        ds = make_mock_ds()
        _, _, x_lo, x_hi, y_lo, y_hi = get_field_coordinates(ds)
        assert x_lo == 0.0
        assert x_hi == 10.0
        assert y_lo == 0.0
        assert y_hi == pytest.approx(1.6)

    def test_coordinate_count(self):
        ds = make_mock_ds()
        x, y, _, _, _, _ = get_field_coordinates(ds)
        assert len(x) == 64
        assert len(y) == 16

    def test_coordinates_are_cell_centred(self):
        ds = make_mock_ds()
        x, y, x_lo, x_hi, _, _ = get_field_coordinates(ds)
        dx = (x_hi - x_lo) / len(x)
        assert x[0] == pytest.approx(x_lo + 0.5 * dx)
        assert x[-1] == pytest.approx(x_hi - 0.5 * dx)


class TestFieldExtraction:
    """The (nx, ny) -> (ny, nx) transpose is the whole point of this layer."""

    def test_shape_is_image_order(self):
        ds = make_mock_ds()
        data = extract_field(ds, "h")
        assert data.shape == (16, 64)  # (ny, nx)

    def test_values_are_transposed(self):
        ds = make_mock_ds()
        data = extract_field(ds, "h")
        # The fake covering grid holds arange(nx*ny).reshape(nx, ny).
        assert data[0, 0] == 0.0
        assert data[0, 1] == 16.0  # second x cell, first y row


class TestAMRRectangles:
    def test_single_level(self):
        ds = make_mock_ds()
        blocks = get_meshblock_rectangles(ds)
        assert len(blocks) == 1
        assert blocks[0].level == 0
        assert abs(blocks[0].x0 - 0.0) < 1e-10
        assert abs(blocks[0].width - 10.0) < 1e-10
        assert abs(blocks[0].height - 1.6) < 1e-10

    def test_level_filter(self):
        ds = make_mock_ds()
        assert get_meshblock_rectangles(ds, level_filter=[1]) == []

    def test_available_levels(self):
        ds = make_mock_ds()
        assert get_available_levels(ds) == [0]

    def test_filter_blocks_by_region(self):
        blocks = [
            MeshBlock(x0=0.0, y0=0.0, width=5.0, height=1.6, level=0),
            MeshBlock(x0=5.0, y0=0.0, width=5.0, height=1.6, level=0),
        ]
        filtered = filter_blocks_by_region(blocks, 2.0, 0.0, 4.0, 1.6)
        assert len(filtered) == 1
        assert filtered[0].x0 == 0.0

    def test_filter_blocks_no_overlap(self):
        blocks = [MeshBlock(x0=0.0, y0=0.0, width=2.0, height=1.6, level=0)]
        assert filter_blocks_by_region(blocks, 5.0, 0.0, 8.0, 1.6) == []


class TestPlotfile:
    def test_is_not_plotfile(self, tmp_path: Path):
        d = tmp_path / "not_a_plotfile"
        d.mkdir()
        assert _is_plotfile(str(d)) is False

    def test_is_plotfile(self, tmp_path: Path):
        d = tmp_path / "plt00000"
        d.mkdir()
        (d / "Header").write_text("test")
        assert _is_plotfile(str(d)) is True

    def test_not_found(self, tmp_path: Path):
        d = tmp_path / "not_a_plotfile"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            Plotfile(str(d))

    def test_dataset_loads_lazily(self, tmp_path: Path):
        d = tmp_path / "plt00001"
        d.mkdir()
        (d / "Header").write_text("test")
        with patch("pyhydroexa.io.plotfile.yt") as fake_yt:
            pf = Plotfile(str(d))
            fake_yt.load.assert_not_called()  # construction only stats the path
            pf.ds  # noqa: B018 - triggers the load
            fake_yt.load.assert_called_once()


class TestPlotfileSeries:
    def test_discovery_empty(self):
        with pytest.raises(FileNotFoundError):
            PlotfileSeries("plt99999")

    def test_discovery_single(self, tmp_path: Path):
        d = tmp_path / "plt00001"
        d.mkdir()
        (d / "Header").write_text("test")
        series = PlotfileSeries(str(d))
        assert len(series) == 1
        assert series.paths == [str(d.resolve())]

    def test_natural_sort_discovery(self, tmp_path: Path):
        for name in ["plt00100", "plt00001", "plt00010", "plt00002"]:
            d = tmp_path / name
            d.mkdir()
            (d / "Header").write_text("test")
        series = PlotfileSeries("plt*", directory=str(tmp_path))
        names = [Path(p).name for p in series.paths]
        assert names == ["plt00001", "plt00002", "plt00010", "plt00100"]

    def test_discover_skips_non_plotfiles(self, tmp_path: Path):
        (tmp_path / "plt00001").mkdir()
        (tmp_path / "plt00001" / "Header").write_text("test")
        (tmp_path / "plt00002").mkdir()  # no Header
        assert [Path(p).name for p in _discover_plotfiles("plt*", str(tmp_path))] == [
            "plt00001"
        ]
