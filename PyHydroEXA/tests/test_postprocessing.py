"""Tests for pyhydroexa.postprocessing.

Uses mocking where actual plotfiles are not available.  No fabricated
AMReX plotfiles are created.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pyhydroexa.postprocessing.fields import (
    DEFAULT_ALIASES,
    resolve_field,
    extract_field,
    get_field_coordinates,
)
from pyhydroexa.postprocessing.plotfile import (
    Plotfile,
    PlotfileSeries,
    _natural_sort_key,
    _discover_plotfiles,
    _is_plotfile,
)
from pyhydroexa.postprocessing.amr import (
    MeshBlock,
    get_meshblock_rectangles,
    get_available_levels,
    filter_blocks_by_region,
)
from pyhydroexa.postprocessing.profiles import (
    LineProfile,
    extract_profile,
    extract_horizontal_cut,
    extract_vertical_cut,
)


# ===================================================================
# Fixtures
# ===================================================================

def _make_mock_ds(field_names: list[str] | None = None) -> MagicMock:
    """Create a minimal mock ``yt.Dataset``.

    The mock exposes:
    - ``field_list`` — list of field tuples
    - ``domain_left_edge``, ``domain_right_edge`` — 3-element arrays
    - ``domain_dimensions`` — 3-element list
    - ``current_time`` — float
    - ``dimensionality`` — int
    - ``num_levels`` — int
    - ``index`` — mock with grid info
    """
    if field_names is None:
        field_names = ["h_fluid", "hu_momentum", "hv_momentum", "z_bathymetry"]

    ds = MagicMock()
    ds.field_list = [("boxlib", n) for n in field_names]
    ds.domain_left_edge = np.array([0.0, 0.0, 0.0])
    ds.domain_right_edge = np.array([10.0, 1.6, 0.0])
    ds.domain_dimensions = [64, 16, 1]
    ds.current_time = 0.15625
    ds.dimensionality = 2
    ds.num_levels = 1
    ds.index.grid_left_edge = np.array([[0.0, 0.0]])
    ds.index.grid_dimensions = np.array([[64, 16]])
    ds.index.grid_levels = np.array([0])
    ds.index.level_dds = {0: np.array([0.15625, 0.1, 0.0])}

    # covering_grid returns a mock with extractable fields
    covering = MagicMock()
    mock_data = np.random.rand(16, 64)  # (ny, nx)
    covering[("boxlib", "h_fluid")] = MagicMock()
    covering[("boxlib", "h_fluid")].to_ndarray.return_value = mock_data
    ds.covering_grid.return_value = covering

    return ds


# ===================================================================
# Natural sort tests
# ===================================================================

class TestNaturalSort:
    def test_basic(self):
        assert _natural_sort_key("plt00010") < _natural_sort_key("plt00009") is False
        assert _natural_sort_key("plt00009") < _natural_sort_key("plt00010")

    def test_mixed(self):
        keys = [
            _natural_sort_key(p) for p in [
                "plt00100", "plt00001", "plt00010", "plt00002", "plt00020",
            ]
        ]
        assert keys == sorted(keys)


# ===================================================================
# Field alias resolution tests
# ===================================================================

class TestFieldAliasResolution:
    def test_exact_match(self):
        ds = _make_mock_ds()
        result = resolve_field(ds, "h")
        assert result == ("boxlib", "h_fluid")

    def test_custom_alias(self):
        ds = _make_mock_ds()
        aliases = {"h": ["water_depth"]}
        # Should fall through to substring match
        result = resolve_field(ds, "h", aliases=aliases)
        assert result == ("boxlib", "h_fluid")

    def test_unresolved_field(self):
        ds = _make_mock_ds()
        with pytest.raises(ValueError, match="Could not resolve field"):
            resolve_field(ds, "nonexistent_field")

    def test_hu_alias(self):
        ds = _make_mock_ds()
        result = resolve_field(ds, "hu")
        assert result == ("boxlib", "hu_momentum")

    def test_hv_alias(self):
        ds = _make_mock_ds()
        result = resolve_field(ds, "hv")
        assert result == ("boxlib", "hv_momentum")

    def test_z_alias(self):
        ds = _make_mock_ds()
        result = resolve_field(ds, "z")
        assert result == ("boxlib", "z_bathymetry")


# ===================================================================
# Domain extraction tests
# ===================================================================

class TestDomainExtraction:
    def test_domain_bounds(self):
        ds = _make_mock_ds()
        x, y, x_lo, x_hi, y_lo, y_hi = get_field_coordinates(ds)
        assert x_lo == 0.0
        assert x_hi == 10.0
        assert y_lo == 0.0
        assert y_hi == 1.6

    def test_coordinate_count(self):
        ds = _make_mock_ds()
        x, y, _, _, _, _ = get_field_coordinates(ds)
        assert len(x) == 64
        assert len(y) == 16


# ===================================================================
# AMR rectangle extraction tests
# ===================================================================

class TestAMRRectangles:
    def test_single_level(self):
        ds = _make_mock_ds()
        blocks = get_meshblock_rectangles(ds)
        assert len(blocks) == 1
        assert blocks[0].level == 0
        assert abs(blocks[0].x0 - 0.0) < 1e-10
        assert abs(blocks[0].width - 10.0) < 1e-10

    def test_level_filter(self):
        ds = _make_mock_ds()
        blocks = get_meshblock_rectangles(ds, level_filter=[1])
        assert len(blocks) == 0

    def test_available_levels(self):
        ds = _make_mock_ds()
        levels = get_available_levels(ds)
        assert len(levels) == 1
        assert levels[0] == 0

    def test_filter_blocks_by_region(self):
        blocks = [
            MeshBlock(x0=0.0, y0=0.0, width=5.0, height=1.6, level=0),
            MeshBlock(x0=5.0, y0=0.0, width=5.0, height=1.6, level=0),
        ]
        filtered = filter_blocks_by_region(blocks, 2.0, 0.0, 4.0, 1.6)
        assert len(filtered) == 1
        assert filtered[0].x0 == 0.0

    def test_filter_blocks_no_overlap(self):
        blocks = [
            MeshBlock(x0=0.0, y0=0.0, width=2.0, height=1.6, level=0),
        ]
        filtered = filter_blocks_by_region(blocks, 5.0, 0.0, 8.0, 1.6)
        assert len(filtered) == 0


# ===================================================================
# Profile extraction tests
# ===================================================================

class TestProfileExtraction:
    def test_horizontal_cuts(self):
        ds = _make_mock_ds()
        x_coords, profiles = extract_profile(ds, "h", axis="x", ny_lines=3)
        assert len(profiles) == 3
        for p in profiles:
            assert isinstance(p.values, np.ndarray)
            assert len(p.values) == 64  # nx

    def test_vertical_cuts(self):
        ds = _make_mock_ds()
        x_coords, profiles = extract_profile(ds, "h", axis="y", ny_lines=3)
        assert len(profiles) == 3
        for p in profiles:
            assert isinstance(p.values, np.ndarray)
            assert len(p.values) == 16  # ny

    def test_custom_positions(self):
        ds = _make_mock_ds()
        x_coords, profiles = extract_profile(
            ds, "h", axis="x", positions=[0.5, 1.0]
        )
        assert len(profiles) == 2

    def test_invalid_axis(self):
        ds = _make_mock_ds()
        with pytest.raises(ValueError, match="axis must be 'x' or 'y'"):
            extract_profile(ds, "h", axis="z")

    def test_single_horizontal_cut(self):
        ds = _make_mock_ds()
        y_val, x_coords, values = extract_horizontal_cut(ds, "h", y=0.8)
        assert len(x_coords) == 64
        assert len(values) == 64

    def test_single_vertical_cut(self):
        ds = _make_mock_ds()
        x_val, y_coords, values = extract_vertical_cut(ds, "h", x=5.0)
        assert len(y_coords) == 16
        assert len(values) == 16


# ===================================================================
# Plotfile / PlotfileSeries tests
# ===================================================================

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
        with patch.dict(sys.modules, {"yt": MagicMock()}):
            with pytest.raises(FileNotFoundError):
                Plotfile(str(d))


class TestPlotfileSeries:
    def test_discovery_empty(self):
        with patch.dict(sys.modules, {"yt": MagicMock()}):
            with pytest.raises(FileNotFoundError):
                PlotfileSeries("plt99999")

    def test_discovery_single(self, tmp_path: Path):
        d = tmp_path / "plt00001"
        d.mkdir()
        (d / "Header").write_text("test")
        with patch.dict(sys.modules, {"yt": MagicMock()}):
            series = PlotfileSeries(str(d))
            assert len(series) == 1

    def test_natural_sort_discovery(self, tmp_path: Path):
        for name in ["plt00100", "plt00001", "plt00010", "plt00002"]:
            d = tmp_path / name
            d.mkdir()
            (d / "Header").write_text("test")
        with patch.dict(sys.modules, {"yt": MagicMock()}):
            series = PlotfileSeries("plt*", directory=str(tmp_path))
            paths = series.paths
            assert len(paths) == 4
            # Verify natural sort order
            assert "plt00001" in paths[0]
            assert "plt00002" in paths[1]
            assert "plt00010" in paths[2]
            assert "plt00100" in paths[3]


# ===================================================================
# Plotting API construction tests
# ===================================================================

class TestPlottingAPI:
    def test_plot_field_signature(self):
        """Verify plot_field accepts expected kwargs."""
        from pyhydroexa.postprocessing.plots import plot_field
        import inspect
        sig = inspect.signature(plot_field)
        params = list(sig.parameters.keys())
        assert "field" in params
        assert "cmap" in params
        assert "colorbar" in params
        assert "output" in params
        assert "show_amr" not in params  # that's in plot_field_with_amr_overlay

    def test_plot_field_with_amr_overlay_signature(self):
        from pyhydroexa.postprocessing.plots import plot_field_with_amr_overlay
        import inspect
        sig = inspect.signature(plot_field_with_amr_overlay)
        params = list(sig.parameters.keys())
        assert "field" in params
        assert "show_amr" in params
        assert "amr_level_colors" in params

    def test_plot_profiles_signature(self):
        from pyhydroexa.postprocessing.plots import plot_profiles
        import inspect
        sig = inspect.signature(plot_profiles)
        params = list(sig.parameters.keys())
        assert "axis_coords" in params
        assert "profiles" in params
        assert "output" in params

    def test_line_profile_dataclass(self):
        vals = np.array([1.0, 2.0, 3.0])
        coords = np.array([0.0, 0.5, 1.0])
        lp = LineProfile(coordinate=0.5, values=vals, axis_coords=coords, label="test")
        assert lp.coordinate == 0.5
        np.testing.assert_array_equal(lp.values, vals)
        assert lp.label == "test"
