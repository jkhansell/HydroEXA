"""Tests for :mod:`pyhydroexa.plotting`.

Two things are worth pinning here: that rendering actually produces an
image headlessly, and that the layering rule holds -- drawing may depend
on reading, never the other way round.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # importing pyplot must be safe with no display

from conftest import make_mock_ds

from pyhydroexa.io.fields import extract_field


class _FakePlotfile:
    """Just enough of :class:`pyhydroexa.io.Plotfile` to render one frame."""

    def __init__(self, ds) -> None:
        self.ds = ds
        self.num_levels = 0

    def extract(self, field_name, level: int = 0, **_):
        return extract_field(self.ds, field_name, level=level)


class TestPlottingAPI:
    def test_plot_field_signature(self):
        from pyhydroexa.plotting import plot_field

        params = list(inspect.signature(plot_field).parameters)
        assert "field" in params
        assert "cmap" in params
        assert "colorbar" in params
        assert "output" in params
        # The overlay variant owns the AMR knobs.
        assert "show_amr" not in params

    def test_plot_field_with_amr_overlay_signature(self):
        from pyhydroexa.plotting import plot_field_with_amr_overlay

        params = list(inspect.signature(plot_field_with_amr_overlay).parameters)
        assert "field" in params
        assert "show_amr" in params
        assert "amr_level_colors" in params

    def test_plot_profiles_signature(self):
        from pyhydroexa.plotting import plot_profiles

        params = list(inspect.signature(plot_profiles).parameters)
        assert "axis_coords" in params
        assert "profiles" in params
        assert "output" in params

    def test_generate_animation_signature(self):
        from pyhydroexa.plotting import generate_animation

        params = list(inspect.signature(generate_animation).parameters)
        assert "series" in params
        assert "field" in params
        assert "duration" in params


class TestRendering:
    def test_frame_is_png_bytes(self):
        from pyhydroexa.plotting.animation import _render_frame

        png = _render_frame(_FakePlotfile(make_mock_ds()), "h", dpi=50, figsize=(4, 2))
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_plot_writes_a_file(self, tmp_path: Path):
        from pyhydroexa.plotting import plot_field_with_amr_overlay

        out = tmp_path / "depth.png"
        plot_field_with_amr_overlay(
            _FakePlotfile(make_mock_ds()), "h", output=str(out), dpi=50, figsize=(4, 2)
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestLayering:
    """The reader layer must stay usable without a plotting stack."""

    @staticmethod
    def _imported_modules(module) -> set[str]:
        tree = ast.parse(inspect.getsource(module))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.add(node.module or "")
        return found

    def test_io_does_not_import_plotting(self):
        import pyhydroexa.io as io_module

        modules = self._imported_modules(io_module)
        assert not any(m.startswith("matplotlib") for m in modules)
        assert not any("plotting" in m for m in modules)

    def test_analysis_does_not_import_plotting(self):
        import pyhydroexa.analysis as analysis_module

        modules = self._imported_modules(analysis_module)
        assert not any(m.startswith("matplotlib") for m in modules)
        assert not any("plotting" in m for m in modules)

    def test_public_facades_expose_expected_names(self):
        import pyhydroexa.analysis
        import pyhydroexa.io
        import pyhydroexa.plotting

        for module in (pyhydroexa.io, pyhydroexa.analysis, pyhydroexa.plotting):
            assert module.__all__
            for name in module.__all__:
                assert hasattr(module, name), f"{module.__name__}.{name} missing"
