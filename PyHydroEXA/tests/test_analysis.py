"""Tests for :mod:`pyhydroexa.analysis` -- profile extraction.

Profiles are the pure-number side of the package: no plotting stack, no
figure objects, just arrays that can be compared or written to disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_mock_ds

from pyhydroexa.analysis import (
    LineProfile,
    extract_horizontal_cut,
    extract_profile,
    extract_vertical_cut,
)


class TestProfileExtraction:
    def test_horizontal_cuts(self):
        ds = make_mock_ds()
        x_coords, profiles = extract_profile(ds, "h", axis="x", ny_lines=3)
        assert len(profiles) == 3
        assert len(x_coords) == 64
        for profile in profiles:
            assert isinstance(profile.values, np.ndarray)
            assert len(profile.values) == 64  # along x

    def test_vertical_cuts(self):
        ds = make_mock_ds()
        y_coords, profiles = extract_profile(ds, "h", axis="y", ny_lines=3)
        assert len(profiles) == 3
        assert len(y_coords) == 16
        for profile in profiles:
            assert isinstance(profile.values, np.ndarray)
            assert len(profile.values) == 16  # along y

    def test_custom_positions(self):
        ds = make_mock_ds()
        _, profiles = extract_profile(ds, "h", axis="x", positions=[0.5, 1.0])
        assert len(profiles) == 2
        assert profiles[0].coordinate <= profiles[1].coordinate

    def test_positions_snap_to_nearest_row(self):
        ds = make_mock_ds()
        _, profiles = extract_profile(ds, "h", axis="x", positions=[0.799])
        # Cell centres are 0.05, 0.15, ... 1.55; 0.799 must land on 0.75.
        assert profiles[0].coordinate == pytest.approx(0.75)

    def test_invalid_axis(self):
        ds = make_mock_ds()
        with pytest.raises(ValueError, match="axis must be 'x' or 'y'"):
            extract_profile(ds, "h", axis="z")

    def test_axis_is_case_insensitive(self):
        ds = make_mock_ds()
        _, profiles = extract_profile(ds, "h", axis="X", ny_lines=1)
        assert len(profiles) == 1

    def test_single_horizontal_cut(self):
        ds = make_mock_ds()
        y_val, x_coords, values = extract_horizontal_cut(ds, "h", y=0.8)
        assert len(x_coords) == 64
        assert len(values) == 64
        assert y_val == pytest.approx(0.75)

    def test_single_vertical_cut(self):
        ds = make_mock_ds()
        x_val, y_coords, values = extract_vertical_cut(ds, "h", x=5.0)
        assert len(y_coords) == 16
        assert len(values) == 16
        assert x_val == pytest.approx(5.0, abs=0.2)


class TestLineProfile:
    def test_dataclass_holds_arrays(self):
        values = np.array([1.0, 2.0, 3.0])
        coords = np.array([0.0, 0.5, 1.0])
        profile = LineProfile(
            coordinate=0.5, values=values, axis_coords=coords, label="test"
        )
        assert profile.coordinate == 0.5
        assert profile.label == "test"
        np.testing.assert_array_equal(profile.values, values)

    def test_label_defaults_to_empty(self):
        profile = LineProfile(
            coordinate=0.0,
            values=np.zeros(2),
            axis_coords=np.zeros(2),
        )
        assert profile.label == ""
