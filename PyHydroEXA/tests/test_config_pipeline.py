"""Tests for config validation and the full preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from pyhydroexa.preprocessing.config import PreprocessConfig, validate_config
from pyhydroexa.preprocessing.pipeline import preprocess
from pathlib import Path


class TestValidateConfig:
    """Test config validation."""

    def test_valid_config(self, tmp_path: Path) -> None:
        inp = tmp_path / "inputs"
        inp.write_text("amr.blocking_factor 16\n")
        config = PreprocessConfig(
            input_file=inp,
            output_file=tmp_path / "out.h5",
            blocking_factor=16,
        )
        errors = validate_config(config)
        assert errors == []

    def test_missing_input_file(self, tmp_path: Path) -> None:
        config = PreprocessConfig(
            input_file=tmp_path / "nonexistent.in",
            output_file=tmp_path / "out.h5",
        )
        errors = validate_config(config)
        assert any("not found" in e for e in errors)

    def test_non_power_of_2_blocking(self, tmp_path: Path) -> None:
        inp = tmp_path / "inputs"
        inp.write_text("")
        config = PreprocessConfig(
            input_file=inp,
            output_file=tmp_path / "out.h5",
            blocking_factor=10,
        )
        errors = validate_config(config)
        assert any("power of 2" in e for e in errors)

    def test_missing_field_source(self, tmp_path: Path) -> None:
        inp = tmp_path / "inputs"
        inp.write_text("")
        config = PreprocessConfig(
            input_file=inp,
            output_file=tmp_path / "out.h5",
            physical_fields={"z": str(tmp_path / "missing.tif")},
        )
        errors = validate_config(config)
        assert any("not found" in e for e in errors)

    def test_invalid_domain(self, tmp_path: Path) -> None:
        inp = tmp_path / "inputs"
        inp.write_text("")
        config = PreprocessConfig(
            input_file=inp,
            output_file=tmp_path / "out.h5",
            domain=(100.0, 0.0, 0.0, 100.0),  # x_hi < x_lo
        )
        errors = validate_config(config)
        assert any("x range" in e for e in errors)


class TestPreprocessConfigProperties:
    """Test PreprocessConfig derived properties."""

    def test_refinement_scale(self) -> None:
        config = PreprocessConfig(
            input_file="/dev/null",
            output_file="/dev/null",
            terrain_ref_lev=3,
        )
        assert config.refinement_scale == 8

    def test_base_nx(self) -> None:
        config = PreprocessConfig(
            input_file="/dev/null",
            output_file="/dev/null",
            domain=(0.0, 0.0, 100.0, 100.0),
            cell_size=10.0,
        )
        assert config.base_nx == 10
        assert config.base_ny == 10

    def test_base_nx_none(self) -> None:
        config = PreprocessConfig(
            input_file="/dev/null",
            output_file="/dev/null",
        )
        assert config.base_nx is None


class TestPreprocessPipeline:
    """Integration test for the full preprocessing pipeline."""

    def test_minimal_pipeline(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Test the full pipeline with minimal inputs."""
        import h5py

        # Create input file
        inp = tmp_path / "inputs"
        inp.write_text(
            "HydroEXA.max_time 0.1\n"
            "amr.max_level 0\n"
            "amr.blocking_factor 16\n"
            "amr.terrain_ref_lev 0\n"
        )

        # Create z (bathymetry) field as a simple NumPy array
        z_data = np.full((32, 32), 5.0, dtype=np.float64)
        z_file = tmp_path / "z.npy"
        np.save(z_file, z_data)

        # Create h (water depth) field
        h_data = np.full((32, 32), 2.0, dtype=np.float64)
        h_file = tmp_path / "h.npy"
        np.save(h_file, h_data)

        # Run preprocessing
        output = tmp_path / "input.h5"
        config = preprocess(
            input_file=inp,
            output_file=output,
            fields={"z": str(z_file), "h": str(h_file)},
            domain=(0.0, 0.0, 320.0, 320.0),
            cell_size=10.0,
            verbose=False,
        )

        # Verify output
        assert output.exists()

        with h5py.File(output, "r") as f:
            assert "terrain" in f
            assert "fluid" in f
            # Shape: (ny, nx * ncomp) = (32, 32 * 2) for terrain
            assert f["terrain"].shape == (32, 64)
            assert f["fluid"].shape == (32, 96)
            # Check attributes
            assert float(f["terrain"].attrs["dx"]) == 10.0
            assert float(f["terrain"].attrs["x_ll"]) == 0.0
            assert float(f["fluid"].attrs["dy"]) == 10.0
            assert float(f["fluid"].attrs["y_ll"]) == 0.0
            # Check z component (component 0)
            z_comp = f["terrain"][:, 0]
            assert np.allclose(z_comp, 5.0)
            # Check h component (component 0 of fluid)
            h_comp = f["fluid"][:, 0]
            assert np.allclose(h_comp, 2.0)

    def test_eta_z_pipeline(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Test pipeline with eta (water level) + z (bathymetry)."""
        import h5py

        inp = tmp_path / "inputs"
        inp.write_text("amr.blocking_factor 16\n")

        z_data = np.full((32, 32), 0.0, dtype=np.float64)
        z_file = tmp_path / "z.npy"
        np.save(z_file, z_data)

        eta_data = np.full((32, 32), 5.0, dtype=np.float64)
        eta_file = tmp_path / "eta.npy"
        np.save(eta_file, eta_data)

        output = tmp_path / "input.h5"
        preprocess(
            input_file=inp,
            output_file=output,
            fields={"z": str(z_file), "eta": str(eta_file)},
            domain=(0.0, 0.0, 320.0, 320.0),
            cell_size=10.0,
        )

        with h5py.File(output, "r") as f:
            # h = eta - z = 5.0 - 0.0 = 5.0
            h_comp = f["fluid"][:, 0]
            assert np.allclose(h_comp, 5.0)

    def test_missing_required_field_raises(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Test that missing z and h both raise an error."""
        inp = tmp_path / "inputs"
        inp.write_text("amr.blocking_factor 16\n")

        eta_data = np.full((32, 32), 5.0)
        eta_file = tmp_path / "eta.npy"
        np.save(eta_file, eta_data)

        output = tmp_path / "input.h5"
        with pytest.raises(ValueError, match="Either 'z'"):
            preprocess(
                input_file=inp,
                output_file=output,
                fields={"eta": str(eta_file)},
                domain=(0.0, 0.0, 320.0, 320.0),
                cell_size=10.0,
            )
