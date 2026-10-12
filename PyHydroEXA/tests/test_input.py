"""Tests for the HydroEXA input file parser."""

from __future__ import annotations

import pytest

from pyhydroexa.preprocessing.input import ParsedInput, parse_input_file, _parse_text


class TestParseText:
    """Test parsing raw input text."""

    def test_scalar_int(self) -> None:
        text = "amr.max_level 3\n"
        result = _parse_text(text)
        assert result.amr["max_level"] == 3

    def test_scalar_float(self) -> None:
        text = "HydroEXA.max_time 0.5\n"
        result = _parse_text(text)
        assert result.hydroexa["max_time"] == 0.5

    def test_scalar_negative(self) -> None:
        text = "HydroEXA.mass_diag_freq -1.0\n"
        result = _parse_text(text)
        assert result.hydroexa["mass_diag_freq"] == -1.0

    def test_quoted_string(self) -> None:
        text = 'HydroEXA.model "Roe"\n'
        result = _parse_text(text)
        assert result.hydroexa["model"] == "Roe"

    def test_vector(self) -> None:
        text = "amr.ref_ratio 2 2 2\n"
        result = _parse_text(text)
        assert result.amr["ref_ratio"] == [2, 2, 2]

    def test_comment(self) -> None:
        text = "# This is a comment\namr.max_level 2\n"
        result = _parse_text(text)
        assert result.amr["max_level"] == 2

    def test_inline_comment(self) -> None:
        text = "amr.max_level 2 # max level\n"
        result = _parse_text(text)
        assert result.amr["max_level"] == 2

    def test_scientific_notation(self) -> None:
        text = "HydroEXA.cfl 5.0e-1\n"
        result = _parse_text(text)
        assert result.hydroexa["cfl"] == 0.5

    def test_empty_lines(self) -> None:
        text = "\n\namr.max_level 1\n\n"
        result = _parse_text(text)
        assert result.amr["max_level"] == 1

    def test_unknown_prefix_ignored(self) -> None:
        text = "unknown.param 42\namr.max_level 1\n"
        result = _parse_text(text)
        assert "param" not in result.hydroexa
        assert "param" not in result.amr
        assert "param" not in result.io
        assert result.amr["max_level"] == 1


class TestParsedInputProperties:
    """Test convenience properties on ParsedInput."""

    def test_max_time_default(self) -> None:
        result = ParsedInput(raw={})
        assert result.max_time == 1.0

    def test_model_default(self) -> None:
        result = ParsedInput(raw={})
        assert result.model == "Roe"

    def test_cfl_default(self) -> None:
        result = ParsedInput(raw={})
        assert result.cfl == 0.5

    def test_h_grad_thresh_default(self) -> None:
        result = ParsedInput(raw={})
        assert result.h_grad_thresh == [0.005]

    def test_lo_bc_default(self) -> None:
        result = ParsedInput(raw={})
        assert result.lo_bc == (0, 0)

    def test_blocking_factor_default(self) -> None:
        result = ParsedInput(raw={})
        assert result.blocking_factor == 16

    def test_terrain_ref_lev_default(self) -> None:
        result = ParsedInput(raw={})
        assert result.terrain_ref_lev == 0

    def test_ref_ratio_default(self) -> None:
        result = ParsedInput(raw={})
        assert result.ref_ratio == (2, 2, 2)

    def test_max_time_from_input(self) -> None:
        result = ParsedInput(raw={}, hydroexa={"max_time": 2.5})
        assert result.max_time == 2.5

    def test_ref_ratio_from_vector(self) -> None:
        result = ParsedInput(raw={}, amr={"ref_ratio": [2, 2, 2]})
        assert result.ref_ratio == (2, 2, 2)


class TestParseInputFile:
    """Test parsing actual files."""

    def test_parse_real_input(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        inp = tmp_path / "test.in"
        inp.write_text(
            "# Test input\n"
            "HydroEXA.max_time 0.1\n"
            "HydroEXA.model \"Roe\"\n"
            "amr.max_level 2\n"
            "amr.blocking_factor 16\n"
            "amr.terrain_ref_lev 1\n"
            "io.input_file \"input.h5\"\n"
        )
        result = parse_input_file(inp)
        assert result.max_time == 0.1
        assert result.model == "Roe"
        assert result.max_level == 2
        assert result.blocking_factor == 16
        assert result.terrain_ref_lev == 1
        assert result.input_file == "input.h5"
