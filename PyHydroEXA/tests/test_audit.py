"""Tests for the source-tree audit: line counting, coupling, entropy.

The audit is itself a measuring instrument, so the tests pin down the cases
where a naive implementation lies: comment markers inside string literals,
docstrings that are indented, and relative Python imports that close a cycle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhydroexa.audit import (
    TreeReport,
    build_graph,
    count_sloc,
    effective_count,
    hhi,
    instability,
    normalized_entropy,
    shannon,
)


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestCLineCounting:
    """The C/C++ counter must not be fooled by literals or block comments."""

    def test_comment_marker_inside_string_is_code(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "a.cpp",
            'const char* url = "http://example.com";\n'
            "int x = 1;\n",
        )
        stats = count_sloc(path)
        assert stats.sloc == 2
        assert stats.comments == 0

    def test_block_comment_spanning_lines(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "b.H",
            "/* header\n"
            " * continued\n"
            " */\n"
            "int f();\n",
        )
        stats = count_sloc(path)
        assert stats.comments == 3
        assert stats.sloc == 1
        assert stats.blanks == 0

    def test_trailing_comment_counts_as_code(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "c.cpp", "int x = 1;  // the answer\n")
        stats = count_sloc(path)
        assert stats.sloc == 1
        assert stats.comments == 0

    def test_blank_and_comment_only_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "d.h", "\n// only a comment\n\n")
        stats = count_sloc(path)
        assert stats.sloc == 0
        assert stats.comments == 1
        assert stats.blanks == 2


class TestPythonLineCounting:
    """Docstrings are documentation wherever they sit."""

    def test_indented_docstring_is_a_comment(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "mod.py",
            '"""Module docstring."""\n'
            "\n"
            "def f():\n"
            '    """Function docstring."""\n'
            "    return 1\n",
        )
        stats = count_sloc(path)
        assert stats.sloc == 2  # def f(): and return 1
        assert stats.comments == 2  # both docstrings

    def test_multiline_docstring_counts_every_line(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "mod.py",
            '"""One.\n'
            "\n"
            'Two.\n'
            '"""\n'
            "x = 1\n",
        )
        stats = count_sloc(path)
        assert stats.comments == 4
        assert stats.sloc == 1

    def test_trailing_comment_counts_as_code(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "mod.py", "x = 1  # trailing\n")
        stats = count_sloc(path)
        assert stats.sloc == 1
        assert stats.comments == 0


class TestDependencyGraph:
    """Include and import resolution, cycles, and fan-in."""

    def test_c_includes_resolve_through_include_root(self, tmp_path: Path) -> None:
        _write(tmp_path, "include/a.H", "#include <b.H>\n")
        _write(tmp_path, "include/b.H", "int g();\n")
        _write(tmp_path, "src/main.cpp", "#include <a.H>\n#include <vector>\n")

        graph = build_graph(tmp_path)
        assert graph.edges["src/main.cpp"] == {"include/a.H"}
        assert graph.edges["include/a.H"] == {"include/b.H"}
        assert graph.fan_in("include/b.H") == 1
        # <vector> is a system header, not a tree file.
        assert "vector" in graph.unresolved["src/main.cpp"]

    def test_pathological_self_include_has_no_edge(self, tmp_path: Path) -> None:
        _write(tmp_path, "include/a.H", "#include <a.H>\n")
        graph = build_graph(tmp_path)
        assert graph.edges["include/a.H"] == set()
        assert graph.cycles() == []

    def test_relative_python_import_cycle(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/pkg/__init__.py", "\n")
        _write(tmp_path, "src/pkg/a.py", "from . import b\n")
        _write(tmp_path, "src/pkg/b.py", "from .a import thing\n")

        graph = build_graph(tmp_path)
        components = graph.cycles()
        assert len(components) == 1
        assert components[0] == ["src/pkg/a.py", "src/pkg/b.py"]
        # A cycle has no depth to speak of: the condensation is a single node.
        assert graph.condensation_depth() == 0

    def test_absolute_import_resolves_to_package_root(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/pkg/__init__.py", "\n")
        _write(tmp_path, "src/pkg/a.py", "import pkg.b\n")
        _write(tmp_path, "src/pkg/b.py", "x = 1\n")

        graph = build_graph(tmp_path)
        assert graph.edges["src/pkg/a.py"] == {"src/pkg/b.py"}
        assert graph.dependents_transitive("src/pkg/b.py") == {"src/pkg/a.py"}


class TestEntropy:
    """The measures have closed forms; test them against those."""

    def test_shannon_of_uniform_distribution(self) -> None:
        assert shannon([1, 1, 1, 1]) == pytest.approx(2.0)

    def test_normalized_entropy_edges(self) -> None:
        assert normalized_entropy([5, 5, 5, 5]) == pytest.approx(1.0)
        assert normalized_entropy([10]) == 0.0
        assert normalized_entropy([]) == 0.0
        assert normalized_entropy([10, 0, 0]) == 0.0  # zeros are not bins

    def test_hhi_and_effective_count(self) -> None:
        assert hhi([1, 1, 1, 1]) == pytest.approx(0.25)
        assert effective_count([1, 1, 1, 1]) == pytest.approx(4.0)
        assert hhi([10, 0]) == pytest.approx(1.0)
        assert effective_count([10, 0]) == pytest.approx(1.0)

    def test_instability_endpoints(self) -> None:
        assert instability(fan_in=5, fan_out=0) == 0.0
        assert instability(fan_in=0, fan_out=5) == 1.0
        assert instability(fan_in=1, fan_out=3) == pytest.approx(0.75)
        assert instability(fan_in=0, fan_out=0) == 0.0  # undefined, not a crash


class TestReport:
    """End-to-end: one small tree, one report, both renderings."""

    def test_report_measures_and_renders(self, tmp_path: Path) -> None:
        _write(tmp_path, "include/a.H", "#include <b.H>\nint f();\n")
        _write(tmp_path, "include/b.H", "int g();\n")
        _write(tmp_path, "src/main.cpp", "#include <a.H>\nint main() {}\n")
        _write(tmp_path, "src/huge.cpp", "int x = 1;\n" * 700)

        report = TreeReport.build(tmp_path, top=5, commit="deadbeef")

        assert report.sloc == 705
        assert report.commit == "deadbeef"
        # Paths are stored relative to the root, never absolute.
        assert all(not entry.path.is_absolute() for entry in report.files)

        kinds = {finding.kind for finding in report.findings()}
        assert "god-file" in kinds
        assert "hub" not in kinds  # fan-in is only 1

        markdown = report.markdown()
        assert "deadbeef" in markdown
        assert "src/huge.cpp" in markdown

        payload = report.as_dict()
        assert payload["totals"]["sloc"] == 705
        assert payload["graph"]["internal_edges"] == 2
        assert payload["mass"]["largest"][0]["path"] == "src/huge.cpp"
