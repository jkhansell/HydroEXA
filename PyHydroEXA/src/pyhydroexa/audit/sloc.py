"""Comment-aware source-line counting.

The interesting number for maintenance is not "lines in the file" but "lines
someone has to read before changing it", so the counter separates:

* ``sloc``     — lines carrying code,
* ``comments`` — lines carrying only a comment (or a docstring),
* ``blanks``   — empty lines.

C/C++ is counted with a small lexer rather than a regex, because ``//`` inside
a string literal and ``/* */`` spanning lines both break naive counters.  Python
is counted with :mod:`tokenize`, so a docstring is a comment and a triple-quoted
SQL blob inside code is still code.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path

C_EXTENSIONS = frozenset(
    {".h", ".H", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".cu"}
)
PYTHON_EXTENSIONS = frozenset({".py"})
SHELL_EXTENSIONS = frozenset({".sh", ".bash"})

#: Directories that hold generated, vendored or installed content.
DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "cmake-build-debug",
        "docs/_build",
        "_build",
        "install",
        "outputs",
        "plotfiles",
        "tpl",
        "node_modules",
        "validation_output",
    }
)


@dataclass(frozen=True)
class FileStats:
    """Line census for one source file.

    Attributes:
        path: Path as found on disk.
        language: ``"c"``, ``"python"``, ``"shell"`` or ``"text"``.
        total: Physical line count, including blanks.
        sloc: Lines carrying code.
        comments: Lines carrying only a comment or docstring.
        blanks: Blank lines.
    """

    path: Path
    language: str
    total: int
    sloc: int
    comments: int
    blanks: int

    @property
    def comment_ratio(self) -> float:
        """Comments per line of code (0 for an empty file)."""
        return self.comments / self.sloc if self.sloc else 0.0


def language_of(path: Path) -> str:
    """Classify *path* by extension."""
    suffix = path.suffix
    if suffix in C_EXTENSIONS:
        return "c"
    if suffix in PYTHON_EXTENSIONS:
        return "python"
    if suffix in SHELL_EXTENSIONS:
        return "shell"
    return "text"


def _count_c(text: str) -> tuple[int, int, int]:
    """Return ``(sloc, comment_lines, blank_lines)`` for C-like source."""
    sloc = comments = blanks = 0
    in_block = False
    has_code = has_comment = False
    line_has_content = False

    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        nxt = text[index + 1] if index + 1 < length else ""

        if char == "\n":
            if has_code:
                sloc += 1
            elif has_comment:
                comments += 1
            else:
                blanks += 1
            has_code = has_comment = line_has_content = False
            index += 1
            continue

        if in_block:
            has_comment = has_comment or not line_has_content
            line_has_content = True
            if char == "*" and nxt == "/":
                in_block = False
                index += 2
                continue
            index += 1
            continue

        if char in " \t\r":
            index += 1
            continue

        if char == "/" and nxt == "*":
            in_block = True
            has_comment = has_comment or not line_has_content
            line_has_content = True
            index += 2
            continue

        if char == "/" and nxt == "/":
            # Line comment: skip to the newline (handled by the loop).
            has_comment = has_comment or not line_has_content
            line_has_content = True
            while index < length and text[index] != "\n":
                index += 1
            continue

        if char in "\"'":
            quote = char
            has_code = True
            line_has_content = True
            index += 1
            while index < length:
                current = text[index]
                if current == "\\":
                    index += 2
                    continue
                if current == quote:
                    index += 1
                    break
                if current == "\n":  # unterminated literal: stop at EOL
                    break
                index += 1
            continue

        has_code = True
        line_has_content = True
        index += 1

    # A trailing newline closes the last line; it does not open a phantom one.
    if line_has_content:
        if has_code:
            sloc += 1
        elif has_comment:
            comments += 1
    return sloc, comments, blanks


def _count_python(path: Path) -> tuple[int, int, int]:
    """Return ``(sloc, comment_lines, blank_lines)`` using :mod:`ast` + :mod:`tokenize`.

    Docstrings are found through the AST rather than by guessing from token
    adjacency: a docstring is whatever ``ast`` says is the first statement of a
    module, class or function, which gets indented docstrings and multi-line
    ones right for free.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    blank_lines = sum(1 for line in lines if not line.strip())

    docstring_lines: set[int] = set()
    try:
        module = ast.parse(text)
    except SyntaxError:
        # Not parseable: fall back to the shell heuristic rather than lying
        # with a zero.
        sloc = sum(1 for line in lines if line.strip() and not line.lstrip().startswith("#"))
        comments = sum(1 for line in lines if line.lstrip().startswith("#"))
        return sloc, comments, len(lines) - sloc - comments

    for node in ast.walk(module):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstring_lines.update(range(first.value.lineno, first.value.end_lineno + 1))

    comment_lines: set[int] = set()
    code_lines: set[int] = set()
    skipped = {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError):
        tokens = []

    for token in tokens:
        if token.type == tokenize.COMMENT:
            comment_lines.add(token.start[0])
            continue
        if token.type in skipped:
            continue
        if token.type == tokenize.STRING and all(
            line in docstring_lines for line in range(token.start[0], token.end[0] + 1)
        ):
            # Documentation, not code: do not let the token claim the lines.
            continue
        code_lines.update(range(token.start[0], token.end[0] + 1))

    comment_lines |= docstring_lines
    sloc = len(code_lines)
    return sloc, len(comment_lines - code_lines), blank_lines


def count_sloc(path: Path) -> FileStats:
    """Count *path* and return a :class:`FileStats`."""
    language = language_of(path)
    if language == "c":
        sloc, comments, blanks = _count_c(path.read_text(encoding="utf-8", errors="replace"))
    elif language == "python":
        sloc, comments, blanks = _count_python(path)
    elif language == "shell":
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        sloc = sum(1 for line in lines if line.strip() and not line.lstrip().startswith("#"))
        comments = sum(1 for line in lines if line.lstrip().startswith("#"))
        blanks = len(lines) - sloc - comments
    else:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        sloc = sum(1 for line in lines if line.strip())
        comments = 0
        blanks = len(lines) - sloc

    return FileStats(
        path=path,
        language=language,
        total=sloc + comments + blanks,
        sloc=sloc,
        comments=comments,
        blanks=blanks,
    )


def iter_source_files(
    root: Path,
    *,
    skip_dirs: frozenset[str] = DEFAULT_SKIP_DIRS,
    suffixes: frozenset[str] = C_EXTENSIONS | PYTHON_EXTENSIONS | SHELL_EXTENSIONS,
) -> list[Path]:
    """Walk *root* and return the source files worth counting.

    Args:
        root: Directory to walk.
        skip_dirs: Directory names (or trailing path fragments) to prune.
        suffixes: File extensions to keep.

    Returns:
        Sorted absolute paths.
    """
    found: list[Path] = []
    stack = [root.resolve()]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            name = entry.name
            if entry.is_dir():
                if name in skip_dirs or name.startswith(("build_", "install_")):
                    continue
                stack.append(entry)
            elif entry.suffix in suffixes:
                found.append(entry)
    return sorted(found)
