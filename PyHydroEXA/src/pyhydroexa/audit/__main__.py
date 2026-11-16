"""Command-line entry point: ``python -m pyhydroexa.audit``.

Examples
--------
Print the Markdown report for the whole repository::

    python -m pyhydroexa.audit

Write the two artefacts CI and the docs consume::

    python -m pyhydroexa.audit --markdown docs/architecture.md --json build/audit.json

Look only at the C++ side while refactoring a solver::

    python -m pyhydroexa.audit --root include --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .report import TreeReport


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyhydroexa.audit",
        description=(
            "Measure SLOC, comment ratio and include/import coupling, then rank "
            "the structural concerns that make the tree hard to change."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root to measure (default: the current directory)",
    )
    parser.add_argument(
        "--markdown",
        metavar="PATH",
        help="write the Markdown report to PATH ('-' for stdout)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="write the machine-readable report to PATH ('-' for stdout)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="rows per ranked table (default: 10)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only report written files; no Markdown on stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the audit and return a process exit code.

    Args:
        argv: Arguments to parse; defaults to :data:`sys.argv`.

    Returns:
        ``0`` on success, ``2`` if the tree cannot be measured.
    """
    args = _build_parser().parse_args(argv)

    if not args.root.is_dir():
        print(f"error: not a directory: {args.root}", file=sys.stderr)
        return 2

    report = TreeReport.build(args.root, top=args.top)

    wrote: list[Path] = []
    if args.markdown:
        text = report.markdown()
        if args.markdown == "-":
            print(text)
        else:
            path = Path(args.markdown)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            wrote.append(path)

    if args.json:
        payload = json.dumps(report.as_dict(), indent=2)
        if args.json == "-":
            print(payload)
        else:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
            wrote.append(path)

    if not args.quiet and not args.markdown:
        print(report.markdown())

    for path in wrote:
        print(f"wrote {path}", file=sys.stderr)

    if not wrote and not args.quiet and not args.markdown:
        findings = report.findings()
        print(
            f"{len(findings)} finding(s); "
            f"{report.sloc:,} SLOC over {len(report.files)} files",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
