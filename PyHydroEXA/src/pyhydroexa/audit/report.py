"""Assemble the SLOC and coupling measurements into one report.

The report is generated, never hand-maintained: file lists and counts are read
from the tree every time.  That is the point -- a hand-written architecture
document rots the moment someone splits a solver, while this one cannot drift
because it is rebuilt from the sources.

Two renderings, one source of truth: :meth:`TreeReport.markdown` for the docs
and :meth:`TreeReport.as_dict` for anything that wants to trend the numbers
over time (JSON, CI artifact, notebook).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from .entropy import (
    dependency_entropy,
    effective_count,
    hhi,
    instability,
    mass_entropy,
)
from .graph import DependencyGraph, build_graph
from .sloc import FileStats, count_sloc, iter_source_files

#: A file above this many SLOC no longer fits in one reading session, which is
#: the practical unit of "can I change this safely?".  Chosen from experience,
#: not from a study: most style guides land between 400 and 800.
GOD_FILE_SLOC = 600

#: Fan-in at which a file stops being a helper and becomes an API commitment.
HUB_FAN_IN = 6

#: Fan-out above which a file is likely to break whenever anything it uses
#: changes.
FRAGILE_FAN_OUT = 14

#: Transitive dependents above which a single edit has repository-wide reach.
AMPLIFIER = 10

#: How many high-fan-out and broad-ripple findings to report.  Cycles and
#: god-files are listed in full because they are always actionable; the two
#: coupling kinds produce long tails where the tenth entry adds nothing.
COUPLING_REPORTED = 3


@dataclass(frozen=True)
class Finding:
    """One ranked structural concern.

    Attributes:
        severity: Sort key, higher is more urgent.  Derived from the offending
            measurement so the ranking is reproducible, not editorial.
        kind: Short machine-readable category (``cycle``, ``god-file``, ...).
        subject: File or group the finding is about.
        detail: One sentence a reader can act on.
    """

    severity: float
    kind: str
    subject: str
    detail: str


@dataclass
class TreeReport:
    """A measured snapshot of one source tree.

    Attributes:
        root: Absolute path the report was built from.
        files: Per-file line census, sorted by descending SLOC.
        graph: Include/import graph over the same files.
        commit: Short commit hash the tree was measured at, if known.
        generated: UTC timestamp of the measurement.
        top: How many rows the ranked tables show.
    """

    root: Path
    files: list[FileStats]
    graph: DependencyGraph
    commit: str | None = None
    generated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    top: int = 10

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, root: Path | str = ".", *, top: int = 10, commit: str | None = None) -> "TreeReport":
        """Measure the tree under *root*.

        Args:
            root: Repository root to walk.
            top: Number of rows in ranked tables.
            commit: Commit hash to record.  When omitted, ``git rev-parse`` is
                tried and silently skipped if git is unavailable -- the audit
                must work on an exported tarball too.

        Returns:
            A populated report.
        """
        resolved = Path(root).resolve()
        stats = [count_sloc(path) for path in iter_source_files(resolved)]
        # Paths are stored relative so reports are comparable across machines,
        # worktrees and checkout locations.
        stats = [
            replace(entry, path=Path(entry.path.resolve().relative_to(resolved)))
            for entry in stats
        ]
        stats.sort(key=lambda entry: (-entry.sloc, str(entry.path)))
        if commit is None:
            commit = _git_commit(resolved)
        return cls(
            root=resolved,
            files=stats,
            graph=build_graph(resolved),
            commit=commit,
            top=top,
        )

    # -- derived numbers ---------------------------------------------------

    @property
    def sloc(self) -> int:
        """Total SLOC across counted files."""
        return sum(entry.sloc for entry in self.files)

    @property
    def comments(self) -> int:
        """Total comment lines across counted files."""
        return sum(entry.comments for entry in self.files)

    def by_language(self) -> dict[str, tuple[int, int, int]]:
        """Group the census by language.

        Returns:
            ``{language: (files, sloc, comment_lines)}``.
        """
        grouped: dict[str, list[int]] = {}
        for entry in self.files:
            bucket = grouped.setdefault(entry.language, [0, 0, 0])
            bucket[0] += 1
            bucket[1] += entry.sloc
            bucket[2] += entry.comments
        return {language: (a, b, c) for language, (a, b, c) in sorted(grouped.items())}

    def largest(self, limit: int | None = None) -> list[FileStats]:
        """The heaviest files by SLOC (the input is already sorted)."""
        return self.files[: limit or self.top]

    @property
    def effective_files(self) -> float:
        """Number of equally sized files with the same concentration."""
        return effective_count([entry.sloc for entry in self.files])

    def findings(self) -> list[Finding]:
        """Ranked structural concerns, most urgent first.

        The rules are deliberately blunt: each one turns a measurement into a
        decision ("this needs splitting", "this is an API now").  Anything that
        reports without demanding an action does not belong here.

        Returns:
            Findings sorted by descending severity.
        """
        found: list[Finding] = []

        for component in self.graph.cycles():
            found.append(
                Finding(
                    severity=1000 + len(component),
                    kind="cycle",
                    subject=" <-> ".join(component),
                    detail=(
                        f"{len(component)} files form an include/import cycle; "
                        "the knot has to be untangled before it can be built "
                        "or reasoned about incrementally."
                    ),
                )
            )

        for entry in self.files:
            if entry.sloc >= GOD_FILE_SLOC:
                found.append(
                    Finding(
                        severity=float(entry.sloc),
                        kind="god-file",
                        subject=str(entry.path),
                        detail=(
                            f"{entry.sloc} SLOC in one file; at "
                            f"{entry.sloc / GOD_FILE_SLOC:.1f}x the "
                            f"{GOD_FILE_SLOC}-line reading budget, split by "
                            "responsibility before extending it."
                        ),
                    )
                )

        for node, count in self.graph.hubs(limit=len(self.graph.nodes)):
            if count >= HUB_FAN_IN:
                found.append(
                    Finding(
                        severity=500 + count,
                        kind="hub",
                        subject=node,
                        detail=(
                            f"{count} files depend on it: every signature change "
                            f"is a {count}-file edit, so treat it as public API."
                        ),
                    )
                )

        for node, count in self.graph.leaves(limit=len(self.graph.nodes)):
            if count >= FRAGILE_FAN_OUT:
                found.append(
                    Finding(
                        severity=300 + count,
                        kind="fragile",
                        subject=node,
                        detail=(
                            f"depends on {count} files; it will need touching "
                            "whenever any of them changes."
                        ),
                    )
                )

        amplifiers = [
            (node, reach)
            for node in self.graph.nodes
            if len(reach := self.graph.dependents_transitive(node)) >= AMPLIFIER
        ]
        amplifiers.sort(key=lambda item: (-len(item[1]), item[0]))
        for node, reach in amplifiers[:COUPLING_REPORTED]:
            found.append(
                Finding(
                    severity=100 + len(reach),
                    kind="amplifier",
                    subject=node,
                    detail=(
                        f"a change here reaches {len(reach)} files "
                        "transitively; the blast radius is a review item."
                    ),
                )
            )

        fragile = [item for item in found if item.kind == "fragile"]
        if len(fragile) > COUPLING_REPORTED:
            fragile.sort(key=lambda finding: (-finding.severity, finding.subject))
            keep = {id(item) for item in fragile[:COUPLING_REPORTED]}
            found = [item for item in found if item.kind != "fragile" or id(item) in keep]

        found.sort(key=lambda finding: (-finding.severity, finding.kind, finding.subject))
        return found

    # -- renderings --------------------------------------------------------

    def as_dict(self) -> dict:
        """Return a JSON-serialisable view of the whole report."""
        graph = self.graph
        return {
            "root": str(self.root),
            "commit": self.commit,
            "generated": self.generated.isoformat(timespec="seconds"),
            "totals": {
                "files": len(self.files),
                "sloc": self.sloc,
                "comments": self.comments,
                "comment_ratio": round(self.comments / self.sloc, 3) if self.sloc else 0.0,
                "by_language": {
                    language: {"files": n, "sloc": sloc, "comments": comments}
                    for language, (n, sloc, comments) in self.by_language().items()
                },
            },
            "mass": {
                "entropy": round(mass_entropy(self.files), 4),
                "hhi": round(hhi([entry.sloc for entry in self.files]), 4),
                "effective_files": round(self.effective_files, 1),
                "largest": [
                    {"path": str(entry.path), "sloc": entry.sloc}
                    for entry in self.largest()
                ],
            },
            "graph": {
                "nodes": len(graph.nodes),
                "internal_edges": graph.internal_edges(),
                "cycles": graph.cycles(),
                "condensation_depth": graph.condensation_depth(),
                "dependency_entropy": round(
                    dependency_entropy({node: graph.fan_out(node) for node in graph.nodes}), 4
                ),
                "hubs": [
                    {"path": node, "fan_in": count} for node, count in graph.hubs(self.top)
                ],
                "leaves": [
                    {"path": node, "fan_out": count} for node, count in graph.leaves(self.top)
                ],
                "unstable": sorted(
                    (
                        {"path": node, "instability": round(instability(graph.fan_in(node), graph.fan_out(node)), 3)}
                        for node in graph.nodes
                        if graph.fan_in(node) or graph.fan_out(node)
                    ),
                    key=lambda row: (-row["instability"], row["path"]),
                )[: self.top],
            },
            "findings": [
                {
                    "kind": finding.kind,
                    "subject": finding.subject,
                    "detail": finding.detail,
                }
                for finding in self.findings()
            ],
        }

    def markdown(self) -> str:
        """Render the report as GitHub-flavoured Markdown."""
        lines: list[str] = []
        add = lines.append

        add("<!-- Generated by `python -m pyhydroexa.audit --markdown`. Do not edit. -->")
        add("")
        add("# Code architecture audit")
        add("")
        add(
            f"Measured at `{self.commit or 'unknown'}` on "
            f"{self.generated.strftime('%Y-%m-%d %H:%M UTC')} from `{self.root.name}`."
        )
        add("")
        add(
            "Counts are comment-aware SLOC: docstrings and comments are counted "
            "separately, blank lines are excluded. Generated and vendored "
            "trees (`tpl/`, `build*/`, `docs/_build/`, plotfiles) are skipped."
        )
        add("")

        add("## Totals")
        add("")
        add("| Language | Files | SLOC | Comment lines | Comments / SLOC |")
        add("| --- | ---: | ---: | ---: | ---: |")
        for language, (count, sloc, comments) in self.by_language().items():
            ratio = f"{comments / sloc:.2f}" if sloc else "--"
            add(f"| {language} | {count} | {sloc:,} | {comments:,} | {ratio} |")
        total_ratio = f"{self.comments / self.sloc:.2f}" if self.sloc else "--"
        add(f"| **all** | **{len(self.files)}** | **{self.sloc:,}** | **{self.comments:,}** | {total_ratio} |")
        add("")

        add("## Code mass distribution")
        add("")
        add("| Measure | Value | Reading |")
        add("| --- | ---: | --- |")
        add(
            f"| Normalized entropy | {mass_entropy(self.files):.3f} | "
            "1.0 = every file the same size; low = a few files hold the code |"
        )
        add(
            f"| Effective file count | {self.effective_files:.1f} | "
            f"the {len(self.files)} files behave like this many equal ones |"
        )
        if self.files:
            heaviest = self.files[0]
            share = heaviest.sloc / self.sloc if self.sloc else 0.0
            add(
                f"| Largest single file | {heaviest.sloc:,} SLOC | "
                f"{share:.1%} of the tree (`{heaviest.path}`) |"
            )
        add("")
        add(f"Heaviest files (top {len(self.largest())}):")
        add("")
        add("| File | SLOC | Comments |")
        add("| --- | ---: | ---: |")
        for entry in self.largest():
            add(f"| `{entry.path}` | {entry.sloc:,} | {entry.comments:,} |")
        add("")

        graph = self.graph
        add("## Dependency structure")
        add("")
        add("| Measure | Value | Reading |")
        add("| --- | ---: | --- |")
        add(f"| Files in graph | {len(graph.nodes)} | counted sources |")
        add(f"| Internal edges | {graph.internal_edges()} | include/import lines resolved inside the tree |")
        add(f"| Cycles | {len(graph.cycles())} | must be zero |")
        add(
            f"| Condensation depth | {graph.condensation_depth()} | "
            "layers a change can ripple through |"
        )
        add(
            f"| Dependency entropy | "
            f"{dependency_entropy({node: graph.fan_out(node) for node in graph.nodes}):.3f} | "
            "1.0 = dependencies spread evenly; low = a few files carry them |"
        )
        add("")

        hubs = graph.hubs(self.top)
        if hubs:
            add(f"Most depended-on files (fan-in, top {len(hubs)}):")
            add("")
            add("| File | Dependents | Instability |")
            add("| --- | ---: | ---: |")
            for node, count in hubs:
                value = instability(graph.fan_in(node), graph.fan_out(node))
                add(f"| `{node}` | {count} | {value:.2f} |")
            add("")

        leaves = graph.leaves(self.top)
        if leaves:
            add(f"Most dependent files (fan-out, top {len(leaves)}):")
            add("")
            add("| File | Dependencies |")
            add("| --- | ---: |")
            for node, count in leaves:
                add(f"| `{node}` | {count} |")
            add("")

        findings = self.findings()
        add("## Findings")
        add("")
        if not findings:
            add(
                "No cycle, no god-file above "
                f"{GOD_FILE_SLOC} SLOC, no hub above {HUB_FAN_IN} dependents, "
                f"no file above {FRAGILE_FAN_OUT} dependencies, and no edit "
                f"reaching {AMPLIFIER} files."
            )
        else:
            reads = {
                "cycle": "cycle",
                "god-file": "god-file",
                "hub": "high fan-in",
                "fragile": "high fan-out",
                "amplifier": "broad ripple",
            }
            add("| # | Kind | Subject | Why it matters |")
            add("| ---: | --- | --- | --- |")
            for position, finding in enumerate(findings, start=1):
                add(
                    f"| {position} | {reads.get(finding.kind, finding.kind)} | "
                    f"`{finding.subject}` | {finding.detail} |"
                )
        add("")
        return "\n".join(lines)


def _git_commit(root: Path) -> str | None:
    """Return the short HEAD hash for *root*, or ``None`` if git is unusable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
