"""Dependency graph over the C++ and Python sources.

The graph answers one question: *if I change this file, what else can break?*
An edge ``a -> b`` means "a depends on b" (a includes b, or a imports b).

Three structural facts matter for maintenance, and none of them show up in a
line count:

* **fan-in** (afferent coupling) — how many files must be re-read/rebuild when
  this one changes.  A header with fan-in 10 is an API commitment; a private
  helper with fan-in 1 is disposable.
* **fan-out** (efferent coupling) — how many things this file needs to compile.
  High fan-out is fragile: every dependency is a potential breaking change.
* **cycles** — strongly connected components larger than one node.  A cycle
  defeats incremental compilation and forces you to hold the whole knot in your
  head at once.

Includes are resolved by *path*, not by guessing: every directory named
``include`` in the tree is an include root, so ``<utils/Logging.H>`` resolves to
``include/utils/Logging.H`` regardless of which subdirectory the includer lives
in.  Python imports are resolved through :mod:`ast`, honouring relative imports
and the package root.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .sloc import (
    C_EXTENSIONS,
    DEFAULT_SKIP_DIRS,
    PYTHON_EXTENSIONS,
    iter_source_files,
)

#: ``#include <a/b.H>`` or ``#include "a/b.H"``.
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


@dataclass
class DependencyGraph:
    """Directed dependency graph with the metrics that matter for maintenance.

    Attributes:
        nodes: Sorted repository-relative POSIX paths.
        edges: ``edges[a]`` is the set of files ``a`` depends on.
        unresolved: Include/import targets that could not be mapped to a file
            in the tree (system headers, third-party packages).  Kept so the
            fan-out numbers can be checked against reality.
    """

    nodes: list[str]
    edges: dict[str, set[str]]
    unresolved: dict[str, set[str]] = field(default_factory=dict)

    # -- adjacency ---------------------------------------------------------

    def dependents(self, node: str) -> set[str]:
        """Files that directly depend on *node* (direct fan-in set)."""
        return {source for source, targets in self.edges.items() if node in targets}

    def fan_in(self, node: str) -> int:
        """Number of files that directly depend on *node*."""
        return len(self.dependents(node))

    def fan_out(self, node: str) -> int:
        """Number of files *node* directly depends on."""
        return len(self.edges.get(node, ()))

    def dependents_transitive(self, node: str) -> set[str]:
        """Everything the change of *node* can reach, directly or not.

        This is the edit-amplification number: the count of files that may need
        a second look after *node* changes.
        """
        seen: set[str] = set()
        stack = list(self.dependents(node))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.dependents(current))
        return seen

    def internal_edges(self) -> int:
        """Number of edges with both endpoints inside the tree."""
        known = set(self.nodes)
        return sum(1 for a, targets in self.edges.items() for b in targets if b in known)

    # -- structure ---------------------------------------------------------

    def strongly_connected_components(self) -> list[list[str]]:
        """Tarjan's algorithm, iterative.

        Iterative on purpose: the solver's include graph is shallow today, but
        a recursive traversal would raise ``RecursionError`` on the first
        accidental deep chain instead of reporting it.
        """
        index_of: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        components: list[list[str]] = []
        counter = 0

        for root in self.nodes:
            if root in index_of:
                continue
            work: list[tuple[str, Iterator[str]]] = [
                (root, iter(sorted(self.edges.get(root, ()))))
            ]
            index_of[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)

            while work:
                node, children = work[-1]
                advanced = False
                for child in children:
                    if child not in index_of:
                        index_of[child] = low[child] = counter
                        counter += 1
                        stack.append(child)
                        on_stack.add(child)
                        work.append((child, iter(sorted(self.edges.get(child, ())))))
                        advanced = True
                        break
                    if child in on_stack:
                        low[node] = min(low[node], index_of[child])
                if advanced:
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index_of[node]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    components.append(sorted(component))
        return components

    def cycles(self) -> list[list[str]]:
        """Strongly connected components that actually form a cycle."""
        return [c for c in self.strongly_connected_components() if len(c) > 1]

    def condensation_depth(self) -> int:
        """Longest dependency chain in the acyclic condensation.

        Reported in edges: a depth of 4 means a change at the bottom of the
        chain can force a rebuild of four layers above it.  Cycles are
        collapsed first, otherwise the "longest path" is undefined.
        """
        component_of: dict[str, int] = {}
        for number, component in enumerate(self.strongly_connected_components()):
            for node in component:
                component_of[node] = number

        dag: dict[int, set[int]] = {}
        for source, targets in self.edges.items():
            here = component_of[source]
            for target in targets:
                there = component_of.get(target)
                if there is not None and there != here:
                    dag.setdefault(here, set()).add(there)

        # Kahn's topological order, then DP on longest path.
        indegree = {node: 0 for node in dag}
        for targets in dag.values():
            for target in targets:
                indegree.setdefault(target, 0)
                indegree[target] += 1
        order: list[int] = [node for node, degree in indegree.items() if degree == 0]
        head = 0
        while head < len(order):
            current = order[head]
            head += 1
            for target in dag.get(current, ()):
                indegree[target] -= 1
                if indegree[target] == 0:
                    order.append(target)

        depth: dict[int, int] = {node: 0 for node in order}
        for node in order:
            base = depth[node]
            for target in dag.get(node, ()):
                if base + 1 > depth.get(target, 0):
                    depth[target] = base + 1
        return max(depth.values(), default=0)

    # -- reporting helpers -------------------------------------------------

    def hubs(self, limit: int = 10) -> list[tuple[str, int]]:
        """Files with the highest fan-in, most-depended-on first."""
        ranked = sorted(((node, self.fan_in(node)) for node in self.nodes), key=lambda kv: (-kv[1], kv[0]))
        return [(node, count) for node, count in ranked[:limit] if count]

    def leaves(self, limit: int = 10) -> list[tuple[str, int]]:
        """Files with the highest fan-out, most-fragile first."""
        ranked = sorted(((node, self.fan_out(node)) for node in self.nodes), key=lambda kv: (-kv[1], kv[0]))
        return [(node, count) for node, count in ranked[:limit] if count]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _include_roots(files: list[Path], root: Path) -> list[Path]:
    """Directories that headers can be named relative to.

    Any directory called ``include`` qualifies, plus the parents of a source
    package root, so this keeps working if the layout is reorganised.
    """
    roots: list[Path] = []
    for path in files:
        parent = path.parent
        if parent.name == "include":
            roots.append(parent)
        # Walk up looking for an ``include`` sibling.
        for ancestor in parent.parents:
            if ancestor == root.parent:
                break
            if ancestor.name == "include":
                roots.append(ancestor)
                break
    return sorted(set(roots))


def _resolve_include(token: str, include_roots: list[Path], known: dict[str, str]) -> str | None:
    """Map an ``#include`` token to a repository path, or ``None``."""
    for base in include_roots:
        candidate = (base / token).resolve()
        mapped = known.get(str(candidate))
        if mapped:
            return mapped
    # Fall back to a suffix match on the relative path: ``utils/Logging.H``
    # matches ``include/utils/Logging.H`` even without a registered root.
    suffix = "/" + token.replace("\\", "/")
    matches = [mapped for path, mapped in known.items() if path.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


def _python_module_name(path: Path, package_roots: list[str]) -> tuple[str, str] | None:
    """Return ``(root, dotted_module)`` for a Python file."""
    parts = path.with_suffix("").parts
    for package_root in package_roots:
        root_parts = tuple(Path(package_root).parts)
        if len(parts) > len(root_parts) and parts[: len(root_parts)] == root_parts:
            module_parts = list(parts[len(root_parts) :])
            if module_parts and module_parts[-1] == "__init__":
                module_parts.pop()
            if module_parts:
                return package_root, ".".join(module_parts)
    return None


def _python_imports(path: Path) -> list[tuple[int, str, list[str]]]:
    """Extract ``(level, module, names)`` from a Python file's import nodes."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    found: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            found.append((node.level or 0, module, [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.append((0, alias.name, []))
    return found


def build_graph(
    root: Path | str = ".",
    *,
    skip_dirs: frozenset[str] = DEFAULT_SKIP_DIRS,
) -> DependencyGraph:
    """Walk *root* and return the include/import graph for all sources.

    Args:
        root: Repository root.
        skip_dirs: Directory names to prune while walking.

    Returns:
        A :class:`DependencyGraph` whose node names are paths relative to
        *root*, so the report is stable across machines and checkout points.
    """
    root = Path(root).resolve()
    files = iter_source_files(root, skip_dirs=skip_dirs)
    known: dict[str, str] = {}
    for path in files:
        resolved = path.resolve()
        known[str(resolved)] = resolved.relative_to(root).as_posix()

    include_roots = _include_roots([p for p in files if p.suffix in C_EXTENSIONS], root)

    # Python resolution: a package root is any directory containing a package
    # (a child directory with an __init__.py) -- i.e. the "src" directory.
    package_roots: list[str] = []
    for path in files:
        if path.suffix in PYTHON_EXTENSIONS and path.name == "__init__.py":
            package_roots.append(str(path.parent.parent.resolve()))
    package_roots = sorted(set(package_roots))

    modules: dict[str, str] = {}
    for path in files:
        if path.suffix not in PYTHON_EXTENSIONS:
            continue
        resolved = path.resolve()
        facts = _python_module_name(resolved, package_roots)
        if facts:
            modules[facts[1]] = resolved.relative_to(root).as_posix()

    nodes = sorted(known.values())
    edges: dict[str, set[str]] = {node: set() for node in nodes}
    unresolved: dict[str, set[str]] = {node: set() for node in nodes}

    for path in files:
        node = known[str(path.resolve())]
        if path.suffix in C_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in _INCLUDE_RE.findall(text):
                target = _resolve_include(token, include_roots, known)
                if target is None:
                    unresolved[node].add(token)
                elif target != node:
                    edges[node].add(target)
        elif path.suffix in PYTHON_EXTENSIONS:
            facts = _python_module_name(path.resolve(), package_roots)
            if facts is None:
                continue
            _, dotted = facts
            package = dotted.rsplit(".", 1)[0] if "." in dotted else ""
            for level, module, names in _python_imports(path):
                if level:
                    # Relative import: climb level-1 packages from this one.
                    base = package.split(".") if package else []
                    climb = level - 1
                    if climb > len(base):
                        continue
                    base = base[: len(base) - climb] if climb else base
                    candidate = ".".join([*base, module]) if module else ".".join(base)
                else:
                    candidate = module
                if not candidate:
                    continue
                # A submodule named in the import wins over the package itself,
                # otherwise "from . import b" would link the package __init__.
                for name in (
                    *[f"{candidate}.{alias}" for alias in names],
                    candidate,
                ):
                    target = modules.get(name)
                    if target is not None and target != node:
                        edges[node].add(target)
                        break
                else:
                    unresolved[node].add(candidate)

    return DependencyGraph(nodes=nodes, edges=edges, unresolved=unresolved)
