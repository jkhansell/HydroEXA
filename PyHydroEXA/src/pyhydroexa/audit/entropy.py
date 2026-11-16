"""Entropy and concentration measures for the source tree.

Why entropy?  A count of 40 000 lines tells you how much code there is, not how
hard it is to hold in your head.  The same 40 000 lines split across 200 evenly
sized files is a different kind of object from the same lines pinned in three
god-files, and the difference shows up in the distribution, not in the total.

The measures here:

``shannon``
    Raw base-2 entropy of a distribution, in bits.
``normalized_entropy``
    Entropy divided by the entropy of a *uniform* distribution over the same
    number of non-empty bins.  1.0 means perfectly even; the lowest possible
    value (``1 / log2(k)``) means one bin holds everything.
``hhi``
    Herfindahl-Hirschman index, ``sum(p**2)``.  Unlike normalized entropy it is
    dominated by the *largest* share, so it answers "how much of the tree hangs
    off one file?".  Its reciprocal is the *effective number of files*: the
    count of equal-sized files that would produce the same concentration.
``instability``
    Martin's ``I = Ce / (Ca + Ce)``: 0 is a stable, widely-depended-on file
    (changing it ripples), 1 is a file that depends on everything and is
    depended on by nothing (changing it is cheap but it breaks easily).

A caveat worth stating once: entropy is scale-free, so it cannot tell a
600-line solver from a 600-line test fixture.  Read it together with the raw
SLOC and the dependency graph, never on its own.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from .sloc import FileStats


def shannon(counts: Iterable[float]) -> float:
    """Return the base-2 Shannon entropy of *counts*.

    Args:
        counts: Non-negative weights (line counts, degree counts, ...).
            Zero entries are ignored, so the number of bins is the number of
            non-zero entries.

    Returns:
        Entropy in bits.  0.0 for an empty or single-bin distribution.
    """
    weights = [float(count) for count in counts if count]
    total = sum(weights)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for weight in weights:
        share = weight / total
        entropy -= share * math.log2(share)
    return entropy


def normalized_entropy(counts: Iterable[float]) -> float:
    """Return Shannon entropy divided by the maximum for that many bins.

    Args:
        counts: Non-negative weights.

    Returns:
        ``H / log2(k)`` in ``[0, 1]``, where ``k`` is the number of non-zero
        bins.  1.0 is perfectly even; 0.0 is one bin holding everything (or one
        bin in total, where "even" is meaningless).
    """
    weights = [float(count) for count in counts if count]
    if len(weights) < 2:
        return 0.0
    ceiling = math.log2(len(weights))
    return shannon(weights) / ceiling if ceiling else 0.0


def hhi(counts: Iterable[float]) -> float:
    """Return the Herfindahl-Hirschman index of *counts*.

    Args:
        counts: Non-negative weights.

    Returns:
        ``sum((count / total) ** 2)`` in ``(0, 1]``.  ``1 / hhi`` is the
        effective number of equally sized bins.
    """
    weights = [float(count) for count in counts if count]
    total = sum(weights)
    if total <= 0:
        return 0.0
    return sum((weight / total) ** 2 for weight in weights)


def effective_count(counts: Iterable[float]) -> float:
    """Return ``1 / hhi``: the number of equally sized bins with the same mass.

    This is the plain-language version of the concentration measure -- "the
    tree behaves like 12 equally sized files" is easier to act on than "the HHI
    is 0.08".
    """
    index = hhi(counts)
    return 1.0 / index if index else 0.0


def mass_entropy(stats: Sequence[FileStats]) -> float:
    """Normalized entropy of the per-file SLOC distribution.

    Args:
        stats: Counted files.

    Returns:
        ``0`` for a tree of one file, ``1`` for perfectly even distribution.
    """
    return normalized_entropy([entry.sloc for entry in stats])


def dependency_entropy(degrees: Mapping[str, int] | Iterable[int]) -> float:
    """Normalized entropy of a degree distribution.

    Args:
        degrees: Either a mapping of node to degree, or the degrees themselves.

    Returns:
        ``0`` when every node has the same degree or no dependencies exist,
        larger when dependency counts are spread out.
    """
    values = list(degrees.values()) if isinstance(degrees, Mapping) else list(degrees)
    return normalized_entropy(values)


def instability(fan_in: int, fan_out: int) -> float:
    """Return Martin's instability ``Ce / (Ca + Ce)``.

    Args:
        fan_in: Afferent coupling -- files that depend on this one.
        fan_out: Efferent coupling -- files this one depends on.

    Returns:
        ``0.0`` for a pure dependency (nothing depends on it, it depends on
        nothing), ``1.0`` for a pure leaf.  ``0.0`` is also returned for an
        isolated node, where the ratio is undefined.
    """
    total = fan_in + fan_out
    return fan_out / total if total else 0.0
